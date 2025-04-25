import os
import torch
from transformers import pipeline
import time
import traceback
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uvicorn
import nest_asyncio
from pyngrok import ngrok

# --- 設定 ---
# モデル名を設定
MODEL_NAME = "google/gemma-2-2b-jpn-it"  # お好みのモデルに変更可能です
print(f"モデル名を設定: {MODEL_NAME}")

# --- モデル設定クラス ---
class Config:
    def __init__(self, model_name=MODEL_NAME):
        self.MODEL_NAME = model_name

config = Config(MODEL_NAME)

# --- FastAPIアプリケーション定義 ---
app = FastAPI(
    title="ローカルLLM APIサービス",
    description="transformersモデルを使用したテキスト生成のためのAPI",
    version="1.0.0"
)

# CORSミドルウェアを追加
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- データモデル定義 ---
class Message(BaseModel):
    role: str
    content: str

# 直接プロンプトを使用した簡略化されたリクエスト
class SimpleGenerationRequest(BaseModel):
    prompt: str
    max_new_tokens: Optional[int] = 512
    do_sample: Optional[bool] = True
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9

class GenerationResponse(BaseModel):
    generated_text: str
    response_time: float

# --- モデル関連の関数 ---
# モデルのグローバル変数
model = None

def load_model():
    """推論用のLLMモデルを読み込む"""
    global model  # グローバル変数を更新するために必要
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用デバイス: {device}")
        pipe = pipeline(
            "text-generation",
            model=config.MODEL_NAME,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device=device
        )
        print(f"モデル '{config.MODEL_NAME}' の読み込みに成功しました")
        model = pipe  # グローバル変数を更新
        return pipe
    except Exception as e:
        error_msg = f"モデル '{config.MODEL_NAME}' の読み込みに失敗: {e}"
        print(error_msg)
        traceback.print_exc()  # 詳細なエラー情報を出力
        return None

def extract_assistant_response(outputs, user_prompt):
    """モデルの出力からアシスタントの応答を抽出する"""
    assistant_response = ""
    try:
        if outputs and isinstance(outputs, list) and len(outputs) > 0 and outputs[0].get("generated_text"):
            generated_output = outputs[0]["generated_text"]
            
            if isinstance(generated_output, list):
                # メッセージフォーマットの場合
                if len(generated_output) > 0:
                    last_message = generated_output[-1]
                    if isinstance(last_message, dict) and last_message.get("role") == "assistant":
                        assistant_response = last_message.get("content", "").strip()
                    else:
                        # 予期しないリスト形式の場合は最後の要素を文字列として試行
                        print(f"警告: 最後のメッセージの形式が予期しないリスト形式です: {last_message}")
                        assistant_response = str(last_message).strip()

            elif isinstance(generated_output, str):
                # 文字列形式の場合
                full_text = generated_output
                
                # 単純なプロンプト入力の場合、プロンプト後の全てを抽出
                if user_prompt:
                    prompt_end_index = full_text.find(user_prompt)
                    if prompt_end_index != -1:
                        prompt_end_pos = prompt_end_index + len(user_prompt)
                        assistant_response = full_text[prompt_end_pos:].strip()
                    else:
                        # 元のプロンプトが見つからない場合は、生成されたテキストをそのまま返す
                        assistant_response = full_text
                else:
                    assistant_response = full_text
            else:
                print(f"警告: 予期しない出力タイプ: {type(generated_output)}")
                assistant_response = str(generated_output).strip()  # 文字列に変換

    except Exception as e:
        print(f"応答の抽出中にエラーが発生しました: {e}")
        traceback.print_exc()
        assistant_response = "応答の抽出に失敗しました。"  # エラーメッセージを設定

    if not assistant_response:
        print("警告: アシスタントの応答を抽出できませんでした。完全な出力:", outputs)
        # デフォルトまたはエラー応答を返す
        assistant_response = "応答を生成できませんでした。"

    return assistant_response

# --- FastAPIエンドポイント定義 ---
@app.on_event("startup")
async def startup_event():
    """起動時にモデルを初期化"""
    load_model_task()  # バックグラウンドではなく同期的に読み込む
    if model is None:
        print("警告: 起動時にモデルの初期化に失敗しました")
    else:
        print("起動時にモデルの初期化が完了しました。")

@app.get("/")
async def root():
    """基本的なAPIチェック用のルートエンドポイント"""
    return {"status": "ok", "message": "Local LLM API is runnning"}

@app.get("/health")
async def health_check():
    """ヘルスチェックエンドポイント"""
    global model
    if model is None:
        return {"status": "error", "message": "No model loaded"}

    return {"status": "ok", "model": config.MODEL_NAME}

# 簡略化されたエンドポイント
@app.post("/generate", response_model=GenerationResponse)
async def generate_simple(request: SimpleGenerationRequest):
    """単純なプロンプト入力に基づいてテキストを生成"""
    global model

    if model is None:
        print("generateエンドポイント: モデルが読み込まれていません。読み込みを試みます...")
        load_model_task()  # 再度読み込みを試みる
        if model is None:
            print("generateエンドポイント: モデルの読み込みに失敗しました。")
            raise HTTPException(status_code=503, detail="モデルが利用できません。後でもう一度お試しください。")

    try:
        start_time = time.time()
        print(f"シンプルなリクエストを受信: prompt={request.prompt[:100]}..., max_new_tokens={request.max_new_tokens}")  # 長いプロンプトは切り捨て

        # プロンプトテキストで直接応答を生成
        print("モデル推論を開始...")
        outputs = model(
            request.prompt,
            max_new_tokens=request.max_new_tokens,
            do_sample=request.do_sample,
            temperature=request.temperature,
            top_p=request.top_p,
        )
        print("モデル推論が完了しました。")

        # アシスタント応答を抽出
        assistant_response = extract_assistant_response(outputs, request.prompt)
        print(f"抽出されたアシスタント応答: {assistant_response[:100]}...")  # 長い場合は切り捨て

        end_time = time.time()
        response_time = end_time - start_time
        print(f"応答生成時間: {response_time:.2f}秒")

        return GenerationResponse(
            generated_text=assistant_response,
            response_time=response_time
        )

    except Exception as e:
        print(f"シンプル応答生成中にエラーが発生しました: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"応答の生成中にエラーが発生しました: {str(e)}")


class ConversationRequest(BaseModel):
    messages: List[Message]

class ConversationResponse(BaseModel):
    response: str
    response_time: float

@app.post("/predict", response_model=ConversationResponse)
async def predict_conversation(request: ConversationRequest):
    """messages形式を受け取って推論するエンドポイント"""
    global model

    if model is None:
        print("predictエンドポイント: モデルが読み込まれていません。")
        load_model_task()
        if model is None:
            raise HTTPException(status_code=503, detail="モデルが利用できません。")

    try:
        start_time = time.time()

        # すべてのmessagesを1つのプロンプトに連結
        prompt = ""
        for message in request.messages:
            role = message.role
            content = message.content
            if role == "user":
                prompt += f"ユーザー: {content}\n"
            elif role == "assistant":
                prompt += f"アシスタント: {content}\n"
        
        # 最後のユーザー発言に対するアシスタント応答を生成
        prompt += "アシスタント:"

        print(f"生成用プロンプト:\n{prompt}")

        outputs = model(
            prompt,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )

        assistant_response = extract_assistant_response(outputs, prompt)

        end_time = time.time()
        response_time = end_time - start_time

        return ConversationResponse(
            response=assistant_response,
            response_time=response_time
        )

    except Exception as e:
        print(f"predictエンドポイントでエラー発生: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="推論中にエラーが発生しました。")


def load_model_task():
    """モデルを読み込むバックグラウンドタスク"""
    global model
    print("load_model_task: モデルの読み込みを開始...")
    # load_model関数を呼び出し、結果をグローバル変数に設定
    loaded_pipe = load_model()
    if loaded_pipe:
        model = loaded_pipe  # グローバル変数を更新
        print("load_model_task: モデルの読み込みが完了しました。")
    else:
        print("load_model_task: モデルの読み込みに失敗しました。")

print("FastAPIエンドポイントを定義しました。")

# --- ngrokでAPIサーバーを実行する関数 ---
def run_with_ngrok(port=8501):
    """ngrokでFastAPIアプリを実行"""
    nest_asyncio.apply()

    ngrok_token = os.environ.get("NGROK_TOKEN")
    if not ngrok_token:
        print("Ngrok認証トークンが'NGROK_TOKEN'環境変数に設定されていません。")
        try:
            print("Colab Secrets(左側の鍵アイコン)で'NGROK_TOKEN'を設定することをお勧めします。")
            ngrok_token = input("Ngrok認証トークンを入力してください (https://dashboard.ngrok.com/get-started/your-authtoken): ")
        except EOFError:
            print("\nエラー: 対話型入力が利用できません。")
            print("Colab Secretsを使用するか、ノートブックセルで`os.environ['NGROK_TOKEN'] = 'あなたのトークン'`でトークンを設定してください")
            return

    if not ngrok_token:
        print("エラー: Ngrok認証トークンを取得できませんでした。中止します。")
        return

    try:
        ngrok.set_auth_token(ngrok_token)

        # 既存のngrokトンネルを閉じる
        try:
            tunnels = ngrok.get_tunnels()
            if tunnels:
                print(f"{len(tunnels)}個の既存トンネルが見つかりました。閉じています...")
                for tunnel in tunnels:
                    print(f"  - 切断中: {tunnel.public_url}")
                    ngrok.disconnect(tunnel.public_url)
                print("すべての既存ngrokトンネルを切断しました。")
            else:
                print("アクティブなngrokトンネルはありません。")
        except Exception as e:
            print(f"トンネル切断中にエラーが発生しました: {e}")
            # エラーにもかかわらず続行を試みる

        # 新しいngrokトンネルを開く
        print(f"ポート{port}に新しいngrokトンネルを開いています...")
        ngrok_tunnel = ngrok.connect(port)
        public_url = ngrok_tunnel.public_url
        print("---------------------------------------------------------------------")
        print(f"公開URL:   {public_url}")
        print(f"APIドキュメント (Swagger UI): {public_url}/docs")
        print("---------------------------------------------------------------------")
        print("(APIクライアントやブラウザからアクセスするためにこのURLをコピーしてください)")
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")  # ログレベルをinfoに設定

    except Exception as e:
        print(f"\n ngrokまたはUvicornの起動中にエラーが発生しました: {e}")
        traceback.print_exc()
        # エラー後に残る可能性のあるngrokトンネルを閉じようとする
        try:
            print("エラーにより残っている可能性のあるngrokトンネルを閉じています...")
            tunnels = ngrok.get_tunnels()
            for tunnel in tunnels:
                ngrok.disconnect(tunnel.public_url)
            print("ngrokトンネルを閉じました。")
        except Exception as ne:
            print(f"ngrokトンネルのクリーンアップ中に別のエラーが発生しました: {ne}")

# --- メイン実行ブロック ---
if __name__ == "__main__":
    # 指定されたポートでサーバーを起動
    run_with_ngrok(port=8501)  # このポート番号を確認
    # run_with_ngrokが終了したときにメッセージを表示
    print("\nサーバープロセスが終了しました。")
