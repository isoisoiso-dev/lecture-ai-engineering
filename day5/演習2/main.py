import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import pickle
import time
import great_expectations as gx

# --- 既存 import に追加 ---
import json
from typing import Dict

# --- 新規クラス ------------------------------------------------------
class ModelComparator:
    """現行モデルとベースラインモデルの比較を行うクラス"""

    # ----------------- 内部ユーティリティ -----------------
    @staticmethod
    def _measure(model, X_test, y_test) -> Dict[str, float]:
        """精度と推論時間を同時取得"""
        start = time.time()
        y_pred = model.predict(X_test)
        elapsed = time.time() - start
        acc = accuracy_score(y_test, y_pred)
        return {"accuracy": acc, "inference_time": elapsed}

    # ----------------- 外部 API ---------------------------
    @staticmethod
    def load_baseline_metrics(path: str = "models/baseline_metrics.json"):
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return json.load(f)

    @staticmethod
    def save_metrics(metrics: Dict[str, float],
                     path: str = "models/baseline_metrics.json"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(metrics, f, indent=2)

    @staticmethod
    def compare(current: Dict[str, float],
                baseline: Dict[str, float] | None,
                acc_tol: float = 0.01,
                time_ratio: float = 1.2) -> bool:
        """
        精度 drop が acc_tol 以下、かつ推論時間が time_ratio 倍以内なら OK
        baseline が無い場合は常に True
        """
        if baseline is None:
            return True
        return (
            current["accuracy"] >= baseline["accuracy"] - acc_tol
            and current["inference_time"] <= baseline["inference_time"] * time_ratio
        )



class DataLoader:
    """データロードを行うクラス"""

    @staticmethod
    def load_titanic_data(path=None):
        """Titanicデータセットを読み込む"""
        if path:
            return pd.read_csv(path)
        else:
            # ローカルのファイル
            local_path = "data/Titanic.csv"
            if os.path.exists(local_path):
                return pd.read_csv(local_path)

    @staticmethod
    def preprocess_titanic_data(data):
        """Titanicデータを前処理する"""
        # 必要な特徴量を選択
        data = data.copy()

        # 不要な列を削除
        columns_to_drop = []
        for col in ["PassengerId", "Name", "Ticket", "Cabin"]:
            if col in data.columns:
                columns_to_drop.append(col)

        if columns_to_drop:
            data.drop(columns_to_drop, axis=1, inplace=True)

        # 目的変数とその他を分離
        if "Survived" in data.columns:
            y = data["Survived"]
            X = data.drop("Survived", axis=1)
            return X, y
        else:
            return data, None


class DataValidator:
    """データバリデーションを行うクラス"""

    @staticmethod
    def validate_titanic_data(data):
        """Titanicデータセットの検証"""
        # DataFrameに変換
        if not isinstance(data, pd.DataFrame):
            return False, ["データはpd.DataFrameである必要があります"]

        # Great Expectationsを使用したバリデーション
        try:
            context = gx.get_context()
            data_source = context.data_sources.add_pandas("pandas")
            data_asset = data_source.add_dataframe_asset(name="pd dataframe asset")

            batch_definition = data_asset.add_batch_definition_whole_dataframe(
                "batch definition"
            )
            batch = batch_definition.get_batch(batch_parameters={"dataframe": data})

            results = []

            # 必須カラムの存在確認
            required_columns = [
                "Pclass",
                "Sex",
                "Age",
                "SibSp",
                "Parch",
                "Fare",
                "Embarked",
            ]
            missing_columns = [
                col for col in required_columns if col not in data.columns
            ]
            if missing_columns:
                print(f"警告: 以下のカラムがありません: {missing_columns}")
                return False, [{"success": False, "missing_columns": missing_columns}]

            expectations = [
                gx.expectations.ExpectColumnDistinctValuesToBeInSet(
                    column="Pclass", value_set=[1, 2, 3]
                ),
                gx.expectations.ExpectColumnDistinctValuesToBeInSet(
                    column="Sex", value_set=["male", "female"]
                ),
                gx.expectations.ExpectColumnValuesToBeBetween(
                    column="Age", min_value=0, max_value=100
                ),
                gx.expectations.ExpectColumnValuesToBeBetween(
                    column="Fare", min_value=0, max_value=600
                ),
                gx.expectations.ExpectColumnDistinctValuesToBeInSet(
                    column="Embarked", value_set=["C", "Q", "S", ""]
                ),
            ]

            for expectation in expectations:
                result = batch.validate(expectation)
                results.append(result)

            # すべての検証が成功したかチェック
            is_successful = all(result.success for result in results)
            return is_successful, results

        except Exception as e:
            print(f"Great Expectations検証エラー: {e}")
            return False, [{"success": False, "error": str(e)}]


class ModelTester:
    """モデルテストを行うクラス"""

    @staticmethod
    def create_preprocessing_pipeline():
        """前処理パイプラインを作成"""
        numeric_features = ["Age", "Fare", "SibSp", "Parch"]
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_features = ["Pclass", "Sex", "Embarked"]
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ],
            remainder="drop",  # 指定されていない列は削除
        )
        return preprocessor

    @staticmethod
    def train_model(X_train, y_train, model_params=None):
        """モデルを学習する"""
        if model_params is None:
            model_params = {"n_estimators": 100, "random_state": 42}

        # 前処理パイプラインを作成
        preprocessor = ModelTester.create_preprocessing_pipeline()

        # モデル作成
        model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", RandomForestClassifier(**model_params)),
            ]
        )

        # 学習
        model.fit(X_train, y_train)
        return model

    @staticmethod
    def evaluate_model(model, X_test, y_test):
        """モデルを評価する"""
        start_time = time.time()
        y_pred = model.predict(X_test)
        inference_time = time.time() - start_time

        accuracy = accuracy_score(y_test, y_pred)
        return {"accuracy": accuracy, "inference_time": inference_time}

    @staticmethod
    def save_model(model, path="models/titanic_model.pkl"):
        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"titanic_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        return path

    @staticmethod
    def load_model(path="models/titanic_model.pkl"):
        """モデルを読み込む"""
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model

    @staticmethod
    def compare_with_baseline(current_metrics, baseline_threshold=0.75):
        """ベースラインと比較する"""
        return current_metrics["accuracy"] >= baseline_threshold


# テスト関数（pytestで実行可能）
def test_data_validation():
    """データバリデーションのテスト"""
    # データロード
    data = DataLoader.load_titanic_data()
    X, y = DataLoader.preprocess_titanic_data(data)

    # 正常なデータのチェック
    success, results = DataValidator.validate_titanic_data(X)
    assert success, "データバリデーションに失敗しました"

    # 異常データのチェック
    bad_data = X.copy()
    bad_data.loc[0, "Pclass"] = 5  # 明らかに範囲外の値
    success, results = DataValidator.validate_titanic_data(bad_data)
    assert not success, "異常データをチェックできませんでした"


def test_model_performance():
    """モデル性能のテスト"""
    # データ準備
    data = DataLoader.load_titanic_data()
    X, y = DataLoader.preprocess_titanic_data(data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # モデル学習
    model = ModelTester.train_model(X_train, y_train)

    # 評価
    metrics = ModelTester.evaluate_model(model, X_test, y_test)

    # ベースラインとの比較
    assert ModelTester.compare_with_baseline(
        metrics, 0.75
    ), f"モデル性能がベースラインを下回っています: {metrics['accuracy']}"

    # 推論時間の確認
    assert (
        metrics["inference_time"] < 1.0
    ), f"推論時間が長すぎます: {metrics['inference_time']}秒"


import sys


if __name__ == "__main__":
    # ─────────────────────────────────────
    # 1. データロード & 前処理
    # ─────────────────────────────────────
    data = DataLoader.load_titanic_data()
    X, y = DataLoader.preprocess_titanic_data(data)

    # ─────────────────────────────────────
    # 2. データバリデーション
    # ─────────────────────────────────────
    success, results = DataValidator.validate_titanic_data(X)
    print(f"データ検証結果: {'成功' if success else '失敗'}")
    for result in results:
        if not result["success"]:
            print(f"異常タイプ: {result['expectation_config']['type']}, 結果: {result}")
    if not success:
        print("データ検証に失敗しました。処理を終了します。")
        sys.exit(1)

    # ─────────────────────────────────────
    # 3. モデルの学習・評価
    # ─────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model_params = {"n_estimators": 100, "random_state": 42}
    model = ModelTester.train_model(X_train, y_train, model_params)

    current_metrics = ModelTester.evaluate_model(model, X_test, y_test)
    print(f"精度: {current_metrics['accuracy']:.4f}")
    print(f"推論時間: {current_metrics['inference_time']:.4f} 秒")

    # ─────────────────────────────────────
    # 4. ベースラインとの比較
    # ─────────────────────────────────────
    baseline_metrics = ModelComparator.load_baseline_metrics()
    is_regression_free = ModelComparator.compare(
        current_metrics,
        baseline_metrics,
        acc_tol=0.01,      # 精度 1pt 以内の低下を許容
        time_ratio=1.2     # 推論時間 20% 以内の悪化を許容
    )
    print(f"ベースライン比較: {'合格' if is_regression_free else '不合格'}")

    if not is_regression_free:
        sys.exit(1)  # CI を fail させる

    # ─────────────────────────────────────
    # 5. モデル／メトリクスの保存
    # ─────────────────────────────────────
    ModelTester.save_model(model, path="models/titanic_model.pkl")

    # ベースラインが無い、または精度が改善した場合のみ更新
    if baseline_metrics is None or current_metrics["accuracy"] > baseline_metrics["accuracy"]:
        ModelComparator.save_metrics(current_metrics)
