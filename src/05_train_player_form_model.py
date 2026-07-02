from pathlib import Path

import pandas as pd
import numpy as np

from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PROJECT_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_DIR / "data" / "model_ready" / "euro2024_player_form_model_ready.parquet"

MODELS_DIR = PROJECT_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

REPORTS_DIR = PROJECT_DIR / "data" / "model_ready"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = MODELS_DIR / "player_form_catboost.cbm"
FEATURE_IMPORTANCE_FILE = REPORTS_DIR / "player_form_feature_importance.csv"
PREDICTIONS_FILE = REPORTS_DIR / "player_form_predictions.csv"


def main():
    print("=== Обучение Player Form Model ===")

    if not INPUT_FILE.exists():
        print("Ошибка: файл с данными не найден")
        print(INPUT_FILE)
        return

    df = pd.read_parquet(INPUT_FILE)

    print("Данные загружены")
    print("Строк:", len(df))
    print("Столбцов:", len(df.columns))

    # Дату переводим в формат даты, чтобы делить train/test по времени
    df["match_date"] = pd.to_datetime(df["match_date"])

    # Целевая переменная
    target_col = "form_score"

    # ВАЖНО:
    # Не используем norm-колонки, потому что они напрямую участвовали в расчёте form_score.
    # Иначе модель будет слишком легко угадывать ответ.
    exclude_cols = [
        "competition_name",
        "season_name",
        "match_id",
        "match_date",
        "player_id",
        "player_name",
        "team_id",
        "form_score",
        "form_level"
    ]

    # Убираем все нормализованные технические признаки
    exclude_cols += [col for col in df.columns if col.endswith("_norm")]

    feature_cols = [col for col in df.columns if col not in exclude_cols]

    # Категориальные признаки
    cat_cols = [
        "team_name",
        "opponent_team",
        "position_name",
        "competition_stage",
        "home_team",
        "away_team"
    ]

    cat_cols = [col for col in cat_cols if col in feature_cols]

    # Числовые признаки — всё остальное
    num_cols = [col for col in feature_cols if col not in cat_cols]

    print("\nПризнаки для модели:")
    print("Всего признаков:", len(feature_cols))
    print("Категориальные:", cat_cols)
    print("Числовых:", len(num_cols))

    # Заполняем пропуски
    for col in cat_cols:
        df[col] = df[col].fillna("unknown").astype(str)

    for col in num_cols:
        df[col] = df[col].fillna(0)

    # Делим не случайно, а по датам матчей.
    # Старые матчи идут в обучение, более поздние — в тест.
    unique_dates = sorted(df["match_date"].unique())

    split_index = int(len(unique_dates) * 0.8)
    train_dates = unique_dates[:split_index]
    test_dates = unique_dates[split_index:]

    train_df = df[df["match_date"].isin(train_dates)].copy()
    test_df = df[df["match_date"].isin(test_dates)].copy()

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]

    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    print("\nРазделение данных:")
    print("Дат всего:", len(unique_dates))
    print("Train дат:", len(train_dates))
    print("Test дат:", len(test_dates))
    print("Train строк:", len(train_df))
    print("Test строк:", len(test_df))

    # Индексы категориальных признаков для CatBoost
    cat_feature_indices = [feature_cols.index(col) for col in cat_cols]

    train_pool = Pool(
        X_train,
        y_train,
        cat_features=cat_feature_indices
    )

    test_pool = Pool(
        X_test,
        y_test,
        cat_features=cat_feature_indices
    )

    # CatBoost — основная сильная модель для табличных данных.
    # Пока параметры умеренные, чтобы всё быстро работало на MacBook.
    model = CatBoostRegressor(
        iterations=800,
        learning_rate=0.03,
        depth=6,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=42,
        verbose=100,
        early_stopping_rounds=80
    )

    print("\nНачинаем обучение CatBoost...")

    model.fit(
        train_pool,
        eval_set=test_pool,
        use_best_model=True
    )

    print("\nОбучение завершено")

    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print("\n=== Качество модели на тесте ===")
    print("MAE:", round(mae, 4))
    print("RMSE:", round(rmse, 4))
    print("R2:", round(r2, 4))

    # Сохраняем модель
    model.save_model(str(MODEL_FILE))

    print("\nМодель сохранена:")
    print(MODEL_FILE)

    # Важность признаков
    importance = model.get_feature_importance(train_pool)
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": importance
    }).sort_values("importance", ascending=False)

    importance_df.to_csv(FEATURE_IMPORTANCE_FILE, index=False)

    print("\nТоп-20 важных признаков:")
    print(importance_df.head(20).to_string(index=False))

    print("\nВажность признаков сохранена:")
    print(FEATURE_IMPORTANCE_FILE)

    # Сохраняем предсказания
    pred_df = test_df[
        [
            "player_name",
            "team_name",
            "opponent_team",
            "match_date",
            "competition_stage",
            "position_name",
            "form_score"
        ]
    ].copy()

    pred_df["predicted_form_score"] = preds
    pred_df["error"] = pred_df["form_score"] - pred_df["predicted_form_score"]

    pred_df = pred_df.sort_values("predicted_form_score", ascending=False)

    pred_df.to_csv(PREDICTIONS_FILE, index=False)

    print("\nПримеры предсказаний:")
    print(pred_df.head(20).to_string(index=False))

    print("\nПредсказания сохранены:")
    print(PREDICTIONS_FILE)


if __name__ == "__main__":
    main()
