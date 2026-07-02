from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_DIR
    / "data"
    / "raw"
    / "skillcorner"
    / "data"
    / "aggregates"
    / "aus1league_physicalaggregates_20242025.csv"
)

OUTPUT_DIR = PROJECT_DIR / "data" / "model_ready"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PARQUET = OUTPUT_DIR / "skillcorner_player_load_model_ready.parquet"
OUTPUT_CSV = OUTPUT_DIR / "skillcorner_player_load_model_ready_sample.csv"


def safe_minmax(series):
    min_value = series.min()
    max_value = series.max()

    if pd.isna(min_value) or pd.isna(max_value):
        return pd.Series(0, index=series.index)

    if max_value == min_value:
        return pd.Series(0, index=series.index)

    return (series - min_value) / (max_value - min_value)


def safe_divide(a, b):
    return np.where(b > 0, a / b, 0)


def main():
    print("=== SkillCorner Load Agent ===")
    print("Файл:", INPUT_FILE)

    if not INPUT_FILE.exists():
        print("Ошибка: файл SkillCorner не найден")
        return

    df = pd.read_csv(INPUT_FILE)

    print("Исходная таблица:")
    print("Строк:", len(df))
    print("Столбцов:", len(df.columns))

    numeric_cols = [
        "minutes_full_all",
        "count_match",
        "total_distance_full_all",
        "total_metersperminute_full_all",
        "running_distance_full_all",
        "hsr_distance_full_all",
        "hsr_count_full_all",
        "sprint_distance_full_all",
        "sprint_count_full_all",
        "hi_distance_full_all",
        "hi_count_full_all",
        "medaccel_count_full_all",
        "highaccel_count_full_all",
        "meddecel_count_full_all",
        "highdecel_count_full_all",
        "explacceltohsr_count_full_all",
        "explacceltosprint_count_full_all",
        "psv99",
        "psv99_top5"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    matches = df["count_match"].replace(0, np.nan)

    # Основные признаки нагрузки
    df["distance_per_match"] = safe_divide(df["total_distance_full_all"], matches)
    df["running_per_match"] = safe_divide(df["running_distance_full_all"], matches)
    df["hsr_per_match"] = safe_divide(df["hsr_distance_full_all"], matches)
    df["sprint_per_match"] = safe_divide(df["sprint_distance_full_all"], matches)
    df["hi_per_match"] = safe_divide(df["hi_distance_full_all"], matches)

    df["hsr_count_per_match"] = safe_divide(df["hsr_count_full_all"], matches)
    df["sprint_count_per_match"] = safe_divide(df["sprint_count_full_all"], matches)
    df["hi_count_per_match"] = safe_divide(df["hi_count_full_all"], matches)

    df["high_accel_per_match"] = safe_divide(df["highaccel_count_full_all"], matches)
    df["high_decel_per_match"] = safe_divide(df["highdecel_count_full_all"], matches)

    df["accel_decel_load_per_match"] = safe_divide(
        df["medaccel_count_full_all"]
        + df["highaccel_count_full_all"]
        + df["meddecel_count_full_all"]
        + df["highdecel_count_full_all"],
        matches
    )

    df["high_intensity_share"] = safe_divide(
        df["hi_distance_full_all"],
        df["total_distance_full_all"].replace(0, np.nan)
    )

    df["sprint_share"] = safe_divide(
        df["sprint_distance_full_all"],
        df["total_distance_full_all"].replace(0, np.nan)
    )

    # psv99 — почти максимальная скорость игрока
    df["top_speed"] = df["psv99"]

    # Заполняем возможные пропуски
    new_numeric_cols = [
        "distance_per_match",
        "running_per_match",
        "hsr_per_match",
        "sprint_per_match",
        "hi_per_match",
        "hsr_count_per_match",
        "sprint_count_per_match",
        "hi_count_per_match",
        "high_accel_per_match",
        "high_decel_per_match",
        "accel_decel_load_per_match",
        "high_intensity_share",
        "sprint_share",
        "top_speed"
    ]

    for col in new_numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Нормализуем признаки от 0 до 1
    score_cols = [
        "distance_per_match",
        "total_metersperminute_full_all",
        "hsr_per_match",
        "sprint_per_match",
        "hi_count_per_match",
        "high_accel_per_match",
        "high_decel_per_match",
        "accel_decel_load_per_match",
        "top_speed"
    ]

    for col in score_cols:
        df[col + "_norm"] = safe_minmax(df[col])

    # Индекс нагрузки.
    # Это НЕ медицинский диагноз, а условная аналитическая оценка интенсивности.
    df["load_index"] = (
        15 * df["distance_per_match_norm"]
        + 15 * df["total_metersperminute_full_all_norm"]
        + 15 * df["hsr_per_match_norm"]
        + 15 * df["sprint_per_match_norm"]
        + 10 * df["hi_count_per_match_norm"]
        + 10 * df["high_accel_per_match_norm"]
        + 10 * df["high_decel_per_match_norm"]
        + 5 * df["accel_decel_load_per_match_norm"]
        + 5 * df["top_speed_norm"]
    )

    df["load_index"] = df["load_index"].clip(0, 100)

    df["load_level"] = pd.cut(
        df["load_index"],
        bins=[-1, 33, 66, 100],
        labels=["low", "medium", "high"]
    ).astype(str)

    df["load_comment"] = np.where(
        df["load_index"] >= 66,
        "Высокая физическая нагрузка. Стоит аккуратно планировать минуты.",
        np.where(
            df["load_index"] >= 33,
            "Средняя нагрузка. Игрок активно вовлечён в интенсивные действия.",
            "Низкая нагрузка по сравнению с другими игроками датасета."
        )
    )

    keep_cols = [
        "player_name",
        "player_short_name",
        "player_id",
        "player_birthdate",
        "team_name",
        "team_id",
        "competition_name",
        "season_name",
        "position_group",
        "minutes_full_all",
        "count_match",
        "distance_per_match",
        "total_metersperminute_full_all",
        "running_per_match",
        "hsr_per_match",
        "sprint_per_match",
        "hi_per_match",
        "hsr_count_per_match",
        "sprint_count_per_match",
        "hi_count_per_match",
        "high_accel_per_match",
        "high_decel_per_match",
        "accel_decel_load_per_match",
        "high_intensity_share",
        "sprint_share",
        "top_speed",
        "load_index",
        "load_level",
        "load_comment"
    ]

    result = df[keep_cols].copy()
    result = result.sort_values("load_index", ascending=False)

    result.to_parquet(OUTPUT_PARQUET, index=False)
    result.head(300).to_csv(OUTPUT_CSV, index=False)

    print("\nИтоговая таблица нагрузки:")
    print("Строк:", len(result))
    print("Столбцов:", len(result.columns))

    print("\nТоп-20 игроков по load_index:")
    print(
        result[
            [
                "player_name",
                "team_name",
                "position_group",
                "count_match",
                "distance_per_match",
                "sprint_per_match",
                "top_speed",
                "load_index",
                "load_level"
            ]
        ].head(20).to_string(index=False)
    )

    print("\nФайл сохранён:")
    print(OUTPUT_PARQUET)

    print("\nПример CSV сохранён:")
    print(OUTPUT_CSV)


if __name__ == "__main__":
    main()
