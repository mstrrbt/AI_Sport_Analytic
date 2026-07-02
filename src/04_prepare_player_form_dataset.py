from pathlib import Path
import json

import pandas as pd
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_DIR / "data" / "processed" / "statsbomb_euro2024_player_match_features.parquet"

MATCHES_FILE = PROJECT_DIR / "data" / "raw" / "statsbomb" / "data" / "matches" / "55" / "282.json"

OUTPUT_DIR = PROJECT_DIR / "data" / "model_ready"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PARQUET = OUTPUT_DIR / "euro2024_player_form_model_ready.parquet"
OUTPUT_CSV = OUTPUT_DIR / "euro2024_player_form_model_ready_sample.csv"


def safe_minmax(series):
    """
    Нормализация от 0 до 1.
    Если все значения одинаковые, возвращаем 0.
    """
    min_value = series.min()
    max_value = series.max()

    if pd.isna(min_value) or pd.isna(max_value):
        return pd.Series(0, index=series.index)

    if max_value == min_value:
        return pd.Series(0, index=series.index)

    return (series - min_value) / (max_value - min_value)


def get_team_name(team_data, team_type):
    """
    В файле матчей StatsBomb команды лежат так:
    home_team -> home_team_name
    away_team -> away_team_name

    Поэтому обычный ключ name не подходит.
    """
    if not isinstance(team_data, dict):
        return None

    if team_type == "home":
        return team_data.get("home_team_name") or team_data.get("name")

    if team_type == "away":
        return team_data.get("away_team_name") or team_data.get("name")

    return None


def main():
    print("=== Подготовка таблицы для Player Form Model ===")

    if not INPUT_FILE.exists():
        print("Ошибка: входной файл не найден")
        print(INPUT_FILE)
        return

    if not MATCHES_FILE.exists():
        print("Ошибка: файл матчей не найден")
        print(MATCHES_FILE)
        return

    df = pd.read_parquet(INPUT_FILE)

    print("Входная таблица:")
    print("Строк:", len(df))
    print("Столбцов:", len(df.columns))

    with open(MATCHES_FILE, "r", encoding="utf-8") as file:
        matches = json.load(file)

    match_info = {}

    for match in matches:
        match_id = match.get("match_id")

        home_team = get_team_name(match.get("home_team"), "home")
        away_team = get_team_name(match.get("away_team"), "away")

        match_info[match_id] = {
            "home_team_fixed": home_team,
            "away_team_fixed": away_team,
            "match_date_fixed": match.get("match_date"),
            "competition_stage": None
        }

        stage = match.get("competition_stage")
        if isinstance(stage, dict):
            match_info[match_id]["competition_stage"] = stage.get("name")

    match_df = pd.DataFrame.from_dict(match_info, orient="index")
    match_df["match_id"] = match_df.index.astype(int)

    df = df.drop(columns=["home_team", "away_team"], errors="ignore")

    df = df.merge(match_df, on="match_id", how="left")

    df = df.rename(columns={
        "home_team_fixed": "home_team",
        "away_team_fixed": "away_team",
        "match_date_fixed": "match_date_from_matches"
    })

    # Если match_date уже был, оставляем исходный, но если там пусто — берём из matches
    if "match_date" in df.columns:
        df["match_date"] = df["match_date"].fillna(df["match_date_from_matches"])
        df = df.drop(columns=["match_date_from_matches"], errors="ignore")
    else:
        df = df.rename(columns={"match_date_from_matches": "match_date"})

    # Соперник игрока
    df["opponent_team"] = np.where(
        df["team_name"] == df["home_team"],
        df["away_team"],
        df["home_team"]
    )

    # Условный флаг: команда игрока указана как домашняя
    df["is_home_team"] = (df["team_name"] == df["home_team"]).astype(int)

    # Заполняем пропуски в числах нулями
    numeric_cols = [
        "total_events",
        "shots",
        "goals",
        "xg",
        "passes",
        "completed_passes",
        "pass_accuracy",
        "progressive_passes",
        "passes_into_final_third",
        "key_passes",
        "carries",
        "progressive_carries",
        "pressures",
        "duels",
        "duels_won",
        "interceptions",
        "clearances",
        "blocks",
        "dribbles",
        "successful_dribbles",
        "avg_x",
        "avg_y"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Дополнительные признаки
    df["duel_success_rate"] = np.where(
        df["duels"] > 0,
        df["duels_won"] / df["duels"],
        0
    )

    df["dribble_success_rate"] = np.where(
        df["dribbles"] > 0,
        df["successful_dribbles"] / df["dribbles"],
        0
    )

    df["xg_per_shot"] = np.where(
        df["shots"] > 0,
        df["xg"] / df["shots"],
        0
    )

    # Нормализация признаков
    score_columns = [
        "xg",
        "shots",
        "goals",
        "key_passes",
        "progressive_passes",
        "passes_into_final_third",
        "progressive_carries",
        "pressures",
        "interceptions",
        "duel_success_rate",
        "dribble_success_rate",
        "total_events"
    ]

    for col in score_columns:
        df[col + "_norm"] = safe_minmax(df[col])

    # Первый вариант form_score.
    # Это не финальная модель, а целевой показатель для первого обучения.
    # Мы создаём понятную оценку формы игрока на основе его действий в матче.
    df["form_score"] = (
        25 * df["xg_norm"] +
        10 * df["shots_norm"] +
        15 * df["goals_norm"] +
        10 * df["key_passes_norm"] +
        10 * df["progressive_passes_norm"] +
        5 * df["passes_into_final_third_norm"] +
        8 * df["progressive_carries_norm"] +
        5 * df["pressures_norm"] +
        4 * df["interceptions_norm"] +
        4 * df["duel_success_rate_norm"] +
        4 * df["dribble_success_rate_norm"]
    )

    # Ограничиваем оценку от 0 до 100
    df["form_score"] = df["form_score"].clip(0, 100)

    # Для удобства добавим категорию формы
    df["form_level"] = pd.cut(
        df["form_score"],
        bins=[-1, 25, 50, 75, 100],
        labels=["low", "medium", "good", "excellent"]
    ).astype(str)

    # Убираем технические norm-колонки из финального файла?
    # Пока оставляем, потому что они пригодятся для анализа.
    
    print("\nПроверка исправленных команд:")
    print(df[["match_id", "home_team", "away_team", "team_name", "opponent_team"]].head(10))

    print("\nТоп-20 игроков по form_score:")
    top_players = df.sort_values("form_score", ascending=False)[
        [
            "player_name",
            "team_name",
            "opponent_team",
            "match_date",
            "competition_stage",
            "position_name",
            "shots",
            "goals",
            "xg",
            "key_passes",
            "progressive_passes",
            "pressures",
            "form_score",
            "form_level"
        ]
    ].head(20)

    print(top_players.to_string(index=False))

    print("\nИтоговая таблица:")
    print("Строк:", len(df))
    print("Столбцов:", len(df.columns))

    df.to_parquet(OUTPUT_PARQUET, index=False)
    df.head(200).to_csv(OUTPUT_CSV, index=False)

    print("\nФайл для модели сохранён:")
    print(OUTPUT_PARQUET)

    print("\nПример CSV сохранён:")
    print(OUTPUT_CSV)


if __name__ == "__main__":
    main()
