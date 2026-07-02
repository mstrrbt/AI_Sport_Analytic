from pathlib import Path
import sys

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor


PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_FILE = PROJECT_DIR / "data" / "model_ready" / "euro2024_player_form_model_ready.parquet"
MODEL_FILE = PROJECT_DIR / "models" / "player_form_catboost.cbm"
IMPORTANCE_FILE = PROJECT_DIR / "data" / "model_ready" / "player_form_feature_importance.csv"

REPORTS_DIR = PROJECT_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def prepare_features(df):
    target_col = "form_score"

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

    exclude_cols += [col for col in df.columns if col.endswith("_norm")]

    feature_cols = [col for col in df.columns if col not in exclude_cols]

    cat_cols = [
        "team_name",
        "opponent_team",
        "position_name",
        "competition_stage",
        "home_team",
        "away_team"
    ]

    cat_cols = [col for col in cat_cols if col in feature_cols]
    num_cols = [col for col in feature_cols if col not in cat_cols]

    for col in cat_cols:
        df[col] = df[col].fillna("unknown").astype(str)

    for col in num_cols:
        df[col] = df[col].fillna(0)

    return df, feature_cols


def make_text_table(df, columns, max_rows=15):
    """
    Простая текстовая таблица для txt-отчёта.
    """
    small = df[columns].head(max_rows).copy()

    for col in small.columns:
        if pd.api.types.is_float_dtype(small[col]):
            small[col] = small[col].round(2)

    return small.to_string(index=False)


def main():
    print("=== Генерация тренерского отчёта ===")

    team_filter = " ".join(sys.argv[1:]).strip()

    if not DATA_FILE.exists():
        print("Ошибка: файл данных не найден")
        print(DATA_FILE)
        return

    if not MODEL_FILE.exists():
        print("Ошибка: файл модели не найден")
        print(MODEL_FILE)
        return

    df = pd.read_parquet(DATA_FILE)
    df["match_date"] = pd.to_datetime(df["match_date"])

    model = CatBoostRegressor()
    model.load_model(str(MODEL_FILE))

    df, feature_cols = prepare_features(df)

    df["predicted_form_score"] = model.predict(df[feature_cols])
    df["predicted_form_score"] = df["predicted_form_score"].clip(0, 100)

    # Все предсказания
    all_predictions_file = REPORTS_DIR / "euro2024_player_predictions_all.csv"
    df.sort_values("predicted_form_score", ascending=False).to_csv(all_predictions_file, index=False)

    # Средняя форма игрока по турниру
    player_avg = (
        df.groupby(["player_name", "team_name", "position_name"], as_index=False)
        .agg(
            matches=("match_id", "nunique"),
            avg_predicted_form=("predicted_form_score", "mean"),
            max_predicted_form=("predicted_form_score", "max"),
            avg_real_form=("form_score", "mean"),
            goals=("goals", "sum"),
            shots=("shots", "sum"),
            xg=("xg", "sum"),
            key_passes=("key_passes", "sum"),
            progressive_passes=("progressive_passes", "sum"),
            pressures=("pressures", "sum")
        )
    )

    # Чтобы топ не ломался из-за игрока с одним случайным матчем,
    # отдельно смотрим игроков, у кого минимум 2 матча.
    player_avg_min2 = player_avg[player_avg["matches"] >= 2].copy()

    top_avg_file = REPORTS_DIR / "euro2024_top_players_by_average.csv"
    player_avg_min2.sort_values("avg_predicted_form", ascending=False).to_csv(top_avg_file, index=False)

    # Важность признаков
    if IMPORTANCE_FILE.exists():
        importance_df = pd.read_csv(IMPORTANCE_FILE)
    else:
        importance_df = pd.DataFrame(columns=["feature", "importance"])

    # Если пользователь указал команду, делаем отчёт только по ней
    report_df = df.copy()
    report_title = "CoachMind Football — отчёт по Euro 2024"

    if team_filter:
        report_df = report_df[report_df["team_name"].str.lower() == team_filter.lower()].copy()
        report_title = f"CoachMind Football — отчёт по команде {team_filter}"

        if len(report_df) == 0:
            print(f"Команда '{team_filter}' не найдена.")
            print("Доступные команды:")
            for team in sorted(df["team_name"].dropna().unique()):
                print("-", team)
            return

        report_file = REPORTS_DIR / f"coach_report_{team_filter.replace(' ', '_')}.txt"
    else:
        report_file = REPORTS_DIR / "coach_report_all_teams.txt"

    # Топ матчей по форме
    top_single_matches = report_df.sort_values("predicted_form_score", ascending=False)

    # Средняя форма по игрокам для выбранной команды или всех команд
    if team_filter:
        team_player_avg = player_avg_min2[
            player_avg_min2["team_name"].str.lower() == team_filter.lower()
        ].sort_values("avg_predicted_form", ascending=False)
    else:
        team_player_avg = player_avg_min2.sort_values("avg_predicted_form", ascending=False)

    # Средняя форма по командам
    team_avg = (
        df.groupby("team_name", as_index=False)
        .agg(
            matches=("match_id", "nunique"),
            avg_team_form=("predicted_form_score", "mean"),
            goals=("goals", "sum"),
            xg=("xg", "sum"),
            shots=("shots", "sum"),
            pressures=("pressures", "sum")
        )
        .sort_values("avg_team_form", ascending=False)
    )

    lines = []

    lines.append(report_title)
    lines.append("=" * len(report_title))
    lines.append("")

    lines.append("1. Что показывает отчёт")
    lines.append("-" * 30)
    lines.append(
        "Отчёт показывает оценку формы игроков на основе событий матча: ударов, xG, голов, "
        "ключевых передач, продвижения мяча, прессинга, единоборств и других действий."
    )
    lines.append(
        "Это не медицинский прогноз и не окончательное решение тренера. "
        "Это аналитическая подсказка для выбора состава."
    )
    lines.append("")

    lines.append("2. Данные")
    lines.append("-" * 30)
    lines.append(f"Источник: StatsBomb Open Data, UEFA Euro 2024")
    lines.append(f"Количество записей игрок-матч: {len(df)}")
    lines.append(f"Количество признаков для модели: {len(feature_cols)}")
    lines.append("Модель: CatBoostRegressor")
    lines.append("")

    lines.append("3. Самые важные признаки модели")
    lines.append("-" * 30)

    if len(importance_df) > 0:
        lines.append(make_text_table(
            importance_df,
            ["feature", "importance"],
            max_rows=12
        ))
    else:
        lines.append("Файл важности признаков не найден.")
    lines.append("")

    if not team_filter:
        lines.append("4. Рейтинг команд по средней форме игроков")
        lines.append("-" * 30)
        lines.append(make_text_table(
            team_avg,
            ["team_name", "matches", "avg_team_form", "goals", "xg", "shots", "pressures"],
            max_rows=16
        ))
        lines.append("")

    lines.append("5. Топ игроков по средней форме за турнир")
    lines.append("-" * 30)
    lines.append(make_text_table(
        team_player_avg,
        [
            "player_name",
            "team_name",
            "position_name",
            "matches",
            "avg_predicted_form",
            "max_predicted_form",
            "goals",
            "xg",
            "key_passes",
            "progressive_passes",
            "pressures"
        ],
        max_rows=20
    ))
    lines.append("")

    lines.append("6. Лучшие отдельные матчи игроков")
    lines.append("-" * 30)
    lines.append(make_text_table(
        top_single_matches,
        [
            "player_name",
            "team_name",
            "opponent_team",
            "match_date",
            "competition_stage",
            "position_name",
            "goals",
            "xg",
            "key_passes",
            "progressive_passes",
            "pressures",
            "predicted_form_score"
        ],
        max_rows=20
    ))
    lines.append("")

    lines.append("7. Как это можно использовать тренеру")
    lines.append("-" * 30)
    lines.append(
        "Тренер может посмотреть, какие игроки сейчас дают больше пользы по действиям, "
        "сравнить игроков одной позиции и выбрать более подходящий состав."
    )
    lines.append(
        "Следующий шаг MVP — добавить модуль подбора состава и what-if сценарии: "
        "например, что изменится, если заменить одного игрока другим."
    )

    report_text = "\n".join(lines)

    with open(report_file, "w", encoding="utf-8") as file:
        file.write(report_text)

    print("Отчёт сохранён:")
    print(report_file)

    print("\nCSV со всеми предсказаниями:")
    print(all_predictions_file)

    print("\nCSV с топом игроков по средней форме:")
    print(top_avg_file)

    print("\nКороткий просмотр отчёта:\n")
    print("\n".join(lines[:60]))


if __name__ == "__main__":
    main()
