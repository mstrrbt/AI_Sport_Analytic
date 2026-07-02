from pathlib import Path
import sys

import pandas as pd
from catboost import CatBoostRegressor


PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_FILE = PROJECT_DIR / "data" / "model_ready" / "euro2024_player_form_model_ready.parquet"
MODEL_FILE = PROJECT_DIR / "models" / "player_form_catboost.cbm"

REPORTS_DIR = PROJECT_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def prepare_features(df):
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


def find_player(players, query):
    """
    Ищем игрока по части имени.
    Например:
    Morata
    Olmo
    Yamal
    """
    query = query.lower().strip()

    found = players[players["player_name"].str.lower().str.contains(query, na=False)].copy()

    return found


def make_text_table(df, columns):
    small = df[columns].copy()

    for col in small.columns:
        if pd.api.types.is_float_dtype(small[col]):
            small[col] = small[col].round(2)

    return small.to_string(index=False)


def main():
    if len(sys.argv) < 4:
        print("Нужно указать команду, кого убрать и кого поставить.")
        print("")
        print("Примеры:")
        print('python src/08_what_if_substitution.py Spain "Morata" "Olmo"')
        print('python src/08_what_if_substitution.py Spain "Morata" "Nico"')
        print('python src/08_what_if_substitution.py Spain "Carvajal" "Grimaldo"')
        return

    team_name = sys.argv[1].strip()
    out_query = sys.argv[2].strip()
    in_query = sys.argv[3].strip()

    safe_team_name = team_name.replace(" ", "_")

    lineup_file = REPORTS_DIR / f"lineup_{safe_team_name}_433.csv"

    print("=== What-if замена игрока ===")
    print("Команда:", team_name)
    print("Убрать:", out_query)
    print("Поставить:", in_query)

    if not lineup_file.exists():
        print("Ошибка: сначала нужно создать состав.")
        print("Запусти:")
        print(f"python src/07_lineup_optimizer.py {team_name}")
        return

    if not DATA_FILE.exists():
        print("Ошибка: файл данных не найден")
        print(DATA_FILE)
        return

    if not MODEL_FILE.exists():
        print("Ошибка: файл модели не найден")
        print(MODEL_FILE)
        return

    lineup = pd.read_csv(lineup_file)

    df = pd.read_parquet(DATA_FILE)
    df["match_date"] = pd.to_datetime(df["match_date"])

    model = CatBoostRegressor()
    model.load_model(str(MODEL_FILE))

    df, feature_cols = prepare_features(df)
    df["predicted_form_score"] = model.predict(df[feature_cols]).clip(0, 100)

    team_df = df[df["team_name"].str.lower() == team_name.lower()].copy()

    if len(team_df) == 0:
        print("Команда не найдена.")
        return

    # Средние показатели игроков по турниру
    players = (
        team_df.groupby(["player_name", "team_name"], as_index=False)
        .agg(
            matches=("match_id", "nunique"),
            avg_predicted_form=("predicted_form_score", "mean"),
            max_predicted_form=("predicted_form_score", "max"),
            goals=("goals", "sum"),
            xg=("xg", "sum"),
            shots=("shots", "sum"),
            key_passes=("key_passes", "sum"),
            progressive_passes=("progressive_passes", "sum"),
            pressures=("pressures", "sum"),
            interceptions=("interceptions", "sum"),
            duels_won=("duels_won", "sum")
        )
    )

    # Ищем игрока, которого убираем, в текущем составе
    out_found = lineup[lineup["player_name"].str.lower().str.contains(out_query.lower(), na=False)].copy()

    if len(out_found) == 0:
        print("\nИгрок для замены не найден в стартовом составе.")
        print("Текущий стартовый состав:")
        print(make_text_table(lineup, ["role", "player_name", "position_name", "avg_predicted_form", "selection_score"]))
        return

    if len(out_found) > 1:
        print("\nНайдено несколько игроков для удаления. Уточни имя:")
        print(make_text_table(out_found, ["role", "player_name", "position_name", "avg_predicted_form", "selection_score"]))
        return

    out_player = out_found.iloc[0]

    # Ищем входящего игрока среди всех игроков команды
    in_found = find_player(players, in_query)

    if len(in_found) == 0:
        print("\nИгрок на замену не найден.")
        print("Доступные игроки команды:")
        print(make_text_table(players.sort_values("avg_predicted_form", ascending=False), ["player_name", "matches", "avg_predicted_form"]))
        return

    if len(in_found) > 1:
        print("\nНайдено несколько игроков на замену. Уточни имя:")
        print(make_text_table(in_found.sort_values("avg_predicted_form", ascending=False), ["player_name", "matches", "avg_predicted_form"]))
        return

    in_player = in_found.iloc[0]

    # Важная проверка: нельзя поставить игрока, который уже есть в стартовом составе.
    # Иначе получится дубль одного и того же футболиста.
    if in_player["player_name"] in lineup["player_name"].values:
        print("\nИгрок на замену уже есть в стартовом составе.")
        print("Такой сценарий пока нельзя считать корректным, потому что один игрок окажется в составе два раза.")
        print("\nПопробуй выбрать игрока не из старта. Например:")
        available = players[~players["player_name"].isin(lineup["player_name"])].copy()
        available = available.sort_values("avg_predicted_form", ascending=False)
        print(make_text_table(available, ["player_name", "matches", "avg_predicted_form"]))
        return

    old_team_score = lineup["selection_score"].sum()

    # У входящего игрока нет selection_score из состава,
    # поэтому считаем его простую оценку для сравнения.
    # Берём среднюю форму + бонус за матчи + полезные действия.
    in_selection_score = (
        in_player["avg_predicted_form"] +
        0.7 * in_player["matches"] +
        0.05 * in_player["pressures"] +
        0.03 * in_player["progressive_passes"] +
        1.2 * in_player["xg"] +
        0.35 * in_player["key_passes"]
    )

    new_team_score = old_team_score - out_player["selection_score"] + in_selection_score
    difference = new_team_score - old_team_score

    new_lineup = lineup.copy()
    mask = new_lineup["player_name"] == out_player["player_name"]

    new_lineup.loc[mask, "player_name"] = in_player["player_name"]
    new_lineup.loc[mask, "position_name"] = "what-if replacement"
    new_lineup.loc[mask, "matches"] = in_player["matches"]
    new_lineup.loc[mask, "avg_predicted_form"] = in_player["avg_predicted_form"]
    new_lineup.loc[mask, "max_predicted_form"] = in_player["max_predicted_form"]
    new_lineup.loc[mask, "selection_score"] = in_selection_score
    new_lineup.loc[mask, "goals"] = in_player["goals"]
    new_lineup.loc[mask, "xg"] = in_player["xg"]
    new_lineup.loc[mask, "shots"] = in_player["shots"]
    new_lineup.loc[mask, "key_passes"] = in_player["key_passes"]
    new_lineup.loc[mask, "progressive_passes"] = in_player["progressive_passes"]
    new_lineup.loc[mask, "pressures"] = in_player["pressures"]

    # Текстовое объяснение
    if difference > 2:
        decision = "Замена выглядит полезной по модели."
    elif difference < -2:
        decision = "Замена выглядит рискованной по модели."
    else:
        decision = "Замена почти нейтральная по модели."

    report_name = f"what_if_{safe_team_name}_{out_query}_to_{in_query}.txt"
    report_name = report_name.replace(" ", "_").replace("/", "_")
    report_file = REPORTS_DIR / report_name

    lines = []

    title = f"CoachMind Football — what-if замена: {out_query} -> {in_query}"
    lines.append(title)
    lines.append("=" * len(title))
    lines.append("")

    lines.append("1. Сценарий")
    lines.append("-" * 40)
    lines.append(f"Команда: {team_name}")
    lines.append(f"Схема: 4-3-3")
    lines.append(f"Убрать из состава: {out_player['player_name']}")
    lines.append(f"Поставить вместо него: {in_player['player_name']}")
    lines.append(f"Роль в составе, где делается замена: {out_player['role']}")
    lines.append("")

    lines.append("2. Сравнение игроков")
    lines.append("-" * 40)

    compare_df = pd.DataFrame([
        {
            "player": out_player["player_name"],
            "type": "current player",
            "matches": out_player["matches"],
            "avg_form": out_player["avg_predicted_form"],
            "selection_score": out_player["selection_score"],
            "goals": out_player["goals"],
            "xg": out_player["xg"],
            "key_passes": out_player["key_passes"],
            "progressive_passes": out_player["progressive_passes"],
            "pressures": out_player["pressures"]
        },
        {
            "player": in_player["player_name"],
            "type": "replacement",
            "matches": in_player["matches"],
            "avg_form": in_player["avg_predicted_form"],
            "selection_score": in_selection_score,
            "goals": in_player["goals"],
            "xg": in_player["xg"],
            "key_passes": in_player["key_passes"],
            "progressive_passes": in_player["progressive_passes"],
            "pressures": in_player["pressures"]
        }
    ])

    lines.append(make_text_table(
        compare_df,
        [
            "type",
            "player",
            "matches",
            "avg_form",
            "selection_score",
            "goals",
            "xg",
            "key_passes",
            "progressive_passes",
            "pressures"
        ]
    ))
    lines.append("")

    lines.append("3. Изменение оценки состава")
    lines.append("-" * 40)
    lines.append(f"Оценка старого состава: {old_team_score:.2f}")
    lines.append(f"Оценка нового состава: {new_team_score:.2f}")
    lines.append(f"Разница: {difference:.2f}")
    lines.append(f"Вывод: {decision}")
    lines.append("")

    lines.append("4. Новый состав после замены")
    lines.append("-" * 40)
    lines.append(make_text_table(
        new_lineup,
        [
            "role",
            "player_name",
            "position_name",
            "matches",
            "avg_predicted_form",
            "selection_score"
        ]
    ))
    lines.append("")

    lines.append("5. Ограничение")
    lines.append("-" * 40)
    lines.append(
        "Сейчас what-if учитывает игровые действия и форму по StatsBomb. "
        "Физическая усталость и нагрузка ещё не подключены. "
        "Это будет следующим модулем через SkillCorner."
    )

    with open(report_file, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))

    print("\nСравнение игроков:")
    print(make_text_table(
        compare_df,
        [
            "type",
            "player",
            "matches",
            "avg_form",
            "selection_score",
            "goals",
            "xg",
            "key_passes",
            "progressive_passes",
            "pressures"
        ]
    ))

    print("\nИзменение оценки состава:")
    print("Старый состав:", round(old_team_score, 2))
    print("Новый состав:", round(new_team_score, 2))
    print("Разница:", round(difference, 2))
    print("Вывод:", decision)

    print("\nФайл what-if отчёта сохранён:")
    print(report_file)


if __name__ == "__main__":
    main()
