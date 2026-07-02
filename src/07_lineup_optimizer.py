from pathlib import Path
import sys

import pandas as pd
import numpy as np
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


def position_to_roles(position):
    """
    Переводим подробные позиции StatsBomb в простые роли для состава.
    Один игрок может подходить под несколько ролей.
    """
    position = str(position).lower()

    roles = []

    if "goalkeeper" in position:
        roles.append("GK")

    if "right back" in position or "right wing back" in position:
        roles.append("RB")

    if "left back" in position or "left wing back" in position:
        roles.append("LB")

    if "center back" in position:
        roles.append("CB")

    if "defensive midfield" in position:
        roles.append("CM")

    if "center midfield" in position:
        roles.append("CM")

    if "attacking midfield" in position:
        roles.append("CM")

    if "right midfield" in position:
        roles.append("RW")
        roles.append("CM")

    if "left midfield" in position:
        roles.append("LW")
        roles.append("CM")

    if "right wing" in position:
        roles.append("RW")

    if "left wing" in position:
        roles.append("LW")

    if "center forward" in position:
        roles.append("CF")

    if "left center forward" in position or "right center forward" in position:
        roles.append("CF")

    return list(dict.fromkeys(roles))


def select_best(candidates, role, already_selected, count):
    """
    Выбираем лучших игроков на конкретную роль.
    already_selected нужен, чтобы один игрок не попал в состав два раза.
    """
    role_candidates = candidates[
        (candidates["role"] == role) &
        (~candidates["player_name"].isin(already_selected))
    ].copy()

    role_candidates = role_candidates.sort_values(
        ["selection_score", "avg_predicted_form", "matches"],
        ascending=False
    )

    selected = role_candidates.head(count).copy()

    for player in selected["player_name"].tolist():
        already_selected.add(player)

    return selected, already_selected


def make_text_table(df, columns):
    small = df[columns].copy()

    for col in small.columns:
        if pd.api.types.is_float_dtype(small[col]):
            small[col] = small[col].round(2)

    return small.to_string(index=False)


def main():
    if len(sys.argv) < 2:
        print("Нужно указать команду.")
        print("Пример:")
        print("python src/07_lineup_optimizer.py Spain")
        return

    team_name = " ".join(sys.argv[1:]).strip()

    print("=== Lineup Optimizer ===")
    print("Команда:", team_name)
    print("Схема: 4-3-3")

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
    df["predicted_form_score"] = model.predict(df[feature_cols]).clip(0, 100)

    team_df = df[df["team_name"].str.lower() == team_name.lower()].copy()

    if len(team_df) == 0:
        print("Команда не найдена.")
        print("Доступные команды:")
        for team in sorted(df["team_name"].dropna().unique()):
            print("-", team)
        return

    # Создаём кандидатов по ролям
    candidate_rows = []

    for _, row in team_df.iterrows():
        roles = position_to_roles(row["position_name"])

        for role in roles:
            candidate_rows.append({
                "player_name": row["player_name"],
                "team_name": row["team_name"],
                "position_name": row["position_name"],
                "role": role,
                "match_id": row["match_id"],
                "match_date": row["match_date"],
                "opponent_team": row["opponent_team"],
                "competition_stage": row["competition_stage"],
                "predicted_form_score": row["predicted_form_score"],
                "goals": row["goals"],
                "xg": row["xg"],
                "shots": row["shots"],
                "key_passes": row["key_passes"],
                "progressive_passes": row["progressive_passes"],
                "pressures": row["pressures"],
                "interceptions": row["interceptions"],
                "duels_won": row["duels_won"]
            })

    candidates = pd.DataFrame(candidate_rows)

    if len(candidates) == 0:
        print("Не удалось собрать кандидатов по позициям.")
        return

    # Агрегируем: игрок + роль
    agg = (
        candidates.groupby(["player_name", "team_name", "position_name", "role"], as_index=False)
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

    # selection_score — оценка для выбора в состав.
    # Делаем небольшой бонус за количество матчей, чтобы не выбирать игрока только из-за одного удачного эпизода.
    agg["selection_score"] = (
        agg["avg_predicted_form"] +
        0.7 * agg["matches"] +
        0.05 * agg["pressures"] +
        0.03 * agg["progressive_passes"]
    )

    # Для защитников добавляем небольшой бонус за оборонительные действия
    defender_mask = agg["role"].isin(["RB", "LB", "CB"])
    agg.loc[defender_mask, "selection_score"] += (
        0.08 * agg.loc[defender_mask, "interceptions"] +
        0.05 * agg.loc[defender_mask, "duels_won"]
    )

    # Для атакующих игроков добавляем небольшой бонус за xG и ключевые передачи
    attack_mask = agg["role"].isin(["RW", "LW", "CF"])
    agg.loc[attack_mask, "selection_score"] += (
        1.5 * agg.loc[attack_mask, "xg"] +
        0.4 * agg.loc[attack_mask, "key_passes"]
    )

    already_selected = set()
    selected_parts = []

    formation = [
        ("GK", 1),
        ("RB", 1),
        ("CB", 2),
        ("LB", 1),
        ("CM", 3),
        ("RW", 1),
        ("LW", 1),
        ("CF", 1)
    ]

    for role, count in formation:
        selected, already_selected = select_best(
            agg,
            role,
            already_selected,
            count
        )
        selected_parts.append(selected)

    lineup = pd.concat(selected_parts, ignore_index=True)

    # Если кого-то не хватает, добираем лучших оставшихся любых ролей
    if len(lineup) < 11:
        need = 11 - len(lineup)

        remaining = agg[~agg["player_name"].isin(already_selected)].copy()
        remaining = remaining.sort_values("selection_score", ascending=False).head(need)

        lineup = pd.concat([lineup, remaining], ignore_index=True)

    lineup = lineup.sort_values(
        ["role", "selection_score"],
        ascending=[True, False]
    )

    # Скамейка запасных
    bench = agg[~agg["player_name"].isin(lineup["player_name"])].copy()
    bench = bench.sort_values("selection_score", ascending=False).head(7)

    safe_team_name = team_name.replace(" ", "_")

    lineup_csv = REPORTS_DIR / f"lineup_{safe_team_name}_433.csv"
    lineup_txt = REPORTS_DIR / f"lineup_{safe_team_name}_433.txt"
    bench_csv = REPORTS_DIR / f"bench_{safe_team_name}_433.csv"

    lineup.to_csv(lineup_csv, index=False)
    bench.to_csv(bench_csv, index=False)

    lines = []

    title = f"CoachMind Football — подбор состава {team_name}, схема 4-3-3"
    lines.append(title)
    lines.append("=" * len(title))
    lines.append("")

    lines.append("1. Рекомендуемый стартовый состав")
    lines.append("-" * 40)
    lines.append(make_text_table(
        lineup,
        [
            "role",
            "player_name",
            "position_name",
            "matches",
            "avg_predicted_form",
            "max_predicted_form",
            "selection_score",
            "goals",
            "xg",
            "key_passes",
            "progressive_passes",
            "pressures"
        ]
    ))
    lines.append("")

    lines.append("2. Запасные кандидаты")
    lines.append("-" * 40)
    lines.append(make_text_table(
        bench,
        [
            "role",
            "player_name",
            "position_name",
            "matches",
            "avg_predicted_form",
            "selection_score",
            "goals",
            "xg",
            "key_passes",
            "progressive_passes",
            "pressures"
        ]
    ))
    lines.append("")

    lines.append("3. Как читать результат")
    lines.append("-" * 40)
    lines.append(
        "avg_predicted_form — средняя оценка формы игрока по модели."
    )
    lines.append(
        "selection_score — итоговая оценка для выбора состава. "
        "Она учитывает форму, стабильность по матчам и действия, важные для роли."
    )
    lines.append(
        "Это не финальное решение вместо тренера, а подсказка: кого модель считает сильными кандидатами под схему 4-3-3."
    )
    lines.append("")

    lines.append("4. Ограничения текущей версии")
    lines.append("-" * 40)
    lines.append(
        "Пока модель использует только событийные данные StatsBomb. "
        "Физическая нагрузка, скорость, усталость и риск перегрузки будут добавлены позже через SkillCorner."
    )

    report_text = "\n".join(lines)

    with open(lineup_txt, "w", encoding="utf-8") as file:
        file.write(report_text)

    print("\nРекомендуемый стартовый состав:")
    print(make_text_table(
        lineup,
        [
            "role",
            "player_name",
            "position_name",
            "matches",
            "avg_predicted_form",
            "selection_score"
        ]
    ))

    print("\nЗапасные:")
    print(make_text_table(
        bench,
        [
            "role",
            "player_name",
            "position_name",
            "matches",
            "avg_predicted_form",
            "selection_score"
        ]
    ))

    print("\nФайлы сохранены:")
    print(lineup_txt)
    print(lineup_csv)
    print(bench_csv)


if __name__ == "__main__":
    main()
