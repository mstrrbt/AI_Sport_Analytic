from pathlib import Path
import json
from collections import defaultdict

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_DIR / "data" / "raw" / "statsbomb" / "data"
MATCHES_FILE = DATA_DIR / "matches" / "55" / "282.json"
EVENTS_DIR = DATA_DIR / "events"

OUTPUT_DIR = PROJECT_DIR / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PARQUET = OUTPUT_DIR / "statsbomb_euro2024_player_match_features.parquet"
OUTPUT_CSV = OUTPUT_DIR / "statsbomb_euro2024_player_match_features_sample.csv"


def get_name(value):
    """
    В StatsBomb многие поля хранятся как словарь:
    {"id": ..., "name": "..."}
    Эта функция безопасно достает name.
    """
    if isinstance(value, dict):
        return value.get("name")
    return None


def get_id(value):
    """
    Безопасно достает id из словаря StatsBomb.
    """
    if isinstance(value, dict):
        return value.get("id")
    return None


def get_location(event):
    """
    Возвращает координаты действия на поле.
    В StatsBomb поле обычно 120 на 80.
    """
    location = event.get("location")
    if isinstance(location, list) and len(location) >= 2:
        return location[0], location[1]
    return None, None


def get_end_location(event):
    """
    Для передач и переносов мяча есть конечная точка.
    Она лежит внутри pass.end_location или carry.end_location.
    """
    event_type = get_name(event.get("type"))

    if event_type == "Pass":
        pass_data = event.get("pass", {})
        end_location = pass_data.get("end_location")
    elif event_type == "Carry":
        carry_data = event.get("carry", {})
        end_location = carry_data.get("end_location")
    else:
        end_location = None

    if isinstance(end_location, list) and len(end_location) >= 2:
        return end_location[0], end_location[1]

    return None, None


def new_player_row():
    """
    Начальные значения признаков игрока в конкретном матче.
    """
    return {
        "competition_name": "UEFA Euro",
        "season_name": "2024",

        "match_id": None,
        "match_date": None,
        "home_team": None,
        "away_team": None,

        "player_id": None,
        "player_name": None,
        "team_id": None,
        "team_name": None,
        "position_name": None,

        "total_events": 0,

        "shots": 0,
        "goals": 0,
        "xg": 0.0,

        "passes": 0,
        "completed_passes": 0,
        "pass_accuracy": 0.0,
        "progressive_passes": 0,
        "passes_into_final_third": 0,
        "key_passes": 0,

        "carries": 0,
        "progressive_carries": 0,

        "pressures": 0,
        "duels": 0,
        "duels_won": 0,
        "interceptions": 0,
        "clearances": 0,
        "blocks": 0,
        "dribbles": 0,
        "successful_dribbles": 0,

        "avg_x": 0.0,
        "avg_y": 0.0,
        "_sum_x": 0.0,
        "_sum_y": 0.0,
        "_location_count": 0,
    }


def main():
    print("=== Подготовка StatsBomb Euro 2024 ===")
    print("Файл матчей:", MATCHES_FILE)

    if not MATCHES_FILE.exists():
        print("Ошибка: файл матчей Euro 2024 не найден.")
        print("Проверь путь:", MATCHES_FILE)
        return

    with open(MATCHES_FILE, "r", encoding="utf-8") as file:
        matches = json.load(file)

    print("Количество матчей Euro 2024:", len(matches))

    all_rows = []

    for i, match in enumerate(matches, start=1):
        match_id = match.get("match_id")
        match_date = match.get("match_date")

        home_team = get_name(match.get("home_team"))
        away_team = get_name(match.get("away_team"))

        events_file = EVENTS_DIR / f"{match_id}.json"

        if not events_file.exists():
            print(f"[{i}/{len(matches)}] Матч {match_id}: файл событий не найден, пропускаю")
            continue

        with open(events_file, "r", encoding="utf-8") as file:
            events = json.load(file)

        print(f"[{i}/{len(matches)}] {home_team} vs {away_team} | events: {len(events)}")

        # Ключ: игрок в конкретном матче
        players = defaultdict(new_player_row)

        for event in events:
            event_type = get_name(event.get("type"))

            player = event.get("player")
            team = event.get("team")

            player_id = get_id(player)
            player_name = get_name(player)

            # События без игрока нам сейчас не нужны
            if player_id is None or player_name is None:
                continue

            team_id = get_id(team)
            team_name = get_name(team)
            position_name = get_name(event.get("position"))

            key = (match_id, player_id)
            row = players[key]

            # Заполняем общие данные
            row["match_id"] = match_id
            row["match_date"] = match_date
            row["home_team"] = home_team
            row["away_team"] = away_team

            row["player_id"] = player_id
            row["player_name"] = player_name
            row["team_id"] = team_id
            row["team_name"] = team_name

            if row["position_name"] is None and position_name is not None:
                row["position_name"] = position_name

            row["total_events"] += 1

            x, y = get_location(event)
            if x is not None and y is not None:
                row["_sum_x"] += x
                row["_sum_y"] += y
                row["_location_count"] += 1

            # Удары
            if event_type == "Shot":
                row["shots"] += 1

                shot_data = event.get("shot", {})
                row["xg"] += float(shot_data.get("statsbomb_xg", 0) or 0)

                shot_outcome = get_name(shot_data.get("outcome"))
                if shot_outcome == "Goal":
                    row["goals"] += 1

            # Передачи
            elif event_type == "Pass":
                row["passes"] += 1

                pass_data = event.get("pass", {})
                pass_outcome = get_name(pass_data.get("outcome"))

                # В StatsBomb, если outcome отсутствует, пас считается успешным
                if pass_outcome is None:
                    row["completed_passes"] += 1

                if pass_data.get("shot_assist") is True:
                    row["key_passes"] += 1

                start_x, start_y = get_location(event)
                end_x, end_y = get_end_location(event)

                if start_x is not None and end_x is not None:
                    if end_x - start_x >= 10:
                        row["progressive_passes"] += 1

                    if start_x < 80 and end_x >= 80:
                        row["passes_into_final_third"] += 1

            # Переносы мяча
            elif event_type == "Carry":
                row["carries"] += 1

                start_x, start_y = get_location(event)
                end_x, end_y = get_end_location(event)

                if start_x is not None and end_x is not None:
                    if end_x - start_x >= 10:
                        row["progressive_carries"] += 1

            # Прессинг
            elif event_type == "Pressure":
                row["pressures"] += 1

            # Единоборства
            elif event_type == "Duel":
                row["duels"] += 1

                duel_data = event.get("duel", {})
                duel_outcome = get_name(duel_data.get("outcome"))

                if duel_outcome in ["Won", "Success", "Success In Play"]:
                    row["duels_won"] += 1

            elif event_type == "Interception":
                row["interceptions"] += 1

            elif event_type == "Clearance":
                row["clearances"] += 1

            elif event_type == "Block":
                row["blocks"] += 1

            elif event_type == "Dribble":
                row["dribbles"] += 1

                dribble_data = event.get("dribble", {})
                dribble_outcome = get_name(dribble_data.get("outcome"))

                if dribble_outcome == "Complete":
                    row["successful_dribbles"] += 1

        # Финальный расчет средних координат и точности передач
        for row in players.values():
            if row["passes"] > 0:
                row["pass_accuracy"] = row["completed_passes"] / row["passes"]

            if row["_location_count"] > 0:
                row["avg_x"] = row["_sum_x"] / row["_location_count"]
                row["avg_y"] = row["_sum_y"] / row["_location_count"]

            # Внутренние технические поля удаляем
            row.pop("_sum_x", None)
            row.pop("_sum_y", None)
            row.pop("_location_count", None)

            all_rows.append(row)

    df = pd.DataFrame(all_rows)

    print("\n=== Итоговая таблица ===")
    print("Строк:", len(df))
    print("Столбцов:", len(df.columns))
    print("\nПервые 10 строк:")
    print(df.head(10))

    print("\nКолонки:")
    print(list(df.columns))

    df.to_parquet(OUTPUT_PARQUET, index=False)
    df.head(100).to_csv(OUTPUT_CSV, index=False)

    print("\nФайл сохранён:")
    print(OUTPUT_PARQUET)
    print("\nПример CSV сохранён:")
    print(OUTPUT_CSV)


if __name__ == "__main__":
    main()
