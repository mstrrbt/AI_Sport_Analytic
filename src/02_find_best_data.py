from pathlib import Path
import json

PROJECT_DIR = Path(__file__).resolve().parents[1]
statsbomb_path = PROJECT_DIR / "data" / "raw" / "statsbomb" / "data"
skillcorner_path = PROJECT_DIR / "data" / "raw" / "skillcorner"

competitions_file = statsbomb_path / "competitions.json"
matches_dir = statsbomb_path / "matches"
events_dir = statsbomb_path / "events"
three_sixty_dir = statsbomb_path / "three-sixty"

with open(competitions_file, "r", encoding="utf-8") as file:
    competitions = json.load(file)

print("=== StatsBomb: свежие и полезные соревнования ===\n")

rows = []

for comp in competitions:
    competition_id = comp.get("competition_id")
    season_id = comp.get("season_id")
    competition_name = comp.get("competition_name")
    season_name = comp.get("season_name")
    gender = comp.get("competition_gender", "unknown")
    country = comp.get("country_name", "")
    match_updated = comp.get("match_updated", "")
    match_available = comp.get("match_available", "")

    matches_file = matches_dir / str(competition_id) / f"{season_id}.json"

    match_count = 0
    event_count_exists = 0
    three_sixty_count_exists = 0

    if matches_file.exists():
        with open(matches_file, "r", encoding="utf-8") as f:
            matches = json.load(f)

        match_count = len(matches)

        for match in matches:
            match_id = match.get("match_id")

            event_file = events_dir / f"{match_id}.json"
            if event_file.exists():
                event_count_exists += 1

            three_file = three_sixty_dir / f"{match_id}.json"
            if three_file.exists():
                three_sixty_count_exists += 1

    rows.append({
        "competition_name": competition_name,
        "season_name": season_name,
        "gender": gender,
        "country": country,
        "competition_id": competition_id,
        "season_id": season_id,
        "match_count": match_count,
        "events_files": event_count_exists,
        "three_sixty_files": three_sixty_count_exists,
        "match_updated": match_updated,
        "match_available": match_available
    })

# Сортируем так, чтобы сверху были соревнования с большим числом матчей и 360-данными
rows_sorted = sorted(
    rows,
    key=lambda x: (x["three_sixty_files"], x["match_count"]),
    reverse=True
)

print("Топ-25 соревнований по количеству матчей и 360-данных:\n")

for row in rows_sorted[:25]:
    print(
        f"{row['competition_name']} | {row['season_name']} | {row['gender']} | "
        f"matches: {row['match_count']} | events: {row['events_files']} | 360: {row['three_sixty_files']} | "
        f"competition_id: {row['competition_id']} | season_id: {row['season_id']}"
    )

print("\n=== Кандидаты для нашего MVP: свежие мужские турниры ===\n")

keywords = [
    "Euro",
    "UEFA",
    "Bundesliga",
    "Champions League",
    "World Cup",
    "La Liga",
    "Premier League"
]

for row in rows_sorted:
    name = str(row["competition_name"])
    season = str(row["season_name"])
    gender = str(row["gender"]).lower()

    is_interesting_name = any(word.lower() in name.lower() for word in keywords)
    is_recent = any(year in season for year in ["2024", "2023", "2022", "2021", "2020"])
    is_male = gender in ["male", "men", "unknown"]

    if is_interesting_name and is_recent and is_male:
        print(
            f"{row['competition_name']} | {row['season_name']} | {row['gender']} | "
            f"matches: {row['match_count']} | events: {row['events_files']} | 360: {row['three_sixty_files']} | "
            f"competition_id: {row['competition_id']} | season_id: {row['season_id']}"
        )

print("\n=== SkillCorner: структура папки data ===\n")

skillcorner_data = skillcorner_path / "data"

if skillcorner_data.exists():
    print("Папка SkillCorner/data найдена:", skillcorner_data)

    print("\nПервые файлы и папки внутри SkillCorner/data:")
    for item in sorted(skillcorner_data.iterdir())[:50]:
        if item.is_dir():
            print("[DIR] ", item.name)
        else:
            print("[FILE]", item.name)

    print("\nПримеры файлов внутри SkillCorner/data на глубине до 4 уровней:")
    count = 0
    for path in skillcorner_data.rglob("*"):
        if path.is_file():
            print("-", path.relative_to(skillcorner_data))
            count += 1
            if count >= 40:
                break
else:
    print("Папка SkillCorner/data НЕ найдена")
