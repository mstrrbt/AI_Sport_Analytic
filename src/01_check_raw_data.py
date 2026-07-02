from pathlib import Path
import json

PROJECT_DIR = Path(__file__).resolve().parents[1]

statsbomb_path = PROJECT_DIR / "data" / "raw" / "statsbomb" / "data"
skillcorner_path = PROJECT_DIR / "data" / "raw" / "skillcorner"

print("Папка проекта:", PROJECT_DIR)

print("\n=== Проверка StatsBomb ===")
print("Путь:", statsbomb_path)

competitions_file = statsbomb_path / "competitions.json"

if competitions_file.exists():
    print("competitions.json найден")

    with open(competitions_file, "r", encoding="utf-8") as file:
        competitions = json.load(file)

    print("Количество соревнований/сезонов:", len(competitions))

    print("\nПервые 20 записей:")
    for item in competitions[:20]:
        print(
            item.get("competition_name"),
            "| сезон:",
            item.get("season_name"),
            "| competition_id:",
            item.get("competition_id"),
            "| season_id:",
            item.get("season_id")
        )
else:
    print("competitions.json НЕ найден")

print("\n=== Проверка SkillCorner ===")
print("Путь:", skillcorner_path)

if skillcorner_path.exists():
    print("Папка SkillCorner найдена")

    print("\nФайлы и папки внутри SkillCorner:")
    for item in skillcorner_path.iterdir():
        print("-", item.name)
else:
    print("Папка SkillCorner НЕ найдена")
