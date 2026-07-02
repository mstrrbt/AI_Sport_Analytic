from pathlib import Path
import json

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]

SKILLCORNER_DIR = PROJECT_DIR / "data" / "raw" / "skillcorner" / "data"
AGG_DIR = SKILLCORNER_DIR / "aggregates"
MATCHES_JSON = SKILLCORNER_DIR / "matches.json"


def print_file_info(file_path):
    print("\n" + "=" * 80)
    print("Файл:", file_path.name)
    print("=" * 80)

    try:
        df = pd.read_csv(file_path)
    except Exception as error:
        print("Не удалось прочитать файл:")
        print(error)
        return

    print("Размер таблицы:")
    print("Строк:", len(df))
    print("Столбцов:", len(df.columns))

    print("\nКолонки:")
    for col in df.columns:
        print("-", col)

    print("\nПервые 5 строк:")
    print(df.head(5).to_string(index=False))

    print("\nТипы данных:")
    print(df.dtypes.to_string())

    # Пробуем найти полезные колонки
    print("\nПохожие на player/team/match/minute/distance/speed/load:")
    keywords = [
        "player", "team", "match", "minute", "minutes",
        "distance", "speed", "sprint", "run", "accel", "decel",
        "load", "physical", "high", "intensity"
    ]

    for col in df.columns:
        col_lower = col.lower()
        if any(word in col_lower for word in keywords):
            print("-", col)

    # Покажем уникальные значения для некоторых текстовых колонок
    for col in df.columns:
        if df[col].dtype == "object":
            unique_count = df[col].nunique(dropna=True)
            if unique_count <= 20:
                print(f"\nУникальные значения в {col}:")
                print(df[col].dropna().unique()[:20])


def main():
    print("=== Проверка SkillCorner ===")
    print("Папка SkillCorner:", SKILLCORNER_DIR)

    if not SKILLCORNER_DIR.exists():
        print("Папка SkillCorner не найдена")
        return

    print("\nСодержимое папки data:")
    for item in sorted(SKILLCORNER_DIR.iterdir()):
        if item.is_dir():
            print("[DIR] ", item.name)
        else:
            print("[FILE]", item.name)

    if MATCHES_JSON.exists():
        print("\n" + "=" * 80)
        print("matches.json найден")
        print("=" * 80)

        with open(MATCHES_JSON, "r", encoding="utf-8") as file:
            matches = json.load(file)

        print("Тип:", type(matches))
        print("Количество записей:", len(matches))

        print("\nПервые 3 записи matches.json:")
        for item in matches[:3]:
            print(json.dumps(item, ensure_ascii=False, indent=2)[:1500])
            print("-" * 40)
    else:
        print("matches.json не найден")

    if not AGG_DIR.exists():
        print("\nПапка aggregates не найдена")
        return

    print("\nФайлы в aggregates:")
    csv_files = sorted(AGG_DIR.glob("*.csv"))

    for file_path in csv_files:
        print("-", file_path.name)

    for file_path in csv_files:
        print_file_info(file_path)


if __name__ == "__main__":
    main()
