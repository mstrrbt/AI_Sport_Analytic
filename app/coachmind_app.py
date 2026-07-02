from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from catboost import CatBoostRegressor


PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_FILE = PROJECT_DIR / "data" / "model_ready" / "euro2024_player_form_model_ready.parquet"
MODEL_FILE = PROJECT_DIR / "models" / "player_form_catboost.cbm"
LOAD_FILE = PROJECT_DIR / "data" / "model_ready" / "skillcorner_player_load_model_ready.parquet"


st.set_page_config(
    page_title="CoachMind Football",
    page_icon="⚽",
    layout="wide"
)


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


@st.cache_data
def load_data():
    df = pd.read_parquet(DATA_FILE)
    df["match_date"] = pd.to_datetime(df["match_date"])

    model = CatBoostRegressor()
    model.load_model(str(MODEL_FILE))

    df, feature_cols = prepare_features(df)
    df["predicted_form_score"] = model.predict(df[feature_cols]).clip(0, 100)

    return df


@st.cache_data
def load_skillcorner_load_data():
    if not LOAD_FILE.exists():
        return pd.DataFrame()

    load_df = pd.read_parquet(LOAD_FILE)

    for col in load_df.columns:
        if pd.api.types.is_float_dtype(load_df[col]):
            load_df[col] = load_df[col].round(2)

    return load_df


def get_player_summary(team_df):
    player_summary = (
        team_df.groupby(["player_name", "team_name"], as_index=False)
        .agg(
            matches=("match_id", "nunique"),
            avg_form=("predicted_form_score", "mean"),
            max_form=("predicted_form_score", "max"),
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

    return player_summary.sort_values("avg_form", ascending=False)


def build_lineup(team_df):
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
        return pd.DataFrame(), pd.DataFrame()

    agg = (
        candidates.groupby(["player_name", "team_name", "position_name", "role"], as_index=False)
        .agg(
            matches=("match_id", "nunique"),
            avg_form=("predicted_form_score", "mean"),
            max_form=("predicted_form_score", "max"),
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

    agg["selection_score"] = (
        agg["avg_form"] +
        0.7 * agg["matches"] +
        0.05 * agg["pressures"] +
        0.03 * agg["progressive_passes"]
    )

    defender_mask = agg["role"].isin(["RB", "LB", "CB"])
    agg.loc[defender_mask, "selection_score"] += (
        0.08 * agg.loc[defender_mask, "interceptions"] +
        0.05 * agg.loc[defender_mask, "duels_won"]
    )

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
        role_candidates = agg[
            (agg["role"] == role) &
            (~agg["player_name"].isin(already_selected))
        ].copy()

        role_candidates = role_candidates.sort_values(
            ["selection_score", "avg_form", "matches"],
            ascending=False
        )

        selected = role_candidates.head(count).copy()

        for player in selected["player_name"].tolist():
            already_selected.add(player)

        selected_parts.append(selected)

    lineup = pd.concat(selected_parts, ignore_index=True)

    if len(lineup) < 11:
        need = 11 - len(lineup)

        remaining = agg[~agg["player_name"].isin(already_selected)].copy()
        remaining = remaining.sort_values("selection_score", ascending=False).head(need)

        lineup = pd.concat([lineup, remaining], ignore_index=True)

    bench = agg[~agg["player_name"].isin(lineup["player_name"])].copy()
    bench = bench.sort_values("selection_score", ascending=False).head(7)

    lineup = lineup.sort_values(["role", "selection_score"], ascending=[True, False])

    return lineup, bench


def make_russian_table(df):
    result = round_table(df)

    rename_map = {
        "role": "Роль",
        "player_name": "Игрок",
        "position_name": "Позиция",
        "matches": "Матчей",
        "avg_form": "Средняя форма",
        "max_form": "Лучшая форма",
        "selection_score": "Оценка выбора",
        "goals": "Голы",
        "xg": "xG",
        "shots": "Удары",
        "key_passes": "Ключевые передачи",
        "progressive_passes": "Прогрессивные передачи",
        "pressures": "Прессинг",
        "team_name": "Команда",
        "opponent_team": "Соперник",
        "match_date": "Дата",
        "avg_team_form": "Средняя форма команды"
    }

    result = result.rename(columns=rename_map)
    return result


def round_table(df):
    result = df.copy()

    for col in result.columns:
        if pd.api.types.is_float_dtype(result[col]):
            result[col] = result[col].round(2)

    return result


st.title("⚽ CoachMind Football")
st.caption("MVP ИИ-аналитики для футбольного тренера на данных StatsBomb Euro 2024")

if not DATA_FILE.exists() or not MODEL_FILE.exists():
    st.error("Не найдены данные или модель. Сначала запусти скрипты подготовки и обучения.")
    st.stop()

df = load_data()

teams = sorted(df["team_name"].dropna().unique())

with st.sidebar:
    st.header("Настройки")
    selected_team = st.selectbox("Команда", teams, index=teams.index("Spain") if "Spain" in teams else 0)

team_df = df[df["team_name"] == selected_team].copy()
player_summary = get_player_summary(team_df)
lineup, bench = build_lineup(team_df)

tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Центр решений",
    "Обзор команды",
    "Топ игроков",
    "Состав 4-3-3",
    "What-if замена",
    "Нагрузка игроков"
])

with tab0:
    st.subheader("Центр решений тренера")

    st.caption(
        "Это краткая сводка по выбранной команде: форма игроков, рекомендуемый состав "
        "и основные подсказки для тренера."
    )

    if len(player_summary) == 0 or len(lineup) == 0:
        st.warning("Недостаточно данных для центра решений.")
    else:
        best_player = player_summary.iloc[0]
        lineup_score = lineup["selection_score"].sum()
        avg_lineup_form = lineup["avg_form"].mean()

        top_attackers = player_summary.sort_values(
            ["goals", "xg", "avg_form"],
            ascending=False
        ).head(3)

        top_creators = player_summary.sort_values(
            ["key_passes", "progressive_passes", "avg_form"],
            ascending=False
        ).head(3)

        top_pressing = player_summary.sort_values(
            ["pressures", "avg_form"],
            ascending=False
        ).head(3)

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Команда", selected_team)
        col2.metric("Оценка состава", round(lineup_score, 2))
        col3.metric("Средняя форма 11 игроков", round(avg_lineup_form, 2))
        col4.metric("Лучший игрок", best_player["player_name"])

        st.markdown("### Краткий вывод")

        st.success(
            f"Модель рекомендует использовать схему 4-3-3. "
            f"Самый сильный игрок по средней форме: {best_player['player_name']} "
            f"с оценкой {best_player['avg_form']:.2f}."
        )

        st.write(
            "Этот вывод не заменяет решение тренера, но помогает быстро увидеть, "
            "кто сейчас даёт больше пользы по событиям матча."
        )

        st.markdown("### Рекомендуемый стартовый состав")

        main_lineup_table = make_russian_table(lineup[
            [
                "role",
                "player_name",
                "position_name",
                "matches",
                "avg_form",
                "selection_score",
                "goals",
                "xg",
                "key_passes",
                "pressures"
            ]
        ])

        st.dataframe(
            main_lineup_table,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            label="Скачать рекомендуемый состав CSV",
            data=main_lineup_table.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"coachmind_{selected_team}_lineup_433.csv",
            mime="text/csv"
        )

        left_col, middle_col, right_col = st.columns(3)

        with left_col:
            st.markdown("### Атака")
            st.caption("Игроки, которые выделяются по голам, xG и форме.")
            st.dataframe(
                make_russian_table(top_attackers[
                    [
                        "player_name",
                        "matches",
                        "avg_form",
                        "goals",
                        "xg",
                        "shots"
                    ]
                ]),
                use_container_width=True,
                hide_index=True
            )

        with middle_col:
            st.markdown("### Создание моментов")
            st.caption("Игроки, которые дают ключевые и прогрессивные передачи.")
            st.dataframe(
                make_russian_table(top_creators[
                    [
                        "player_name",
                        "matches",
                        "avg_form",
                        "key_passes",
                        "progressive_passes"
                    ]
                ]),
                use_container_width=True,
                hide_index=True
            )

        with right_col:
            st.markdown("### Прессинг")
            st.caption("Игроки, которые активнее работают без мяча.")
            st.dataframe(
                make_russian_table(top_pressing[
                    [
                        "player_name",
                        "matches",
                        "avg_form",
                        "pressures"
                    ]
                ]),
                use_container_width=True,
                hide_index=True
            )

        st.markdown("### Автоматическая подсказка по составу")

        weak_starters = lineup.sort_values("selection_score").head(3)
        strong_bench = bench.sort_values("selection_score", ascending=False).head(3)

        col_a, col_b = st.columns(2)

        with col_a:
            st.write("Игроки старта с самой низкой оценкой выбора")
            st.dataframe(
                make_russian_table(weak_starters[
                    [
                        "role",
                        "player_name",
                        "position_name",
                        "avg_form",
                        "selection_score"
                    ]
                ]),
                use_container_width=True,
                hide_index=True
            )

        with col_b:
            st.write("Сильные кандидаты со скамейки")
            st.dataframe(
                make_russian_table(strong_bench[
                    [
                        "role",
                        "player_name",
                        "position_name",
                        "avg_form",
                        "selection_score"
                    ]
                ]),
                use_container_width=True,
                hide_index=True
            )

        if len(strong_bench) > 0 and len(weak_starters) > 0:
            candidate_in = strong_bench.iloc[0]
            candidate_out = weak_starters.iloc[0]
            diff = candidate_in["selection_score"] - candidate_out["selection_score"]

            if diff > 2:
                st.info(
                    f"Возможный what-if: заменить {candidate_out['player_name']} "
                    f"на {candidate_in['player_name']}. "
                    f"Разница по selection_score: +{diff:.2f}."
                )
            else:
                st.info(
                    "По текущей модели нет очевидной замены, которая сильно усиливает стартовый состав."
                )

        st.markdown("### Что сказать тренеру простыми словами")

        st.markdown(
            f"""
            **{selected_team}** можно анализировать через три блока:

            1. **Форма игроков** — кто чаще создаёт полезные действия.
            2. **Состав 4-3-3** — кого модель предлагает поставить в старт.
            3. **What-if** — что изменится, если заменить одного игрока другим.

            Сейчас система показывает не окончательный ответ, а **подсказку для тренера**.  
            Тренер может сравнить игроков, посмотреть слабые места состава и проверить замену до матча.
            """
        )



with tab1:
    st.subheader(f"Обзор команды: {selected_team}")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Матчей", team_df["match_id"].nunique())
    col2.metric("Игроков", team_df["player_name"].nunique())
    col3.metric("Голов", int(team_df["goals"].sum()))
    col4.metric("Средняя форма", round(team_df["predicted_form_score"].mean(), 2))

    st.write("Средняя форма по матчам")

    match_form = (
        team_df.groupby(["match_id", "match_date", "opponent_team"], as_index=False)
        .agg(
            avg_team_form=("predicted_form_score", "mean"),
            goals=("goals", "sum"),
            xg=("xg", "sum"),
            shots=("shots", "sum"),
            pressures=("pressures", "sum")
        )
        .sort_values("match_date")
    )

    match_form["match_date"] = match_form["match_date"].dt.strftime("%Y-%m-%d")

    st.write("График средней формы по матчам")
    st.line_chart(match_form.set_index("match_date")[["avg_team_form"]])

    st.dataframe(round_table(match_form), use_container_width=True)

with tab2:
    st.subheader("Топ игроков по средней форме")

    st.write("График топ-10 игроков по средней форме")
    top_players_chart = player_summary.head(10).copy()
    st.bar_chart(top_players_chart.set_index("player_name")[["avg_form"]])

    top_players_table = make_russian_table(player_summary[
        [
            "player_name",
            "matches",
            "avg_form",
            "max_form",
            "goals",
            "xg",
            "shots",
            "key_passes",
            "progressive_passes",
            "pressures"
        ]
    ])

    st.dataframe(
        top_players_table,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        label="Скачать топ игроков CSV",
        data=top_players_table.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"coachmind_{selected_team}_top_players.csv",
        mime="text/csv"
    )

with tab3:
    st.subheader("Рекомендуемый состав 4-3-3")

    if len(lineup) == 0:
        st.warning("Не удалось построить состав.")
    else:
        lineup_score = lineup["selection_score"].sum()
        avg_lineup_form = lineup["avg_form"].mean()
        lineup_goals = lineup["goals"].sum()

        col1, col2, col3 = st.columns(3)
        col1.metric("Оценка состава", round(lineup_score, 2))
        col2.metric("Средняя форма состава", round(avg_lineup_form, 2))
        col3.metric("Голов у выбранных игроков", int(lineup_goals))

        st.write("Рекомендуемые 11 игроков под схему 4-3-3")

        st.dataframe(
            round_table(lineup[
                [
                    "role",
                    "player_name",
                    "position_name",
                    "matches",
                    "avg_form",
                    "max_form",
                    "selection_score",
                    "goals",
                    "xg",
                    "key_passes",
                    "progressive_passes",
                    "pressures"
                ]
            ]),
            use_container_width=True
        )

        st.subheader("Запасные кандидаты")
        st.dataframe(
            round_table(bench[
                [
                    "role",
                    "player_name",
                    "position_name",
                    "matches",
                    "avg_form",
                    "selection_score",
                    "goals",
                    "xg",
                    "key_passes",
                    "progressive_passes",
                    "pressures"
                ]
            ]),
            use_container_width=True
        )

with tab4:
    st.subheader("What-if замена игрока")

    if len(lineup) == 0:
        st.warning("Сначала нужен стартовый состав.")
    else:
        current_players = lineup["player_name"].tolist()

        available_players = player_summary[
            ~player_summary["player_name"].isin(current_players)
        ]["player_name"].tolist()

        out_player_name = st.selectbox("Кого убрать из состава", current_players)
        in_player_name = st.selectbox("Кого поставить вместо него", available_players)

        out_row = lineup[lineup["player_name"] == out_player_name].iloc[0]
        in_row = player_summary[player_summary["player_name"] == in_player_name].iloc[0]

        old_score = lineup["selection_score"].sum()

        in_score = (
            in_row["avg_form"] +
            0.7 * in_row["matches"] +
            0.05 * in_row["pressures"] +
            0.03 * in_row["progressive_passes"] +
            1.2 * in_row["xg"] +
            0.35 * in_row["key_passes"]
        )

        new_score = old_score - out_row["selection_score"] + in_score
        diff = new_score - old_score

        col1, col2, col3 = st.columns(3)
        col1.metric("Старый состав", round(old_score, 2))
        col2.metric("Новый состав", round(new_score, 2))
        col3.metric("Разница", round(diff, 2))

        if diff > 2:
            st.success("Замена выглядит полезной по модели.")
            st.write("Модель считает, что новый игрок может усилить состав по текущим игровым показателям.")
        elif diff < -2:
            st.error("Замена выглядит рискованной по модели.")
            st.write("Модель считает, что текущий игрок лучше подходит для состава по суммарной оценке.")
        else:
            st.info("Замена почти нейтральная по модели.")
            st.write("Разница небольшая, поэтому решение лучше принимать тренеру с учётом тактики матча.")

        compare = pd.DataFrame([
            {
                "type": "current player",
                "player": out_row["player_name"],
                "matches": out_row["matches"],
                "avg_form": out_row["avg_form"],
                "selection_score": out_row["selection_score"],
                "goals": out_row["goals"],
                "xg": out_row["xg"],
                "key_passes": out_row["key_passes"],
                "progressive_passes": out_row["progressive_passes"],
                "pressures": out_row["pressures"]
            },
            {
                "type": "replacement",
                "player": in_row["player_name"],
                "matches": in_row["matches"],
                "avg_form": in_row["avg_form"],
                "selection_score": in_score,
                "goals": in_row["goals"],
                "xg": in_row["xg"],
                "key_passes": in_row["key_passes"],
                "progressive_passes": in_row["progressive_passes"],
                "pressures": in_row["pressures"]
            }
        ])

        st.subheader("Сравнение игроков")
        st.dataframe(round_table(compare), use_container_width=True)

with tab5:
    st.subheader("Load Agent — физическая нагрузка игроков")

    st.caption(
        "Здесь используются открытые данные SkillCorner по AUS A-League 2024/2025. "
        "Это отдельный демо-модуль физической нагрузки, он не смешивается с игроками Euro 2024."
    )

    load_df = load_skillcorner_load_data()

    if len(load_df) == 0:
        st.warning("Файл нагрузки ещё не создан. Сначала запусти: python src/10_prepare_skillcorner_load.py")
    else:
        load_teams = sorted(load_df["team_name"].dropna().unique())

        selected_load_team = st.selectbox(
            "Команда SkillCorner",
            load_teams,
            key="skillcorner_team"
        )

        team_load = load_df[load_df["team_name"] == selected_load_team].copy()
        team_load = team_load.sort_values("load_index", ascending=False)

        st.info(
            "Эти клубы взяты из SkillCorner: AUS - A-League 2024/2025. "
            "Это отдельная лига и отдельный набор данных для демонстрации нагрузки."
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Игроков", team_load["player_name"].nunique())
        col2.metric("Средний load_index", round(team_load["load_index"].mean(), 2))
        col3.metric("Игроков с высокой нагрузкой", int((team_load["load_level"] == "high").sum()))
        col4.metric("Средняя скорость, м/мин", round(team_load["total_metersperminute_full_all"].mean(), 2))

        load_view1, load_view2, load_view3 = st.tabs([
            "Рейтинг нагрузки",
            "Карта интенсивности",
            "Рекомендации тренеру"
        ])

        with load_view1:
            st.write("Топ-10 игроков по индексу нагрузки")

            top_load = team_load.head(10).copy()
            top_load = top_load.sort_values("load_index", ascending=True)

            fig = px.bar(
                top_load,
                x="load_index",
                y="player_name",
                orientation="h",
                color="load_level",
                text="load_index",
                hover_data=[
                    "position_group",
                    "count_match",
                    "distance_per_match",
                    "sprint_per_match",
                    "top_speed"
                ],
                labels={
                    "load_index": "Индекс нагрузки",
                    "player_name": "Игрок",
                    "load_level": "Уровень нагрузки"
                }
            )

            fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
            fig.update_layout(
                height=520,
                xaxis_range=[0, 100],
                yaxis_title="",
                xaxis_title="load_index от 0 до 100",
                showlegend=True
            )

            st.plotly_chart(fig, use_container_width=True)

            st.write("Таблица нагрузки игроков")

            display_cols = [
                "player_name",
                "position_group",
                "count_match",
                "distance_per_match",
                "running_per_match",
                "hsr_per_match",
                "sprint_per_match",
                "top_speed",
                "load_index",
                "load_level",
                "load_comment"
            ]

            renamed = team_load[display_cols].copy()
            renamed = renamed.rename(columns={
                "player_name": "Игрок",
                "position_group": "Позиционная группа",
                "count_match": "Матчей",
                "distance_per_match": "Дистанция за матч, м",
                "running_per_match": "Бег за матч, м",
                "hsr_per_match": "Быстрый бег за матч, м",
                "sprint_per_match": "Спринт за матч, м",
                "top_speed": "Макс. скорость, км/ч",
                "load_index": "Индекс нагрузки",
                "load_level": "Уровень",
                "load_comment": "Комментарий"
            })

            st.dataframe(renamed, use_container_width=True)

        with load_view2:
            st.write("Карта интенсивности: скорость команды и спринтовая нагрузка")

            fig2 = px.scatter(
                team_load,
                x="total_metersperminute_full_all",
                y="sprint_per_match",
                size="distance_per_match",
                color="load_level",
                hover_name="player_name",
                hover_data=[
                    "position_group",
                    "load_index",
                    "top_speed",
                    "hsr_per_match"
                ],
                labels={
                    "total_metersperminute_full_all": "Средняя интенсивность, м/мин",
                    "sprint_per_match": "Спринтовая дистанция за матч, м",
                    "distance_per_match": "Дистанция за матч",
                    "load_level": "Уровень нагрузки"
                }
            )

            fig2.update_layout(height=560)

            st.plotly_chart(fig2, use_container_width=True)

            st.caption(
                "Чем правее игрок — тем выше средняя интенсивность. "
                "Чем выше точка — тем больше спринтовой дистанции. "
                "Размер точки показывает общую дистанцию за матч."
            )

        with load_view3:
            st.write("Игроки с высокой нагрузкой")

            high_load = team_load[team_load["load_level"] == "high"].copy()

            if len(high_load) == 0:
                st.success("В выбранной команде нет игроков с высокой нагрузкой по текущему индексу.")
            else:
                for _, row in high_load.head(8).iterrows():
                    st.warning(
                        f"{row['player_name']} — load_index {row['load_index']:.1f}. "
                        f"Позиция: {row['position_group']}. "
                        f"Дистанция за матч: {row['distance_per_match']:.0f} м, "
                        f"спринт: {row['sprint_per_match']:.0f} м, "
                        f"макс. скорость: {row['top_speed']:.1f} км/ч."
                    )

            st.write("Как читать load_index")

            st.markdown(
                """
                **load_index** — это условный индекс интенсивности от 0 до 100.

                Он учитывает:
                - общую дистанцию за матч;
                - метры в минуту;
                - быстрый бег;
                - спринты;
                - ускорения и торможения;
                - максимальную скорость.

                **Важно:** это не медицинский диагноз и не прогноз травмы.  
                Это подсказка тренеру: кто выполняет много интенсивной работы и кому нужно внимательнее планировать минуты.
                """
            )

            st.write("Практическая логика для тренера")

            st.markdown(
                """
                - **Высокая нагрузка** — игрок много бегает на высокой скорости. Перед следующим матчем стоит осторожнее планировать минуты.
                - **Средняя нагрузка** — игрок активно вовлечён, но без явного перегруза по индексу.
                - **Низкая нагрузка** — игрок меньше вовлечён физически, но это может быть связано с позицией или ролью.
                """
            )


st.divider()
st.caption(
    "MVP использует StatsBomb для формы и состава, а SkillCorner — для отдельного демо-модуля физической нагрузки."
)
