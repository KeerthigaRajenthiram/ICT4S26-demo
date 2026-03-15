from __future__ import annotations

import json
from typing import Dict, Any, Optional, List

import pandas as pd
import streamlit as st
import plotly.express as px

from data.access import (
    get_db_stats,
    list_datasets,
    get_dataset_by_task_id,
    get_best_per_framework,
    get_candidates_flat,
    filter_topk_by_metric,
    pareto_front
)
from data.recommend import (
    Constraints,
    pick_winner,
    pick_alternatives,
    ACC_COL,
    LAT_COL,
    ENERGY_COL,
)


def _dataset_card(ds: Dict[str, Any]) -> None:
    st.markdown("### Dataset")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Name", ds.get("name", "—"))
    c2.metric("Task ID", str(ds.get("task_id", "—")))
    c3.metric("Rows", str(ds.get("rows", "—")))
    c4.metric("Cols", str(ds.get("cols", "—")))
    st.caption(
        f"dataset_id: {ds.get('dataset_id')} • task_type: {ds.get('task_type')} • target: {ds.get('target_col')}"
    )


def _winner_card(row: pd.Series) -> None:
    st.markdown("### 🏆 Winner")
    st.write(
        f"**{row.get('algorithm')}**  • framework: `{row.get('framework')}`  • candidate: `{row.get('candidate_id')}`"
    )

    acc = row.get(ACC_COL, None)
    lat = row.get(LAT_COL, None)
    energy = row.get(ENERGY_COL, None)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{acc:.4f}" if pd.notna(acc) else "—")
    c2.metric("Energy (kWh)", f"{energy:.2e}" if pd.notna(energy) else "—")
    c3.metric("Latency (ms/row)", f"{lat:.2f}" if pd.notna(lat) else "—")
    c4.metric("Train time (s)", f"{row.get('training_duration_secs', '—')}")


def _pretty_json(text: str) -> None:
    try:
        obj = json.loads(text) if text else {}
        st.json(obj)
    except Exception:
        st.code(text or "", language="json")


def render_home_page(db_ok: bool) -> None:
    st.title("🤖 AutoML Knowledge Base Agent")
    st.write("A conversational, intent-aware assistant for selecting ML model configurations.")
    st.write("- Uses historical AutoML runs stored in `runs.db`")
    st.write("- Optimizes for your goal: accuracy / energy / latency")
    st.write("- Supports hard constraints (e.g., min accuracy, max latency)")

    st.divider()

    if db_ok:
        stats = get_db_stats()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Datasets", stats["datasets"])
        c2.metric("Experiments", stats["experiments"])
        c3.metric("Candidates", stats["candidates"])
        c4.metric("Evaluations", stats["topk_evaluations"])
    else:
        st.warning("DB not connected. Check `automl_kb/config.py` DB_PATH and ensure runs.db exists.")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🔎 Explore Sustainability-Aware Knowledge Base")
        st.caption("Browse datasets already stored, inspect metrics, and view trade-offs.")
    with col2:
        st.markdown("#### 🤖 Get recommendation")
        st.caption("Enter an OpenML Task ID (v1 requires it exists in the DB).")


def render_explore_page(db_ok: bool) -> None:
    st.title("Explore Sustainability-Aware Knowledge Base")
    if not db_ok:
        st.error("DB not connected.")
        return

    ds_df = list_datasets()
    if ds_df.empty:
        st.info("No datasets found in DB.")
        return

    # Controls row
    labels = ds_df.apply(lambda r: f"{r['name']} — {r['dataset_id']} (task {r['task_id']})", axis=1).tolist()
    selected_label = st.selectbox("Select dataset", labels)
    ds = ds_df.iloc[labels.index(selected_label)].to_dict()
    dataset_id = ds["dataset_id"]

    _dataset_card(ds)

    df = get_candidates_flat(dataset_id)


    st.divider()
    st.subheader("🏆 Framework Leaderboard (Best per framework)")

    metric_choice = st.selectbox(
        "Leaderboard metric",
        ["Accuracy", "Energy (kWh)", "Latency (ms/row)"],
        index=0
    )

    if metric_choice == "Accuracy":
        metric_type, metric_name, higher_is_better = "performance", "accuracy", True
    elif metric_choice == "Energy (kWh)":
        metric_type, metric_name, higher_is_better = "energy", "inference_energy_kwh", False
    else:
        metric_type, metric_name, higher_is_better = "timing", "inference_latency_per_row_ms", False

    leaderboard = get_best_per_framework(
        dataset_id,
        metric_type=metric_type,
        metric_name=metric_name,
        higher_is_better=higher_is_better
    )


    st.dataframe(leaderboard, use_container_width=True, hide_index=True)


    frameworks = sorted(df["framework"].dropna().unique().tolist())

    st.divider()
    st.subheader("Filter by framework")
    colA, colB = st.columns([2, 1])
    with colA:
        framework_filter = st.multiselect(
            "Framework filter",
            options=frameworks,
            default=frameworks,
        )
    with colB:
        top_k = st.slider(
            "Best models (Top-K evaluated candidates)",
            min_value=1,
            max_value=20,
            value=10,
            step=1,
            help="Top-K candidates ranked by the selected leaderboard metric"
        )

    view = df.copy()
    if framework_filter:
        view = view[view["framework"].isin(framework_filter)]

    # metric_col depends on selected leaderboard metric
    metric_col = {
        "Accuracy": ACC_COL,
        "Energy (kWh)": ENERGY_COL,
        "Latency (ms/row)": LAT_COL
    }.get(metric_choice, ACC_COL)

    higher_is_better = (metric_choice == "Accuracy")

    view = filter_topk_by_metric(view, metric_col=metric_col, k=top_k, higher_is_better=higher_is_better)

    tabs = st.tabs(["Trade-offs", "Candidates"])

    with tabs[0]:

        # # ---------------------------
        # # Pareto front toggle
        # # ---------------------------
        # show_pareto = st.checkbox(
        #     "Highlight Pareto-optimal models",
        #     value=True,
        #     help="Models that are not dominated in Accuracy–Energy–Latency space"
        # )
        #
        # plot_df = view.copy()
        #
        # if show_pareto:
        #     pareto_mask = pareto_front(
        #         plot_df,
        #         maximize=[ACC_COL],
        #         minimize=[ENERGY_COL, LAT_COL]
        #     )
        #     plot_df["is_pareto"] = pareto_mask
        # else:
        #     plot_df["is_pareto"] = False
        #
        # # ---------------------------
        # # Scatter: Accuracy vs Energy
        # # ---------------------------
        # fig = px.scatter(
        #     plot_df,
        #     x=ENERGY_COL,
        #     y=ACC_COL,
        #     color="framework",
        #     symbol="is_pareto",
        #     symbol_map={True: "star", False: "circle"},
        #     hover_data=["framework", "algorithm", "candidate_id"],
        # )
        #
        # # Emphasize Pareto points
        # fig.update_traces(
        #     marker=dict(size=11),
        #     selector=dict(marker_symbol="star")
        # )
        #
        # st.plotly_chart(fig, use_container_width=True)
        #
        # st.caption(
        #     f"Pareto-optimal models: {plot_df['is_pareto'].sum()} / {len(plot_df)} "
        #     "(maximize accuracy, minimize energy & latency)"
        # )
        #
        # --- Highlight winners toggle (added) ---
        highlight_winners = st.checkbox(
            f"Highlight framework winners (by {metric_choice})",
            value=True
        )

        winner_ids = set()
        if highlight_winners and metric_col in view.columns:
            # pick 1 winner per framework using the selected metric
            ascending = not higher_is_better
            for fw, grp in view.dropna(subset=[metric_col]).groupby("framework"):
                top = grp.sort_values(metric_col, ascending=ascending).head(1)
                if not top.empty:
                    winner_ids.add(top.iloc[0]["candidate_id"])

        st.markdown("#### Accuracy vs Energy (kWh)")

        if ACC_COL in view.columns and ENERGY_COL in view.columns:
            fig1 = px.scatter(
                view,
                x=ENERGY_COL,
                y=ACC_COL,
                color="framework",
                hover_data=["algorithm", "candidate_id"],
            )

            # --- Overlay winners (added) ---
            if highlight_winners and winner_ids:
                winners_df = view[view["candidate_id"].isin(winner_ids)].dropna(subset=[ACC_COL, ENERGY_COL])
                if not winners_df.empty:
                    fig1.add_scatter(
                        x=winners_df[ENERGY_COL],
                        y=winners_df[ACC_COL],
                        mode="markers",
                        name=f"Winners ({metric_choice})",
                        marker=dict(size=14, symbol="circle-open"),
                        text=winners_df["candidate_id"],
                    )

            st.plotly_chart(fig1, use_container_width=True)
        else:
            st.info("Missing columns for Accuracy vs Energy plot.")

        st.markdown("#### Accuracy vs Latency (ms/row)")
        if ACC_COL in view.columns and LAT_COL in view.columns:
            fig2 = px.scatter(
                view,
                x=LAT_COL,
                y=ACC_COL,
                color="framework",
                hover_data=["algorithm", "candidate_id"],
            )

            # --- Overlay winners (added) ---
            if highlight_winners and winner_ids:
                winners_df = view[view["candidate_id"].isin(winner_ids)].dropna(subset=[ACC_COL, LAT_COL])
                if not winners_df.empty:
                    fig2.add_scatter(
                        x=winners_df[LAT_COL],
                        y=winners_df[ACC_COL],
                        mode="markers",
                        name=f"Winners ({metric_choice})",
                        marker=dict(size=14, symbol="circle-open"),
                        text=winners_df["candidate_id"],
                    )

            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Missing columns for Accuracy vs Latency plot.")

    with tabs[1]:
        st.subheader("Candidates")
        q = st.text_input("Search (algorithm/framework/candidate_id)", "")
        table = view.copy()
        if q.strip():
            mask = (
                table["algorithm"].astype(str).str.contains(q, case=False, na=False)
                | table["framework"].astype(str).str.contains(q, case=False, na=False)
                | table["candidate_id"].astype(str).str.contains(q, case=False, na=False)
            )
            table = table[mask]

        st.caption(f"Showing {min(len(table), 200)} of {len(table)} rows")
        st.dataframe(table.head(200), use_container_width=True, hide_index=True)

        st.divider()
        st.markdown("#### Inspect one candidate")
        cand_ids = table["candidate_id"].dropna().unique().tolist()
        if cand_ids:
            chosen = st.selectbox("Candidate ID", cand_ids)
            row = table[table["candidate_id"] == chosen].head(1)
            if not row.empty:
                st.markdown("**Params JSON**")
                _pretty_json(row.iloc[0].get("params_json", ""))


def render_recommend_page(db_ok: bool) -> None:
    st.title("Get recommendation")
    if not db_ok:
        st.error("DB not connected.")
        return

    left, right = st.columns([1.6, 1.0], gap="large")

    with left:
        if st.button("🔄 New conversation"):
            for k in list(st.session_state.keys()):
                if k.startswith("rec_") or k == "chat":
                    del st.session_state[k]
            st.session_state.rec_state = "ASK_TASK"
            st.rerun()

        # -------------------------
        # Session state init
        # -------------------------
        if "chat" not in st.session_state:
            st.session_state.chat = []

        if "rec_state" not in st.session_state:
            st.session_state.rec_state = "ASK_TASK"

        # Conversation context
        if "rec_task_id" not in st.session_state:
            st.session_state.rec_task_id = None
        if "rec_dataset" not in st.session_state:
            st.session_state.rec_dataset = None

        # Decision variables (CLI-like)
        if "rec_show_leaderboard" not in st.session_state:
            st.session_state.rec_show_leaderboard = None  # True/False
        if "rec_goal" not in st.session_state:
            st.session_state.rec_goal = None  # accuracy/energy/latency

        if "rec_has_constraints" not in st.session_state:
            st.session_state.rec_has_constraints = None  # True/False
        if "rec_use_max_latency" not in st.session_state:
            st.session_state.rec_use_max_latency = None  # True/False
        if "rec_max_latency" not in st.session_state:
            st.session_state.rec_max_latency = 10.0

        if "rec_use_min_accuracy" not in st.session_state:
            st.session_state.rec_use_min_accuracy = None  # True/False
        if "rec_min_accuracy" not in st.session_state:
            st.session_state.rec_min_accuracy = 0.90

        # Use same "Top-K evaluated candidates" concept as Explore (data has up to 20)
        if "rec_top_k" not in st.session_state:
            st.session_state.rec_top_k = 20

        # Persisted artifacts so they don't vanish on rerun
        if "rec_leaderboard_df" not in st.session_state:
            st.session_state.rec_leaderboard_df = None
        if "rec_winner_row" not in st.session_state:
            st.session_state.rec_winner_row = None
        if "rec_alts_df" not in st.session_state:
            st.session_state.rec_alts_df = None
        if "rec_winner_params_json" not in st.session_state:
            st.session_state.rec_winner_params_json = None

        if "rec_show_config" not in st.session_state:
            st.session_state.rec_show_config = False
        if "rec_config_json" not in st.session_state:
            st.session_state.rec_config_json = None


        # -------------------------
        # Helpers
        # -------------------------
        def say(role: str, text: str):
            st.session_state.chat.append({"role": role, "text": text})

        def render_chat():
            for m in st.session_state.chat:
                st.chat_message(m["role"]).write(m["text"])

        def parse_yes_no(x: str) -> str:
            x = (x or "").strip().lower()
            if x in ["y", "yes"]:
                return "yes"
            if x in ["n", "no"]:
                return "no"
            return "invalid"

        def metric_col_for_goal(goal: str) -> tuple[str, bool]:
            # Uses the same pivot columns as Explore (from topk_evaluations)
            if goal == "accuracy":
                return ACC_COL, True
            if goal == "energy":
                return ENERGY_COL, False
            return LAT_COL, False  # latency

        def winner_summary_line(row) -> str:
            acc = row.get(ACC_COL, None)
            en = row.get(ENERGY_COL, None)
            lat = row.get(LAT_COL, None)

            acc_s = f"{acc:.4f}" if acc is not None and pd.notna(acc) else "—"
            en_s = f"{en:.2e} kWh" if en is not None and pd.notna(en) else "—"

            if lat is None or pd.isna(lat):
                lat_s = "—"
            elif lat < 0.01:
                lat_s = f"{lat:.4f} ms"  # ✅ avoids 0.00 for tiny values
            else:
                lat_s = f"{lat:.2f} ms"

            return (
                f"**Top Recommendation:** `{row.get('algorithm')}` ({row.get('framework')})  \n"
                f"Score (Accuracy): **{acc_s}**  |  Energy: **{en_s}**  |  Latency: **{lat_s}**"
            )

        def build_alts_table(alts_df: pd.DataFrame) -> pd.DataFrame:
            out = pd.DataFrame()
            out["Type"] = alts_df["type"]
            out["Framework"] = alts_df["framework"]
            out["Algo"] = alts_df["algorithm"]
            out["Acc"] = alts_df[ACC_COL] if ACC_COL in alts_df.columns else None
            out["Energy"] = alts_df[ENERGY_COL] if ENERGY_COL in alts_df.columns else None
            out["Latency"] = alts_df[LAT_COL] if LAT_COL in alts_df.columns else None
            return out

        # -------------------------
        # Start message
        # -------------------------
        if not st.session_state.chat:
            say("assistant", "AutoML Knowledge Base Agent 🤖\n\nEnter OpenML **Task ID** (e.g., 14965).")

        # Render chat history
        render_chat()

        # -------------------------
        # State machine (ONE chat_input per state)
        # -------------------------

        # 1) Ask Task ID
        if st.session_state.rec_state == "ASK_TASK":
            task_id = st.chat_input("Task ID")
            if task_id:
                try:
                    task_id_int = int(task_id.strip())
                except ValueError:
                    say("user", task_id)
                    say("assistant", "That doesn’t look like a number. Please enter a numeric Task ID (e.g., 14965).")
                    st.rerun()

                say("user", str(task_id_int))

                ds = get_dataset_by_task_id(task_id_int)
                st.session_state.rec_task_id = task_id_int
                st.session_state.rec_dataset = ds

                # Clear old artifacts
                st.session_state.rec_leaderboard_df = None
                st.session_state.rec_winner_row = None
                st.session_state.rec_alts_df = None
                st.session_state.rec_winner_params_json = None
                st.session_state.rec_show_config = False
                st.session_state.rec_config_json = None

                if not ds:
                    say("assistant", f"I couldn’t find Task ID **{task_id_int}** in the knowledge base.")
                    say("assistant", "Try another Task ID. (V2 will support dataset similarity search for unseen datasets.)")
                    st.rerun()

                say("assistant", f"Dataset found: **{ds['name']}** ({ds['rows']} rows).")
                say("assistant", "Show the leaderboard (best from each tool)? **[y/n]**")
                st.session_state.rec_state = "ASK_SHOW_LEADERBOARD"
                st.rerun()

        # 2) Ask whether to show leaderboard
        if st.session_state.rec_state == "ASK_SHOW_LEADERBOARD":
            ans = st.chat_input("y / n")
            if ans:
                say("user", ans)
                yn = parse_yes_no(ans)
                if yn == "invalid":
                    say("assistant", "Please answer **y** or **n**.")
                    st.rerun()

                st.session_state.rec_show_leaderboard = (yn == "yes")

                ds = st.session_state.rec_dataset
                if st.session_state.rec_show_leaderboard:
                    lb = get_best_per_framework(
                        ds["dataset_id"],
                        metric_type="performance",
                        metric_name="accuracy",
                        higher_is_better=True
                    )
                    st.session_state.rec_leaderboard_df = lb
                    say("assistant", "Leaderboard loaded ✅")

                say("assistant", "What is your primary goal? **[accuracy/energy/latency]**")
                st.session_state.rec_state = "ASK_GOAL"
                st.rerun()

        # 3) Ask goal
        if st.session_state.rec_state == "ASK_GOAL":
            goal = st.chat_input("accuracy / energy / latency")
            if goal:
                goal_norm = goal.strip().lower()
                say("user", goal_norm)

                if goal_norm not in ["accuracy", "energy", "latency"]:
                    say("assistant", "Please choose one of: **accuracy**, **energy**, **latency**.")
                    st.rerun()

                st.session_state.rec_goal = goal_norm
                say("assistant", "Do you have hard constraints? **[y/n]**")
                st.session_state.rec_state = "ASK_HAS_CONSTRAINTS"
                st.rerun()

        # 4) Ask hard constraints?
        if st.session_state.rec_state == "ASK_HAS_CONSTRAINTS":
            ans = st.chat_input("y / n")
            if ans:
                say("user", ans)
                yn = parse_yes_no(ans)
                if yn == "invalid":
                    say("assistant", "Please answer **y** or **n**.")
                    st.rerun()

                st.session_state.rec_has_constraints = (yn == "yes")

                if not st.session_state.rec_has_constraints:
                    say("assistant", "Okay — computing recommendation…")
                    st.session_state.rec_state = "COMPUTE"
                    st.rerun()

                say("assistant", "Max inference latency constraint? **[y/n]**")
                st.session_state.rec_state = "ASK_USE_MAX_LAT"
                st.rerun()

        # 5) Ask: use max latency?
        if st.session_state.rec_state == "ASK_USE_MAX_LAT":
            ans = st.chat_input("y / n")
            if ans:
                say("user", ans)
                yn = parse_yes_no(ans)
                if yn == "invalid":
                    say("assistant", "Please answer **y** or **n**.")
                    st.rerun()

                st.session_state.rec_use_max_latency = (yn == "yes")

                if st.session_state.rec_use_max_latency:
                    say("assistant", "Max ms (e.g., 10)")
                    st.session_state.rec_state = "ASK_MAX_LAT_VAL"
                    st.rerun()

                say("assistant", "Min accuracy constraint? **[y/n]**")
                st.session_state.rec_state = "ASK_USE_MIN_ACC"
                st.rerun()

        # 6) Ask max latency value
        if st.session_state.rec_state == "ASK_MAX_LAT_VAL":
            ans = st.chat_input("Max latency (ms/row)")
            if ans:
                say("user", ans)
                try:
                    st.session_state.rec_max_latency = float(ans)
                except ValueError:
                    say("assistant", "Please enter a valid number (e.g., 10).")
                    st.rerun()

                say("assistant", "Min accuracy constraint? **[y/n]**")
                st.session_state.rec_state = "ASK_USE_MIN_ACC"
                st.rerun()

        # 7) Ask: use min accuracy?
        if st.session_state.rec_state == "ASK_USE_MIN_ACC":
            ans = st.chat_input("y / n")
            if ans:
                say("user", ans)
                yn = parse_yes_no(ans)
                if yn == "invalid":
                    say("assistant", "Please answer **y** or **n**.")
                    st.rerun()

                st.session_state.rec_use_min_accuracy = (yn == "yes")

                if st.session_state.rec_use_min_accuracy:
                    say("assistant", "Min accuracy value (0–1), e.g., 0.90")
                    st.session_state.rec_state = "ASK_MIN_ACC_VAL"
                    st.rerun()

                say("assistant", "Great — computing recommendation…")
                st.session_state.rec_state = "COMPUTE"
                st.rerun()

        # 8) Ask min accuracy value
        if st.session_state.rec_state == "ASK_MIN_ACC_VAL":
            ans = st.chat_input("Min accuracy (0–1)")
            if ans:
                say("user", ans)
                try:
                    v = float(ans)
                    if v < 0 or v > 1:
                        raise ValueError()
                    st.session_state.rec_min_accuracy = v
                except ValueError:
                    say("assistant", "Please enter a number between 0 and 1 (e.g., 0.90).")
                    st.rerun()

                say("assistant", "Great — computing recommendation…")
                st.session_state.rec_state = "COMPUTE"
                st.rerun()

        # 9) Compute and persist results
        if st.session_state.rec_state == "COMPUTE":
            ds = st.session_state.rec_dataset
            goal = st.session_state.rec_goal or "accuracy"

            df = get_candidates_flat(ds["dataset_id"])

            metric_col, higher_is_better = metric_col_for_goal(goal)
            top_k = int(st.session_state.rec_top_k)
            df = filter_topk_by_metric(df, metric_col=metric_col, k=top_k, higher_is_better=higher_is_better)

            cons = Constraints()
            if st.session_state.rec_has_constraints:
                if st.session_state.rec_use_max_latency:
                    cons.max_latency_ms = float(st.session_state.rec_max_latency)
                if st.session_state.rec_use_min_accuracy:
                    cons.min_accuracy = float(st.session_state.rec_min_accuracy)

            winner_df, filtered = pick_winner(df, goal, cons)
            st.write("Winner latency raw:", float(winner_df.iloc[0][LAT_COL]))

            if winner_df.empty:
                say("assistant", "No candidates satisfy the constraints. Try relaxing them.")
                st.session_state.rec_state = "ASK_HAS_CONSTRAINTS"
                st.rerun()

            winner_row = winner_df.iloc[0].to_dict()
            winner_row["_summary_md"] = winner_summary_line(winner_row)

            st.session_state.rec_winner_row = winner_row
            st.session_state.rec_winner_params_json = winner_row.get("params_json", "")

            alts_df = pick_alternatives(filtered)
            if not alts_df.empty:
                st.session_state.rec_alts_df = build_alts_table(alts_df)
            else:
                st.session_state.rec_alts_df = None

            say("assistant", "Do you want to explore the configuration of the winner? **[y/n]**")
            st.session_state.rec_state = "ASK_EXPLORE_CFG"
            st.rerun()

        # 10) Explore config?
        if st.session_state.rec_state == "ASK_EXPLORE_CFG":
            ans = st.chat_input("y / n")
            if ans:
                say("user", ans)
                yn = parse_yes_no(ans)
                if yn == "invalid":
                    say("assistant", "Please answer **y** or **n**.")
                    st.rerun()

                if yn == "yes":
                    with st.expander("Winner configuration", expanded=True):
                        st.session_state.rec_show_config = True
                        st.session_state.rec_config_json = st.session_state.rec_winner_params_json
                        st.markdown("**Params JSON**")
                        _pretty_json(st.session_state.rec_winner_params_json)

                say("assistant", "Done ✅ Enter another Task ID to try again.")
                st.session_state.rec_state = "ASK_TASK"
                st.rerun()

    # Right panel (context + tables) — kept here, no duplicate rendering below chat
    with right:
        # st.subheader("Context")
        if st.session_state.get("rec_dataset"):
            _dataset_card(st.session_state.rec_dataset)

        if st.session_state.get("rec_leaderboard_df") is not None:
            st.markdown("### 🏆 Framework Leaderboard")
            st.dataframe(st.session_state.rec_leaderboard_df, use_container_width=True, hide_index=True)

        if st.session_state.get("rec_winner_row") is not None:
            st.markdown("### Winner")
            st.markdown(st.session_state.rec_winner_row.get("_summary_md", ""))

        if st.session_state.get("rec_alts_df") is not None:
            st.markdown("### Alternatives")
            st.dataframe(st.session_state.rec_alts_df, use_container_width=True, hide_index=True)

        if st.session_state.get("rec_show_config") and st.session_state.get("rec_config_json"):
            st.markdown("### Winner configuration")
            with st.expander("Params JSON", expanded=True):
                _pretty_json(st.session_state.rec_config_json)


def render_about_page() -> None:
    st.title("About")
    st.write(
        """
        This Streamlit app is a v1 GUI for your AutoML Knowledge Base.

        **What it does**
        - Reads historical AutoML runs from `runs.db`
        - Lets users explore datasets + candidate models
        - Recommends a configuration based on an intent (accuracy/energy/latency) and constraints

        **Metrics source**
        - `topk_evaluations` table (phase='inference') stores performance/energy/timing metrics.

        **V2 plan**
        - If a task is not present in DB, route to dataset similarity and infer promising configs.
        """
    )
