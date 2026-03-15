from __future__ import annotations

import os
import sqlite3
from typing import Optional, Tuple, Dict, Any

import pandas as pd
import streamlit as st


def _get_db_path() -> Optional[str]:
    try:
        from automl_kb.config import DB_PATH
        return DB_PATH
    except Exception:
        return None


def db_health() -> Tuple[bool, str]:
    db_path = _get_db_path()
    if not db_path:
        return False, "Could not import automl_kb.config.DB_PATH"
    if not os.path.exists(db_path):
        return False, f"DB file not found at: {db_path}"
    try:
        conn = sqlite3.connect(db_path)
        conn.execute("SELECT 1;")
        conn.close()
        return True, f"Using: {db_path}"
    except Exception as e:
        return False, str(e)


def _connect() -> sqlite3.Connection:
    db_path = _get_db_path()
    if not db_path:
        raise RuntimeError("Could not import DB_PATH from automl_kb.config")
    return sqlite3.connect(db_path)


@st.cache_data(show_spinner=False)
def get_db_stats() -> Dict[str, int]:
    with _connect() as conn:
        stats = {}
        for table in ["datasets", "experiments", "candidates", "topk_evaluations"]:
            stats[table] = int(pd.read_sql_query(f"SELECT COUNT(*) AS n FROM {table}", conn)["n"].iloc[0])
        return stats


@st.cache_data(show_spinner=False)
def list_datasets() -> pd.DataFrame:
    with _connect() as conn:
        return pd.read_sql_query(
            """
            SELECT dataset_id, task_id, name, task_type, rows, cols, target_col
            FROM datasets
            ORDER BY name
            """,
            conn,
        )


@st.cache_data(show_spinner=False)
def get_dataset_by_task_id(task_id: int) -> Optional[Dict[str, Any]]:
    with _connect() as conn:
        df = pd.read_sql_query(
            """
            SELECT dataset_id, task_id, name, task_type, rows, cols, target_col
            FROM datasets
            WHERE task_id = ?
            """,
            conn,
            params=(task_id,),
        )
    if df.empty:
        return None
    return df.iloc[0].to_dict()


@st.cache_data(show_spinner=False)
def get_best_per_framework(
    dataset_id: str,
    metric_type: str,
    metric_name: str,
    higher_is_better: bool,
) -> pd.DataFrame:
    order = "DESC" if higher_is_better else "ASC"

    with _connect() as conn:
        df = pd.read_sql_query(
            f"""
            WITH m AS (
              SELECT
                e.framework,
                c.algorithm,
                c.candidate_id,
                te.value AS metric_value
              FROM experiments e
              JOIN candidates c
                ON c.exp_id = e.exp_id
              JOIN topk_evaluations te
                ON te.exp_id = c.exp_id
               AND te.candidate_id = c.candidate_id
              WHERE e.dataset_id = ?
                AND te.phase = 'inference'
                AND te.metric_type = ?
                AND te.metric_name = ?
            ),
            ranked AS (
              SELECT *,
                     ROW_NUMBER() OVER (
                       PARTITION BY framework
                       ORDER BY metric_value {order}
                     ) AS rn
              FROM m
            )
            SELECT framework, algorithm AS best_algo, metric_value AS best_metric, candidate_id
            FROM ranked
            WHERE rn = 1
            ORDER BY best_metric {order}
            """,
            conn,
            params=(dataset_id, metric_type, metric_name),
        )
    return df


@st.cache_data(show_spinner=False)
def get_candidates_flat(dataset_id: str) -> pd.DataFrame:
    """
    Denormalized view similar to your Excel:
    experiments + candidates + environments + pivoted topk_evaluations.
    Produces pivot columns like: inference_performance_accuracy, inference_energy_inference_energy_kwh, etc.
    """
    with _connect() as conn:
        base = pd.read_sql_query(
            """
            SELECT
              e.exp_id,
              e.timestamp,
              e.dataset_id,
              e.framework,
              c.candidate_id,
              c.algorithm,
              c.is_best_model,
              c.training_duration_secs,
              c.params_json,
              env.cpu_info,
              env.gpu_info
            FROM experiments e
            JOIN candidates c ON c.exp_id = e.exp_id
            LEFT JOIN environments env ON env.env_id = e.env_id
            WHERE e.dataset_id = ?
            """,
            conn,
            params=(dataset_id,),
        )

        metrics = pd.read_sql_query(
            """
            SELECT
              te.exp_id,
              te.candidate_id,
              te.phase,
              te.metric_type,
              te.metric_name,
              te.value
            FROM topk_evaluations te
            JOIN experiments e ON e.exp_id = te.exp_id
            WHERE e.dataset_id = ?
            """,
            conn,
            params=(dataset_id,),
        )

    if metrics.empty:
        return base

    metrics["col"] = metrics.apply(lambda r: f"{r['phase']}_{r['metric_type']}_{r['metric_name']}", axis=1)

    pivot = metrics.pivot_table(
        index=["exp_id", "candidate_id"],
        columns="col",
        values="value",
        aggfunc="max",
    ).reset_index()

    df = base.merge(pivot, on=["exp_id", "candidate_id"], how="left")
    return df


@st.cache_data(show_spinner=False)
def filter_topk_by_metric(df: pd.DataFrame, metric_col: str, k: int, higher_is_better: bool) -> pd.DataFrame:
    if k <= 0:
        return df
    if metric_col not in df.columns:
        return df

    work = df.dropna(subset=[metric_col]).copy()
    work = work.sort_values(metric_col, ascending=not higher_is_better).head(k)

    # Keep rows without the metric out (since Top-K means “ranked by metric”)
    return work


def pareto_front(df: pd.DataFrame,
                 maximize: list[str],
                 minimize: list[str]) -> pd.Series:
    """
    Returns a boolean mask indicating Pareto-optimal rows.
    """
    data = df.copy()

    # Drop rows with missing objectives
    cols = maximize + minimize
    data = data.dropna(subset=cols)

    values = data[cols].values
    is_pareto = [True] * len(values)

    for i, row in enumerate(values):
        for j, other in enumerate(values):
            if i == j:
                continue

            better_or_equal = True
            strictly_better = False

            # maximize objectives
            for k, col in enumerate(maximize):
                if other[k] < row[k]:
                    better_or_equal = False
                    break
                if other[k] > row[k]:
                    strictly_better = True

            # minimize objectives
            for k, col in enumerate(minimize):
                idx = len(maximize) + k
                if other[idx] > row[idx]:
                    better_or_equal = False
                    break
                if other[idx] < row[idx]:
                    strictly_better = True

            if better_or_equal and strictly_better:
                is_pareto[i] = False
                break

    mask = pd.Series(False, index=df.index)
    mask.loc[data.index] = is_pareto
    return mask
