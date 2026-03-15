from __future__ import annotations

import pandas as pd
from dataclasses import dataclass
from typing import Optional, Tuple


# Canonical columns in YOUR DB (after pivot)
ACC_COL = "inference_performance_accuracy"
LAT_COL = "inference_timing_inference_latency_per_row_ms"
ENERGY_COL = "inference_energy_inference_energy_kwh"


@dataclass
class Constraints:
    max_latency_ms: Optional[float] = None
    min_accuracy: Optional[float] = None
    max_energy_kwh: Optional[float] = None


def apply_constraints(df: pd.DataFrame, c: Constraints) -> pd.DataFrame:
    work = df.copy()

    if c.min_accuracy is not None and ACC_COL in work.columns:
        work = work[work[ACC_COL].notna() & (work[ACC_COL] >= c.min_accuracy)]

    if c.max_latency_ms is not None and LAT_COL in work.columns:
        work = work[work[LAT_COL].notna() & (work[LAT_COL] <= c.max_latency_ms)]

    if c.max_energy_kwh is not None and ENERGY_COL in work.columns:
        work = work[work[ENERGY_COL].notna() & (work[ENERGY_COL] <= c.max_energy_kwh)]

    return work


def pick_winner(df: pd.DataFrame, objective: str, constraints: Constraints) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (winner_df (1 row), filtered_df).
    objective: accuracy | latency | energy
    """
    filtered = apply_constraints(df, constraints)
    if filtered.empty:
        return pd.DataFrame(), filtered

    if objective == "accuracy" and ACC_COL in filtered.columns:
        winner = filtered.sort_values(ACC_COL, ascending=False).head(1)
    elif objective == "latency" and LAT_COL in filtered.columns:
        winner = filtered.sort_values(LAT_COL, ascending=True).head(1)
    elif objective == "energy" and ENERGY_COL in filtered.columns:
        winner = filtered.sort_values(ENERGY_COL, ascending=True).head(1)
    else:
        # Fallback to accuracy if objective column missing
        if ACC_COL in filtered.columns:
            winner = filtered.sort_values(ACC_COL, ascending=False).head(1)
        else:
            winner = filtered.head(1)

    return winner, filtered


def pick_alternatives(filtered: pd.DataFrame) -> pd.DataFrame:
    """
    Returns up to 3 rows labeled:
      - Alt: Max Acc
      - Alt: Greenest
      - Alt: Fastest
    """
    if filtered.empty:
        return pd.DataFrame()

    rows = []
    if ACC_COL in filtered.columns:
        r = filtered.sort_values(ACC_COL, ascending=False).head(1).copy()
        r.insert(0, "type", "Alt: Max Acc")
        rows.append(r)

    if ENERGY_COL in filtered.columns:
        r = filtered.sort_values(ENERGY_COL, ascending=True).head(1).copy()
        r.insert(0, "type", "Alt: Greenest")
        rows.append(r)

    if LAT_COL in filtered.columns:
        r = filtered.sort_values(LAT_COL, ascending=True).head(1).copy()
        r.insert(0, "type", "Alt: Fastest")
        rows.append(r)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)
