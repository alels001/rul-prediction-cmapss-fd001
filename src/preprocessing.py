"""
preprocessing.py — Shared preprocessing utilities for the C-MAPSS FD001 study.

This module provides:
  - Column lists (sensors to drop, final 13 features)
  - RUL label construction (linear and piecewise-linear with cap)
  - Min-Max scaling (fit on training data only to avoid leakage)
  - Test set evaluation builder (last-cycle extraction)

All functions are imported by the Jupyter notebooks in ../notebooks/.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# ── Column configuration ─────────────────────────────────────────────────────

# Columns removed from the raw FD001 data:
#   op1-op3  — single operating condition (constant)
#   s1, s5, s6, s10, s16, s18, s19 — near-zero variance
#   s14 — redundant with s9 (Pearson r = 0.963)
COLUMNS_TO_DROP_FD001 = [
    "op1", "op2", "op3",
    "s1", "s5", "s6", "s10", "s16", "s18", "s19",
    "s14",
]

# The 13 informative sensor features retained for modelling
FINAL_FEATURES_FD001 = [
    "s2", "s3", "s4", "s7", "s8", "s9",
    "s11", "s12", "s13", "s15", "s17", "s20", "s21",
]


# ── Column dropping ──────────────────────────────────────────────────────────

def drop_non_informative_columns(df, cols_to_drop=COLUMNS_TO_DROP_FD001):
    """Drop constant / redundant columns from a C-MAPSS dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe with all 24 sensor + 3 operational-setting columns.
    cols_to_drop : list of str
        Column names to remove (default: COLUMNS_TO_DROP_FD001).

    Returns
    -------
    pd.DataFrame
        Copy of *df* with the specified columns removed.
    """
    return df.drop(columns=[c for c in cols_to_drop if c in df.columns])


# ── RUL label construction ───────────────────────────────────────────────────

def add_linear_rul(df, unit_col="unit", cycle_col="cycle"):
    """Add a *linear* RUL column: max_cycle − current_cycle per engine.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain *unit_col* and *cycle_col*.
    unit_col, cycle_col : str
        Column names for the engine id and the time-step counter.

    Returns
    -------
    pd.DataFrame
        Same dataframe with a new ``RUL_linear`` column appended.
    """
    max_cycles = df.groupby(unit_col)[cycle_col].transform("max")
    df = df.copy()
    df["RUL_linear"] = max_cycles - df[cycle_col]
    return df


def add_piecewise_rul(df, cap=125, source_col="RUL_linear",
                      target_col="RUL_piecewise125"):
    """Clip the linear RUL at *cap* to form a piecewise-linear target.

    Early cycles where the engine is still healthy are capped to *cap*,
    creating a flat-then-linear degradation label.  The structural
    justification for cap = 125 is that the shortest engine in FD001
    reaches a maximum linear RUL of 127.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain *source_col* (e.g. ``RUL_linear``).
    cap : int
        Maximum RUL value (default 125).
    source_col : str
        Column with uncapped linear RUL.
    target_col : str
        Name for the new capped column.

    Returns
    -------
    pd.DataFrame
        Same dataframe with *target_col* appended.
    """
    df = df.copy()
    df[target_col] = df[source_col].clip(upper=cap)
    return df


# ── Scaling ──────────────────────────────────────────────────────────────────

def fit_minmax_scaler(train_df, feature_cols):
    """Fit a MinMaxScaler on the training set only (no leakage).

    Parameters
    ----------
    train_df : pd.DataFrame
        Training data.
    feature_cols : list of str
        Columns to scale (default usage: FINAL_FEATURES_FD001).

    Returns
    -------
    MinMaxScaler
        Fitted scaler object (can be persisted with ``joblib.dump``).
    """
    scaler = MinMaxScaler()
    scaler.fit(train_df[feature_cols])
    return scaler


def apply_minmax_scaler(df, scaler, feature_cols):
    """Apply a previously fitted MinMaxScaler to a dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Data to transform (train or test).
    scaler : MinMaxScaler
        Scaler fitted on training data via :func:`fit_minmax_scaler`.
    feature_cols : list of str
        Columns to transform.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with *feature_cols* replaced by their scaled values.
    """
    df = df.copy()
    df[feature_cols] = scaler.transform(df[feature_cols])
    return df


# ── Test-set evaluation helper ───────────────────────────────────────────────

def build_test_last_cycle(test_df, y_test_df, unit_col="unit",
                          cycle_col="cycle"):
    """Extract the last observed cycle per test engine and attach the true RUL.

    The C-MAPSS test set is truncated at an unknown point before failure.
    This function picks each engine's final row and merges the ground-truth
    RUL labels from ``RUL_FD001.txt``.

    Parameters
    ----------
    test_df : pd.DataFrame
        Full test data (all cycles per engine).
    y_test_df : pd.DataFrame
        Ground-truth RUL values, one row per engine, with a column
        ``RUL_true`` (or the first column is used).
    unit_col, cycle_col : str
        Column names for engine id and time-step.

    Returns
    -------
    pd.DataFrame
        One row per engine: the last-cycle features plus ``RUL_true``.
    """
    idx = test_df.groupby(unit_col)[cycle_col].idxmax()
    last = test_df.loc[idx].reset_index(drop=True)

    # Align the ground-truth labels
    if "RUL_true" in y_test_df.columns:
        last["RUL_true"] = y_test_df["RUL_true"].values
    else:
        last["RUL_true"] = y_test_df.iloc[:, 0].values

    return last
