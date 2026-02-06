#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Synthetic population generator for Hungarian census data.

Implements a pragmatic version of the agreed plan:
- Parses flat and hierarchical KSH tables.
- Builds settlement → (vármegye, településtípus) crosswalk.
- Uses IPF to match town-level marginals for core demographics.
- Assigns secondary attributes from hierarchical conditional distributions.
- Generates households with template+IPU (size + age-composition constraints).
- Writes Parquet outputs and a validation report.

Note: This is a large-scale pipeline. Use --max-settlements or --county for dry runs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# -----------------------------
# Constants / Category Labels
# -----------------------------

SEXES = ["Férfi", "Nő"]

EDUCATION_CATS = [
    "Általános iskola 8. évfolyamnál alacsonyabb",
    "Általános iskola 8. évfolyam",
    "Középfokú iskola érettségi nélkül, szakmai oklevéllel",
    "Érettségi",
    "Egyetem, főiskola stb. oklevéllel",
]
ACTIVITY_EDU_CATS = EDUCATION_CATS + ["15 évesnél fiatalabb személy"]

EMPLOYMENT_CATS = [
    "Foglalkoztatott",
    "Munkanélküli",
    "Ellátásban részesülő inaktív",
    "Eltartott",
]

ACTIVITY_AGE_BANDS = [
    "15 évesnél fiatalabb",
    "15–19 éves",
    "20–24 éves",
    "25–29 éves",
    "30–34 éves",
    "35–39 éves",
    "40–44 éves",
    "45–49 éves",
    "50–54 éves",
    "55–59 éves",
    "60–64 éves",
    "65–69 éves",
    "70–74 éves",
    "75 éves és idősebb",
]

ACTIVITY_AGE_BANDS_15PLUS = ACTIVITY_AGE_BANDS[1:]

OCCUPATION_AGE_BANDS = [
    "15–19 éves",
    "20–24 éves",
    "25–29 éves",
    "30–34 éves",
    "35–39 éves",
    "40–44 éves",
    "45–49 éves",
    "50–54 éves",
    "55–59 éves",
    "60–64 éves",
    "65–69 éves",
    "70 éves és idősebb",
]

MARITAL_AGE_BANDS = [
    "15–19 éves",
    "20–24 éves",
    "25–29 éves",
    "30–34 éves",
    "35–39 éves",
    "40–44 éves",
    "45–49 éves",
    "50–54 éves",
    "55–59 éves",
    "60–64 éves",
    "65–69 éves",
    "70 éves és idősebb",
]
SCHOOL_AGE_BANDS = [
    "6–9 éves",
    "10–14 éves",
    "15–19 éves",
    "20–24 éves",
    "25–29 éves",
    "30–34 éves",
    "35–39 éves",
    "40 éves és idősebb",
]

HEALTH_AGE_BANDS = [
    "5–14 éves",
    "15–24 éves",
    "25–34 éves",
    "35–44 éves",
    "45–54 éves",
    "55–64 éves",
    "65–74 éves",
    "75–84 éves",
    "85 éves és idősebb",
]

HOUSEHOLD_SIZE_CATS = [
    "Egyszemélyes háztartás",
    "Kétszemélyes háztartás",
    "Háromszemélyes háztartás",
    "Négyszemélyes háztartás",
    "Ötszemélyes háztartás",
    "Hat vagy többszemélyes háztartás",
]

HOUSEHOLD_AGE_COMPS = [
    "Csak 30 évesnél fiatalabb személy van a háztartásban",
    "Csak 30–64 éves személy van a háztartásban",
    "Csak 65 éves és idősebb személy van a háztartásban",
    "30 évesnél fiatalabb és 30–64 éves személyek vannak a háztartásban",
    "30 évesnél fiatalabb és 65 éves és idősebb személyek vannak a háztartásban",
    "30–64 éves és 65 éves és idősebb személyek vannak a háztartásban",
    "30 évesnél fiatalabb, 30–64 éves és 65 éves és idősebb személyek vannak a háztartásban",
]

HOUSEHOLD_EMP_COMPS = [
    "Egy foglalkoztatott van a háztartásban",
    "Két foglalkoztatott van a háztartásban",
    "Három vagy több foglalkoztatott van a háztartásban",
    "Nincs foglalkoztatott, van munkanélküli és lehet ellátásban részesülő inaktív és/vagy eltartott a háztartásban",
    "Nincs foglalkoztatott, nincs munkanélküli, van ellátásban részesülő inaktív és lehet eltartott a háztartásban",
    "Csak eltartott van a háztartásban",
]

HOUSEHOLD_U15_CATS = [
    "Nincs 15 évesnél fiatalabb személy a háztartásban",
    "1 személy 15 évesnél fiatalabb a háztartásban",
    "2 személy 15 évesnél fiatalabb a háztartásban",
    "3 vagy több személy 15 évesnél fiatalabb a háztartásban",
]

HOUSEHOLD_O65_CATS = [
    "Nincs 65 éves és idősebb személy a háztartásban",
    "1 személy 65 éves és idősebb a háztartásban",
    "2 személy 65 éves és idősebb a háztartásban",
    "3 vagy több személy 65 éves és idősebb a háztartásban",
]

HOUSEHOLD_EMPLOYED_CATS = [
    "Nincs foglalkoztatott személy a háztartásban",
    "1 foglalkoztatott személy van a háztartásban",
    "2 foglalkoztatott személy van a háztartásban",
    "3 vagy több foglalkoztatott személy van a háztartásban",
]

MARITAL_CATS = ["Nőtlen, hajadon", "Házas", "Özvegy", "Elvált"]

FERTILITY_CATS = [
    "15 éves és idősebb nő 0 élve született gyermekkel",
    "15 éves és idősebb nő 1 élve született gyermekkel",
    "15 éves és idősebb nő 2 élve született gyermekkel",
    "15 éves és idősebb nő 3 élve született gyermekkel",
    "15 éves és idősebb nő 4 élve született gyermekkel",
    "15 éves és idősebb nő 5 vagy több élve született gyermekkel",
]

DISABILITY_CATS = [
    "Fogyatékossága van vagy súlyosan korlátozott",
    "Nincs fogyatékossága és nem súlyosan korlátozott",
    "Nem válaszolt a fogyatékossági és a korlátozottsági kérdésekre",
]

CHRONIC_CATS = [
    "Van tartós betegsége",
    "Nincs tartós betegsége",
    "Nem válaszolt a tartós betegségre vonatkozó kérdésre",
]

LIMITATION_CATS = [
    "Nem korlátozott",
    "Mérsékelten korlátozott",
    "Súlyosan korlátozott",
    "Nem válaszolt a korlátozottsági kérdésekre",
]

BP_UNASSIGNED_COL = "Budapest kerületre nem bontható adatai"


@dataclass
class Config:
    max_age: int = 100
    ipf_max_iter: int = 200
    ipf_tol: float = 1e-5
    seed: int = 42
    household_swap_max_iter: int = 200000
    household_swap_patience: int = 20000


# -----------------------------
# Utility Functions
# -----------------------------

def ensure_dir(path: Path) -> None:
    """Create a directory (and its parents) if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


def to_numeric_series(s: pd.Series) -> pd.Series:
    """Coerce a Series to numeric values, returning NaN for non-numeric cells."""
    return pd.to_numeric(s, errors="coerce")


def split_budapest_unassigned(df: pd.DataFrame) -> pd.DataFrame:
    """Redistribute Budapest-level unassigned counts proportionally across districts."""
    if BP_UNASSIGNED_COL not in df.columns:
        return df
    district_cols = [c for c in df.columns if c.startswith("Budapest ") and "ker." in c]
    if not district_cols:
        return df

    df = df.copy()
    unassigned = df[BP_UNASSIGNED_COL].fillna(0)
    district_sum = df[district_cols].sum(axis=1)

    for i in df.index:
        ua = unassigned.loc[i]
        if ua == 0:
            continue
        if district_sum.loc[i] > 0:
            shares = df.loc[i, district_cols] / district_sum.loc[i]
        else:
            shares = pd.Series(1 / len(district_cols), index=district_cols)
        df.loc[i, district_cols] = df.loc[i, district_cols] + ua * shares

    df = df.drop(columns=[BP_UNASSIGNED_COL])
    return df


def read_flat(path: Path, sheet: str = "Adattábla") -> pd.DataFrame:
    """Read a flat settlement-level table and coerce all measure columns to numeric."""
    df = pd.read_excel(path, sheet_name=sheet)
    df = split_budapest_unassigned(df)
    # Ensure numeric for value columns
    for c in df.columns[1:]:
        df[c] = to_numeric_series(df[c])
    return df


def melt_flat(df: pd.DataFrame, value_name: str) -> pd.DataFrame:
    """Convert a wide settlement-by-category flat table into long format."""
    cat_col = df.columns[0]
    out = df.melt(id_vars=[cat_col], var_name="settlement", value_name=value_name)
    out = out.rename(columns={cat_col: "category"})
    out = out.dropna(subset=[value_name])
    return out


def parse_hier_table(path: Path, sheet: str, label_names: List[str]) -> pd.DataFrame:
    """Parse a generic hierarchical table with multi-row headers into tidy long format."""
    raw = pd.read_excel(path, sheet_name=sheet, header=None)
    header0 = raw.iloc[0]
    header1 = raw.iloc[1]
    first_data_col = header0.first_valid_index()
    if first_data_col is None:
        raise ValueError(f"Could not locate data columns in {path}")
    label_cols = list(range(first_data_col))

    data = raw.iloc[2:].copy()
    labels = data[label_cols].ffill()
    labels.columns = label_names
    labels = labels.copy()
    labels["row_id"] = labels.index

    data_cols = data.columns[first_data_col:]
    counties = header0.ffill().iloc[first_data_col:]
    telep = header1.iloc[first_data_col:]
    col_index = pd.MultiIndex.from_arrays([counties.values, telep.values], names=["varmegye", "telepules_tipus"])

    values = data[data_cols]
    values.columns = col_index

    long = values.stack(["varmegye", "telepules_tipus"]).reset_index()
    long = long.rename(columns={"level_0": "row_id", 0: "value"})
    long = long.merge(labels, on="row_id", how="left")
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long = long.dropna(subset=["value"])
    return long


def parse_hier_household_table(path: Path, sheet: str) -> pd.DataFrame:
    """Parse the household hierarchy table (size x age composition x employment composition)."""
    # Expected layout: header row 0 = county, row 1 = telepules_tipus, data from row 2
    raw = pd.read_excel(path, sheet_name=sheet, header=None)
    header0 = raw.iloc[0]
    header1 = raw.iloc[1]
    first_data_col = header0.first_valid_index()
    if first_data_col is None:
        raise ValueError(f"Could not locate data columns in {path}")
    label_cols = list(range(first_data_col))

    data = raw.iloc[2:].copy()
    labels = data[label_cols].ffill()
    labels.columns = ["household_size", "age_comp", "employment_comp"]
    labels = labels.copy()
    labels["row_id"] = labels.index

    data_cols = data.columns[first_data_col:]
    counties = header0.ffill().iloc[first_data_col:]
    telep = header1.iloc[first_data_col:]
    col_index = pd.MultiIndex.from_arrays([counties.values, telep.values], names=["varmegye", "telepules_tipus"])

    values = data[data_cols]
    values.columns = col_index

    long = values.stack(["varmegye", "telepules_tipus"]).reset_index()
    long = long.rename(columns={"level_0": "row_id", 0: "value"})
    long = long.merge(labels, on="row_id", how="left")
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long = long.dropna(subset=["value"])
    return long


def ipf(seed: np.ndarray, targets: List[np.ndarray], max_iter: int, tol: float) -> np.ndarray:
    """Run iterative proportional fitting over all axes to match provided marginals."""
    arr = seed.astype(float).copy()
    for _ in range(max_iter):
        max_diff = 0.0
        for axis, target in enumerate(targets):
            # Compute current marginal along axis
            axes = tuple(i for i in range(arr.ndim) if i != axis)
            current = arr.sum(axis=axes)
            # Avoid division by zero
            with np.errstate(divide="ignore", invalid="ignore"):
                factors = np.where(current > 0, target / current, 0)
            # Broadcast factors to arr
            shape = [1] * arr.ndim
            shape[axis] = factors.shape[0]
            arr = arr * factors.reshape(shape)
            max_diff = max(max_diff, np.nanmax(np.abs(current - target)))
        if max_diff < tol:
            break
    return arr


def integerize_counts(arr: np.ndarray, total: int) -> np.ndarray:
    """Convert fractional cell weights to integers while preserving the global total."""
    flat = arr.flatten()
    floor = np.floor(flat)
    remainder = int(round(total - floor.sum()))
    if remainder > 0:
        frac = flat - floor
        if remainder <= len(flat):
            idx = np.argsort(frac)[-remainder:]
            floor[idx] += 1
        else:
            idx = np.argsort(frac)[::-1]
            for i in range(remainder):
                floor[idx[i % len(idx)]] += 1
    return floor.reshape(arr.shape).astype(int)


def adjust_household_status_by_swaps(
    df: pd.DataFrame,
    households_df: pd.DataFrame,
    status: str,
    target_vec: np.ndarray,
    rng: np.random.Generator,
    max_iter: int,
    patience: int,
) -> None:
    """Greedy swap-based calibration to match household status-count marginals."""
    if max_iter <= 0:
        return
    if target_vec.sum() <= 0:
        return

    hh_ids = households_df["household_id"].astype(int).tolist()
    if not hh_ids:
        return
    total_hh = len(hh_ids)

    # Scale targets to total households
    target = target_vec.astype(float)
    if target.sum() != total_hh:
        if target.sum() > 0:
            target = target * (total_hh / target.sum())
        else:
            target = np.ones_like(target, dtype=float) * (total_hh / len(target))

    def count_to_cat(c: int) -> int:
        if c <= 0:
            return 0
        if c == 1:
            return 1
        if c == 2:
            return 2
        return 3

    # Current counts per household
    status_counts = df[df["employment_status"] == status].groupby("household_id").size().to_dict()
    hh_counts = {hid: int(status_counts.get(hid, 0) or 0) for hid in hh_ids}

    curr = np.zeros(4, dtype=int)
    for hid in hh_ids:
        curr[count_to_cat(hh_counts[hid])] += 1

    error = float(np.abs(curr - target).sum())
    if error == 0:
        return

    # Build per-household person lists by age group for status and non-status
    age_groups = ["u15", "u30", "a30_64", "o65"]
    persons_s = {hid: {ag: [] for ag in age_groups} for hid in hh_ids}
    persons_not = {hid: {ag: [] for ag in age_groups} for hid in hh_ids}

    mask = df["household_id"].notna()
    for pid, ag, emp, hid in df.loc[mask, ["_age_house", "employment_status", "household_id"]].itertuples(
        index=True, name=None
    ):
        hid = int(hid)
        if emp == status:
            persons_s[hid][ag].append(pid)
        else:
            persons_not[hid][ag].append(pid)

    donor_set = {hid for hid in hh_ids if hh_counts[hid] > 0}
    donor_list = list(donor_set)
    recip_set_by_age = {ag: {hid for hid in hh_ids if len(persons_not[hid][ag]) > 0} for ag in age_groups}
    recip_list_by_age = {ag: list(s) for ag, s in recip_set_by_age.items()}

    def sample_from_list(lst, valid_set):
        if not valid_set:
            return None
        for _ in range(5):
            if not lst:
                break
            hid = lst[int(rng.integers(len(lst)))]
            if hid in valid_set:
                return hid
        # refresh list
        lst[:] = list(valid_set)
        if not lst:
            return None
        return lst[int(rng.integers(len(lst)))]

    no_improve = 0
    for _ in range(max_iter):
        if error == 0 or no_improve >= patience:
            break

        # Pick donor household
        a = sample_from_list(donor_list, donor_set)
        if a is None:
            break
        # Choose age group with available status person
        donor_ages = [ag for ag in age_groups if persons_s[a][ag]]
        if not donor_ages:
            donor_set.discard(a)
            continue
        ag = donor_ages[int(rng.integers(len(donor_ages)))]

        # Pick recipient household with non-status person in same age group
        b = sample_from_list(recip_list_by_age[ag], recip_set_by_age[ag])
        if b is None or b == a:
            no_improve += 1
            continue
        if not persons_not[b][ag]:
            recip_set_by_age[ag].discard(b)
            no_improve += 1
            continue

        # Select people to swap
        p_s = persons_s[a][ag][int(rng.integers(len(persons_s[a][ag])))]
        p_n = persons_not[b][ag][int(rng.integers(len(persons_not[b][ag])))]

        old_cat_a = count_to_cat(hh_counts[a])
        old_cat_b = count_to_cat(hh_counts[b])
        new_count_a = hh_counts[a] - 1
        new_count_b = hh_counts[b] + 1
        new_cat_a = count_to_cat(new_count_a)
        new_cat_b = count_to_cat(new_count_b)

        if old_cat_a == new_cat_a and old_cat_b == new_cat_b:
            no_improve += 1
            continue

        curr_new = curr.copy()
        curr_new[old_cat_a] -= 1
        curr_new[new_cat_a] += 1
        curr_new[old_cat_b] -= 1
        curr_new[new_cat_b] += 1
        new_error = float(np.abs(curr_new - target).sum())

        if new_error < error:
            # Apply swap (swap household ids)
            df.at[p_s, "household_id"] = b
            df.at[p_n, "household_id"] = a

            # Update counts
            hh_counts[a] = new_count_a
            hh_counts[b] = new_count_b
            curr = curr_new
            error = new_error

            # Update donor set
            if hh_counts[a] <= 0:
                donor_set.discard(a)
            if hh_counts[b] > 0:
                if b not in donor_set:
                    donor_set.add(b)
                    donor_list.append(b)

            # Update persons lists for status and non-status
            persons_s[a][ag].remove(p_s)
            persons_s[b][ag].append(p_s)
            persons_not[b][ag].remove(p_n)
            persons_not[a][ag].append(p_n)

            # Update recipient sets
            if len(persons_not[b][ag]) == 0:
                recip_set_by_age[ag].discard(b)
            if len(persons_not[a][ag]) > 0 and a not in recip_set_by_age[ag]:
                recip_set_by_age[ag].add(a)
                recip_list_by_age[ag].append(a)

            no_improve = 0
        else:
            no_improve += 1


def scale_to_total(arr: np.ndarray, total: float) -> np.ndarray:
    """Scale an array so its sum matches a target total."""
    s = arr.sum()
    if s > 0 and abs(s - total) > 1e-6:
        return arr * (total / s)
    return arr


def uniform_age_array(start: int, end: int, total: int, rng: np.random.Generator) -> np.ndarray:
    """Sample integer ages approximately uniformly in [start, end] with fixed length."""
    n = end - start + 1
    if n <= 0 or total <= 0:
        return np.array([], dtype=int)
    base = total // n
    rem = total % n
    ages = np.repeat(np.arange(start, end + 1), base)
    if rem > 0:
        extra = rng.choice(np.arange(start, end + 1), size=rem, replace=False if rem <= n else True)
        ages = np.concatenate([ages, extra])
    rng.shuffle(ages)
    return ages


def safe_pct_diff(orig: np.ndarray, gen: np.ndarray) -> np.ndarray:
    """Compute percentage difference with safe handling of zero denominators."""
    orig = orig.astype(float)
    gen = gen.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        diff = np.where(orig != 0, (gen - orig) / orig * 100.0, np.nan)
    return diff


# -----------------------------
# Parsing and Precomputation
# -----------------------------

def build_crosswalk(telepules_path: Path) -> pd.DataFrame:
    """Build settlement -> (KSH code, county, settlement type) mapping."""
    cols = ["Helység megnevezése", "Helység KSH kódja", "Vármegye megnevezése", "Településtípus"]
    df = pd.read_excel(telepules_path, sheet_name="Helységek 2024.01.01.", usecols=cols)
    df = df.rename(columns={
        "Helység megnevezése": "settlement",
        "Helység KSH kódja": "ksh_code",
        "Vármegye megnevezése": "varmegye",
        "Településtípus": "telepules_tipus",
    })
    # Normalize Budapest naming for alignment with hier tables
    df["varmegye"] = df["varmegye"].astype(str).str.strip()
    df.loc[df["varmegye"].isin(["nan", "NaN", "None", ""]), "varmegye"] = np.nan
    df.loc[df["varmegye"].str.lower().str.contains("főváros", na=False), "varmegye"] = "Budapest"
    df.loc[df["settlement"].str.startswith("Budapest", na=False), "varmegye"] = df.loc[
        df["settlement"].str.startswith("Budapest", na=False), "varmegye"
    ].fillna("Budapest")
    return df


def build_activity_seed(activity_long: pd.DataFrame) -> Tuple[Dict[Tuple[str, str], np.ndarray], np.ndarray]:
    """Create county+type seed tensors for sex x age x education x activity."""
    # Map to fixed categories
    activity_long = activity_long.copy()
    activity_long = activity_long[activity_long["sex"].isin(SEXES)]
    activity_long = activity_long[activity_long["age"].isin(ACTIVITY_AGE_BANDS)]
    activity_long = activity_long[activity_long["education"].isin(EDUCATION_CATS)]
    activity_long = activity_long[activity_long["activity"].isin(EMPLOYMENT_CATS)]

    shape = (len(SEXES), len(ACTIVITY_AGE_BANDS), len(EDUCATION_CATS), len(EMPLOYMENT_CATS))
    seeds: Dict[Tuple[str, str], np.ndarray] = {}

    for (varmegye, telepules_tipus), sub in activity_long.groupby(["varmegye", "telepules_tipus"]):
        arr = np.zeros(shape, dtype=float)
        for _, row in sub.iterrows():
            i = SEXES.index(row["sex"])
            j = ACTIVITY_AGE_BANDS.index(row["age"])
            k = EDUCATION_CATS.index(row["education"])
            l = EMPLOYMENT_CATS.index(row["activity"])
            arr[i, j, k, l] += row["value"]
        seeds[(varmegye, telepules_tipus)] = arr

    # Global fallback seed
    total = np.zeros(shape, dtype=float)
    for arr in seeds.values():
        total += arr
    return seeds, total


def build_activity_seed_nem_kor(activity_long: pd.DataFrame) -> Tuple[Dict[Tuple[str, str], np.ndarray], np.ndarray]:
    """Create activity+education seed tensors from the nem-kor hierarchical table."""
    # Expected labels: sex, age (15+), activity, education
    activity_long = activity_long.copy()
    activity_long = activity_long[activity_long["sex"].isin(SEXES)]
    activity_long = activity_long[activity_long["age"].notna()]
    activity_long = activity_long[activity_long["education"].isin(EDUCATION_CATS)]
    activity_long = activity_long[activity_long["activity"].isin(EMPLOYMENT_CATS)]

    shape = (len(SEXES), len(ACTIVITY_AGE_BANDS), len(EDUCATION_CATS), len(EMPLOYMENT_CATS))
    seeds: Dict[Tuple[str, str], np.ndarray] = {}

    for (varmegye, telepules_tipus), sub in activity_long.groupby(["varmegye", "telepules_tipus"]):
        arr = np.zeros(shape, dtype=float)
        for _, row in sub.iterrows():
            i = SEXES.index(row["sex"])
            k = EDUCATION_CATS.index(row["education"])
            l = EMPLOYMENT_CATS.index(row["activity"])
            age = row["age"]
            if age in ACTIVITY_AGE_BANDS_15PLUS:
                j = ACTIVITY_AGE_BANDS.index(age)
                arr[i, j, k, l] += row["value"]
            elif age == "70 éves és idősebb":
                j1 = ACTIVITY_AGE_BANDS.index("70–74 éves")
                j2 = ACTIVITY_AGE_BANDS.index("75 éves és idősebb")
                arr[i, j1, k, l] += row["value"] * 0.5
                arr[i, j2, k, l] += row["value"] * 0.5
        seeds[(varmegye, telepules_tipus)] = arr

    total = np.zeros(shape, dtype=float)
    for arr in seeds.values():
        total += arr
    return seeds, total


def build_age_split_profile(activity_long: pd.DataFrame, health_long: pd.DataFrame) -> Dict[Tuple[str, str], Dict[str, float]]:
    """Estimate within-decade age split ratios per county+type using hier signals."""
    # Activity age counts
    act = (
        activity_long.groupby(["varmegye", "telepules_tipus", "age"])["value"]
        .sum()
        .reset_index()
    )
    # Health age counts
    health_age = health_long[health_long["group"].isin(HEALTH_AGE_BANDS)].copy()
    health = (
        health_age.groupby(["varmegye", "telepules_tipus", "group"])["value"]
        .sum()
        .reset_index()
    )

    profile: Dict[Tuple[str, str], Dict[str, float]] = {}

    for key in set(zip(act["varmegye"], act["telepules_tipus"])):
        varmegye, telep = key
        sub_act = act[(act["varmegye"] == varmegye) & (act["telepules_tipus"] == telep)]
        sub_health = health[(health["varmegye"] == varmegye) & (health["telepules_tipus"] == telep)]

        def get_act(age_label: str) -> float:
            row = sub_act[sub_act["age"] == age_label]
            return float(row["value"].sum())

        def get_health(age_label: str) -> float:
            row = sub_health[sub_health["group"] == age_label]
            return float(row["value"].sum())

        ratios = {}
        # Use activity for 20-69 splits
        def ratio(low_label: str, high_label: str) -> float:
            a = get_act(low_label)
            b = get_act(high_label)
            if a + b == 0:
                return 0.5
            return a / (a + b)

        ratios["20-29"] = ratio("20–24 éves", "25–29 éves")
        ratios["30-39"] = ratio("30–34 éves", "35–39 éves")
        ratios["40-49"] = ratio("40–44 éves", "45–49 éves")
        ratios["50-59"] = ratio("50–54 éves", "55–59 éves")
        ratios["60-69"] = ratio("60–64 éves", "65–69 éves")

        # 10-19: estimate 10-14 from health 5-14
        health_5_14 = get_health("5–14 éves")
        est_10_14 = 0.5 * health_5_14
        act_15_19 = get_act("15–19 éves")
        if est_10_14 + act_15_19 == 0:
            ratios["10-19"] = 0.5
        else:
            ratios["10-19"] = est_10_14 / (est_10_14 + act_15_19)

        # 70-79 and 80-89 are split evenly (fallback)
        ratios["70-79"] = 0.5
        ratios["80-89"] = 0.5

        profile[(varmegye, telep)] = ratios

    return profile


def build_prob_table(df: pd.DataFrame, key_cols: List[str], outcome_col: str) -> Dict[Tuple, Tuple[List[str], np.ndarray]]:
    """Build conditional outcome probability tables keyed by selected dimensions."""
    grouped = df.groupby(key_cols + [outcome_col])["value"].sum().reset_index()
    probs: Dict[Tuple, Tuple[List[str], np.ndarray]] = {}
    for key, sub in grouped.groupby(key_cols):
        total = sub["value"].sum()
        if total <= 0:
            continue
        outcomes = sub[outcome_col].tolist()
        p = (sub["value"] / total).to_numpy()
        probs[key] = (outcomes, p)
    return probs


def global_distribution(df: pd.DataFrame, outcome_col: str) -> Tuple[List[str], np.ndarray]:
    """Build an overall fallback distribution for an outcome column."""
    totals = df.groupby(outcome_col)["value"].sum()
    if totals.sum() <= 0:
        return [], np.array([])
    outcomes = totals.index.tolist()
    probs = (totals / totals.sum()).to_numpy()
    return outcomes, probs


def build_school_rates(school_long: pd.DataFrame, pop_by_ct: pd.DataFrame) -> Dict[Tuple, float]:
    """Estimate school-attendance rates by county+type+sex+age from hier totals."""
    # school_long: sex, age, school_level, varmegye, telepules_tipus, value
    school_totals = (
        school_long.groupby(["varmegye", "telepules_tipus", "sex", "age"])["value"]
        .sum()
        .reset_index()
    )
    rates = {}
    for _, row in school_totals.iterrows():
        key = (row["varmegye"], row["telepules_tipus"], row["sex"], row["age"])
        denom_row = pop_by_ct[
            (pop_by_ct["varmegye"] == row["varmegye"]) &
            (pop_by_ct["telepules_tipus"] == row["telepules_tipus"]) &
            (pop_by_ct["age_group"] == row["age"])
        ]
        denom = float(denom_row["count"].sum())
        if denom <= 0:
            rate = 0.0
        else:
            rate = min(1.0, row["value"] / denom)
        rates[key] = rate
    return rates


def age_to_activity_band(age: int) -> str:
    """Map single-year age to activity-table age bands."""
    if age < 15:
        return "15 évesnél fiatalabb"
    if age < 20:
        return "15–19 éves"
    if age < 25:
        return "20–24 éves"
    if age < 30:
        return "25–29 éves"
    if age < 35:
        return "30–34 éves"
    if age < 40:
        return "35–39 éves"
    if age < 45:
        return "40–44 éves"
    if age < 50:
        return "45–49 éves"
    if age < 55:
        return "50–54 éves"
    if age < 60:
        return "55–59 éves"
    if age < 65:
        return "60–64 éves"
    if age < 70:
        return "65–69 éves"
    if age < 75:
        return "70–74 éves"
    return "75 éves és idősebb"


def age_to_occupation_band(age: int) -> str:
    """Map single-year age to occupation/sector/commuting age bands."""
    if age < 20:
        return "15–19 éves"
    if age < 25:
        return "20–24 éves"
    if age < 30:
        return "25–29 éves"
    if age < 35:
        return "30–34 éves"
    if age < 40:
        return "35–39 éves"
    if age < 45:
        return "40–44 éves"
    if age < 50:
        return "45–49 éves"
    if age < 55:
        return "50–54 éves"
    if age < 60:
        return "55–59 éves"
    if age < 65:
        return "60–64 éves"
    if age < 70:
        return "65–69 éves"
    return "70 éves és idősebb"


def age_to_school_band(age: int) -> str:
    """Map single-year age to school-table age bands."""
    if age < 6:
        return "6–9 éves"
    if age < 10:
        return "6–9 éves"
    if age < 15:
        return "10–14 éves"
    if age < 20:
        return "15–19 éves"
    if age < 25:
        return "20–24 éves"
    if age < 30:
        return "25–29 éves"
    if age < 35:
        return "30–34 éves"
    if age < 40:
        return "35–39 éves"
    return "40 éves és idősebb"


def age_to_health_band(age: int) -> str:
    """Map single-year age to health-table age bands."""
    if age < 15:
        return "5–14 éves"
    if age < 25:
        return "15–24 éves"
    if age < 35:
        return "25–34 éves"
    if age < 45:
        return "35–44 éves"
    if age < 55:
        return "45–54 éves"
    if age < 65:
        return "55–64 éves"
    if age < 75:
        return "65–74 éves"
    if age < 85:
        return "75–84 éves"
    return "85 éves és idősebb"


def occupation_band_for_age_array(ages: np.ndarray) -> np.ndarray:
    """Vectorized age-to-occupation-band mapping."""
    return np.select(
        [
            ages < 20,
            ages < 25,
            ages < 30,
            ages < 35,
            ages < 40,
            ages < 45,
            ages < 50,
            ages < 55,
            ages < 60,
            ages < 65,
            ages < 70,
        ],
        [
            "15–19 éves",
            "20–24 éves",
            "25–29 éves",
            "30–34 éves",
            "35–39 éves",
            "40–44 éves",
            "45–49 éves",
            "50–54 éves",
            "55–59 éves",
            "60–64 éves",
            "65–69 éves",
        ],
        default="70 éves és idősebb",
    )


def activity_band_for_age_array(ages: np.ndarray) -> np.ndarray:
    """Vectorized age-to-activity-band mapping."""
    return np.select(
        [
            ages < 15,
            ages < 20,
            ages < 25,
            ages < 30,
            ages < 35,
            ages < 40,
            ages < 45,
            ages < 50,
            ages < 55,
            ages < 60,
            ages < 65,
            ages < 70,
            ages < 75,
        ],
        [
            "15 évesnél fiatalabb",
            "15–19 éves",
            "20–24 éves",
            "25–29 éves",
            "30–34 éves",
            "35–39 éves",
            "40–44 éves",
            "45–49 éves",
            "50–54 éves",
            "55–59 éves",
            "60–64 éves",
            "65–69 éves",
            "70–74 éves",
        ],
        default="75 éves és idősebb",
    )


def marital_band_for_age_array(ages: np.ndarray) -> np.ndarray:
    """Vectorized age-to-marital-table-band mapping (15+ bands)."""
    return np.select(
        [
            ages < 20,
            ages < 25,
            ages < 30,
            ages < 35,
            ages < 40,
            ages < 45,
            ages < 50,
            ages < 55,
            ages < 60,
            ages < 65,
            ages < 70,
        ],
        [
            "15–19 éves",
            "20–24 éves",
            "25–29 éves",
            "30–34 éves",
            "35–39 éves",
            "40–44 éves",
            "45–49 éves",
            "50–54 éves",
            "55–59 éves",
            "60–64 éves",
            "65–69 éves",
        ],
        default="70 éves és idősebb",
    )


def age_band_70plus_for_age_array(ages: np.ndarray) -> np.ndarray:
    """Vectorized mapping for nem-kor tables with a single 70+ band."""
    return np.select(
        [
            ages < 20,
            ages < 25,
            ages < 30,
            ages < 35,
            ages < 40,
            ages < 45,
            ages < 50,
            ages < 55,
            ages < 60,
            ages < 65,
            ages < 70,
        ],
        [
            "15–19 éves",
            "20–24 éves",
            "25–29 éves",
            "30–34 éves",
            "35–39 éves",
            "40–44 éves",
            "45–49 éves",
            "50–54 éves",
            "55–59 éves",
            "60–64 éves",
            "65–69 éves",
        ],
        default="70 éves és idősebb",
    )


def school_band_for_age_array(ages: np.ndarray) -> np.ndarray:
    """Vectorized age-to-school-band mapping."""
    return np.select(
        [
            ages < 10,
            ages < 15,
            ages < 20,
            ages < 25,
            ages < 30,
            ages < 35,
            ages < 40,
        ],
        [
            "6–9 éves",
            "10–14 éves",
            "15–19 éves",
            "20–24 éves",
            "25–29 éves",
            "30–34 éves",
            "35–39 éves",
        ],
        default="40 éves és idősebb",
    )


def health_band_for_age_array(ages: np.ndarray) -> np.ndarray:
    """Vectorized age-to-health-band mapping."""
    return np.select(
        [
            ages < 15,
            ages < 25,
            ages < 35,
            ages < 45,
            ages < 55,
            ages < 65,
            ages < 75,
            ages < 85,
        ],
        [
            "5–14 éves",
            "15–24 éves",
            "25–34 éves",
            "35–44 éves",
            "45–54 éves",
            "55–64 éves",
            "65–74 éves",
            "75–84 éves",
        ],
        default="85 éves és idősebb",
    )


def household_age_comp_label(has_u30: bool, has_a30_64: bool, has_o65: bool) -> str:
    """Encode household age-presence flags into the census age-composition label."""
    if has_u30 and not has_a30_64 and not has_o65:
        return "Csak 30 évesnél fiatalabb személy van a háztartásban"
    if not has_u30 and has_a30_64 and not has_o65:
        return "Csak 30–64 éves személy van a háztartásban"
    if not has_u30 and not has_a30_64 and has_o65:
        return "Csak 65 éves és idősebb személy van a háztartásban"
    if has_u30 and has_a30_64 and not has_o65:
        return "30 évesnél fiatalabb és 30–64 éves személyek vannak a háztartásban"
    if has_u30 and not has_a30_64 and has_o65:
        return "30 évesnél fiatalabb és 65 éves és idősebb személyek vannak a háztartásban"
    if not has_u30 and has_a30_64 and has_o65:
        return "30–64 éves és 65 éves és idősebb személyek vannak a háztartásban"
    return "30 évesnél fiatalabb, 30–64 éves és 65 éves és idősebb személyek vannak a háztartásban"


def household_emp_comp_label(emp_f: int, emp_u: int, emp_i: int, emp_d: int) -> str:
    """Encode household employment counts into the census employment-composition label."""
    if emp_f >= 3:
        return "Három vagy több foglalkoztatott van a háztartásban"
    if emp_f == 2:
        return "Két foglalkoztatott van a háztartásban"
    if emp_f == 1:
        return "Egy foglalkoztatott van a háztartásban"
    # No employed
    if emp_u > 0:
        return "Nincs foglalkoztatott, van munkanélküli és lehet ellátásban részesülő inaktív és/vagy eltartott a háztartásban"
    if emp_i > 0:
        return "Nincs foglalkoztatott, nincs munkanélküli, van ellátásban részesülő inaktív és lehet eltartott a háztartásban"
    return "Csak eltartott van a háztartásban"


# -----------------------------
# Household Template + IPU
# -----------------------------

def build_household_templates() -> pd.DataFrame:
    """Construct a feasible household template library for IPU-based weighting."""
    templates = []
    template_id = 0
    for size in range(1, 7):
        max_u15 = min(3, size)
        max_o65 = min(3, size)
        for u15 in range(0, max_u15 + 1):
            for o65 in range(0, max_o65 + 1):
                a30_64 = size - u15 - o65
                if a30_64 < 0:
                    continue
                # Simple feasibility: at least one adult if children exist
                if u15 > 0 and a30_64 == 0:
                    continue
                if size == 1 and u15 == 1:
                    continue

                if size == 1:
                    fam = "Egyszemélyes háztartás"
                elif u15 > 0 and a30_64 == 1 and size == 2:
                    fam = "Egy szülő gyermekkel"
                elif u15 == 0 and a30_64 == 2 and size == 2:
                    fam = "Házaspár és élettársi kapcsolat"
                elif u15 > 0 and a30_64 >= 1:
                    fam = "Egy családból álló háztartás"
                elif size >= 3 and a30_64 >= 2:
                    fam = "Egyéb összetételű háztartás"
                else:
                    fam = "Nem családháztartás"

                templates.append({
                    "template_id": template_id,
                    "size": size,
                    "u15": u15,
                    "o65": o65,
                    "a30_64": a30_64,
                    "family_structure": fam,
                })
                template_id += 1

    return pd.DataFrame(templates)


# -----------------------------
# Diagnostics
# -----------------------------

def write_flat_diagnostic(orig_df: pd.DataFrame, gen_df: pd.DataFrame, out_path: Path) -> None:
    """Write flat-table diagnostics with adjacent orig/gen/diff columns per settlement."""
    cat_col = orig_df.columns[0]
    categories = orig_df[cat_col].tolist()
    parts = [pd.DataFrame({cat_col: categories})]
    for col in orig_df.columns[1:]:
        orig = orig_df[col].to_numpy(dtype=float)
        gen = gen_df.reindex(categories).get(col, pd.Series([np.nan] * len(categories))).to_numpy(dtype=float)
        diff = safe_pct_diff(orig, gen)
        parts.append(pd.DataFrame({
            f"{col} | orig": orig,
            f"{col} | gen": gen,
            f"{col} | diff%": diff,
        }))
    out = pd.concat(parts, axis=1)
    out.to_excel(out_path, index=False)


def write_hier_diagnostic(
    orig_long: pd.DataFrame,
    gen_long: pd.DataFrame,
    label_cols: List[str],
    out_path: Path,
) -> None:
    """Write hierarchical diagnostics with interleaved orig/gen/diff per county+type."""
    merge_cols = ["varmegye", "telepules_tipus"] + label_cols
    merged = orig_long.merge(gen_long, on=merge_cols, how="left", suffixes=("_orig", "_gen"))
    merged["value_gen"] = merged["value_gen"].fillna(0)
    merged["diff_pct"] = safe_pct_diff(merged["value_orig"].to_numpy(), merged["value_gen"].to_numpy())

    temp_orig = merged.pivot_table(
        index=label_cols, columns=["varmegye", "telepules_tipus"], values="value_orig", aggfunc="first"
    )
    temp_gen = merged.pivot_table(
        index=label_cols, columns=["varmegye", "telepules_tipus"], values="value_gen", aggfunc="first"
    )
    temp_diff = merged.pivot_table(
        index=label_cols, columns=["varmegye", "telepules_tipus"], values="diff_pct", aggfunc="first"
    )

    # Ensure column alignment
    cols = list(temp_orig.columns)
    temp_gen = temp_gen.reindex(columns=cols)
    temp_diff = temp_diff.reindex(columns=cols)

    # Interleave orig/gen/diff per (varmegye, telepules_tipus)
    wide_cols = {}
    for v, t in cols:
        wide_cols[(v, t, "orig")] = temp_orig[(v, t)]
        wide_cols[(v, t, "gen")] = temp_gen[(v, t)]
        wide_cols[(v, t, "diff%")] = temp_diff[(v, t)]

    out = pd.DataFrame(wide_cols, index=temp_orig.index).reset_index()
    # Flatten MultiIndex columns for Excel export
    flat_cols = []
    for c in out.columns:
        if isinstance(c, tuple):
            flat_cols.append(" | ".join([str(x) for x in c if x is not None and str(x) != ""]))
        else:
            flat_cols.append(c)
    out.columns = flat_cols
    out.to_excel(out_path, index=False)


def compute_flat_population_generated(persons: pd.DataFrame, flat_pop: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct flat population statistics from synthetic persons."""
    categories = flat_pop.iloc[:, 0].tolist()
    settlements = flat_pop.columns[1:]
    gen = pd.DataFrame(index=categories, columns=settlements, dtype=float)

    # Precompute masks
    age = persons["age"].to_numpy()
    sex = persons["sex"].to_numpy()
    edu = persons["education"].to_numpy()
    emp = persons["employment_status"].to_numpy()
    marital = persons["marital_status"].to_numpy()
    fert = persons["children_count"].to_numpy()

    for settlement in settlements:
        sub = persons[persons["settlement"] == settlement]
        if sub.empty:
            gen[settlement] = 0
            continue
        a = sub["age"].to_numpy()
        s = sub["sex"].to_numpy()
        e = sub["education"].to_numpy()
        em = sub["employment_status"].to_numpy()
        m = sub["marital_status"].to_numpy()
        f = sub["children_count"].to_numpy()

        counts = {}
        counts["Férfi"] = int((s == "Férfi").sum())
        counts["Nő"] = int((s == "Nő").sum())
        counts["10 évesnél fiatalabb"] = int((a < 10).sum())
        counts["10–19 éves"] = int(((a >= 10) & (a <= 19)).sum())
        counts["20–29 éves"] = int(((a >= 20) & (a <= 29)).sum())
        counts["30–39 éves"] = int(((a >= 30) & (a <= 39)).sum())
        counts["40–49 éves"] = int(((a >= 40) & (a <= 49)).sum())
        counts["50–59 éves"] = int(((a >= 50) & (a <= 59)).sum())
        counts["60–69 éves"] = int(((a >= 60) & (a <= 69)).sum())
        counts["70–79 éves"] = int(((a >= 70) & (a <= 79)).sum())
        counts["80–89 éves"] = int(((a >= 80) & (a <= 89)).sum())
        counts["90 éves és idősebb"] = int((a >= 90).sum())
        counts["15 évesnél fiatalabb férfi"] = int(((s == "Férfi") & (a < 15)).sum())
        counts["15–64 éves férfi"] = int(((s == "Férfi") & (a >= 15) & (a <= 64)).sum())
        counts["65 éves és idősebb férfi"] = int(((s == "Férfi") & (a >= 65)).sum())
        counts["15 évesnél fiatalabb nő"] = int(((s == "Nő") & (a < 15)).sum())
        counts["15–64 éves nő"] = int(((s == "Nő") & (a >= 15) & (a <= 64)).sum())
        counts["65 éves és idősebb nő"] = int(((s == "Nő") & (a >= 65)).sum())

        # Marital status (15+)
        for cat in MARITAL_CATS:
            counts[cat] = int(((a >= 15) & (m == cat)).sum())

        counts["15 évesnél fiatalabb személy"] = int((a < 15).sum())

        # Fertility (women 15+)
        for cat in FERTILITY_CATS:
            counts[cat] = int(((s == "Nő") & (a >= 15) & (f == cat)).sum())

        # Education (7+)
        for cat in EDUCATION_CATS:
            counts[cat] = int(((a >= 7) & (e == cat)).sum())
        counts["7 évesnél fiatalabb személy"] = int((a < 7).sum())

        # Employment (15+)
        for cat in EMPLOYMENT_CATS:
            counts[cat] = int(((a >= 15) & (em == cat)).sum())

        gen[settlement] = pd.Series(counts)

    return gen


def compute_flat_household_generated(
    persons: pd.DataFrame,
    households: pd.DataFrame,
    flat_house: pd.DataFrame,
) -> pd.DataFrame:
    """Reconstruct flat household statistics from synthetic persons/households."""
    categories = flat_house.iloc[:, 0].tolist()
    settlements = flat_house.columns[1:]
    gen = pd.DataFrame(index=categories, columns=settlements, dtype=float)

    # Household aggregates from persons
    hh = persons.dropna(subset=["household_id"]).copy()
    hh["household_id"] = hh["household_id"].astype(int)
    hh = hh.groupby("household_id").agg(
        settlement=("settlement", "first"),
        size=("person_id", "count"),
        u15=("age", lambda x: int((x < 15).sum())),
        u30=("age", lambda x: int((x < 30).sum())),
        a30_64=("age", lambda x: int(((x >= 30) & (x <= 64)).sum())),
        o65=("age", lambda x: int((x >= 65).sum())),
        emp_f=("employment_status", lambda x: int((x == "Foglalkoztatott").sum())),
        emp_u=("employment_status", lambda x: int((x == "Munkanélküli").sum())),
        emp_i=("employment_status", lambda x: int((x == "Ellátásban részesülő inaktív").sum())),
        emp_d=("employment_status", lambda x: int((x == "Eltartott").sum())),
    ).reset_index()

    if households is not None and "family_structure" in households.columns:
        hh = hh.merge(households[["household_id", "family_structure"]], on="household_id", how="left")
    else:
        hh["family_structure"] = None

    for settlement in settlements:
        sub = hh[hh["settlement"] == settlement]
        if sub.empty:
            gen[settlement] = 0
            continue

        counts = {}
        # Size categories
        counts["Egyszemélyes háztartás"] = int((sub["size"] == 1).sum())
        counts["Kétszemélyes háztartás"] = int((sub["size"] == 2).sum())
        counts["Háromszemélyes háztartás"] = int((sub["size"] == 3).sum())
        counts["Négyszemélyes háztartás"] = int((sub["size"] == 4).sum())
        counts["Ötszemélyes háztartás"] = int((sub["size"] == 5).sum())
        counts["Hat vagy többszemélyes háztartás"] = int((sub["size"] >= 6).sum())

        # Family structure categories (approximate)
        fam = sub["family_structure"]
        counts["Egy családból álló háztartás"] = int(fam.isin(["Egy családból álló háztartás", "Házaspár és élettársi kapcsolat", "Egy szülő gyermekkel"]).sum())
        counts["Házaspár és élettársi kapcsolat"] = int((fam == "Házaspár és élettársi kapcsolat").sum())
        counts["Egy szülő gyermekkel"] = int((fam == "Egy szülő gyermekkel").sum())
        counts["Több családból álló háztartás"] = 0
        counts["Két családból álló háztartás"] = 0
        counts["Három és több családból álló háztartás"] = 0
        counts["Nem családháztartás"] = int((fam == "Nem családháztartás").sum())
        counts["Egyéb összetételű háztartás"] = int((fam == "Egyéb összetételű háztartás").sum())

        # Under-15 presence
        counts["Nincs 15 évesnél fiatalabb személy a háztartásban"] = int((sub["u15"] == 0).sum())
        counts["1 személy 15 évesnél fiatalabb a háztartásban"] = int((sub["u15"] == 1).sum())
        counts["2 személy 15 évesnél fiatalabb a háztartásban"] = int((sub["u15"] == 2).sum())
        counts["3 vagy több személy 15 évesnél fiatalabb a háztartásban"] = int((sub["u15"] >= 3).sum())

        # Under-30 presence
        counts["Nincs 30 évesnél fiatalabb személy a háztartásban"] = int((sub["u30"] == 0).sum())
        counts["1 személy 30 évesnél fiatalabb a háztartásban"] = int((sub["u30"] == 1).sum())
        counts["2 személy 30 évesnél fiatalabb a háztartásban"] = int((sub["u30"] == 2).sum())
        counts["3 vagy több személy 30 évesnél fiatalabb a háztartásban"] = int((sub["u30"] >= 3).sum())

        # 30-64 presence
        counts["Nincs 30–64 éves személy a háztartásban"] = int((sub["a30_64"] == 0).sum())
        counts["1 személy 30–64 éves a háztartásban"] = int((sub["a30_64"] == 1).sum())
        counts["2 személy 30–64 éves a háztartásban"] = int((sub["a30_64"] == 2).sum())
        counts["3 vagy több személy 30–64 éves a háztartásban"] = int((sub["a30_64"] >= 3).sum())

        # 65+ presence
        counts["Nincs 65 éves és idősebb személy a háztartásban"] = int((sub["o65"] == 0).sum())
        counts["1 személy 65 éves és idősebb a háztartásban"] = int((sub["o65"] == 1).sum())
        counts["2 személy 65 éves és idősebb a háztartásban"] = int((sub["o65"] == 2).sum())
        counts["3 vagy több személy 65 éves és idősebb a háztartásban"] = int((sub["o65"] >= 3).sum())

        # Age composition combos
        u30 = sub["u30"] > 0
        a30_64 = sub["a30_64"] > 0
        o65 = sub["o65"] > 0
        counts["Csak 30 évesnél fiatalabb személy van a háztartásban"] = int((u30 & ~a30_64 & ~o65).sum())
        counts["Csak 30–64 éves személy van a háztartásban"] = int((~u30 & a30_64 & ~o65).sum())
        counts["Csak 65 éves és idősebb személy van a háztartásban"] = int((~u30 & ~a30_64 & o65).sum())
        counts["30 évesnél fiatalabb és 30–64 éves személyek vannak a háztartásban"] = int((u30 & a30_64 & ~o65).sum())
        counts["30 évesnél fiatalabb és 65 éves és idősebb személyek vannak a háztartásban"] = int((u30 & ~a30_64 & o65).sum())
        counts["30–64 éves és 65 éves és idősebb személyek vannak a háztartásban"] = int((~u30 & a30_64 & o65).sum())
        counts["30 évesnél fiatalabb, 30–64 éves és 65 éves és idősebb személyek vannak a háztartásban"] = int((u30 & a30_64 & o65).sum())

        # Employment composition
        for label, col in [
            ("foglalkoztatott", "emp_f"),
            ("munkanélküli", "emp_u"),
            ("ellátásban részesülő inaktív", "emp_i"),
            ("eltartott", "emp_d"),
        ]:
            counts[f"Nincs {label} személy a háztartásban"] = int((sub[col] == 0).sum())
            counts[f"1 {label} személy van a háztartásban"] = int((sub[col] == 1).sum())
            counts[f"2 {label} személy van a háztartásban"] = int((sub[col] == 2).sum())
            key3 = f"3 vagy több {label} személy van a háztartásban"
            if key3 not in categories and label == "foglalkoztatott":
                key3 = "3 vagy több foglalkoztatott szeméy van a háztartásban"
            counts[key3] = int((sub[col] >= 3).sum())

        counts["Van foglalkoztatott a háztartásban"] = int((sub["emp_f"] > 0).sum())
        counts["Nincs foglalkoztatott a háztartásban"] = int((sub["emp_f"] == 0).sum())

        gen[settlement] = pd.Series(counts)

    return gen


def compute_flat_housing_generated(dwellings: pd.DataFrame, flat_housing: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct flat housing statistics from synthetic dwellings."""
    categories = flat_housing.iloc[:, 0].tolist()
    settlements = flat_housing.columns[1:]
    gen = pd.DataFrame(index=categories, columns=settlements, dtype=float)

    if dwellings is None or dwellings.empty:
        return gen.fillna(0)

    for settlement in settlements:
        sub = dwellings[dwellings["settlement"] == settlement]
        if sub.empty:
            gen[settlement] = 0
            continue
        counts = {
            "Lakott lakás": int(len(sub)),
            "1 szoba": int((sub["rooms"] == 1).sum()),
            "2 szoba": int((sub["rooms"] == 2).sum()),
            "3 szoba": int((sub["rooms"] == 3).sum()),
            "4 vagy több szoba": int((sub["rooms"] >= 4).sum()),
        }
        gen[settlement] = pd.Series(counts)

    return gen


def generate_diagnostics(
    out_dir: Path,
    flat_pop: pd.DataFrame,
    flat_house: pd.DataFrame,
    flat_housing: pd.DataFrame,
    activity_long: pd.DataFrame,
    activity_edu_long: pd.DataFrame,
    marital_long: pd.DataFrame,
    household_hier_long: pd.DataFrame,
    occupation_long: pd.DataFrame,
    sector_long: pd.DataFrame,
    commute_long: pd.DataFrame,
    school_long: pd.DataFrame,
    health_long: pd.DataFrame,
) -> None:
    """Generate all flat and hierarchical diagnostic workbooks for an output directory."""
    diag_dir = out_dir / "diagnostics"
    ensure_dir(diag_dir)

    pop_path = out_dir / "synthetic_population.parquet"
    if not pop_path.exists():
        return
    persons = pq.read_table(pop_path).to_pandas()

    households = None
    house_path = out_dir / "synthetic_households.parquet"
    if house_path.exists():
        households = pq.read_table(house_path).to_pandas()

    dwellings = None
    dwell_path = out_dir / "synthetic_dwellings.parquet"
    if dwell_path.exists():
        dwellings = pq.read_table(dwell_path).to_pandas()

    # Limit flat tables to settlements present in the synthetic population
    settlements_present = sorted(persons["settlement"].dropna().unique().tolist())
    flat_cols = [flat_pop.columns[0]] + [c for c in flat_pop.columns[1:] if c in settlements_present]
    flat_pop = flat_pop[flat_cols]
    flat_house = flat_house[flat_cols] if flat_house is not None else flat_house
    flat_housing = flat_housing[flat_cols] if flat_housing is not None else flat_housing

    # Flat diagnostics
    gen_flat_pop = compute_flat_population_generated(persons, flat_pop)
    write_flat_diagnostic(flat_pop, gen_flat_pop, diag_dir / "flat_A_nepesseg_adatok_telepulesenkent_diagnostic.xlsx")

    gen_flat_house = compute_flat_household_generated(persons, households, flat_house)
    write_flat_diagnostic(flat_house, gen_flat_house, diag_dir / "flat_A_haztartasok_adatai_telepulesenkent_diagnostic.xlsx")

    gen_flat_housing = compute_flat_housing_generated(dwellings, flat_housing)
    write_flat_diagnostic(flat_housing, gen_flat_housing, diag_dir / "flat_A_lakasok_legfontosabb_jellemzok_telepulesenkent_diagnostic.xlsx")

    # Household hier diagnostics (size x age_comp x employment_comp)
    if households is not None and not households.empty:
        hh = persons.dropna(subset=["household_id"]).copy()
        hh["household_id"] = hh["household_id"].astype(int)
        agg = hh.groupby("household_id").agg(
            varmegye=("varmegye", "first"),
            telepules_tipus=("telepules_tipus", "first"),
            size=("person_id", "count"),
            u30=("age", lambda x: int((x < 30).sum())),
            a30_64=("age", lambda x: int(((x >= 30) & (x <= 64)).sum())),
            o65=("age", lambda x: int((x >= 65).sum())),
            emp_f=("employment_status", lambda x: int((x == "Foglalkoztatott").sum())),
            emp_u=("employment_status", lambda x: int((x == "Munkanélküli").sum())),
            emp_i=("employment_status", lambda x: int((x == "Ellátásban részesülő inaktív").sum())),
            emp_d=("employment_status", lambda x: int((x == "Eltartott").sum())),
        ).reset_index()
        # Map size to label
        size_label_map = {
            1: "Egyszemélyes háztartás",
            2: "Kétszemélyes háztartás",
            3: "Háromszemélyes háztartás",
            4: "Négyszemélyes háztartás",
            5: "Ötszemélyes háztartás",
        }
        agg["household_size"] = agg["size"].map(size_label_map).fillna("Hat vagy többszemélyes háztartás")
        agg["age_comp"] = agg.apply(
            lambda r: household_age_comp_label(r["u30"] > 0, r["a30_64"] > 0, r["o65"] > 0),
            axis=1,
        )
        agg["employment_comp"] = agg.apply(
            lambda r: household_emp_comp_label(r["emp_f"], r["emp_u"], r["emp_i"], r["emp_d"]),
            axis=1,
        )
        gen_home = (
            agg.groupby(["varmegye", "telepules_tipus", "household_size", "age_comp", "employment_comp"])
            .size()
            .reset_index(name="value")
        )
        write_hier_diagnostic(
            household_hier_long,
            gen_home,
            ["household_size", "age_comp", "employment_comp"],
            diag_dir / "hier_homesize_agegroup_employment_diagnostic.xlsx",
        )

    # Hier diagnostics
    # Activity (full, incl under-15)
    activity_df = persons.copy()
    activity_df["age_band"] = activity_band_for_age_array(activity_df["age"].to_numpy())
    activity_df["education_adj"] = np.where(activity_df["age"] < 15, "15 évesnél fiatalabb személy", activity_df["education"])
    gen_activity = (
        activity_df.groupby(["varmegye", "telepules_tipus", "sex", "age_band", "education_adj", "employment_status"])
        .size()
        .reset_index(name="value")
        .rename(columns={"age_band": "age", "education_adj": "education", "employment_status": "activity"})
    )
    write_hier_diagnostic(
        activity_long,
        gen_activity,
        ["sex", "age", "education", "activity"],
        diag_dir / "hier_gazdasagi_aktivitas_varmegyenkent_telepulestipusonkent_diagnostic.xlsx",
    )

    # Activity + education (nem-kor)
    nemkor_df = persons[persons["age"] >= 15].copy()
    nemkor_df["age_band"] = age_band_70plus_for_age_array(nemkor_df["age"].to_numpy())
    gen_nemkor = (
        nemkor_df.groupby(["varmegye", "telepules_tipus", "sex", "age_band", "employment_status", "education"])
        .size()
        .reset_index(name="value")
        .rename(columns={"age_band": "age", "employment_status": "activity"})
    )
    write_hier_diagnostic(
        activity_edu_long,
        gen_nemkor,
        ["sex", "age", "activity", "education"],
        diag_dir / "hier_nem_kor_gazdasagiaktivitas_iskolaivegzettseg_diagnostic.xlsx",
    )

    # Marital status (nem-kor)
    mar_df = persons[persons["age"] >= 15].copy()
    mar_df["age_band"] = age_band_70plus_for_age_array(mar_df["age"].to_numpy())
    gen_mar = (
        mar_df.groupby(["varmegye", "telepules_tipus", "sex", "age_band", "marital_status"])
        .size()
        .reset_index(name="value")
        .rename(columns={"age_band": "age"})
    )
    write_hier_diagnostic(
        marital_long,
        gen_mar,
        ["sex", "age", "marital_status"],
        diag_dir / "hier_nem_kor_csaladiallapot_diagnostic.xlsx",
    )

    # Occupation
    emp = persons[persons["employment_status"] == "Foglalkoztatott"].copy()
    emp["age_band"] = occupation_band_for_age_array(emp["age"].to_numpy())
    gen_occ = (
        emp.groupby(["varmegye", "telepules_tipus", "sex", "age_band", "education", "occupation_group"])
        .size()
        .reset_index(name="value")
        .rename(columns={"age_band": "age", "occupation_group": "occupation"})
    )
    write_hier_diagnostic(
        occupation_long,
        gen_occ,
        ["sex", "age", "education", "occupation"],
        diag_dir / "hier_foglalkoztatott_nepesseg_foglalkozasi_focsoport_diagnostic.xlsx",
    )

    # Sector
    gen_sector = (
        emp.groupby(["varmegye", "telepules_tipus", "sex", "age_band", "education", "sector"])
        .size()
        .reset_index(name="value")
        .rename(columns={"age_band": "age"})
    )
    write_hier_diagnostic(
        sector_long,
        gen_sector,
        ["sex", "age", "education", "sector"],
        diag_dir / "hier_foglalkoztatott_nemzetgazdasagi_agazat_diagnostic.xlsx",
    )

    # Commuting
    gen_commute = (
        emp.groupby(["varmegye", "telepules_tipus", "sex", "age_band", "education", "commute_region", "commute_type"])
        .size()
        .reset_index(name="value")
        .rename(columns={"age_band": "age", "commute_region": "employment_region"})
    )
    write_hier_diagnostic(
        commute_long,
        gen_commute,
        ["sex", "age", "education", "employment_region", "commute_type"],
        diag_dir / "hier_foglalkoztatott_ingazas_diagnostic.xlsx",
    )

    # School attendance
    school_df = persons.copy()
    school_df = school_df[school_df["age"] >= 6]
    school_df = school_df[school_df["school_level"].notna()]
    school_df = school_df[school_df["school_level"] != "Nem iskolázott"]
    school_df["age_band"] = school_band_for_age_array(school_df["age"].to_numpy())
    gen_school = (
        school_df.groupby(["varmegye", "telepules_tipus", "sex", "age_band", "school_level"])
        .size()
        .reset_index(name="value")
        .rename(columns={"age_band": "age"})
    )
    write_hier_diagnostic(
        school_long,
        gen_school,
        ["sex", "age", "school_level"],
        diag_dir / "hier_iskolaba_jaro_nepesseg_diagnostic.xlsx",
    )

    # Health
    health_frames = []
    # Sex groups
    sex_health = persons.groupby(["varmegye", "telepules_tipus", "sex", "disability_status"]).size().reset_index(name="value")
    sex_health = sex_health.rename(columns={"sex": "group", "disability_status": "health_category"})
    health_frames.append(sex_health)
    sex_health = persons.groupby(["varmegye", "telepules_tipus", "sex", "chronic_status"]).size().reset_index(name="value")
    sex_health = sex_health.rename(columns={"sex": "group", "chronic_status": "health_category"})
    health_frames.append(sex_health)
    sex_health = persons.groupby(["varmegye", "telepules_tipus", "sex", "limitation_status"]).size().reset_index(name="value")
    sex_health = sex_health.rename(columns={"sex": "group", "limitation_status": "health_category"})
    health_frames.append(sex_health)

    # Age groups
    tmp = persons.copy()
    tmp["age_band"] = health_band_for_age_array(tmp["age"].to_numpy())
    age_health = tmp.groupby(["varmegye", "telepules_tipus", "age_band", "disability_status"]).size().reset_index(name="value")
    age_health = age_health.rename(columns={"age_band": "group", "disability_status": "health_category"})
    health_frames.append(age_health)
    age_health = tmp.groupby(["varmegye", "telepules_tipus", "age_band", "chronic_status"]).size().reset_index(name="value")
    age_health = age_health.rename(columns={"age_band": "group", "chronic_status": "health_category"})
    health_frames.append(age_health)
    age_health = tmp.groupby(["varmegye", "telepules_tipus", "age_band", "limitation_status"]).size().reset_index(name="value")
    age_health = age_health.rename(columns={"age_band": "group", "limitation_status": "health_category"})
    health_frames.append(age_health)

    # Education groups
    edu_health = persons.groupby(["varmegye", "telepules_tipus", "education", "disability_status"]).size().reset_index(name="value")
    edu_health = edu_health.rename(columns={"education": "group", "disability_status": "health_category"})
    health_frames.append(edu_health)
    edu_health = persons.groupby(["varmegye", "telepules_tipus", "education", "chronic_status"]).size().reset_index(name="value")
    edu_health = edu_health.rename(columns={"education": "group", "chronic_status": "health_category"})
    health_frames.append(edu_health)
    edu_health = persons.groupby(["varmegye", "telepules_tipus", "education", "limitation_status"]).size().reset_index(name="value")
    edu_health = edu_health.rename(columns={"education": "group", "limitation_status": "health_category"})
    health_frames.append(edu_health)

    # Activity groups
    act_health = persons.groupby(["varmegye", "telepules_tipus", "employment_status", "disability_status"]).size().reset_index(name="value")
    act_health = act_health.rename(columns={"employment_status": "group", "disability_status": "health_category"})
    health_frames.append(act_health)
    act_health = persons.groupby(["varmegye", "telepules_tipus", "employment_status", "chronic_status"]).size().reset_index(name="value")
    act_health = act_health.rename(columns={"employment_status": "group", "chronic_status": "health_category"})
    health_frames.append(act_health)
    act_health = persons.groupby(["varmegye", "telepules_tipus", "employment_status", "limitation_status"]).size().reset_index(name="value")
    act_health = act_health.rename(columns={"employment_status": "group", "limitation_status": "health_category"})
    health_frames.append(act_health)

    gen_health = pd.concat(health_frames, ignore_index=True)
    # Filter to categories present in the source table
    gen_health = gen_health[gen_health["health_category"].isin(health_long["health_category"].unique())]

    write_hier_diagnostic(
        health_long,
        gen_health,
        ["group", "health_category"],
        diag_dir / "hier_Egeszseg_allapot_varmegyenkent_telepulestipusonkent_diagnostic.xlsx",
    )


def ipu_weights(templates: pd.DataFrame, targets: Dict[str, Dict[int, float]], max_iter: int = 100, tol: float = 1e-4) -> np.ndarray:
    """Estimate template weights that satisfy selected household marginals."""
    # targets: mapping from dimension name to desired counts by category index
    w = np.ones(len(templates), dtype=float)

    # Precompute indicator matrices per constraint
    indicators = {}
    for dim, mapping in targets.items():
        inds = {}
        for cat, _ in mapping.items():
            if dim == "size":
                inds[cat] = (templates["size"] == cat).to_numpy()
            elif dim == "u15":
                if cat == 0:
                    inds[cat] = (templates["u15"] == 0).to_numpy()
                elif cat == 1:
                    inds[cat] = (templates["u15"] == 1).to_numpy()
                elif cat == 2:
                    inds[cat] = (templates["u15"] == 2).to_numpy()
                else:
                    inds[cat] = (templates["u15"] >= 3).to_numpy()
            elif dim == "o65":
                if cat == 0:
                    inds[cat] = (templates["o65"] == 0).to_numpy()
                elif cat == 1:
                    inds[cat] = (templates["o65"] == 1).to_numpy()
                elif cat == 2:
                    inds[cat] = (templates["o65"] == 2).to_numpy()
                else:
                    inds[cat] = (templates["o65"] >= 3).to_numpy()
        indicators[dim] = inds

    for _ in range(max_iter):
        max_diff = 0
        for dim, mapping in targets.items():
            inds = indicators[dim]
            for cat, target in mapping.items():
                current = (w * inds[cat]).sum()
                if current > 0:
                    factor = target / current
                    w = w * np.where(inds[cat], factor, 1.0)
                    max_diff = max(max_diff, abs(current - target))
        if max_diff < tol:
            break
    return w


# -----------------------------
# Main Pipeline
# -----------------------------

def main() -> None:
    """Run the end-to-end synthetic population generation pipeline."""
    parser = argparse.ArgumentParser(description="Synthetic population generator")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument("--max-settlements", type=int, default=None, help="Limit number of settlements")
    parser.add_argument("--county", default=None, help="Filter by vármegye name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--skip-households", action="store_true", help="Skip household/dwelling generation")
    parser.add_argument("--diagnostics", action="store_true", help="Write diagnostic XLSX files comparing source vs synthetic")
    args = parser.parse_args()

    cfg = Config(seed=args.seed)
    rng = np.random.default_rng(cfg.seed)

    root = Path(__file__).resolve().parent
    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)

    # Load flat tables
    flat_pop = read_flat(root / "flat_A_nepesseg_adatok_telepulesenkent.xlsx")
    flat_house = read_flat(root / "flat_A_haztartasok_adatai_telepulesenkent.xlsx")
    flat_housing = read_flat(root / "flat_A_lakasok_legfontosabb_jellemzok_telepulesenkent.xlsx")

    # Load hierarchy tables
    activity_long = parse_hier_table(
        root / "hier_gazdasagi_aktivitas_varmegyenkent_telepulestipusonkent.xlsx",
        "Adattábla",
        ["sex", "age", "education", "activity"],
    )
    marital_long = parse_hier_table(
        root / "hier_nem_kor_csaladiallapot.xlsx",
        "Adattábla",
        ["sex", "age", "marital_status"],
    )
    activity_edu_long = parse_hier_table(
        root / "hier_nem_kor_gazdasagiaktivitas_iskolaivegzettseg.xlsx",
        "Adattábla",
        ["sex", "age", "activity", "education"],
    )
    occupation_long = parse_hier_table(
        root / "hier_foglalkoztatott_nepesseg_foglalkozasi_focsoport.xlsx",
        "Adattábla",
        ["sex", "age", "education", "occupation"],
    )
    sector_long = parse_hier_table(
        root / "hier_foglalkoztatott_nemzetgazdasagi_agazat.xlsx",
        "Adattábla",
        ["sex", "age", "education", "sector"],
    )
    commute_long = parse_hier_table(
        root / "hier_foglalkoztatott_ingazas.xlsx",
        "Sheet1",
        ["sex", "age", "education", "employment_region", "commute_type"],
    )
    school_long = parse_hier_table(
        root / "hier_iskolaba_jaro_nepesseg.xlsx",
        "Adattábla",
        ["sex", "age", "school_level"],
    )
    health_long = parse_hier_table(
        root / "hier_Egeszseg_allapot_varmegyenkent_telepulestipusonkent.xlsx",
        "Adattábla",
        ["group", "health_category"],
    )
    household_hier_long = parse_hier_household_table(
        root / "hier_homesize_agegroup_employment.xlsx",
        "Adattábla",
    )

    # Crosswalk
    crosswalk = build_crosswalk(root / "telepules_hierarchia.xlsx")

    # Apply county filter if requested
    if args.county:
        crosswalk = crosswalk[crosswalk["varmegye"] == args.county]

    # Settlement list
    settlements = [c for c in flat_pop.columns[1:] if c in crosswalk["settlement"].values]
    if args.max_settlements:
        settlements = settlements[: args.max_settlements]

    # Precompute seed distributions
    seeds, global_seed = build_activity_seed(activity_long)
    seeds_nk, global_seed_nk = build_activity_seed_nem_kor(activity_edu_long)
    # Structural validity mask: only combinations observed in hier activity tables
    activity_valid_mask = (global_seed > 0) | (global_seed_nk > 0)

    # Build reference counts for hier activity validation (includes under-15 category)
    activity_ref_counts: Dict[Tuple[str, str], np.ndarray] = {}
    activity_shape = (len(SEXES), len(ACTIVITY_AGE_BANDS), len(ACTIVITY_EDU_CATS), len(EMPLOYMENT_CATS))
    sex_index = {s: i for i, s in enumerate(SEXES)}
    age_index = {a: i for i, a in enumerate(ACTIVITY_AGE_BANDS)}
    edu_index = {e: i for i, e in enumerate(ACTIVITY_EDU_CATS)}
    emp_index = {e: i for i, e in enumerate(EMPLOYMENT_CATS)}
    for _, row in activity_long.iterrows():
        sex = row.get("sex")
        age = row.get("age")
        edu = row.get("education")
        emp = row.get("activity")
        if sex not in sex_index or age not in age_index or edu not in edu_index or emp not in emp_index:
            continue
        key = (row.get("varmegye"), row.get("telepules_tipus"))
        if key not in activity_ref_counts:
            activity_ref_counts[key] = np.zeros(activity_shape, dtype=float)
        activity_ref_counts[key][sex_index[sex], age_index[age], edu_index[edu], emp_index[emp]] += float(row.get("value", 0))

    # Accumulator for generated activity counts (for validation)
    activity_gen_counts: Dict[Tuple[str, str], np.ndarray] = {}
    age_split_profile = build_age_split_profile(activity_long, health_long)

    # Build conditional probability tables
    occupation_probs = build_prob_table(
        occupation_long[occupation_long["age"].isin(OCCUPATION_AGE_BANDS)],
        ["varmegye", "telepules_tipus", "sex", "age", "education"],
        "occupation",
    )
    sector_probs = build_prob_table(
        sector_long[sector_long["age"].isin(OCCUPATION_AGE_BANDS)],
        ["varmegye", "telepules_tipus", "sex", "age", "education"],
        "sector",
    )
    commute_long = commute_long.copy()
    commute_long["commute_combo"] = commute_long["employment_region"].astype(str) + "||" + commute_long["commute_type"].astype(str)
    commute_probs = build_prob_table(
        commute_long[commute_long["age"].isin(OCCUPATION_AGE_BANDS)],
        ["varmegye", "telepules_tipus", "sex", "age", "education"],
        "commute_combo",
    )
    occupation_default = global_distribution(occupation_long, "occupation")
    sector_default = global_distribution(sector_long, "sector")
    commute_default = global_distribution(commute_long, "commute_combo")
    if not occupation_default[0]:
        occupation_default = (["Szakképzettséget nem igénylő (egyszerű) foglalkozások"], np.array([1.0]))
    if not sector_default[0]:
        sector_default = (["Egyéb szolgáltatás"], np.array([1.0]))
    if not commute_default[0]:
        commute_default = (["Ismeretlen||Lakóhely településén foglalkoztatott"], np.array([1.0]))

    # Build school probabilities + attendance rates
    school_probs = build_prob_table(
        school_long[school_long["age"].isin(SCHOOL_AGE_BANDS)],
        ["varmegye", "telepules_tipus", "sex", "age"],
        "school_level",
    )

    # Approximate population by county+type+sex+school age group from flat
    pop_long = melt_flat(flat_pop, "count")
    pop_long = pop_long.merge(crosswalk, on="settlement", how="left")
    pop_long = pop_long.dropna(subset=["varmegye", "telepules_tipus"])
    # Map flat age categories to school bands
    school_age_rows = []
    for _, row in pop_long.iterrows():
        cat = row["category"]
        count = row["count"]
        if cat == "10 évesnél fiatalabb":
            # split into 6-9 and 0-5 roughly; assume 40% in 6-9
            school_age_rows.append({**row, "age_group": "6–9 éves", "count": count * 0.4})
        elif cat == "10–19 éves":
            school_age_rows.append({**row, "age_group": "10–14 éves", "count": count * 0.5})
            school_age_rows.append({**row, "age_group": "15–19 éves", "count": count * 0.5})
        elif cat == "20–29 éves":
            school_age_rows.append({**row, "age_group": "20–24 éves", "count": count * 0.5})
            school_age_rows.append({**row, "age_group": "25–29 éves", "count": count * 0.5})
        elif cat == "30–39 éves":
            school_age_rows.append({**row, "age_group": "30–34 éves", "count": count * 0.5})
            school_age_rows.append({**row, "age_group": "35–39 éves", "count": count * 0.5})
        elif cat in ["40–49 éves", "50–59 éves", "60–69 éves", "70–79 éves", "80–89 éves", "90 éves és idősebb"]:
            school_age_rows.append({**row, "age_group": "40 éves és idősebb", "count": count})

    pop_by_ct = pd.DataFrame(school_age_rows)
    pop_by_ct = pop_by_ct.groupby(["varmegye", "telepules_tipus", "age_group"])["count"].sum().reset_index()

    school_rates = build_school_rates(school_long, pop_by_ct)

    # Marital status probabilities by county+type+sex+age (15+ only)
    marital_probs = build_prob_table(
        marital_long[marital_long["age"].isin(MARITAL_AGE_BANDS)],
        ["varmegye", "telepules_tipus", "sex", "age"],
        "marital_status",
    )
    marital_default = global_distribution(marital_long, "marital_status")

    # Health distributions
    health_long = health_long.copy()
    health_long["group_type"] = "other"
    health_long.loc[health_long["group"].isin(SEXES), "group_type"] = "sex"
    health_long.loc[health_long["group"].isin(HEALTH_AGE_BANDS), "group_type"] = "age"
    health_long.loc[health_long["group"].isin(EDUCATION_CATS), "group_type"] = "education"
    health_long.loc[health_long["group"].isin(EMPLOYMENT_CATS), "group_type"] = "activity"

    def build_health_prob(var_cats: List[str]) -> Dict[Tuple, Tuple[List[str], np.ndarray]]:
        sub = health_long[health_long["health_category"].isin(var_cats)]
        return build_prob_table(sub, ["varmegye", "telepules_tipus", "group_type", "group"], "health_category")

    disability_probs = build_health_prob(DISABILITY_CATS)
    chronic_probs = build_health_prob(CHRONIC_CATS)
    limitation_probs = build_health_prob(LIMITATION_CATS)

    # Household hier distributions (size x age_comp x employment_comp)
    size_label_to_int = {
        "Egyszemélyes háztartás": 1,
        "Kétszemélyes háztartás": 2,
        "Háromszemélyes háztartás": 3,
        "Négyszemélyes háztartás": 4,
        "Ötszemélyes háztartás": 5,
        "Hat vagy többszemélyes háztartás": 6,
    }
    household_hier_long = household_hier_long.copy()
    household_hier_long = household_hier_long[
        household_hier_long["household_size"].isin(size_label_to_int.keys())
        & household_hier_long["age_comp"].isin(HOUSEHOLD_AGE_COMPS)
        & household_hier_long["employment_comp"].isin(HOUSEHOLD_EMP_COMPS)
    ]
    household_hier_long["size"] = household_hier_long["household_size"].map(size_label_to_int)

    size_vals = [1, 2, 3, 4, 5, 6]
    age_vals = HOUSEHOLD_AGE_COMPS
    emp_vals = HOUSEHOLD_EMP_COMPS

    household_seed: Dict[Tuple[str, str], np.ndarray] = {}
    global_household_seed = np.zeros((len(size_vals), len(age_vals), len(emp_vals)), dtype=float)

    for (varmegye, telep), sub in household_hier_long.groupby(["varmegye", "telepules_tipus"]):
        arr = np.zeros_like(global_household_seed)
        for _, row in sub.iterrows():
            i = size_vals.index(int(row["size"]))
            j = age_vals.index(row["age_comp"])
            k = emp_vals.index(row["employment_comp"])
            arr[i, j, k] += float(row["value"])
        household_seed[(varmegye, telep)] = arr
        global_household_seed += arr

    # Precompute conditional employment distributions
    household_emp_cond: Dict[Tuple[str, str, int, str], np.ndarray] = {}
    household_emp_cond_size: Dict[Tuple[str, str, int], np.ndarray] = {}
    household_emp_global = None
    if global_household_seed.sum() > 0:
        emp_totals = global_household_seed.sum(axis=(0, 1))
        household_emp_global = emp_totals / emp_totals.sum()

    for key, arr in household_seed.items():
        # size+age conditional
        for i, size_val in enumerate(size_vals):
            for j, age_val in enumerate(age_vals):
                total = arr[i, j, :].sum()
                if total > 0:
                    household_emp_cond[(key[0], key[1], size_val, age_val)] = arr[i, j, :] / total
            total_size = arr[i, :, :].sum()
            if total_size > 0:
                household_emp_cond_size[(key[0], key[1], size_val)] = arr[i, :, :].sum(axis=0) / total_size

    # Household templates
    templates = build_household_templates()

    # Output writers
    person_writer = None
    household_writer = None
    dwelling_writer = None

    person_schema = None
    household_schema = None
    dwelling_schema = None

    person_id = 1
    household_id = 1
    dwelling_id = 1

    validation = {
        "settlements": {},
        "summary": {},
    }

    # Pre-extract flat category rows for fast lookup
    pop_df = flat_pop.set_index(flat_pop.columns[0])
    house_df = flat_house.set_index(flat_house.columns[0])
    housing_df = flat_housing.set_index(flat_housing.columns[0])

    for idx, settlement in enumerate(settlements):
        cw = crosswalk[crosswalk["settlement"] == settlement]
        if cw.empty:
            continue
        varmegye = cw.iloc[0]["varmegye"]
        telep = cw.iloc[0]["telepules_tipus"]
        ksh_code = cw.iloc[0]["ksh_code"]

        # Flat population counts
        pop_counts = pop_df[settlement].to_dict()

        total_pop = int(pop_counts.get("Férfi", 0) + pop_counts.get("Nő", 0))
        if total_pop <= 0:
            continue

        # Sex targets (total)
        sex_targets = np.array([pop_counts.get("Férfi", 0), pop_counts.get("Nő", 0)], dtype=float)

        # Age bands for activity (target marginals)
        under10 = pop_counts.get("10 évesnél fiatalabb", 0)
        age10_19 = pop_counts.get("10–19 éves", 0)
        age20_29 = pop_counts.get("20–29 éves", 0)
        age30_39 = pop_counts.get("30–39 éves", 0)
        age40_49 = pop_counts.get("40–49 éves", 0)
        age50_59 = pop_counts.get("50–59 éves", 0)
        age60_69 = pop_counts.get("60–69 éves", 0)
        age70_79 = pop_counts.get("70–79 éves", 0)
        age80_89 = pop_counts.get("80–89 éves", 0)
        age90_plus = pop_counts.get("90 éves és idősebb", 0)
        lt15 = pop_counts.get("15 évesnél fiatalabb személy", under10 + age10_19 * 0.5)
        lt7 = pop_counts.get("7 évesnél fiatalabb személy", under10 * 0.7)

        u15_male = pop_counts.get("15 évesnél fiatalabb férfi", np.nan)
        u15_female = pop_counts.get("15 évesnél fiatalabb nő", np.nan)
        if pd.notna(u15_male) and pd.notna(u15_female):
            under15_total = int(round(u15_male + u15_female))
        else:
            under15_total = int(round(lt15))
            # split by sex proportionally
            total_sex = max(1, int(sex_targets.sum()))
            u15_male = under15_total * (sex_targets[0] / total_sex)
            u15_female = under15_total - u15_male

        # Ensure under-15 counts are consistent
        under15_total = min(under15_total, total_pop)
        lt15 = under15_total
        lt7 = min(int(round(lt7)), under15_total)

        profile = age_split_profile.get((varmegye, telep), {
            "10-19": 0.5,
            "20-29": 0.5,
            "30-39": 0.5,
            "40-49": 0.5,
            "50-59": 0.5,
            "60-69": 0.5,
            "70-79": 0.5,
            "80-89": 0.5,
        })

        ten14 = max(0, min(age10_19, lt15 - under10))
        fifteen19 = age10_19 - ten14

        a20_24 = age20_29 * profile["20-29"]
        a25_29 = age20_29 - a20_24

        a30_34 = age30_39 * profile["30-39"]
        a35_39 = age30_39 - a30_34

        a40_44 = age40_49 * profile["40-49"]
        a45_49 = age40_49 - a40_44

        a50_54 = age50_59 * profile["50-59"]
        a55_59 = age50_59 - a50_54

        a60_64 = age60_69 * profile["60-69"]
        a65_69 = age60_69 - a60_64

        a70_74 = age70_79 * profile["70-79"]
        a75_79 = max(0.0, age70_79 - a70_74)
        a75_plus = a75_79 + age80_89 + age90_plus

        total_15plus = max(0, total_pop - under15_total)

        age_band_targets = np.array([
            fifteen19,
            a20_24,
            a25_29,
            a30_34,
            a35_39,
            a40_44,
            a45_49,
            a50_54,
            a55_59,
            a60_64,
            a65_69,
            a70_74,
            a75_plus,
        ], dtype=float)
        age_band_targets = scale_to_total(age_band_targets, total_15plus)

        sex_targets_15p = np.array([
            max(0, sex_targets[0] - u15_male),
            max(0, sex_targets[1] - u15_female),
        ], dtype=float)
        sex_targets_15p = scale_to_total(sex_targets_15p, total_15plus)

        # Education targets (assume 15+ unless totals match total_pop)
        edu_targets = np.array([pop_counts.get(c, 0) for c in EDUCATION_CATS], dtype=float)
        edu_total = edu_targets.sum()
        if edu_total > 0:
            if abs(edu_total - total_pop) <= abs(edu_total - total_15plus):
                low_idx = EDUCATION_CATS.index("Általános iskola 8. évfolyamnál alacsonyabb")
                edu_targets[low_idx] = max(0, edu_targets[low_idx] - under15_total)
        edu_targets = scale_to_total(edu_targets, total_15plus)

        # Employment targets (assume 15+ unless totals match total_pop)
        emp_targets = np.array([pop_counts.get(c, 0) for c in EMPLOYMENT_CATS], dtype=float)
        emp_total = emp_targets.sum()
        if emp_total > 0:
            if abs(emp_total - total_pop) <= abs(emp_total - total_15plus):
                dep_idx = EMPLOYMENT_CATS.index("Eltartott")
                emp_targets[dep_idx] = max(0, emp_targets[dep_idx] - under15_total)
        emp_targets = scale_to_total(emp_targets, total_15plus)

        # Seed distribution for county+type (prefer nem-kor activity+education table)
        seed = seeds_nk.get((varmegye, telep))
        if seed is None or seed.sum() == 0:
            seed = seeds.get((varmegye, telep))
        if seed is None or seed.sum() == 0:
            seed = global_seed_nk if global_seed_nk.sum() > 0 else global_seed
        seed_15p = seed[:, 1:, :, :].copy()
        valid_mask_15p = activity_valid_mask[:, 1:, :, :]
        # Enforce structural zeros from hier table
        seed_15p = seed_15p * valid_mask_15p
        if seed_15p.sum() == 0:
            fallback = global_seed[:, 1:, :, :].copy()
            fallback = fallback * valid_mask_15p
            if fallback.sum() == 0:
                fallback = valid_mask_15p.astype(float)
            seed_15p = fallback
        else:
            # Smooth only within valid cells to allow IPF to satisfy marginals
            eps = 1e-3
            seed_15p = seed_15p + ((seed_15p == 0) & valid_mask_15p) * eps

        # IPF for 15+
        if total_15plus > 0:
            ipf_result = ipf(seed_15p, [sex_targets_15p, age_band_targets, edu_targets, emp_targets], cfg.ipf_max_iter, cfg.ipf_tol)
            ipf_counts = integerize_counts(ipf_result, int(round(total_15plus)))
        else:
            ipf_counts = np.zeros((len(SEXES), len(ACTIVITY_AGE_BANDS_15PLUS), len(EDUCATION_CATS), len(EMPLOYMENT_CATS)), dtype=int)

        # Expand individuals (15+)
        rows = []
        for i, sex in enumerate(SEXES):
            for j, age_band in enumerate(ACTIVITY_AGE_BANDS_15PLUS):
                for k, edu in enumerate(EDUCATION_CATS):
                    for l, emp in enumerate(EMPLOYMENT_CATS):
                        n = ipf_counts[i, j, k, l]
                        if n <= 0:
                            continue
                        rows.append({
                            "sex": sex,
                            "age_band": age_band,
                            "education": edu,
                            "employment_status": emp,
                            "count": n,
                        })

        df_15p = pd.DataFrame(rows)
        if not df_15p.empty:
            df_15p = df_15p.loc[df_15p.index.repeat(df_15p["count"])].drop(columns=["count"]).reset_index(drop=True)
        else:
            df_15p = pd.DataFrame(columns=["sex", "age_band", "education", "employment_status"])

        # Assign ages for 15+
        band_to_range = {
            "15–19 éves": (15, 19),
            "20–24 éves": (20, 24),
            "25–29 éves": (25, 29),
            "30–34 éves": (30, 34),
            "35–39 éves": (35, 39),
            "40–44 éves": (40, 44),
            "45–49 éves": (45, 49),
            "50–54 éves": (50, 54),
            "55–59 éves": (55, 59),
            "60–64 éves": (60, 64),
            "65–69 éves": (65, 69),
            "70–74 éves": (70, 74),
            "75 éves és idősebb": (75, cfg.max_age),
        }

        if not df_15p.empty:
            ages = np.empty(len(df_15p), dtype=int)
            for band, (a, b) in band_to_range.items():
                idxs = df_15p.index[df_15p["age_band"] == band].to_numpy()
                if len(idxs) == 0:
                    continue
                if band == "75 éves és idősebb":
                    total = len(idxs)
                    weights = np.array([a75_79, age80_89, age90_plus], dtype=float)
                    if weights.sum() <= 0:
                        weights = np.array([1.0, 1.0, 1.0], dtype=float)
                    weights = weights / weights.sum()
                    n75_79 = int(round(total * weights[0]))
                    n80_89 = int(round(total * weights[1]))
                    n90_plus = total - n75_79 - n80_89
                    pool = np.concatenate([
                        uniform_age_array(75, 79, n75_79, rng),
                        uniform_age_array(80, 89, n80_89, rng),
                        uniform_age_array(90, cfg.max_age, n90_plus, rng),
                    ])
                    rng.shuffle(pool)
                    ages[idxs] = pool[: len(idxs)]
                else:
                    total = len(idxs)
                    pool = uniform_age_array(a, b, total, rng)
                    rng.shuffle(pool)
                    ages[idxs] = pool[: len(idxs)]
            df_15p["age"] = ages

        # Build under-15 population
        under15_total = int(round(under15_total))
        df_under = pd.DataFrame(columns=["sex", "age", "age_band", "education", "employment_status"])
        if under15_total > 0:
            lt7_total = min(int(round(lt7)), under15_total)
            ages_0_6 = uniform_age_array(0, 6, lt7_total, rng)
            ages_7_14 = uniform_age_array(7, 14, under15_total - lt7_total, rng)
            ages_under = np.concatenate([ages_0_6, ages_7_14])
            u15_male_int = int(round(u15_male))
            u15_female_int = int(round(u15_female))
            sexes = np.array(["Férfi"] * u15_male_int + ["Nő"] * u15_female_int, dtype=object)
            if len(sexes) < len(ages_under):
                extra = len(ages_under) - len(sexes)
                extra_sex = "Férfi" if sex_targets[0] >= sex_targets[1] else "Nő"
                sexes = np.concatenate([sexes, np.array([extra_sex] * extra, dtype=object)])
            if len(sexes) > len(ages_under):
                sexes = sexes[: len(ages_under)]
            rng.shuffle(sexes)
            df_under = pd.DataFrame({
                "sex": sexes,
                "age": ages_under,
                "age_band": "15 évesnél fiatalabb",
                "education": "Általános iskola 8. évfolyamnál alacsonyabb",
                "employment_status": "Eltartott",
            })

        df = pd.concat([df_15p, df_under], ignore_index=True)

        # Fix education for <15
        child_mask = df["age"] < 15
        low_edu = "Általános iskola 8. évfolyamnál alacsonyabb"
        if child_mask.any():
            wrong = child_mask & (df["education"] != low_edu)
            if wrong.any():
                # Swap with adults having low_edu if possible
                adults_low = (~child_mask) & (df["education"] == low_edu)
                swap_n = min(wrong.sum(), adults_low.sum())
                if swap_n > 0:
                    wrong_idx = df.index[wrong].to_numpy()[:swap_n]
                    adult_idx = df.index[adults_low].to_numpy()[:swap_n]
                    prev = df.loc[wrong_idx, "education"].values.copy()
                    df.loc[wrong_idx, "education"] = low_edu
                    df.loc[adult_idx, "education"] = prev
                else:
                    df.loc[wrong, "education"] = low_edu

        # Fix employment status for <15
        if child_mask.any():
            wrong = child_mask & (df["employment_status"] != "Eltartott")
            if wrong.any():
                adults_dep = (~child_mask) & (df["employment_status"] == "Eltartott")
                swap_n = min(wrong.sum(), adults_dep.sum())
                if swap_n > 0:
                    wrong_idx = df.index[wrong].to_numpy()[:swap_n]
                    adult_idx = df.index[adults_dep].to_numpy()[:swap_n]
                    prev = df.loc[wrong_idx, "employment_status"].values.copy()
                    df.loc[wrong_idx, "employment_status"] = "Eltartott"
                    df.loc[adult_idx, "employment_status"] = prev
                else:
                    df.loc[wrong, "employment_status"] = "Eltartott"

        # Accumulate activity counts for hier validation
        key_ct = (varmegye, telep)
        if key_ct not in activity_gen_counts:
            activity_gen_counts[key_ct] = np.zeros(activity_shape, dtype=int)
        activity_age = activity_band_for_age_array(df["age"].to_numpy())
        activity_edu = np.where(df["age"].to_numpy() < 15, "15 évesnél fiatalabb személy", df["education"].to_numpy())
        sex_idx_arr = pd.Series(df["sex"]).map(sex_index).to_numpy()
        age_idx_arr = pd.Series(activity_age).map(age_index).to_numpy()
        edu_idx_arr = pd.Series(activity_edu).map(edu_index).to_numpy()
        emp_idx_arr = pd.Series(df["employment_status"]).map(emp_index).to_numpy()
        valid = (~pd.isna(sex_idx_arr) & ~pd.isna(age_idx_arr) & ~pd.isna(edu_idx_arr) & ~pd.isna(emp_idx_arr))
        if valid.any():
            np.add.at(
                activity_gen_counts[key_ct],
                (
                    sex_idx_arr[valid].astype(int),
                    age_idx_arr[valid].astype(int),
                    edu_idx_arr[valid].astype(int),
                    emp_idx_arr[valid].astype(int),
                ),
                1,
            )

        # Assign occupation/sector/commuting for employed (grouped sampling)
        df["occupation_group"] = "Nem foglalkoztatott"
        df["sector"] = "Nem foglalkoztatott"
        df["commute_type"] = "Nem foglalkoztatott"
        df["commute_region"] = "Nem foglalkoztatott"

        employed_mask = df["employment_status"] == "Foglalkoztatott"
        if employed_mask.any():
            sub = df.loc[employed_mask].copy()
            sub["_occ_age_band"] = occupation_band_for_age_array(sub["age"].to_numpy())
            for (sex, age_band, edu), idxs in sub.groupby(["sex", "_occ_age_band", "education"]).groups.items():
                key = (varmegye, telep, sex, age_band, edu)
                # occupation
                outcomes, p = occupation_probs.get(
                    key,
                    occupation_default if occupation_default[0] else (["Szakképzettséget nem igénylő (egyszerű) foglalkozások"], np.array([1.0])),
                )
                df.loc[idxs, "occupation_group"] = rng.choice(outcomes, size=len(idxs), p=p)
                # sector
                outcomes, p = sector_probs.get(
                    key,
                    sector_default if sector_default[0] else (["Egyéb szolgáltatás"], np.array([1.0])),
                )
                df.loc[idxs, "sector"] = rng.choice(outcomes, size=len(idxs), p=p)
                # commute
                outcomes, p = commute_probs.get(
                    key,
                    commute_default if commute_default[0] else (["Ismeretlen||Lakóhely településén foglalkoztatott"], np.array([1.0])),
                )
                combos = rng.choice(outcomes, size=len(idxs), p=p)
                regions = []
                types = []
                for combo in combos:
                    if "||" in combo:
                        region, ctype = combo.split("||", 1)
                    else:
                        region, ctype = "Ismeretlen", combo
                    regions.append(region)
                    types.append(ctype)
                df.loc[idxs, "commute_region"] = regions
                df.loc[idxs, "commute_type"] = types

        # School assignment (grouped sampling)
        df["school_level"] = "Nem jár iskolába"
        age_mask = df["age"] >= 6
        if age_mask.any():
            sub = df.loc[age_mask].copy()
            sub["_school_band"] = school_band_for_age_array(sub["age"].to_numpy())
            for (sex, band), idxs in sub.groupby(["sex", "_school_band"]).groups.items():
                key_rate = (varmegye, telep, sex, band)
                p_in = school_rates.get(key_rate, 0.0)
                if p_in <= 0:
                    continue
                idxs_arr = np.array(list(idxs))
                take = rng.random(len(idxs_arr)) < p_in
                if not take.any():
                    continue
                student_idxs = idxs_arr[take]
                key = (varmegye, telep, sex, band)
                outcomes, p = school_probs.get(key, (["Alapfokú képzésre jár"], np.array([1.0])))
                df.loc[student_idxs, "school_level"] = rng.choice(outcomes, size=len(student_idxs), p=p)

        # Marital status (use hier by sex+age, then rake to settlement totals)
        # Flat marital counts are for 15+ population (they sum to total - under15)
        marital_counts = {c: int(pop_counts.get(c, 0)) for c in MARITAL_CATS}

        marital = np.full(len(df), "Nőtlen, hajadon", dtype=object)
        adult_idx = df.index[df["age"] >= 15].to_numpy()
        if len(adult_idx) > 0:
            adult_df = df.loc[adult_idx, ["sex", "age"]].copy()
            adult_df["_mar_band"] = marital_band_for_age_array(adult_df["age"].to_numpy())

            group_labels = [(sex, band) for sex in SEXES for band in MARITAL_AGE_BANDS]
            group_counts = adult_df.groupby(["sex", "_mar_band"]).size().to_dict()
            row_totals = np.array([group_counts.get(g, 0) for g in group_labels], dtype=float)
            total_adults = int(row_totals.sum())

            # Seed from hier probabilities
            seed = np.zeros((len(group_labels), len(MARITAL_CATS)), dtype=float)
            seed_probs = np.zeros_like(seed)
            for r, (sex, band) in enumerate(group_labels):
                key = (varmegye, telep, sex, band)
                outcomes, p = marital_probs.get(key, marital_default)
                prob_vec = np.zeros(len(MARITAL_CATS), dtype=float)
                if outcomes:
                    for o, prob in zip(outcomes, p):
                        if o in MARITAL_CATS:
                            prob_vec[MARITAL_CATS.index(o)] = prob
                if prob_vec.sum() == 0:
                    prob_vec = np.ones(len(MARITAL_CATS), dtype=float) / len(MARITAL_CATS)
                else:
                    prob_vec = prob_vec / prob_vec.sum()
                seed_probs[r] = prob_vec
                seed[r] = prob_vec * row_totals[r]

            col_targets = np.array([marital_counts.get(c, 0) for c in MARITAL_CATS], dtype=float)
            col_targets = scale_to_total(col_targets, total_adults)
            col_targets_int = integerize_counts(col_targets, total_adults)

            if total_adults > 0:
                ipf_result = ipf(seed, [row_totals, col_targets], cfg.ipf_max_iter, cfg.ipf_tol)
                # Row-constrained rounding
                counts = np.zeros_like(ipf_result, dtype=int)
                for r in range(len(group_labels)):
                    counts[r] = integerize_counts(ipf_result[r], int(round(row_totals[r])))

                # Adjust columns to match targets while preserving row totals
                col_diff = col_targets_int - counts.sum(axis=0)
                if col_diff.sum() == 0 and (col_diff != 0).any():
                    for c_def in range(len(MARITAL_CATS)):
                        need = int(col_diff[c_def])
                        if need <= 0:
                            continue
                        row_order = np.argsort(-seed_probs[:, c_def])
                        for r in row_order:
                            if need <= 0:
                                break
                            if seed_probs[r, c_def] <= 0:
                                continue
                            surplus_cats = [c for c in range(len(MARITAL_CATS)) if col_diff[c] < 0 and counts[r, c] > 0]
                            if not surplus_cats:
                                continue
                            # take from the most surplus category
                            c_sur = min(surplus_cats, key=lambda c: col_diff[c])
                            counts[r, c_sur] -= 1
                            counts[r, c_def] += 1
                            col_diff[c_sur] += 1
                            col_diff[c_def] -= 1
                            need -= 1

                # Assign to individuals by group
                group_indices = adult_df.groupby(["sex", "_mar_band"]).groups
                for r, group in enumerate(group_labels):
                    idxs = group_indices.get(group)
                    if idxs is None:
                        continue
                    idxs = np.array(list(idxs))
                    pool = []
                    for c_idx, cat in enumerate(MARITAL_CATS):
                        pool.extend([cat] * int(counts[r, c_idx]))
                    rng.shuffle(pool)
                    if len(pool) < len(idxs):
                        pool.extend(["Nőtlen, hajadon"] * (len(idxs) - len(pool)))
                    marital[idxs[: len(pool)]] = pool[: len(idxs)]

        df["marital_status"] = marital

        # Fertility
        fertility_counts = {c: int(pop_counts.get(c, 0)) for c in FERTILITY_CATS}
        women_idx = df.index[(df["sex"] == "Nő") & (df["age"] >= 15)].to_numpy()
        pool = []
        for cat, cnt in fertility_counts.items():
            pool.extend([cat] * cnt)
        rng.shuffle(pool)
        fert = np.full(len(df), None, dtype=object)
        n_assign = min(len(pool), len(women_idx))
        if n_assign > 0:
            fert[women_idx[:n_assign]] = pool[:n_assign]
        # Fill remaining eligible women with 0 children
        remaining_idx = women_idx[n_assign:]
        if len(remaining_idx) > 0:
            fert[remaining_idx] = "15 éves és idősebb nő 0 élve született gyermekkel"
        df["children_count"] = fert

        # Health variables using naive combination of marginals (grouped sampling)
        df["_health_band"] = health_band_for_age_array(df["age"].to_numpy())

        def assign_health_column(var_probs: Dict[Tuple, Tuple[List[str], np.ndarray]],
                                 var_cats: List[str],
                                 out_col: str) -> None:
            df[out_col] = var_cats[0]
            for (sex, hband, edu, emp), idxs in df.groupby(["sex", "_health_band", "education", "employment_status"]).groups.items():
                weights = np.ones(len(var_cats), dtype=float)
                for gtype, gval in [
                    ("sex", sex),
                    ("age", hband),
                    ("education", edu),
                    ("activity", emp),
                ]:
                    key = (varmegye, telep, gtype, gval)
                    if key in var_probs:
                        outcomes, p = var_probs[key]
                        p_map = {o: pv for o, pv in zip(outcomes, p)}
                        weights *= np.array([p_map.get(c, 1e-6) for c in var_cats])
                if weights.sum() <= 0:
                    weights = np.ones(len(var_cats), dtype=float)
                weights = weights / weights.sum()
                df.loc[idxs, out_col] = rng.choice(var_cats, size=len(idxs), p=weights)

        assign_health_column(disability_probs, DISABILITY_CATS, "disability_status")
        assign_health_column(chronic_probs, CHRONIC_CATS, "chronic_status")
        assign_health_column(limitation_probs, LIMITATION_CATS, "limitation_status")
        df = df.drop(columns=["_health_band"])

        # Add settlement metadata
        df["settlement"] = settlement
        df["ksh_code"] = ksh_code
        df["varmegye"] = varmegye
        df["telepules_tipus"] = telep

        # Assign IDs (household id later)
        df["person_id"] = np.arange(person_id, person_id + len(df))
        person_id += len(df)
        df["household_id"] = pd.NA

        # Households
        households_df = None
        dwellings_df = None
        if not args.skip_households:
            # Household targets
            house_counts = house_df[settlement].to_dict()
            size_targets = {
                1: house_counts.get("Egyszemélyes háztartás", 0),
                2: house_counts.get("Kétszemélyes háztartás", 0),
                3: house_counts.get("Háromszemélyes háztartás", 0),
                4: house_counts.get("Négyszemélyes háztartás", 0),
                5: house_counts.get("Ötszemélyes háztartás", 0),
                6: house_counts.get("Hat vagy többszemélyes háztartás", 0),
            }
            age_comp_targets = {cat: house_counts.get(cat, 0) for cat in HOUSEHOLD_AGE_COMPS}

            seed_arr = household_seed.get((varmegye, telep), global_household_seed)
            if seed_arr.sum() == 0:
                seed_arr = np.ones_like(seed_arr, dtype=float)

            # Employment composition targets: use flat employed counts + hier split for no-employed
            emp_targets = {
                "Egy foglalkoztatott van a háztartásban": house_counts.get(
                    "1 foglalkoztatott személy van a háztartásban", 0
                ),
                "Két foglalkoztatott van a háztartásban": house_counts.get(
                    "2 foglalkoztatott személy van a háztartásban", 0
                ),
                "Három vagy több foglalkoztatott van a háztartásban": house_counts.get(
                    "3 vagy több foglalkoztatott szeméy van a háztartásban", 0
                )
                or house_counts.get("3 vagy több foglalkoztatott személy van a háztartásban", 0),
            }
            no_employed_total = house_counts.get("Nincs foglalkoztatott személy a háztartásban", 0)
            emp_seed = seed_arr.sum(axis=(0, 1))
            no_emp_idx = [3, 4, 5]
            no_emp_seed = emp_seed[no_emp_idx].sum()
            if no_emp_seed > 0:
                no_emp_shares = emp_seed[no_emp_idx] / no_emp_seed
            else:
                no_emp_shares = np.ones(len(no_emp_idx), dtype=float) / len(no_emp_idx)
            for idx, share in zip(no_emp_idx, no_emp_shares):
                emp_targets[emp_vals[idx]] = no_employed_total * share

            size_vec = np.array([size_targets[i] for i in size_vals], dtype=float)
            age_vec = np.array([age_comp_targets[c] for c in age_vals], dtype=float)
            emp_vec = np.array([emp_targets.get(c, 0) for c in emp_vals], dtype=float)

            total_households = int(round(size_vec.sum()))
            if total_households <= 0:
                total_households = int(round(len(df) / 2.5))

            def scale_vec(vec: np.ndarray, total: int) -> np.ndarray:
                s = vec.sum()
                if s > 0 and abs(s - total) > 1e-6:
                    return vec * (total / s)
                if s <= 0:
                    return np.ones_like(vec, dtype=float) * (total / len(vec))
                return vec

            age_vec = scale_vec(age_vec, total_households)
            emp_vec = scale_vec(emp_vec, total_households)

            # Build size x age_comp x employment_comp table via 3D IPF using hier seed
            ipf_3d = ipf(seed_arr, [size_vec, age_vec, emp_vec], cfg.ipf_max_iter, cfg.ipf_tol)
            counts_3d = integerize_counts(ipf_3d, total_households)

            # Expand to household specs directly from 3D counts
            specs = []
            for i, size_val in enumerate(size_vals):
                for j, age_val in enumerate(age_vals):
                    for k, emp_val in enumerate(emp_vals):
                        cell_total = int(counts_3d[i, j, k])
                        if cell_total <= 0:
                            continue
                        specs.extend(
                            [{"size": size_val, "age_comp": age_val, "employment_comp": emp_val} for _ in range(cell_total)]
                        )

            if not specs:
                num_households = int(total_pop / 2.5)
                specs = [{"size": 2, "age_comp": age_vals[1], "employment_comp": emp_vals[0]} for _ in range(num_households)]

            rng.shuffle(specs)
            households_df = pd.DataFrame(specs)
            households_df["household_id"] = np.arange(household_id, household_id + len(households_df))
            household_id += len(households_df)
            households_df["settlement"] = settlement
            households_df["family_structure"] = "Ismeretlen"

            # Assign people to households based on age and employment composition
            df["_age_house"] = np.select(
                [df["age"] < 15, df["age"] < 30, df["age"] < 65],
                ["u15", "u30", "a30_64"],
                default="o65",
            )
            pools = {}
            for ag in ["u15", "u30", "a30_64", "o65"]:
                for emp in EMPLOYMENT_CATS:
                    pool = df[(df["_age_house"] == ag) & (df["employment_status"] == emp)].index.tolist()
                    rng.shuffle(pool)
                    pools[(ag, emp)] = pool

            def pop_candidate(age_groups, emp_statuses):
                for ag in age_groups:
                    for es in emp_statuses:
                        pool = pools.get((ag, es))
                        if pool:
                            return pool.pop()
                return None

            def pop_any():
                for pool in pools.values():
                    if pool:
                        return pool.pop()
                return None

            assignments = {}
            for _, h in households_df.iterrows():
                hid = h["household_id"]
                size = int(h["size"])
                age_comp = h["age_comp"]
                emp_comp = h["employment_comp"]

                need_u30 = age_comp in [
                    "Csak 30 évesnél fiatalabb személy van a háztartásban",
                    "30 évesnél fiatalabb és 30–64 éves személyek vannak a háztartásban",
                    "30 évesnél fiatalabb és 65 éves és idősebb személyek vannak a háztartásban",
                    "30 évesnél fiatalabb, 30–64 éves és 65 éves és idősebb személyek vannak a háztartásban",
                ]
                need_a30_64 = age_comp in [
                    "Csak 30–64 éves személy van a háztartásban",
                    "30 évesnél fiatalabb és 30–64 éves személyek vannak a háztartásban",
                    "30–64 éves és 65 éves és idősebb személyek vannak a háztartásban",
                    "30 évesnél fiatalabb, 30–64 éves és 65 éves és idősebb személyek vannak a háztartásban",
                ]
                need_o65 = age_comp in [
                    "Csak 65 éves és idősebb személy van a háztartásban",
                    "30 évesnél fiatalabb és 65 éves és idősebb személyek vannak a háztartásban",
                    "30–64 éves és 65 éves és idősebb személyek vannak a háztartásban",
                    "30 évesnél fiatalabb, 30–64 éves és 65 éves és idősebb személyek vannak a háztartásban",
                ]

                if age_comp == "Csak 30 évesnél fiatalabb személy van a háztartásban":
                    allowed_age = ["u15", "u30"]
                elif age_comp == "Csak 30–64 éves személy van a háztartásban":
                    allowed_age = ["a30_64"]
                elif age_comp == "Csak 65 éves és idősebb személy van a háztartásban":
                    allowed_age = ["o65"]
                elif age_comp == "30 évesnél fiatalabb és 30–64 éves személyek vannak a háztartásban":
                    allowed_age = ["u15", "u30", "a30_64"]
                elif age_comp == "30 évesnél fiatalabb és 65 éves és idősebb személyek vannak a háztartásban":
                    allowed_age = ["u15", "u30", "o65"]
                elif age_comp == "30–64 éves és 65 éves és idősebb személyek vannak a háztartásban":
                    allowed_age = ["a30_64", "o65"]
                else:
                    allowed_age = ["u15", "u30", "a30_64", "o65"]

                if emp_comp == "Egy foglalkoztatott van a háztartásban":
                    required_emp = ["Foglalkoztatott"]
                    allowed_emp = EMPLOYMENT_CATS
                elif emp_comp == "Két foglalkoztatott van a háztartásban":
                    required_emp = ["Foglalkoztatott", "Foglalkoztatott"]
                    allowed_emp = EMPLOYMENT_CATS
                elif emp_comp == "Három vagy több foglalkoztatott van a háztartásban":
                    required_emp = ["Foglalkoztatott", "Foglalkoztatott", "Foglalkoztatott"]
                    allowed_emp = EMPLOYMENT_CATS
                elif emp_comp.startswith("Nincs foglalkoztatott, van munkanélküli"):
                    required_emp = ["Munkanélküli"]
                    allowed_emp = ["Munkanélküli", "Ellátásban részesülő inaktív", "Eltartott"]
                elif emp_comp.startswith("Nincs foglalkoztatott, nincs munkanélküli"):
                    required_emp = ["Ellátásban részesülő inaktív"]
                    allowed_emp = ["Ellátásban részesülő inaktív", "Eltartott"]
                else:
                    required_emp = ["Eltartott"]
                    allowed_emp = ["Eltartott"]

                members = []
                if len(required_emp) > size:
                    required_emp = required_emp[:size]
                for emp in required_emp:
                    if len(members) >= size:
                        break
                    pid_sel = pop_candidate(allowed_age, [emp])
                    if pid_sel is None:
                        pid_sel = pop_candidate(allowed_age, allowed_emp)
                    if pid_sel is None:
                        pid_sel = pop_any()
                    if pid_sel is not None:
                        members.append(pid_sel)

                def has_under30(members_list):
                    return any(df.loc[m, "_age_house"] in ["u15", "u30"] for m in members_list)

                if need_u30 and not has_under30(members) and len(members) < size:
                    pid_sel = pop_candidate(["u15", "u30"], allowed_emp)
                    if pid_sel is None:
                        pid_sel = pop_any()
                    if pid_sel is not None:
                        members.append(pid_sel)
                if need_a30_64 and not any(df.loc[m, "_age_house"] == "a30_64" for m in members) and len(members) < size:
                    pid_sel = pop_candidate(["a30_64"], allowed_emp)
                    if pid_sel is None:
                        pid_sel = pop_any()
                    if pid_sel is not None:
                        members.append(pid_sel)
                if need_o65 and not any(df.loc[m, "_age_house"] == "o65" for m in members) and len(members) < size:
                    pid_sel = pop_candidate(["o65"], allowed_emp)
                    if pid_sel is None:
                        pid_sel = pop_any()
                    if pid_sel is not None:
                        members.append(pid_sel)

                while len(members) < size:
                    pid_sel = pop_candidate(allowed_age, allowed_emp)
                    if pid_sel is None:
                        pid_sel = pop_any()
                    if pid_sel is None:
                        break
                    members.append(pid_sel)

                assignments[hid] = members

            remaining = []
            for pool in pools.values():
                remaining.extend(pool)
                pool.clear()
            if remaining:
                rng.shuffle(remaining)
                house_ids = list(assignments.keys())
                for i, pid in enumerate(remaining):
                    hid = house_ids[i % len(house_ids)]
                    assignments[hid].append(pid)

            for hid, members in assignments.items():
                df.loc[members, "household_id"] = hid

            # Post-assignment swaps to improve household counts of unemployment/inactive/dependent
            status_target_rows = {
                "Munkanélküli": [
                    "Nincs munkanélküli személy a háztartásban",
                    "1 munkanélküli személy van a háztartásban",
                    "2 munkanélküli személy van a háztartásban",
                    "3 vagy több munkanélküli személy van a háztartásban",
                ],
                "Ellátásban részesülő inaktív": [
                    "Nincs ellátásban részesülő inaktív személy a háztartásban",
                    "1 ellátásban részesülő inaktív személy van a háztartásban",
                    "2 ellátásban részesülő inaktív személy van a háztartásban",
                    "3 vagy több ellátásban részesülő inaktív személy van a háztartásban",
                ],
                "Eltartott": [
                    "Nincs eltartott személy a háztartásban",
                    "1 eltartott személy van a háztartásban",
                    "2 eltartott személy van a háztartásban",
                    "3 vagy több eltartott személy van a háztartásban",
                ],
            }
            for status, rows in status_target_rows.items():
                target_vec = np.array([house_counts.get(r, 0) for r in rows], dtype=float)
                if target_vec.sum() <= 0:
                    continue
                adjust_household_status_by_swaps(
                    df,
                    households_df,
                    status,
                    target_vec,
                    rng,
                    cfg.household_swap_max_iter,
                    cfg.household_swap_patience,
                )

            size_map = df.groupby("household_id").size().to_dict()
            households_df["size"] = households_df["household_id"].map(size_map).fillna(0).astype(int)

            # Compute actual age and employment composition from assigned members
            age_bins = pd.cut(df["age"], bins=[-1, 14, 64, 200], labels=["u15", "a30_64", "o65"])
            age_counts = pd.crosstab(df["household_id"], age_bins).reset_index()
            age_counts = age_counts.rename(columns={
                "u15": "u15_count",
                "a30_64": "a30_64_count",
                "o65": "o65_count",
            })
            emp_counts = pd.crosstab(df["household_id"], df["employment_status"]).reset_index()

            households_df = households_df.merge(age_counts, on="household_id", how="left").fillna(0)
            households_df = households_df.merge(emp_counts, on="household_id", how="left").fillna(0)

            # Build flag strings
            households_df["age_composition_flags"] = households_df.apply(
                lambda r: f"u15={int(r.get('u15_count',0))};a30_64={int(r.get('a30_64_count',0))};o65={int(r.get('o65_count',0))}",
                axis=1,
            )
            households_df["employment_composition"] = households_df.apply(
                lambda r: f"E={int(r.get('Foglalkoztatott',0))};U={int(r.get('Munkanélküli',0))};I={int(r.get('Ellátásban részesülő inaktív',0))};D={int(r.get('Eltartott',0))}",
                axis=1,
            )

            # Housing units
            housing_counts = housing_df[settlement].to_dict()
            def safe_int(val: Any) -> int:
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    return 0
                return int(val)
            room_counts = {
                1: safe_int(housing_counts.get("1 szoba", 0)),
                2: safe_int(housing_counts.get("2 szoba", 0)),
                3: safe_int(housing_counts.get("3 szoba", 0)),
                4: safe_int(housing_counts.get("4 vagy több szoba", 0)),
            }
            # Generate dwellings
            dwellings = []
            for rooms, cnt in room_counts.items():
                for _ in range(cnt):
                    dwellings.append({"dwelling_id": dwelling_id, "rooms": rooms})
                    dwelling_id += 1
            if len(dwellings) < len(households_df):
                # Add more with room=2 fallback
                for _ in range(len(households_df) - len(dwellings)):
                    dwellings.append({"dwelling_id": dwelling_id, "rooms": 2})
                    dwelling_id += 1
            dwellings_df = pd.DataFrame(dwellings)
            dwellings_df["settlement"] = settlement
            # Assign dwellings to households
            dwellings_df = dwellings_df.sample(frac=1.0, random_state=cfg.seed).reset_index(drop=True)
            households_df["dwelling_id"] = dwellings_df["dwelling_id"].values[: len(households_df)]

        # Validation (aligned to flat category bases)
        gen_sex = df["sex"].value_counts().to_dict()
        gen_edu_7p = df[df["age"] >= 7]["education"].value_counts().to_dict()
        gen_emp_15p = df[df["age"] >= 15]["employment_status"].value_counts().to_dict()

        edu_full = sum(pop_counts.get(c, 0) for c in EDUCATION_CATS) + pop_counts.get("7 évesnél fiatalabb személy", 0)
        emp_full = sum(pop_counts.get(c, 0) for c in EMPLOYMENT_CATS) + pop_counts.get("15 évesnél fiatalabb személy", 0)

        validation["settlements"][settlement] = {
            "total": len(df),
            "sex_diff": {s: int(gen_sex.get(s, 0) - sex_targets[i]) for i, s in enumerate(SEXES)},
            "edu_diff_7plus": {c: int(gen_edu_7p.get(c, 0) - pop_counts.get(c, 0)) for c in EDUCATION_CATS},
            "emp_diff_15plus": {c: int(gen_emp_15p.get(c, 0) - pop_counts.get(c, 0)) for c in EMPLOYMENT_CATS},
            "edu_full_minus_total": float(edu_full - total_pop),
            "emp_full_minus_total": float(emp_full - total_pop),
        }

        # Output
        # Add age group columns
        df["age_group_5y"] = df["age"].apply(lambda a: f"{(a // 5) * 5}–{(a // 5) * 5 + 4}" if a < cfg.max_age else f"{cfg.max_age}+")
        df["age_group_10y"] = df["age"].apply(lambda a: f"{(a // 10) * 10}–{(a // 10) * 10 + 9}" if a < cfg.max_age else f"{cfg.max_age}+")
        df["children_count"] = df["children_count"].astype("string").fillna("")

        out_cols = [
            "person_id",
            "household_id",
            "settlement",
            "ksh_code",
            "varmegye",
            "telepules_tipus",
            "sex",
            "age",
            "age_group_5y",
            "age_group_10y",
            "education",
            "employment_status",
            "occupation_group",
            "sector",
            "commute_region",
            "commute_type",
            "school_level",
            "marital_status",
            "children_count",
            "disability_status",
            "chronic_status",
            "limitation_status",
        ]

        table = pa.Table.from_pandas(df[out_cols])
        if person_writer is None:
            person_schema = table.schema
            person_writer = pq.ParquetWriter(out_dir / "synthetic_population.parquet", person_schema)
        person_writer.write_table(table)

        if households_df is not None:
            house_out = households_df[
                ["household_id", "settlement", "size", "family_structure", "age_composition_flags", "employment_composition", "dwelling_id"]
            ].copy()
            ht = pa.Table.from_pandas(house_out)
            if household_writer is None:
                household_schema = ht.schema
                household_writer = pq.ParquetWriter(out_dir / "synthetic_households.parquet", household_schema)
            household_writer.write_table(ht)

        if dwellings_df is not None:
            dt = pa.Table.from_pandas(dwellings_df)
            if dwelling_writer is None:
                dwelling_schema = dt.schema
                dwelling_writer = pq.ParquetWriter(out_dir / "synthetic_dwellings.parquet", dwelling_schema)
            dwelling_writer.write_table(dt)

        print(f"Processed {settlement} ({idx + 1}/{len(settlements)})")

    if person_writer is not None:
        person_writer.close()
    if household_writer is not None:
        household_writer.close()
    if dwelling_writer is not None:
        dwelling_writer.close()

    # Hier activity validation (distributional fit by county+type)
    hier_activity: Dict[str, Dict[str, float]] = {}
    l1_vals: List[float] = []
    rmse_vals: List[float] = []
    invalid_vals: List[float] = []
    for key, gen_arr in activity_gen_counts.items():
        ref_arr = activity_ref_counts.get(key)
        if ref_arr is None:
            continue
        valid_mask = ref_arr > 0
        gen_total = float(gen_arr.sum())
        gen_valid = float(gen_arr[valid_mask].sum())
        ref_total = float(ref_arr[valid_mask].sum())
        invalid_mass = 0.0 if gen_total == 0 else max(0.0, (gen_total - gen_valid) / gen_total)
        l1 = None
        rmse = None
        if ref_total > 0 and gen_valid > 0:
            ref_share = ref_arr[valid_mask] / ref_total
            gen_share = gen_arr[valid_mask] / gen_valid
            diff = gen_share - ref_share
            l1 = float(np.abs(diff).sum())
            rmse = float(np.sqrt((diff ** 2).mean()))
            l1_vals.append(l1)
            rmse_vals.append(rmse)
        invalid_vals.append(float(invalid_mass))
        hier_activity[f"{key[0]}|{key[1]}"] = {
            "gen_total": gen_total,
            "ref_total": ref_total,
            "gen_valid": gen_valid,
            "invalid_mass": float(invalid_mass),
            "l1_share": l1,
            "rmse_share": rmse,
        }

    if hier_activity:
        validation["hier_activity"] = hier_activity
        validation["summary"]["hier_activity_fit"] = {
            "mean_l1_share": float(np.mean(l1_vals)) if l1_vals else None,
            "max_l1_share": float(np.max(l1_vals)) if l1_vals else None,
            "mean_rmse_share": float(np.mean(rmse_vals)) if rmse_vals else None,
            "max_rmse_share": float(np.max(rmse_vals)) if rmse_vals else None,
            "max_invalid_mass": float(np.max(invalid_vals)) if invalid_vals else None,
        }

    # Summary validation
    validation["summary"]["settlement_count"] = len(validation["settlements"])
    validation["summary"]["total_population"] = person_id - 1

    with open(out_dir / "validation_report.json", "w", encoding="utf-8") as f:
        json.dump(validation, f, ensure_ascii=False, indent=2)

    if args.diagnostics:
        generate_diagnostics(
            out_dir,
            flat_pop,
            flat_house,
            flat_housing,
            activity_long,
            activity_edu_long,
            marital_long,
            household_hier_long,
            occupation_long,
            sector_long,
            commute_long,
            school_long,
            health_long,
        )

    print("Done.")


if __name__ == "__main__":
    main()
