"""Shared training and prediction code for the Octagon Analytics app.

All performance features are lagged: a row only uses results and in-fight
statistics from bouts that occurred before that row's event date.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
STAT_COLUMNS = [
    "sig_str_landed", "sig_str_attempted", "td_landed", "td_attempted",
    "sub_att", "rev", "ctrl_seconds",
]
BASE_FEATURES = [
    "pre_bouts", "pre_win_pct", "last_3_win_pct", "age", "height_inches",
    "reach_inches", "weight_lbs", *[f"pre_{column}_avg" for column in STAT_COLUMNS],
]
FEATURES = [f"{column}_difference" for column in BASE_FEATURES]


def clean_name(value: object) -> str:
    """Create a reliable matching key without changing the display name."""
    if pd.isna(value):
        return ""
    value = unicodedata.normalize("NFKD", str(value))
    value = "".join(c for c in value if not unicodedata.combining(c))
    value = re.sub(r"[^A-Za-z\s]", "", value)
    return re.sub(r"\s+", " ", value).strip().upper()


def parse_number(value: object) -> float:
    if pd.isna(value):
        return np.nan
    value = str(value).replace("%", "").replace("lbs.", "").replace('"', "").strip()
    return pd.to_numeric(value.replace("---", ""), errors="coerce")


def parse_of(value: object, position: int) -> float:
    """Extract landed (0) or attempted (1) from strings like '12 of 30'."""
    match = re.match(r"^\s*(\d+)\s+of\s+(\d+)\s*$", str(value))
    return float(match.group(position + 1)) if match else np.nan


def parse_time(value: object) -> float:
    """Return control time in seconds from an M:SS value."""
    match = re.match(r"^\s*(\d+):(\d+)\s*$", str(value))
    return float(int(match.group(1)) * 60 + int(match.group(2))) if match else np.nan


def _load_bios() -> pd.DataFrame:
    bios = pd.read_csv(ROOT / "fighters_raw.csv")
    bios.columns = bios.columns.str.strip().str.lower().str.replace(".", "", regex=False)
    bios["fighter_key"] = bios["name"].map(clean_name)
    height = bios["height"].astype("string").str.extract(r"(\d+)'(\d+)")
    bios["height_inches"] = pd.to_numeric(height[0], errors="coerce") * 12 + pd.to_numeric(height[1], errors="coerce")
    bios["reach_inches"] = bios["reach"].map(parse_number)
    bios["weight_lbs"] = bios["weight"].map(parse_number)
    bios["dob"] = pd.to_datetime(bios["dob"], format="%b%d,%Y", errors="coerce")
    return bios[["fighter_key", "name", "height_inches", "reach_inches", "weight_lbs", "dob", "stance"]].drop_duplicates("fighter_key")


def _build_fighter_fights() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Make two fighter-perspective rows per completed fight with lagged data."""
    fights = pd.read_csv(ROOT / "merged_stats_n_scorecards.csv")
    fights.columns = fights.columns.str.strip().str.lower()
    fights["event_date"] = pd.to_datetime(fights["event_date"], dayfirst=True, errors="coerce")
    fights = fights.dropna(subset=["event_date", "red_fighter_name", "blue_fighter_name"])
    fights = fights.loc[
        fights["red_fighter_result"].isin(["W", "L"])
        & fights["blue_fighter_result"].isin(["W", "L"])
    ].copy()
    fights["fight_id"] = np.arange(len(fights))

    def side(prefix: str, other: str) -> pd.DataFrame:
        frame = pd.DataFrame({
            "fight_id": fights["fight_id"], "event_date": fights["event_date"],
            "fighter": fights[f"{prefix}_fighter_name"],
            "opponent": fights[f"{other}_fighter_name"],
            "won_fight": fights[f"{prefix}_fighter_result"].eq("W").astype(int),
        })
        frame["sig_str_landed"] = fights[f"{prefix}_fighter_sig_str"].map(lambda x: parse_of(x, 0))
        frame["sig_str_attempted"] = fights[f"{prefix}_fighter_sig_str"].map(lambda x: parse_of(x, 1))
        frame["td_landed"] = fights[f"{prefix}_fighter_td"].map(lambda x: parse_of(x, 0))
        frame["td_attempted"] = fights[f"{prefix}_fighter_td"].map(lambda x: parse_of(x, 1))
        frame["sub_att"] = fights[f"{prefix}_fighter_sub_att"].map(parse_number)
        frame["rev"] = fights[f"{prefix}_fighter_rev"].map(parse_number)
        frame["ctrl_seconds"] = fights[f"{prefix}_fighter_ctrl"].map(parse_time)
        return frame

    history = pd.concat([side("red", "blue"), side("blue", "red")], ignore_index=True)
    history["fighter_key"] = history["fighter"].map(clean_name)
    history["opponent_key"] = history["opponent"].map(clean_name)
    history = history.sort_values(["fighter_key", "event_date", "fight_id"]).reset_index(drop=True)
    grouped = history.groupby("fighter_key", group_keys=False)
    history["pre_bouts"] = grouped.cumcount()
    history["pre_wins"] = grouped["won_fight"].transform(lambda s: s.shift().fillna(0).cumsum())
    # Laplace smoothing keeps debut and 1-fight records from appearing certain.
    history["pre_win_pct"] = (history["pre_wins"] + 1) / (history["pre_bouts"] + 2)
    history["last_3_win_pct"] = grouped["won_fight"].transform(
        lambda s: s.shift().rolling(3, min_periods=1).mean()
    ).fillna(0.5)
    for column in STAT_COLUMNS:
        history[f"pre_{column}_avg"] = grouped[column].transform(
            lambda s: s.shift().expanding(min_periods=1).mean()
        )

    bios = _load_bios()
    history = history.merge(bios.drop(columns="name"), on="fighter_key", how="left")
    history["age"] = (history["event_date"] - history["dob"]).dt.days / 365.25
    return history, bios


def _matchup_table(history: pd.DataFrame) -> pd.DataFrame:
    left = history[["fight_id", "event_date", "fighter", "opponent", "fighter_key", "opponent_key", "won_fight", *BASE_FEATURES]].copy()
    right = history[["fight_id", "fighter_key", *BASE_FEATURES]].copy()
    right = right.rename(columns={"fighter_key": "opponent_key", **{c: f"opponent_{c}" for c in BASE_FEATURES}})
    table = left.merge(right, on=["fight_id", "opponent_key"], how="inner")
    for feature in BASE_FEATURES:
        table[f"{feature}_difference"] = table[feature] - table[f"opponent_{feature}"]
    # First professional appearances have no observed UFC history.  They remain usable
    # for live predictions, but are left out of evaluation/training to avoid all-default rows.
    return table.loc[table["pre_bouts"].gt(0) & table[f"opponent_pre_bouts"].gt(0)].copy()


def _make_models() -> dict[str, Pipeline]:
    prep_scaled = ColumnTransformer([("numeric", Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ]), FEATURES)])
    prep_tree = ColumnTransformer([("numeric", Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
    ]), FEATURES)])
    return {
        "Logistic regression": Pipeline([
            ("prep", prep_scaled),
            # No intercept preserves symmetry for an A-vs-B difference vector.
            ("model", LogisticRegression(C=0.4, max_iter=3000, fit_intercept=False, random_state=42)),
        ]),
        "Gradient boosting": Pipeline([
            ("prep", prep_tree),
            ("model", GradientBoostingClassifier(
                n_estimators=120, learning_rate=0.035, max_depth=2,
                min_samples_leaf=18, subsample=0.8, random_state=42,
            )),
        ]),
    }


def _chronological_split(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    dates = np.sort(table["event_date"].dropna().unique())
    cutoff = pd.Timestamp(dates[int(len(dates) * 0.80)])
    train = table.loc[table["event_date"] < cutoff]
    test = table.loc[table["event_date"] >= cutoff]
    return train, test, cutoff


@lru_cache(maxsize=1)
def load_artifacts() -> dict:
    history, bios = _build_fighter_fights()
    table = _matchup_table(history)
    train, test, cutoff = _chronological_split(table)
    models = _make_models()
    metrics = []
    for name, model in models.items():
        model.fit(train[FEATURES], train["won_fight"])
        probability = model.predict_proba(test[FEATURES])[:, 1]
        metrics.append({
            "Model": name,
            "Accuracy": accuracy_score(test["won_fight"], probability >= 0.5),
            "ROC AUC": roc_auc_score(test["won_fight"], probability),
            "Brier score": brier_score_loss(test["won_fight"], probability),
        })
    # Refit on all available historical rows before serving live matchups.
    for model in models.values():
        model.fit(table[FEATURES], table["won_fight"])
    names = bios[["fighter_key", "name"]].copy()
    names = pd.concat([names, history[["fighter_key", "fighter"]].rename(columns={"fighter": "name"})])
    names = names.dropna().drop_duplicates("fighter_key").sort_values("name")
    return {
        "history": history, "bios": bios, "models": models, "metrics": pd.DataFrame(metrics),
        "names": names, "cutoff": cutoff, "latest_date": history["event_date"].max(),
        "training_rows": len(train), "test_rows": len(test), "fights": table["fight_id"].nunique(),
    }


def _profile_at(history: pd.DataFrame, bios: pd.DataFrame, fighter_key: str, when: pd.Timestamp) -> dict:
    earlier = history.loc[(history["fighter_key"] == fighter_key) & (history["event_date"] < when)].sort_values(["event_date", "fight_id"])
    bio = bios.loc[bios["fighter_key"] == fighter_key].head(1)
    profile = {feature: np.nan for feature in BASE_FEATURES}
    profile["pre_bouts"] = len(earlier)
    if len(earlier):
        wins = earlier["won_fight"].sum()
        profile["pre_win_pct"] = (wins + 1) / (len(earlier) + 2)
        profile["last_3_win_pct"] = earlier["won_fight"].tail(3).mean()
        for column in STAT_COLUMNS:
            profile[f"pre_{column}_avg"] = earlier[column].mean()
    else:
        profile["pre_win_pct"] = 0.5
        profile["last_3_win_pct"] = 0.5
    if len(bio):
        row = bio.iloc[0]
        for col in ["height_inches", "reach_inches", "weight_lbs"]:
            profile[col] = row[col]
        profile["age"] = (when - row["dob"]).days / 365.25 if pd.notna(row["dob"]) else np.nan
    return profile


def predict_matchup(fighter_a: str, fighter_b: str, event_date: object) -> dict:
    """Generate symmetrical probabilities and feature values for a hypothetical fight."""
    artifacts = load_artifacts()
    names = artifacts["names"]
    key_a = names.loc[names["name"] == fighter_a, "fighter_key"].iloc[0]
    key_b = names.loc[names["name"] == fighter_b, "fighter_key"].iloc[0]
    when = pd.Timestamp(event_date)
    a = _profile_at(artifacts["history"], artifacts["bios"], key_a, when)
    b = _profile_at(artifacts["history"], artifacts["bios"], key_b, when)
    forward = pd.DataFrame([{f"{f}_difference": a[f] - b[f] for f in BASE_FEATURES}])
    reverse = -forward
    result = {}
    for name, model in artifacts["models"].items():
        raw_a = model.predict_proba(forward[FEATURES])[:, 1][0]
        raw_b = model.predict_proba(reverse[FEATURES])[:, 1][0]
        # Normalization makes displayed A/B probabilities complementary even for trees.
        result[name] = float(raw_a / (raw_a + raw_b))
    return {"probabilities": result, "a": a, "b": b, "date": when, "latest_date": artifacts["latest_date"]}
