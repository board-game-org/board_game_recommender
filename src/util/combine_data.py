"""
combine_data.py
Create a master dataset by merging BGG metadata with descriptions,
game attributes, and aggregated user ratings.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
DEFAULT_OUTPUT = DATA_DIR / "bgg_master_data.csv"


def _aggregate_user_ratings(path: Path) -> pd.DataFrame:
    """Summarize user ratings per game."""
    ratings_df = pd.read_csv(path)
    if ratings_df.empty:
        return pd.DataFrame(
            columns=[
                "BGGId",
                "UserRatingCount",
                "UserRatingMean",
                "UserRatingStd",
                "UserRatingMin",
                "UserRatingMax",
            ]
        )

    agg_df = (
        ratings_df.groupby("BGGId")["Rating"]
        .agg(
            UserRatingCount="count",
            UserRatingMean="mean",
            UserRatingStd="std",
            UserRatingMin="min",
            UserRatingMax="max",
        )
        .reset_index()
    )
    return agg_df


def build_master_dataframe() -> pd.DataFrame:
    """Create the combined dataset using bgg_data as the master table."""
    bgg_df = pd.read_csv(DATA_DIR / "bgg_data.csv").rename(columns={"id": "BGGId"})
    games_df = pd.read_csv(DATA_DIR / "games.csv")
    descriptions_df = pd.read_csv(
        DATA_DIR / "game_descriptions.csv",
        encoding="utf-8-sig",
    ).rename(columns={"bgg_id": "BGGId", "full_description": "FullDescription"})
    ratings_summary_df = _aggregate_user_ratings(DATA_DIR / "user_ratings.csv")

    master_df = bgg_df.merge(games_df, on="BGGId", how="left", suffixes=("", "_games"))
    master_df = master_df.merge(descriptions_df, on="BGGId", how="left")
    master_df = master_df.merge(ratings_summary_df, on="BGGId", how="left")

    return master_df


def main(output_path: Path | None = None) -> None:
    output_path = (output_path or DEFAULT_OUTPUT).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    master_df = build_master_dataframe()
    master_df.to_csv(output_path, index=False)

    try:
        display_path = output_path.relative_to(Path.cwd())
    except ValueError:
        display_path = output_path

    print(f"Wrote master dataset to {display_path} ({len(master_df)} rows, {len(master_df.columns)} columns)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine board game datasets into a single CSV.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=f"Destination path for the combined CSV (default: {DEFAULT_OUTPUT.name}).",
    )
    args = parser.parse_args()
    main(args.output)
