"""
Build (or rebuild) the local demonstration SQLite database from scratch.

Usage:
    python scripts/generate_data.py [--reset]

--reset drops and recreates the schema before loading; otherwise existing
rows are upserted (safe to re-run).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import DATABASE_PATH, RANDOM_SEED  # noqa: E402
from src import database  # noqa: E402
from src.data_generator import generate_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reset", action="store_true", help="Drop and recreate tables first")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed (default: config.settings.RANDOM_SEED)")
    args = parser.parse_args()

    t0 = time.time()
    print(f"Initialising database at {DATABASE_PATH} (reset={args.reset}) ...")
    database.init_db(reset=args.reset)

    print(f"Generating synthetic dataset (seed={args.seed}) ...")
    df = generate_dataset(seed=args.seed)
    print(f"  {len(df):,} rows across {df['station_id'].nunique()} stations "
          f"({df['timestamp'].min()} -> {df['timestamp'].max()})")

    print("Loading into SQLite ...")
    n_inserted = database.insert_readings(df)
    elapsed = time.time() - t0

    print(f"Done: {n_inserted:,} rows written in {elapsed:.1f}s.")
    print(f"Total rows now in database: {database.row_count():,}")


if __name__ == "__main__":
    main()
