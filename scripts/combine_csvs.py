
import argparse
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Recursively concatenate CSV files into a single output CSV."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory to scan recursively for CSV files",
    )
    parser.add_argument(
        "output_csv",
        type=Path,
        help="Path to the output CSV file",
    )

    args = parser.parse_args()

    if not args.input_dir.is_dir():
        parser.error(f"Input directory does not exist: {args.input_dir}")

    output_path = args.output_csv.resolve()
    csv_files = sorted(
        file for file in args.input_dir.rglob("*.csv")
        if file.resolve() != output_path
    )

    if not csv_files:
        parser.error(f"No CSV files found in {args.input_dir}")

    print(f"Found {len(csv_files)} CSV files.")

    dataframes = []
    for file in csv_files:
        print(f"Reading: {file}")
        dataframes.append(pd.read_csv(file))

    combined_df = pd.concat(dataframes, ignore_index=True)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(args.output_csv, index=False, float_format="%.2f")

    print(f"Combined {len(combined_df)} rows into: {args.output_csv}")


if __name__ == "__main__":
    main()