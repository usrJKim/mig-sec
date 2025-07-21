#!/usr/bin/env python3
import argparse
import random
import csv
import os

def main():
    parser = argparse.ArgumentParser(
        description="Generate a random sequence of integers [0–14] and write to CSV, with 10 numbers per line."
    )
    parser.add_argument(
        "-n", "--count",
        type=int,
        default=100,
        help="Total number of random symbols to generate (default: 100)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="sequence.csv",
        help="Output CSV filename (placed under ./data/)"
    )
    args = parser.parse_args()

    # ensure data directory exists
    out_dir = "./input_files"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, args.output)

    # 1) Generate the random sequence
    seq = [random.randint(0, 14) for _ in range(args.count)]

    # 2) Write to CSV, 10 symbols per row
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        for i in range(0, len(seq), 10):
            writer.writerow(seq[i:i+10])

    print(f"Wrote {len(seq)} symbols to '{out_path}', 10 per line.")

if __name__ == "__main__":
    main()
