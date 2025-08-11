#!/usr/bin/env python3
"""
NER Dataset Feature Engineering Script
Processes the names_featured.csv dataset to create position-independent variations
"""

import argparse
import os

from processing.ner.ner_engineering import NEREngineering


def main():
    parser = argparse.ArgumentParser(description='Engineer NER dataset for position-independent learning')
    parser.add_argument('--input', default='data/dataset/names_featured.csv', help='Input CSV file path')
    parser.add_argument('--output', default='data/dataset/names_featured_engineered.csv', help='Output CSV file path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    args = parser.parse_args()

    print("=== NER Dataset Feature Engineering ===")
    print(f"Input file: {args.input}")
    print(f"Output file: {args.output}")
    print(f"Random seed: {args.seed}")

    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} not found!")
        return

    # Initialize engineering class
    engineering = NEREngineering()

    try:
        # Load data with progress indication
        print("\n1. Loading NER-tagged data...")
        data = engineering.load_ner_data(args.input)
        print(f"   Dataset size: {len(data):,} rows")

        # Show sample of original data
        print("\n2. Sample original data:")
        for i, row in data.head(3).iterrows():
            print(f"   {row['name']} -> Native: '{row['probable_native']}', Surname: '{row['probable_surname']}'")

        # Apply transformations
        print("\n3. Applying feature engineering transformations...")
        engineered_data = engineering.engineer_dataset(data, random_seed=args.seed)

        # Save results
        print(f"\n4. Saving engineered dataset to {args.output}...")
        engineering.save_engineered_dataset(engineered_data, args.output)

        # Show statistics
        print(f"\n=== RESULTS SUMMARY ===")
        print(f"Original dataset: {len(data):,} rows")
        print(f"Engineered dataset: {len(engineered_data):,} rows")
        print(f"Transformation distribution:")
        counts = engineered_data['transformation_type'].value_counts().sort_index()
        for trans_type, count in counts.items():
            percentage = (count / len(engineered_data)) * 100
            print(f"  {trans_type}: {count:,} rows ({percentage:.1f}%)")

        print(f"\nDataset successfully engineered and saved!")

    except Exception as e:
        print(f"Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
