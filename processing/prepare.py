import os
import argparse
import pandas as pd
from misc import DATA_DIR, REGION_MAPPING, logging


def clean(filepath) -> pd.DataFrame:
    """
    Clean the CSV file by removing null bytes, non-breaking spaces, and extra spaces.
    Also, it attempts to read the file with different encodings to handle potential encoding issues.
    """

    encodings = ['utf-8', 'utf-16', 'latin1']
    for enc in encodings:
        try:
            logging.info(f"Trying to read {filepath} with encoding: {enc}")
            # Use chunked reading to handle large files
            chunks = pd.read_csv(filepath, encoding=enc, chunksize=100_000, on_bad_lines='skip')
            cleaned_chunks = []

            for chunk in chunks:
                # Drop rows with essential missing values early
                chunk = chunk.dropna(subset=['name', 'sex', 'region'])

                # Clean string columns in-place
                for col in chunk.select_dtypes(include='object').columns:
                    chunk[col] = (
                        chunk[col]
                        .astype(str)
                        .str.replace('\x00', ' ', regex=False)
                        .str.replace('\u00a0', ' ', regex=False)
                        .str.replace(' +', ' ', regex=True)
                        .str.strip()
                        .str.lower()
                    )

                cleaned_chunks.append(chunk)

            df = pd.concat(cleaned_chunks, ignore_index=True)
            df.to_csv(filepath, index=False, encoding='utf-8')
            logging.info(f"Successfully read with encoding: {enc}")
            return df
        except Exception:
            continue
    raise UnicodeDecodeError(f"Unable to decode {filepath} with common encodings.")


def process(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the DataFrame to extract features and clean data.
    This includes counting words, calculating name length, and extracting probable native names and surnames.
    Also maps regions to provinces based on REGION_MAPPING.
    """

    logging.info("Preprocessing names")
    df['words'] = df['name'].str.count(' ') + 1
    df['length'] = df['name'].str.replace(' ', '', regex=False).str.len()
    df['year'] = df['year'].astype(int)

    # Calculate probable_native and probable_surname
    name_split = df['name'].str.split()
    df['probable_native'] = name_split.apply(lambda x: ' '.join(x[:-1]) if len(x) > 1 else '')
    df['probable_surname'] = name_split.apply(lambda x: x[-1] if x else '')
    df['identified_category'] = df['words'].apply(lambda x: 'compose' if x > 3 else 'simple')
    df['identified_name'] = None
    df['identified_surname'] = None
    df['annotated'] = 0

    # We can assume that if a name has exactly 3 words, the first two are the native name and the last is the surname
    # This is a common pattern in Congolese names
    three_word_mask = df['words'] == 3
    df.loc[three_word_mask, 'identified_name'] = df.loc[three_word_mask, 'probable_native']
    df.loc[three_word_mask, 'identified_surname'] = df.loc[three_word_mask, 'probable_surname']
    df.loc[three_word_mask, 'annotated'] = 1

    logging.info("Mapping regions to provinces")
    df['province'] = df['region'].map(lambda r: REGION_MAPPING.get(r, ('AUTRES', 'AUTRES'))[1])
    df['province'] = df['province'].str.lower()

    return df


def save_artifacts(df: pd.DataFrame, split_eval: bool = True, split_by_sex: bool = True) -> None:
    """
    Splits the input DataFrame into evaluation and featured datasets, saves them as CSV files, 
    and additionally saves separate CSV files for male and female entries if requested.
    """

    if split_eval:
        logging.info("Saving evaluation and featured datasets")
        eval_idx = df.sample(frac=0.2, random_state=42).index
        df_evaluation = df.loc[eval_idx]
        df_featured = df.drop(index=eval_idx)
        df_evaluation.to_csv(os.path.join(DATA_DIR, 'names_evaluation.csv'), index=False)
        df_featured.to_csv(os.path.join(DATA_DIR, 'names_featured.csv'), index=False)
    else:
        df.to_csv(os.path.join(DATA_DIR, 'names_featured.csv'), index=False)

    if split_by_sex:
        logging.info("Saving by sex")
        df[df['sex'] == 'm'].to_csv(os.path.join(DATA_DIR, 'names_males.csv'), index=False)
        df[df['sex'] == 'f'].to_csv(os.path.join(DATA_DIR, 'names_females.csv'), index=False)


def main(split_eval: bool = True, split_by_sex: bool = True):
    df = process(clean(os.path.join(DATA_DIR, 'names.csv')))
    save_artifacts(df, split_eval=split_eval, split_by_sex=split_by_sex)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prepare name datasets with optional splits.")
    
    parser.add_argument('--split_eval', action='store_true', default=True, help="Split into evaluation and featured datasets (default: True)")
    parser.add_argument('--no-split_eval', action='store_false', dest='split_eval', help="Do not split into evaluation and featured datasets")
    parser.add_argument('--split_by_sex', action='store_true', default=True, help="Split by sex into male/female datasets (default: True)")
    parser.add_argument('--no-split_by_sex', action='store_false', dest='split_by_sex', help="Do not split by sex into male/female datasets")
    
    args = parser.parse_args()
    main(split_eval=args.split_eval, split_by_sex=args.split_by_sex)
