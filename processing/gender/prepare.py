import os
import pandas as pd
from misc import DATA_DIR


def clean(filepath):
    encodings = ['utf-8', 'utf-16', 'latin1']
    for enc in encodings:
        try:
            print(f">> Trying to read {filepath} with encoding: {enc}")
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
                    )

                cleaned_chunks.append(chunk)

            df = pd.concat(cleaned_chunks, ignore_index=True)
            df.to_csv(filepath, index=False, encoding='utf-8')
            print(f">> Successfully read with encoding: {enc}")
            return df
        except Exception:
            continue
    raise UnicodeDecodeError(f"Unable to decode {filepath} with common encodings.")


def process(df: pd.DataFrame):
    print(">> Preprocessing names")
    df['name'] = df['name'].str.strip().str.lower()

    df['words'] = df['name'].str.count(' ') + 1
    df['length'] = df['name'].str.replace(' ', '', regex=False).str.len()

    name_split = df['name'].str.split()
    df['probable_native'] = name_split.apply(lambda x: ' '.join(x[:-1]) if len(x) > 1 else '')
    df['probable_surname'] = name_split.apply(lambda x: x[-1] if x else '')
    df['llm_annotated'] = 0

    return df


def split_and_save(df: pd.DataFrame):
    print(">> Saving evaluation and featured datasets")
    eval_idx = df.sample(frac=0.2, random_state=42).index

    df_evaluation = df.loc[eval_idx]
    df_featured = df.drop(index=eval_idx)

    df_evaluation.to_csv(os.path.join(DATA_DIR, 'names_evaluation.csv'), index=False)
    df_featured.to_csv(os.path.join(DATA_DIR, 'names_featured.csv'), index=False)

    print(">> Saving by sex")
    df[df['sex'].str.lower() == 'm'].to_csv(os.path.join(DATA_DIR, 'names_males.csv'), index=False)
    df[df['sex'].str.lower() == 'f'].to_csv(os.path.join(DATA_DIR, 'names_females.csv'), index=False)


def main():
    filepath = os.path.join(DATA_DIR, 'names.csv')
    df = clean(filepath)
    df = process(df)
    split_and_save(df)


if __name__ == '__main__':
    main()
