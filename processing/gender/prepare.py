import os

import pandas as pd

from misc import DATA_DIR


def clean(filepath):
    encodings = ['utf-8', 'utf-16', 'latin1']
    for enc in encodings:
        try:
            print(f">> Trying to read {filepath} with encoding: {enc}")
            df = pd.read_csv(filepath, encoding=enc, on_bad_lines='skip')

            print(">> Remove null bytes and non-breaking spaces from all string columns")
            for col in df.select_dtypes(include=['object']).columns:
                df[col] = df[col].astype(str).str.replace('\x00', ' ', regex=False)
                df[col] = df[col].str.replace('\u00a0', ' ', regex=False)
                df[col] = df[col].str.replace(' +', ' ', regex=True)

            print(f">> Successfully read with encoding: {enc}")
            df = df.dropna(subset=['name', 'sex', 'region'])
            df.to_csv(filepath, index=False, encoding='utf-8')
            return df
        except Exception:
            continue
    raise UnicodeDecodeError(f"Unable to decode {filepath} with common encodings.")


def main():
    df = clean(os.path.join(DATA_DIR, 'names.csv'))

    df['name'] = df['name'].str.strip().str.lower()
    df['words'] = df['name'].str.split().apply(len)
    df['length'] = df['name'].str.replace(' ', '', regex=False).str.len()
    df['probable_native'] = df['name'].str.split().apply(lambda x: ' '.join(x[:-1]) if len(x) > 1 else '')
    df['probable_surname'] = df['name'].str.split().apply(lambda x: x[-1] if len(x) > 0 else '')

    print(f">> Arranging columns")
    cols = [c for c in df.columns if c != 'sex'] + ['sex']
    df = df[cols]

    print(f">> Saving featured dataset")
    df.to_csv(os.path.join(DATA_DIR, 'names_featured.csv'), index=False)

    print(f">> Splitting dataset by sex")
    df[df['sex'].str.lower() == 'm'].to_csv(os.path.join(DATA_DIR, 'names_males.csv'), index=False)
    df[df['sex'].str.lower() == 'f'].to_csv(os.path.join(DATA_DIR, 'names_females.csv'), index=False)


if __name__ == '__main__':
    main()
