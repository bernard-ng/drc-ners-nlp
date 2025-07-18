import os

import ollama
import pandas as pd
from pydantic import BaseModel, ValidationError
from tqdm import tqdm

from misc import load_prompt, load_csv_dataset, DATA_DIR


class NameAnalysis(BaseModel):
    identified_name: str | None
    identified_surname: str | None
    identified_category: str | None


def main():
    dataset = pd.DataFrame(load_csv_dataset('names_featured.csv'))
    prompt = load_prompt()

    print(">> Filtering dataset for names that need analysis...")
    to_analyze = dataset[dataset['llm_annotated'] == 0].copy()
    if to_analyze.empty:
        print(">> No names to analyze.")
        return

    client = ollama.Client()
    updates = []

    print(">> Starting name analysis with LLM...")
    for row in tqdm(to_analyze.itertuples(index=True), total=len(to_analyze)):
        name = row.name
        try:
            response = client.chat(
                model="llama3.2:3b",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": name}
                ],
                format=NameAnalysis.model_json_schema()
            )
            analysis = NameAnalysis.model_validate_json(response.message.content)
            result = analysis.model_dump()
        except (ValidationError, Exception):
            result = {
                "identified_name": None,
                "identified_surname": None,
                "identified_category": None
            }

        updates.append({
            "index": row.Index,
            "identified_name": result["identified_name"],
            "identified_surname": result["identified_surname"],
            "identified_category": result["identified_category"],
            "llm_annotated": 1
        })

    print(">> Updating dataset with results...")
    updates_df = pd.DataFrame(updates).set_index("index")
    dataset.update(updates_df)

    print(">> Saving updated dataset...")
    dataset.to_csv(os.path.join(DATA_DIR, 'names_featured.csv'), index=False)
    print(">> Done.")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f">> Fatal error: {e}")
