import os
import argparse

import ollama
import pandas as pd
from pydantic import BaseModel, ValidationError
from tqdm import tqdm
from typing import Optional

from misc import load_prompt, load_csv_dataset, DATA_DIR, logging


class NameAnalysis(BaseModel):
    identified_name: Optional[str]
    identified_surname: Optional[str]


def analyze_name(client: ollama.Client, model: str, prompt: str, name: str) -> dict:
    """
    Analyze a name using the specified model and prompt.
    Returns a dictionary with identified name, surname, and category.
    """
    try:
        response = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": name}
            ],
            format=NameAnalysis.model_json_schema()
        )
        analysis = NameAnalysis.model_validate_json(response.message.content)
        return analysis.model_dump()
    except ValidationError as ve:
        logging.warning(f"Validation error: {ve}")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
    return {
        "identified_name": None,
        "identified_surname": None
    }


def build_updates(client: ollama.Client, prompt: str, llm_model: str, rows: pd.DataFrame) -> pd.DataFrame:
    """
    Build updates for the DataFrame by analyzing names.
    Iterates through the DataFrame rows, analyzes each name, and returns a DataFrame with updates.
    """
    logging.getLogger("httpx").setLevel(logging.WARNING)
    updates = []

    for idx, row in rows.iterrows():
        entry = analyze_name(client, llm_model, prompt, row['name'])
        entry["annotated"] = 1
        updates.append((idx, entry))
        logging.info(f"Analyzed name: {row['name']} - {entry}")

    return pd.DataFrame.from_dict(dict(updates), orient='index')


def main(llm_model: str = "llama3.2:3b"):
    df = pd.DataFrame(load_csv_dataset('names_featured.csv'))
    prompt = load_prompt()

    entries = df[df['annotated'].astype("Int8") == 0]
    if entries.empty:
        logging.info("No names to analyze.")
        return

    logging.info(f"Found {len(entries)} names to analyze.")
    client = ollama.Client()

    df.update(build_updates(client, prompt, llm_model, entries))
    df.to_csv(os.path.join(DATA_DIR, 'names_featured.csv'), index=False)
    logging.info("Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analyze names using an LLM model.")
    parser.add_argument('--llm_model', type=str, default="llama3.2:3b", help="Ollama model name to use (default: llama3.2:3b)")
    args = parser.parse_args()
    
    try:
        main(llm_model=args.llm_model)
    except Exception as e:
        logging.error(f"Fatal error: {e}", exc_info=True)
