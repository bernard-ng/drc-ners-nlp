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
                {"role": "user", "content": name},
            ],
            format=NameAnalysis.model_json_schema(),
        )
        analysis = NameAnalysis.model_validate_json(response.message.content)
        return analysis.model_dump()
    except ValidationError as ve:
        logging.warning(f"Validation error: {ve}")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
    return {"identified_name": None, "identified_surname": None}


def save_checkpoint(df: pd.DataFrame):
    df.to_csv(os.path.join(DATA_DIR, "names_featured.csv"), index=False)
    logging.critical(f"Checkpoint saved")


def build_updates(llm_model: str, df: pd.DataFrame, entries: pd.DataFrame) -> pd.DataFrame:
    BATCH_SIZE = 10

    client = ollama.Client()
    prompt = load_prompt()
    updates = []
    
    # Set logging level for HTTP client to reduce noise
    # This is useful to avoid excessive logging from the HTTP client used by Ollama
    logging.getLogger("httpx").setLevel(logging.WARNING)

    
    for idx, (row_idx, row) in enumerate(entries.iterrows(), 1):
        try:
            entry = analyze_name(client, llm_model, prompt, row["name"])
            entry["annotated"] = 1
            updates.append((row_idx, entry))
            logging.info(f"Analyzed: {row['name']} - {entry}")
        except Exception as e:
            logging.warning(f"Failed to analyze '{row['name']}': {e}")
            continue

        if idx % BATCH_SIZE == 0 or idx == len(entries):
            update_df = pd.DataFrame.from_dict(dict(updates), orient="index")
            update_df["annotated"] = pd.to_numeric(update_df["annotated"], errors="coerce").fillna(0).astype("Int8")

            df.update(update_df)
            save_checkpoint(df)
            updates.clear()  # avoid re-applying same updates

    return df


def main(llm_model: str = "llama3.2:3b"):
    df = pd.DataFrame(load_csv_dataset(os.path.join(DATA_DIR, "names_featured.csv")))

    # Safely cast 'annotated' column to Int8, handling float-like strings (e.g., '1.0')
    df["annotated"] = pd.to_numeric(df["annotated"], errors="coerce").fillna(0).astype(float).astype("Int8")

    entries = df[df["annotated"] == 0]
    if entries.empty:
        logging.info("No names to analyze.")
        return

    logging.info(f"Found {len(entries)} names to analyze.")
    df = build_updates(llm_model, df, entries)
    save_checkpoint(df)
    logging.info("Analysis complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze names using an LLM model.")
    parser.add_argument(
        "--llm_model",
        type=str,
        default="mistral:7b",
        help="Ollama model name to use (default: mistral:7b)",
    )
    args = parser.parse_args()

    try:
        main(llm_model=args.llm_model)
    except Exception as e:
        logging.error(f"Fatal error: {e}", exc_info=True)
