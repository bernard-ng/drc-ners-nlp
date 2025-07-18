import ollama
from pydantic import BaseModel

from misc import load_prompt


class NameAnalysis(BaseModel):
    identified_name: str | None
    identified_surname: str | None
    identified_category: str | None


name = input("Enter name: ")

client = ollama.Client()
response = client.chat(
    model="mistral:7b",
    messages=[
        {"role": "system", "content": load_prompt()},
        {"role": "user", "content": name}
    ],
    format=NameAnalysis.model_json_schema()
)
analysis = NameAnalysis.model_validate_json(response.message.content)
result = analysis.model_dump()

print(result)
