import json
import urllib.request
from pathlib import Path
from typing import TypedDict

DEFAULT_DATA_PATH = Path(__file__).with_name("instruction-data.json")
DATA_URL = "https://raw.githubusercontent.com/rasbt/LLms-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json"


class InstructionEntry(TypedDict):
    instruction: str
    input: str
    output: str


def load_instruction_data(file_path: str | Path) -> list[InstructionEntry]:
    with Path(file_path).open(encoding="utf-8") as file:
        raw_data = json.load(file)
    if not isinstance(raw_data, list) or not raw_data:
        raise ValueError("Instruction data must be a non-empty JSON list.")

    data: list[InstructionEntry] = []
    for index, entry in enumerate(raw_data):
        if not isinstance(entry, dict) or any(
            not isinstance(entry.get(key), str)
            for key in ("instruction", "input", "output")
        ):
            raise ValueError(
                f"Entry {index} must have string instruction, input, and output fields."
            )
        data.append(
            InstructionEntry(
                instruction=entry["instruction"],
                input=entry["input"],
                output=entry["output"],
            )
        )
    return data


def download_and_load_file(file_path: str | Path, url: str) -> list[InstructionEntry]:
    path = Path(file_path)
    if not path.exists():
        with urllib.request.urlopen(url, timeout=60) as response:
            text_data = response.read().decode("utf-8")
        json.loads(text_data)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text_data, encoding="utf-8")
    return load_instruction_data(path)


def download_and_load_file_default() -> list[InstructionEntry]:
    return download_and_load_file(DEFAULT_DATA_PATH, DATA_URL)


def load_data(file_path: str | Path = DEFAULT_DATA_PATH) -> list[InstructionEntry]:
    if Path(file_path) == DEFAULT_DATA_PATH:
        return download_and_load_file_default()
    return load_instruction_data(file_path)


# Function to format the input into a Alpaca prompt style template
def format_input(entry: InstructionEntry) -> str:
    instruction_text = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

    return instruction_text + input_text


def preview_data(file_path: str | Path = DEFAULT_DATA_PATH) -> None:
    data = load_data(file_path)
    print("Number of entries:", len(data))
    print("First entry:", data[0])
    for index in sorted({0, min(50, len(data) - 1), min(999, len(data) - 1)}):
        print(f"\nExample {index}:")
        print(format_input(data[index]) + f"\n\n### Response:\n{data[index]['output']}")


if __name__ == "__main__":
    preview_data()
