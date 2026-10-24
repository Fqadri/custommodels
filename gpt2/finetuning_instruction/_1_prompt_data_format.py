import json
from pathlib import Path
from typing import TypedDict

DEFAULT_DATA_PATH = Path(__file__).with_name("instruction-data.json")


class InstructionEntry(TypedDict):
    instruction: str
    input: str
    output: str


def load_data(file_path: str | Path = DEFAULT_DATA_PATH) -> list[InstructionEntry]:
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


# Function to format the input into a Alpaca prompt style template
def format_input_to_alpaca_prompt(entry: InstructionEntry) -> str:
    instruction_text = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

    return instruction_text + input_text


def preview_formatted_data(file_path: str | Path = DEFAULT_DATA_PATH) -> None:
    data = load_data(file_path)
    print("Number of entries:", len(data))
    print("First entry:", data[0])
    for index in sorted({0, min(50, len(data) - 1), min(999, len(data) - 1)}):
        print(f"\nExample {index}:")
        print(format_input_to_alpaca_prompt(data[index]) + f"\n\n### Response:\n{data[index]['output']}")