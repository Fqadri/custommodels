import argparse
from pathlib import Path

from ._1_rawdataandpromptformatconcept import DEFAULT_DATA_PATH, preview_data


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def context_length(value: str) -> int:
    number = positive_int(value)
    if number > 1024:
        raise argparse.ArgumentTypeError("GPT-2 supports at most 1024 context tokens")
    return number


def worker_count(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return number


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="GPT-2 instruction fine-tuning workflows."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    preview = commands.add_parser(
        "preview", help="Inspect raw entries and formatted prompts."
    )
    prepare = commands.add_parser(
        "prepare", help="Inspect dataset splits and batch shapes."
    )
    train = commands.add_parser(
        "train", help="Fine-tune pretrained GPT-2 and save test responses."
    )
    for command in (preview, prepare, train):
        command.add_argument(
            "--data",
            type=Path,
            default=DEFAULT_DATA_PATH,
            help="Instruction JSON file.",
        )
    for command in (prepare, train):
        command.add_argument("--batch-size", type=positive_int, default=8)
        command.add_argument("--context-length", type=context_length, default=1024)
        command.add_argument("--num-workers", type=worker_count, default=0)
    train.add_argument(
        "--model", choices=("124M", "355M", "774M", "1558M"), default="355M"
    )
    train.add_argument("--epochs", type=positive_int, default=2)
    train.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    train.add_argument(
        "--models-dir",
        type=Path,
        help="Override the pretrained-weight cache directory.",
    )
    train.add_argument(
        "--output-dir", type=Path, help="Override the fine-tuning output directory."
    )
    train.add_argument("--max-new-tokens", type=positive_int, default=256)
    args = parser.parse_args(argv)

    if args.command == "preview":
        preview_data(args.data)
    elif args.command == "prepare":
        from ._2_preparedatasetandloadmodel import prepare_dataset

        prepare_dataset(
            args.data,
            batch_size=args.batch_size,
            context_length=args.context_length,
            num_workers=args.num_workers,
        )
    else:
        from ..save_load_model.gpt_download import DEFAULT_MODELS_DIR
        from ._3_instructionfinetuningandeval import DEFAULT_OUTPUT_DIR, run_finetuning

        run_finetuning(
            data_path=args.data,
            model_size=args.model,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            context_length=args.context_length,
            num_workers=args.num_workers,
            device_name=args.device,
            models_dir=args.models_dir
            if args.models_dir is not None
            else DEFAULT_MODELS_DIR,
            output_dir=args.output_dir
            if args.output_dir is not None
            else DEFAULT_OUTPUT_DIR,
            max_new_tokens=args.max_new_tokens,
        )


if __name__ == "__main__":
    main()
