"""Warm the HuggingFace cache with the contrastive A/B datasets.

The `train` Docker target and any run that loads pairs with ``source="truthful_qa"``
or ``source="Anthropic/hh-rlhf"`` read these from the mounted HF cache volume. This
downloads them once so the first run does not block on the network, and so an
offline run finds them locally. It is deliberately *not* baked into the image: the
cache is a volume shared across the dev, train and validation containers, which
keeps the image lean and avoids tying an image build to HuggingFace availability.

Run:  uv run python scripts/warm_hf_datasets.py
"""

import sys


def main() -> int:
    try:
        from datasets import load_dataset
    except ImportError:
        print("datasets is not installed; install the 'train' extra: uv sync --extra train")
        return 1

    print("Warming truthful_qa (multiple_choice, validation) ...")
    truthful = load_dataset("truthful_qa", "multiple_choice", split="validation")
    print(f"  cached {len(truthful)} rows")

    print("Warming Anthropic/hh-rlhf (harmless-base, train) ...")
    # Stream and touch a bounded prefix: the converter streams too, so the files
    # land in the cache without materialising the whole split.
    harmless = load_dataset(
        "Anthropic/hh-rlhf", data_dir="harmless-base", split="train", streaming=True
    )
    touched = sum(1 for _, _ in zip(range(256), harmless, strict=False))
    print(f"  streamed {touched} rows into the cache")

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
