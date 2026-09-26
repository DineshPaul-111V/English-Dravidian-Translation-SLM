from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = PROJECT_ROOT / "data" / "processed" / "opus100_en_ta_final"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "opus100_en_ta_tokenized"

MODEL_NAME = "Helsinki-NLP/opus-mt-en-dra"

MAX_SOURCE_LENGTH = 128
MAX_TARGET_LENGTH = 128

BATCH_SIZE = 256


# ============================================================
# TOKENIZE ONE SPLIT
# ============================================================

def tokenize_split(split_name, tokenizer):

    input_path = INPUT_DIR / f"{split_name}.parquet"
    output_path = OUTPUT_DIR / f"{split_name}.parquet"

    print()
    print("=" * 70)
    print(f"Tokenizing: {split_name.upper()}")
    print("=" * 70)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input dataset not found:\n{input_path}"
        )

    # Remove previous output if it exists
    if output_path.exists():
        output_path.unlink()

    parquet_file = pq.ParquetFile(input_path)

    total_rows = parquet_file.metadata.num_rows

    print(f"Input rows : {total_rows:,}")
    print(f"Batch size : {BATCH_SIZE}")
    print()

    writer = None
    processed_rows = 0
    batch_number = 0

    try:

        for batch in parquet_file.iter_batches(
            batch_size=BATCH_SIZE
        ):

            batch_number += 1

            rows = batch.to_pylist()

            english_texts = []
            tamil_texts = []

            for row in rows:

                english = row["translation"]["en"]
                tamil = row["translation"]["ta"]

                # IMPORTANT:
                # Target-language token goes on SOURCE side.
                source_text = f">>tam<< {english}"

                english_texts.append(source_text)
                tamil_texts.append(tamil)

            # ------------------------------------------------
            # TOKENIZE SOURCE
            # ------------------------------------------------

            source_tokens = tokenizer(
                english_texts,
                max_length=MAX_SOURCE_LENGTH,
                truncation=True,
                padding=False,
            )

            # ------------------------------------------------
            # TOKENIZE TARGET
            # ------------------------------------------------

            target_tokens = tokenizer(
                text_target=tamil_texts,
                max_length=MAX_TARGET_LENGTH,
                truncation=True,
                padding=False,
            )

            # ------------------------------------------------
            # CREATE OUTPUT RECORDS
            # ------------------------------------------------

            tokenized_rows = []

            for input_ids, attention_mask, labels in zip(
                source_tokens["input_ids"],
                source_tokens["attention_mask"],
                target_tokens["input_ids"],
            ):

                tokenized_rows.append(
                    {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "labels": labels,
                    }
                )

            # ------------------------------------------------
            # WRITE BATCH TO PARQUET
            # ------------------------------------------------

            output_table = pa.Table.from_pylist(tokenized_rows)

            if writer is None:

                writer = pq.ParquetWriter(
                    output_path,
                    output_table.schema,
                    compression="zstd",
                )

            writer.write_table(output_table)

            processed_rows += len(rows)

            if (
                batch_number == 1
                or batch_number % 50 == 0
                or processed_rows == total_rows
            ):

                percentage = (
                    processed_rows / total_rows
                ) * 100

                print(
                    f"Processed: "
                    f"{processed_rows:,}/{total_rows:,} "
                    f"({percentage:.1f}%)"
                )

    finally:

        if writer is not None:
            writer.close()

    print()
    print(f"Saved: {output_path}")
    print(f"Rows : {processed_rows:,}")

    return processed_rows


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("TamilNLP - Full Dataset Tokenization")
    print("=" * 70)

    print()
    print(f"Model : {MODEL_NAME}")
    print(f"Input : {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")

    print()
    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    print("Tokenizer loaded successfully.")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    total_processed = 0

    for split_name in [
        "train",
        "validation",
        "test",
    ]:

        processed = tokenize_split(
            split_name,
            tokenizer
        )

        total_processed += processed

    print()
    print("=" * 70)
    print("TOKENIZATION COMPLETED")
    print("=" * 70)

    print(f"Total rows tokenized: {total_processed:,}")

    print()
    print("Output files:")

    for split_name in [
        "train",
        "validation",
        "test",
    ]:

        output_path = OUTPUT_DIR / f"{split_name}.parquet"

        print(f"  {output_path}")


if __name__ == "__main__":
    main()