from pathlib import Path

import pyarrow.parquet as pq
import sacrebleu
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "opus100_en_ta_final"
)

BASE_MODEL_NAME = "Helsinki-NLP/opus-mt-en-dra"

FINETUNED_MODEL = (
    PROJECT_ROOT
    / "models"
    / "finetuned"
    / "final"
)

NUM_SAMPLES = 100

BATCH_SIZE = 4

MAX_SOURCE_LENGTH = 128
MAX_TARGET_LENGTH = 128


# ============================================================
# LOAD TEST DATA
# ============================================================

def load_test_data():

    path = DATA_DIR / "test.parquet"

    table = pq.read_table(path)

    rows = table.to_pylist()

    return rows[:NUM_SAMPLES]


# ============================================================
# GENERATE TRANSLATIONS
# ============================================================

def generate_predictions(
    model,
    tokenizer,
    texts,
):

    predictions = []

    for start in range(
        0,
        len(texts),
        BATCH_SIZE,
    ):

        batch_texts = texts[
            start:start + BATCH_SIZE
        ]

        source_texts = [
            f">>tam<< {text}"
            for text in batch_texts
        ]

        inputs = tokenizer(
            source_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_SOURCE_LENGTH,
        )

        inputs = {
            key: value.to("cpu")
            for key, value in inputs.items()
        }

        with torch.no_grad():

            generated = model.generate(
                **inputs,
                max_new_tokens=MAX_TARGET_LENGTH,
                num_beams=2,
            )

        decoded = tokenizer.batch_decode(
            generated,
            skip_special_tokens=True,
        )

        predictions.extend(decoded)

        print(
            f"Generated "
            f"{min(start + BATCH_SIZE, len(texts)):,}"
            f"/{len(texts):,}"
        )

    return predictions


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    predictions,
    references,
):

    bleu = sacrebleu.corpus_bleu(
        predictions,
        [references],
    )

    chrf = sacrebleu.corpus_chrf(
        predictions,
        [references],
    )

    return bleu, chrf


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("TamilNLP - Base vs Fine-Tuned Evaluation")
    print("=" * 70)

    print()
    print(f"Evaluation samples: {NUM_SAMPLES}")
    print(f"Batch size        : {BATCH_SIZE}")

    # --------------------------------------------------------
    # Load tokenizer
    # --------------------------------------------------------

    print()
    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_NAME
    )

    print("Tokenizer loaded.")

    # --------------------------------------------------------
    # Load test data
    # --------------------------------------------------------

    print()
    print("Loading test data...")

    rows = load_test_data()

    english_texts = [
        row["translation"]["en"]
        for row in rows
    ]

    references = [
        row["translation"]["ta"]
        for row in rows
    ]

    print(
        f"Loaded {len(rows):,} test examples."
    )

    # --------------------------------------------------------
    # Base model
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("BASE MODEL")
    print("=" * 70)

    print()
    print("Loading base model...")

    base_model = (
        AutoModelForSeq2SeqLM
        .from_pretrained(BASE_MODEL_NAME)
    )

    base_model.to("cpu")
    base_model.eval()

    print("Base model loaded.")

    print()
    print("Generating base predictions...")

    base_predictions = generate_predictions(
        base_model,
        tokenizer,
        english_texts,
    )

    base_bleu, base_chrf = calculate_metrics(
        base_predictions,
        references,
    )

    print()
    print("Base model metrics:")
    print(
        f"BLEU : {base_bleu.score:.4f}"
    )
    print(
        f"chrF : {base_chrf.score:.4f}"
    )

    del base_model

    # --------------------------------------------------------
    # Fine-tuned model
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FINE-TUNED MODEL")
    print("=" * 70)

    print()
    print("Loading fine-tuned model...")

    finetuned_model = (
        AutoModelForSeq2SeqLM
        .from_pretrained(FINETUNED_MODEL)
    )

    finetuned_model.to("cpu")
    finetuned_model.eval()

    print("Fine-tuned model loaded.")

    print()
    print("Generating fine-tuned predictions...")

    finetuned_predictions = generate_predictions(
        finetuned_model,
        tokenizer,
        english_texts,
    )

    finetuned_bleu, finetuned_chrf = (
        calculate_metrics(
            finetuned_predictions,
            references,
        )
    )

    print()
    print("Fine-tuned model metrics:")
    print(
        f"BLEU : {finetuned_bleu.score:.4f}"
    )
    print(
        f"chrF : {finetuned_chrf.score:.4f}"
    )

    # --------------------------------------------------------
    # Comparison
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("METRIC COMPARISON")
    print("=" * 70)

    print()
    print(
        f"Base BLEU       : "
        f"{base_bleu.score:.4f}"
    )

    print(
        f"Fine-tuned BLEU : "
        f"{finetuned_bleu.score:.4f}"
    )

    print()

    print(
        f"Base chrF       : "
        f"{base_chrf.score:.4f}"
    )

    print(
        f"Fine-tuned chrF : "
        f"{finetuned_chrf.score:.4f}"
    )

    print()
    print("=" * 70)
    print("METRIC EVALUATION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()