from pathlib import Path

import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "opus100_en_ta_tokenized"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "models"
    / "finetuned"
)

MODEL_NAME = "Helsinki-NLP/opus-mt-en-dra"


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

# Controlled first production run.
# Increase later after checking results.
MAX_STEPS = 500

TRAIN_BATCH_SIZE = 2
EVAL_BATCH_SIZE = 2

GRADIENT_ACCUMULATION_STEPS = 1

LEARNING_RATE = 3e-5

WARMUP_STEPS = 50

LOGGING_STEPS = 25

EVAL_STEPS = 100
SAVE_STEPS = 100

SAVE_TOTAL_LIMIT = 2


# ============================================================
# PARQUET DATASET
# ============================================================

class TokenizedParquetDataset(Dataset):
    """
    Lightweight PyTorch Dataset backed directly by Parquet.

    This avoids converting the complete training dataset into
    a huge Python list in memory.
    """

    def __init__(self, parquet_path):

        self.parquet_path = Path(parquet_path)

        if not self.parquet_path.exists():
            raise FileNotFoundError(
                f"Dataset file not found:\n{self.parquet_path}"
            )

        self.table = pq.read_table(self.parquet_path)

        required_columns = {
            "input_ids",
            "attention_mask",
            "labels",
        }

        actual_columns = set(
            self.table.column_names
        )

        missing = required_columns - actual_columns

        if missing:
            raise ValueError(
                f"Missing columns in {self.parquet_path}: "
                f"{sorted(missing)}"
            )

        self.input_ids = self.table["input_ids"]
        self.attention_mask = self.table["attention_mask"]
        self.labels = self.table["labels"]

    def __len__(self):

        return self.table.num_rows

    def __getitem__(self, index):

        return {
            "input_ids": self.input_ids[index].as_py(),
            "attention_mask": self.attention_mask[index].as_py(),
            "labels": self.labels[index].as_py(),
        }


# ============================================================
# DEVICE
# ============================================================

def get_device():

    if torch.cuda.is_available():

        device = torch.device("cuda")

        print()
        print("CUDA GPU detected.")

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

        return device

    device = torch.device("cpu")

    print()
    print("No CUDA GPU detected.")
    print("Using CPU.")

    return device


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("TamilNLP - Full Fine-Tuning Pipeline")
    print("=" * 70)

    print()
    print(f"Model: {MODEL_NAME}")

    print()
    print("Training configuration:")
    print(f"  Maximum steps            : {MAX_STEPS}")
    print(f"  Train batch size         : {TRAIN_BATCH_SIZE}")
    print(f"  Eval batch size          : {EVAL_BATCH_SIZE}")
    print(
        f"  Gradient accumulation    : "
        f"{GRADIENT_ACCUMULATION_STEPS}"
    )
    print(f"  Learning rate            : {LEARNING_RATE}")
    print(f"  Warmup steps             : {WARMUP_STEPS}")
    print()

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = get_device()

    # --------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------

    print()
    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    print("Tokenizer loaded.")

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print()
    print("Loading base model...")

    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME
    )

    model.to(device)

    print("Model loaded.")

    # --------------------------------------------------------
    # Training dataset
    # --------------------------------------------------------

    print()
    print("Loading training dataset...")

    train_dataset = TokenizedParquetDataset(
        DATA_DIR / "train.parquet"
    )

    print(
        f"Training rows: "
        f"{len(train_dataset):,}"
    )

    # --------------------------------------------------------
    # Validation dataset
    # --------------------------------------------------------

    print()
    print("Loading validation dataset...")

    validation_dataset = TokenizedParquetDataset(
        DATA_DIR / "validation.parquet"
    )

    print(
        f"Validation rows: "
        f"{len(validation_dataset):,}"
    )

    # --------------------------------------------------------
    # Data collator
    # --------------------------------------------------------

    print()
    print("Creating data collator...")

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
    )

    print("Data collator ready.")

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Training arguments
    # --------------------------------------------------------

    training_args = Seq2SeqTrainingArguments(

        output_dir=str(OUTPUT_DIR),

        # Controlled training run
        max_steps=MAX_STEPS,

        per_device_train_batch_size=TRAIN_BATCH_SIZE,

        per_device_eval_batch_size=EVAL_BATCH_SIZE,

        gradient_accumulation_steps=(
            GRADIENT_ACCUMULATION_STEPS
        ),

        learning_rate=LEARNING_RATE,

        warmup_steps=WARMUP_STEPS,

        # Logging
        logging_strategy="steps",
        logging_steps=LOGGING_STEPS,

        # Evaluation
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,

        # Checkpoints
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,

        # Select best checkpoint by validation loss
        load_best_model_at_end=True,

        metric_for_best_model="eval_loss",

        greater_is_better=False,

        # CPU/GPU settings
        fp16=False,
        bf16=False,

        dataloader_num_workers=0,

        # Our dataset explicitly returns the model columns
        remove_unused_columns=False,

        # No external experiment tracking
        report_to="none",

        # Use CPU when CUDA isn't available
        use_cpu=(device.type == "cpu"),
    )

    # --------------------------------------------------------
    # Trainer
    # --------------------------------------------------------

    print()
    print("Creating trainer...")

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    print("Trainer ready.")

    # --------------------------------------------------------
    # Start training
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("STARTING FINE-TUNING")
    print("=" * 70)

    print()
    print(
        f"Training for up to {MAX_STEPS} steps."
    )

    train_result = trainer.train()

    # --------------------------------------------------------
    # Final evaluation
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("RUNNING FINAL VALIDATION")
    print("=" * 70)

    evaluation_result = trainer.evaluate()

    # --------------------------------------------------------
    # Save final model
    # --------------------------------------------------------

    final_model_dir = OUTPUT_DIR / "final"

    print()
    print("Saving final model...")

    trainer.save_model(
        str(final_model_dir)
    )

    tokenizer.save_pretrained(
        str(final_model_dir)
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FINE-TUNING COMPLETED")
    print("=" * 70)

    print()
    print("Training metrics:")
    print(train_result.metrics)

    print()
    print("Final validation metrics:")
    print(evaluation_result)

    print()
    print("Final model saved to:")

    print(final_model_dir)

    print()
    print("=" * 70)
    print("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()