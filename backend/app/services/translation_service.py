from pathlib import Path

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

BASE_MODEL_NAME = "Helsinki-NLP/opus-mt-en-dra"

TAMIL_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "finetuned"
    / "final"
)

MAX_SOURCE_LENGTH = 128
MAX_NEW_TOKENS = 128


# ============================================================
# SUPPORTED LANGUAGES
# ============================================================

LANGUAGE_CODES = {
    "Tamil": "tam",
    "Telugu": "tel",
    "Kannada": "kan",
    "Malayalam": "mal",
}


class TranslationService:

    def __init__(self):

        print("=" * 70)
        print("TamilNLP Translation Service")
        print("=" * 70)

        if torch.cuda.is_available():

            self.device = torch.device("cuda")

        else:

            self.device = torch.device("cpu")

        print()
        print(f"Device: {self.device}")

        self.current_model_name = None
        self.tokenizer = None
        self.model = None

        # Start with Tamil because it is the primary
        # fine-tuned language.
        self._load_model("Tamil")

    # ========================================================
    # LOAD MODEL
    # ========================================================

    def _load_model(self, language):

        if language not in LANGUAGE_CODES:

            raise ValueError(
                f"Unsupported language: {language}"
            )

        if language == self.current_model_name:
            return

        # ----------------------------------------------------
        # Release previous model
        # ----------------------------------------------------

        if self.model is not None:

            del self.model
            self.model = None

            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        # ----------------------------------------------------
        # Select model
        # ----------------------------------------------------

        if (
            language == "Tamil"
            and TAMIL_MODEL_PATH.exists()
        ):

            model_source = str(
                TAMIL_MODEL_PATH
            )

            print()
            print(
                "Using fine-tuned Tamil model."
            )

        else:

            model_source = BASE_MODEL_NAME

            if language == "Tamil":

                print()
                print(
                    "Fine-tuned Tamil model not found."
                )

            else:

                print()
                print(
                    f"Using multilingual base model "
                    f"for {language}."
                )

        print()
        print(
            f"Loading model for {language}..."
        )

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                model_source
            )
        )

        self.model = (
            AutoModelForSeq2SeqLM.from_pretrained(
                model_source
            )
        )

        self.model.to(self.device)

        self.model.eval()

        self.current_model_name = language

        print(
            f"{language} model ready."
        )

    # ========================================================
    # TRANSLATE
    # ========================================================

    def translate(
        self,
        text: str,
        target_language: str = "Tamil",
    ):

        if not isinstance(
            text,
            str,
        ):

            raise TypeError(
                "text must be a string."
            )

        text = text.strip()

        if not text:

            raise ValueError(
                "Text cannot be empty."
            )

        if target_language not in LANGUAGE_CODES:

            raise ValueError(
                "Unsupported target language: "
                f"{target_language}"
            )

        # Load the correct model.
        self._load_model(
            target_language
        )

        language_code = (
            LANGUAGE_CODES[
                target_language
            ]
        )

        source_text = (
            f">>{language_code}<< {text}"
        )

        inputs = self.tokenizer(
            source_text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_SOURCE_LENGTH,
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.no_grad():

            generated = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                num_beams=4,
            )

        translation = (
            self.tokenizer.decode(
                generated[0],
                skip_special_tokens=True,
            )
        )

        return translation.strip()


# ============================================================
# SINGLE SERVICE INSTANCE
# ============================================================

translation_service = TranslationService()