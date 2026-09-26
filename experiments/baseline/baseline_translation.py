from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


MODEL_NAME = "Helsinki-NLP/opus-mt-en-dra"


def main():
    print("=" * 70)
    print("TamilNLP - Baseline Translation")
    print("=" * 70)

    print(f"Loading model: {MODEL_NAME}")
    print("Execution device: CPU")
    print()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME
    )

    model.to("cpu")
    model.eval()

    print("Model loaded successfully.")
    print()

    sentences = [
        "Hello, how are you?",
        "My name is Leela.",
        "I am learning artificial intelligence.",
        "The weather is very good today.",
        "I want to build a machine learning project."
    ]

    print("-" * 70)
    print("Translations")
    print("-" * 70)

    for sentence in sentences:
        inputs = tokenizer(
            sentence,
            return_tensors="pt",
            truncation=True
        )

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=128
            )

        translation = tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True
        )

        print(f"English : {sentence}")
        print(f"Output  : {translation}")
        print("-" * 70)

    print("Baseline translation completed.")


if __name__ == "__main__":
    main()