from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)


MODEL_NAME = "Helsinki-NLP/opus-mt-en-dra"


def translate(
    model,
    tokenizer,
    text,
    language_code,
):

    source_text = (
        f">>{language_code}<< {text}"
    )

    inputs = tokenizer(
        source_text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    )

    generated = model.generate(
        **inputs,
        max_new_tokens=128,
        num_beams=4,
    )

    return tokenizer.decode(
        generated[0],
        skip_special_tokens=True,
    )


def main():

    print("=" * 70)
    print("TamilNLP - Multilingual Test")
    print("=" * 70)

    print()
    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    print("Tokenizer loaded.")

    print()
    print("Loading model...")

    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME
    )

    model.to("cpu")
    model.eval()

    print("Model loaded.")

    english = "How are you?"

    languages = {
        "tam": "Tamil",
        "tel": "Telugu",
        "kan": "Kannada",
        "mal": "Malayalam",
    }

    print()
    print("=" * 70)
    print("MULTILINGUAL TRANSLATION TEST")
    print("=" * 70)

    print()
    print("English:")
    print(english)

    for code, language in languages.items():

        translation = translate(
            model,
            tokenizer,
            english,
            code,
        )

        print()
        print(
            f"{language} ({code}):"
        )

        print(translation)

    print()
    print("=" * 70)
    print("MULTILINGUAL TEST COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()