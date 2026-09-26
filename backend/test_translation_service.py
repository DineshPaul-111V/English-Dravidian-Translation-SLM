from app.services.translation_service import (
    translation_service,
)


def main():

    print("=" * 70)
    print("TamilNLP - Multilingual Backend Test")
    print("=" * 70)

    english = "How are you?"

    languages = [
        "Tamil",
        "Telugu",
        "Kannada",
        "Malayalam",
    ]

    for language in languages:

        print()
        print("-" * 70)
        print(f"Target: {language}")
        print("-" * 70)

        print(
            "English:",
            english,
        )

        result = (
            translation_service.translate(
                english,
                language,
            )
        )

        print(
            f"{language}:",
            result,
        )

    print()
    print("=" * 70)
    print("MULTILINGUAL BACKEND TEST COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()