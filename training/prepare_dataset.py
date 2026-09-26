from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


def main():
    print("=" * 70)
    print("TamilNLP - Dataset Preparation")
    print("=" * 70)

    print(f"Project root   : {PROJECT_ROOT}")
    print(f"Raw data       : {RAW_DATA_DIR}")
    print(f"Processed data : {PROCESSED_DATA_DIR}")

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print()
    print("Dataset directories verified.")
    print("Ready for English-Tamil parallel dataset preparation.")
    print("=" * 70)


if __name__ == "__main__":
    main()