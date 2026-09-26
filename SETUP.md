# TamilNLP Development Setup

This document contains the commands required to create and configure the local development environment.

---

# 1. Create Project

Open the VS Code terminal.

Navigate to the location where the project should be created.

Example:

```powershell
cd F:\

Create the project:

mkdir English-Tamil_Trans_SLM

cd English-Tamil_Trans_SLM

Open the project in VS Code:

code .
2. Create Python Virtual Environment

Check Python:

python --version

The project currently uses:

Python 3.12.0

Create the virtual environment:

python -m venv .venv

Activate it:

.venv\Scripts\activate

Expected terminal:

(.venv) PS F:\English-Tamil_Trans_SLM>
3. Create Project Directories

The project uses the following main directories:

backend/
frontend/
data/
training/
experiments/
models/

Create them if starting from a fresh project:

mkdir backend
mkdir frontend
mkdir data
mkdir training
mkdir experiments
mkdir models

Create backend directories:

mkdir backend\app
mkdir backend\app\services

Create data directories:

mkdir data\raw
mkdir data\processed
mkdir data\evaluation

Create experiment directories:

mkdir experiments\baseline

Create model directories:

mkdir models\finetuned

The project does not require the old FastAPI-specific directories such as:

backend/app/api/
backend/app/schemas/
backend/app/utils/
4. Create Root Files

The root project contains:

ROADMAP.md

SETUP.md

README.md

.gitignore

requirements.txt

These files contain the project roadmap, environment setup, documentation, dependency definitions, and Git configuration.

5. Current Project Structure

The current project structure is:

English-Tamil_Trans_SLM/

├── backend/
│   ├── app/
│   │   └── services/
│   │       └── translation_service.py
│   │
│   └── test_translation_service.py
│
├── frontend/
│   └── app.py
│
├── training/
│   ├── prepare_dataset.py
│   ├── tokenize_dataset.py
│   ├── train.py
│   └── evaluate_metrics.py
│
├── experiments/
│   └── baseline/
│       ├── baseline_translation.py
│       └── test_multilingual.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── evaluation/
│
├── models/
│   └── finetuned/
│       └── final/
│
├── ROADMAP.md
├── SETUP.md
├── README.md
├── requirements.txt
└── .gitignore

Dataset files and trained model files are intentionally excluded from Git through .gitignore.

6. Python Dependencies

The project uses a CPU-based PyTorch environment.

The current requirements.txt contains:

--extra-index-url https://download.pytorch.org/whl/cpu

torch==2.14.0+cpu
transformers==5.17.0
sentencepiece==0.2.2
accelerate==1.15.0
sacrebleu==2.6.0
streamlit
pandas
numpy
pyarrow

The project does not require:

fastapi
uvicorn
react
vite
torchvision

The translation application currently runs using Streamlit and a Python translation service.

7. Environment Verification

Before installing the project dependencies, verify Python:

python --version

Verify pip:

python -m pip --version

The expected Python version is:

Python 3.12.0

The project is currently configured for CPU execution.

A dedicated NVIDIA GPU or CUDA installation is not required for the current setup.

8. Install Project Dependencies

Upgrade pip:

python -m pip install --upgrade pip

Install the project requirements:

pip install -r requirements.txt

Do not install additional packages unless they are required by the project.

9. Verify PyTorch

Check the installed PyTorch version:

python -c "import torch; print(torch.__version__)"

Expected:

2.14.0+cpu

Check CUDA availability:

python -c "import torch; print(torch.cuda.is_available())"

Expected:

False

This is expected because the current project uses the CPU version of PyTorch.

10. Verify Transformers

Run:

python -c "import transformers; print(transformers.__version__)"

Expected:

5.17.0
11. Verify Supporting Libraries

Verify SentencePiece:

python -c "import sentencepiece; print(sentencepiece.__version__)"

Expected:

0.2.2

Verify Accelerate:

python -c "import accelerate; print(accelerate.__version__)"

Expected:

1.15.0

Verify SacreBLEU:

python -c "import sacrebleu; print(sacrebleu.__version__)"

Expected:

2.6.0

Verify Streamlit:

python -c "import streamlit; print(streamlit.__version__)"
12. Streamlit Configuration

The project uses Streamlit for the frontend.

Create the Streamlit configuration directory:

mkdir .streamlit

Create:

.streamlit/config.toml

Add:

[server]
fileWatcherType = "none"

This disables Streamlit's file watcher.

The setting is required because the local Transformers installation can expose optional vision modules that may trigger repeated file-watcher warnings.

Do not install torchvision just to solve the Streamlit watcher issue.

13. Verify Baseline Model

The project uses:

Helsinki-NLP/opus-mt-en-dra

for English → Dravidian translation.

The supported language codes are:

Tamil       → tam

Telugu      → tel

Kannada     → kan

Malayalam   → mal

The model is loaded automatically through the translation service.

The base model can be tested using:

python experiments\baseline\test_multilingual.py

This verifies translation for:

Tamil
Telugu
Kannada
Malayalam
14. Verify Fine-Tuned Tamil Model

The Tamil translation service uses the fine-tuned model when it is available at:

models/finetuned/final

The fine-tuned model is based on:

Helsinki-NLP/opus-mt-en-dra

and was fine-tuned using the cleaned OPUS-100 English–Tamil dataset.

The model directory is intentionally excluded from Git.

If the local model is available, the translation service automatically loads it for Tamil translation.

15. Backend Translation Service

The translation logic is located at:

backend/app/services/translation_service.py

The service:

1. Selects the target language

2. Selects the appropriate model

3. Loads the model

4. Formats the source text with the target language token

5. Performs translation

6. Returns the translated text

The main translation function is:

translate(text, target_language="Tamil")

The supported target languages are:

Tamil
Telugu
Kannada
Malayalam
16. Streamlit Frontend

The application frontend is located at:

frontend/app.py

Start the Streamlit application from the project root:

streamlit run frontend\app.py

Streamlit will display a local URL such as:

http://localhost:8501

The application provides:

English input
        ↓
Target language selection
        ↓
Translation
        ↓
Translated output

The frontend currently supports:

Tamil

Telugu

Kannada

Malayalam
17. Test Translation Service

The backend translation service can be tested directly using:

python backend\test_translation_service.py

This test checks translation for the supported target languages.

The application should successfully load the appropriate model and return translated text.

18. Dataset Preparation

The project uses the:

Helsinki-NLP/opus-100

English–Tamil dataset.

The dataset is prepared using:

python training\prepare_dataset.py

The cleaned dataset is stored under:

data/processed/

The dataset preparation process includes:

Dataset loading
        ↓
Data cleaning
        ↓
English/Tamil validation
        ↓
Tamil Unicode validation
        ↓
Clean dataset creation
19. Dataset Tokenization

The cleaned dataset is tokenized using:

python training\tokenize_dataset.py

The tokenized dataset is stored under:

data/processed/opus100_en_ta_tokenized

The project uses a maximum source and target token length of:

128

The target language token is included in the source text for multilingual translation.

20. Model Fine-Tuning

Training is performed using:

python training\train.py

The training process uses:

Base Model
    ↓
Prepared Dataset
    ↓
Tokenized Dataset
    ↓
Fine-Tuning
    ↓
Fine-Tuned Tamil Model

The final local model is stored at:

models/finetuned/final

Model files are not committed to Git.

21. Model Evaluation

Evaluation is performed using:

python training\evaluate_metrics.py

The project uses translation evaluation metrics including:

BLEU

chrF

Evaluation should compare model outputs against reference Tamil translations.

The current project uses evaluation results to determine whether fine-tuning actually improves translation quality compared with the strong multilingual baseline.

22. Git Setup

Initialize Git when creating the project from scratch:

git init

Check the repository:

git status

Add files:

git add .

Create the initial commit:

git commit -m "Initial clean project setup"

The project should not commit:

.venv/

data/raw/

data/processed/

models/

*.safetensors

*.bin

*.pt

*.pth

These files are excluded through .gitignore.

23. VS Code Setup

Recommended VS Code extensions:

Python

Pylance

Jupyter

GitLens

Select the project Python interpreter:

.venv

The project currently uses a Python-based Streamlit frontend, so React-specific extensions are not required.

24. Project Execution

After activating the virtual environment:

.venv\Scripts\activate

Start the application:

streamlit run frontend\app.py

The application should open at:

http://localhost:8501

The complete runtime architecture is:

Streamlit Frontend
        ↓
Translation Service
        ↓
Tamil Fine-Tuned Model
        ↓
Translated Tamil Text

For Telugu, Kannada, and Malayalam:

Streamlit Frontend
        ↓
Translation Service
        ↓
Helsinki-NLP/opus-mt-en-dra
        ↓
Translated Text
25. Development Order

Setup must be performed in this order:

Create project

        ↓

Create virtual environment

        ↓

Verify Python

        ↓

Install CPU PyTorch

        ↓

Verify PyTorch

        ↓

Install Transformers

        ↓

Verify Hugging Face libraries

        ↓

Configure Streamlit

        ↓

Verify baseline model

        ↓

Prepare dataset

        ↓

Tokenize dataset

        ↓

Fine-tune Tamil model

        ↓

Evaluate model

        ↓

Run translation service tests

        ↓

Start Streamlit application
26. Important Rule

Do not start fine-tuning until:

Python is working
Virtual environment is working
PyTorch is working
Transformers is working
SentencePiece is working
Baseline model can be loaded
Baseline translation has been tested
Dataset preparation has completed
Tokenization has completed