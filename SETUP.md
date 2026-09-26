# TamilNLP Development Setup

This document contains the commands required to create and configure the local development environment.

---

# 1. Create Project

Open the VS Code terminal.

Navigate to the location where the project should be created.

Example:

```powershell
cd C:\Users\hp\Downloads
```

Create the project:

```powershell
mkdir TamilNLP
cd TamilNLP
```

Open the project in VS Code:

```powershell
code .
```

---

# 2. Create Python Virtual Environment

Check Python:

```powershell
python --version
```

Create virtual environment:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\activate
```

Expected terminal:

```text
(.venv) PS C:\...\TamilNLP>
```

---

# 3. Create Project Directories

Create the main directories:

```powershell
mkdir backend
mkdir frontend
mkdir data
mkdir training
mkdir experiments
mkdir models
mkdir notebooks
```

Create backend directories:

```powershell
mkdir backend\app
mkdir backend\app\api
mkdir backend\app\schemas
mkdir backend\app\services
mkdir backend\app\utils
mkdir backend\models
mkdir backend\tests
```

Create data directories:

```powershell
mkdir data\raw
mkdir data\processed
mkdir data\evaluation
mkdir data\evaluation\baseline
mkdir data\evaluation\finetuned
```

Create experiment directories:

```powershell
mkdir experiments\baseline
mkdir experiments\finetuning
```

Create model directories:

```powershell
mkdir models\baseline
mkdir models\finetuned
```

---

# 4. Create Root Files

Create:

```text
ROADMAP.md
SETUP.md
README.md
.gitignore
requirements.txt
```

---

# 5. Create Python Package Files

Create:

```text
backend/app/__init__.py
backend/app/api/__init__.py
backend/app/schemas/__init__.py
backend/app/services/__init__.py
backend/app/utils/__init__.py
```

---

# 6. Create Backend Structure

The backend will eventually contain:

```text
backend/
├── app/
│   ├── main.py
│   ├── config.py
│   │
│   ├── api/
│   │   └── routes.py
│   │
│   ├── schemas/
│   │   └── translation.py
│   │
│   ├── services/
│   │   ├── translator.py
│   │   └── preprocessing.py
│   │
│   └── utils/
│       └── logging.py
│
├── models/
└── tests/
```

These files will be implemented later according to the roadmap.

---

# 7. Python Dependencies

The root `requirements.txt` will eventually contain the project's Python dependencies.

Initial dependencies:

```text
torch
transformers
datasets
sentencepiece
evaluate
sacrebleu
fastapi
uvicorn
pydantic
python-dotenv
numpy
pandas
```

Do not install blindly before verifying the local PyTorch/CUDA environment.

---

# 8. Environment Verification

Before installing the full ML stack, verify Python:

```powershell
python --version
```

Verify pip:

```powershell
python -m pip --version
```

Verify GPU visibility through NVIDIA:

```powershell
nvidia-smi
```

Record:

* GPU name
* VRAM
* NVIDIA driver
* CUDA information

---

# 9. Install Project Dependencies

After environment verification, install the required packages.

```powershell
python -m pip install --upgrade pip
```

Then install the project requirements:

```powershell
pip install -r requirements.txt
```

---

# 10. Verify PyTorch

Run:

```powershell
python -c "import torch; print(torch.__version__)"
```

Check CUDA:

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

Check GPU:

```powershell
python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA GPU not available')"
```

---

# 11. Verify Transformers

Run:

```powershell
python -c "import transformers; print(transformers.__version__)"
```

---

# 12. Git Setup

Initialize Git:

```powershell
git init
```

Check:

```powershell
git status
```

Add files:

```powershell
git add .
```

Initial commit:

```powershell
git commit -m "Initialize TamilNLP project"
```

---

# 13. VS Code

Recommended extensions:

* Python
* Pylance
* Jupyter
* GitLens
* ESLint
* Prettier

Select the project Python interpreter:

```text
.venv
```

---

# 14. Frontend Setup

Frontend will be created using Vite.

From the project root:

```powershell
npm create vite@latest frontend -- --template react
```

Then:

```powershell
cd frontend
npm install
```

Start development server:

```powershell
npm run dev
```

Return to project root:

```powershell
cd ..
```

---

# 15. Development Servers

Backend:

```powershell
uvicorn backend.app.main:app --reload
```

Frontend:

```powershell
cd frontend
npm run dev
```

Backend and frontend will eventually run independently during development.

---

# 16. Development Order

Setup must be performed in this order:

```text
Create project
      ↓
Create virtual environment
      ↓
Verify Python
      ↓
Verify NVIDIA GPU
      ↓
Install PyTorch
      ↓
Verify CUDA
      ↓
Install Transformers
      ↓
Verify Hugging Face
      ↓
Initialize Git
      ↓
Create frontend
      ↓
Verify frontend
      ↓
Begin ML baseline
```

---

# 17. Important Rule

Do not start model fine-tuning until:

* Python is working
* Virtual environment is working
* PyTorch is working
* CUDA is working if GPU training is intended
* Transformers is working
* Baseline model can be loaded
* Baseline translation has been tested

The model must be established as a baseline before fine-tuning.
