# TamilNLP

## English → Dravidian Translation SLM

---

# 1. Project Vision

Build an end-to-end English → Dravidian machine translation system using a pretrained Hugging Face Seq2Seq model, with Tamil as the primary fine-tuning language and support for multiple Dravidian target languages.

The current product will contain:

* Multilingual translation
* Fine-tuned Tamil translation model
* Dataset preparation pipeline
* Dataset tokenization pipeline
* Training pipeline
* Evaluation pipeline
* Translation service
* Streamlit frontend
* Multilingual target-language selection
* Backend testing
* Future translation history
* Future PostgreSQL integration
* Deployment support

---

# 2. Core Objective

The project will investigate how effectively a pretrained multilingual English/Dravidian translation model can be used and fine-tuned for high-quality translation.

Initial baseline model:

`Helsinki-NLP/opus-mt-en-dra`

The project currently supports:

```text
English → Tamil
English → Telugu
English → Kannada
English → Malayalam

Tamil is the primary fine-tuning target.

The pretrained multilingual model will always be treated as the baseline.

Fine-tuning results will be evaluated against the baseline instead of assuming that fine-tuning automatically improves translation quality.

3. High-Level Architecture
                         TamilNLP
                            │
                            ▼
                   Streamlit Frontend
                            │
                            ▼
                  Translation Service
                            │
              ┌─────────────┴─────────────┐
              │                           │
              ▼                           ▼
      Fine-tuned Tamil Model      Multilingual Base Model
      models/finetuned/final      Helsinki-NLP/opus-mt-en-dra
              │                           │
              ▼                           ▼
            Tamil             Telugu / Kannada / Malayalam

The translation service dynamically selects the appropriate model based on the selected target language.

4. Project Components
4.1 Frontend

Technology:

Python
Streamlit

Current frontend:

frontend/app.py

Main responsibilities:

English text input
Target language selection
Translation button
Translation output
Loading state
Empty-input validation
Error handling

Supported target languages:

Tamil
Telugu
Kannada
Malayalam

Main user flow:

English Input
      ↓
Target Language Selection
      ↓
Translate
      ↓
Translation Service
      ↓
Translated Output

Future features:

Copy translation
Clear input
Translation history
Model information
Translation latency
Batch translation
Document translation
Dark mode
Speech input
Text-to-speech
4.2 Translation Service

Technology:

Python
PyTorch
Hugging Face Transformers

Main file:

backend/app/services/translation_service.py

Responsibilities:

Target-language validation
Model selection
Tokenizer loading
Model loading
Model replacement
Target-language token handling
Model inference
Output decoding
CPU/GPU device selection
Memory-aware model switching

Current model selection:

Tamil
  ↓
Fine-tuned Tamil model when available

Telugu
  ↓
Multilingual base model

Kannada
  ↓
Multilingual base model

Malayalam
  ↓
Multilingual base model
4.3 Translation Model

Initial model:

Helsinki-NLP/opus-mt-en-dra

Model architecture:

Encoder-decoder Transformer
Seq2Seq
Multilingual translation model

Target language tokens:

Tamil      → >>tam<<
Telugu     → >>tel<<
Kannada    → >>kan<<
Malayalam  → >>mal<<

Pipeline:

English

   ↓

Target Language Token

   ↓

Tokenizer

   ↓

Encoder

   ↓

Decoder

   ↓

Target Language Tokens

   ↓

Decoded Translation

Tamil uses the locally fine-tuned model when available.

The fine-tuned Tamil model is stored locally at:

models/finetuned/final/

Large model files are intentionally excluded from GitHub.

4.4 Dataset

Primary dataset:

Helsinki-NLP/opus-100

Primary language pair:

English → Tamil

Dataset requirements:

English sentence
Tamil translation
Correct alignment
No empty samples
Valid Tamil Unicode
Minimal corrupted data
Appropriate sentence length
Reliable source-target pairing

The dataset is processed into training-ready Parquet files.

4.5 Dataset Splitting

Dataset:

Training
Validation
Test

Final main dataset used in the training workflow:

Training      177,483
Validation      1,538
Test            1,563
-----------------------
Total         180,584

The test set remains isolated from training and validation.

The training pipeline uses:

train.parquet
validation.parquet

The evaluation pipeline uses:

test.parquet

The test set is not used by the fine-tuning process.

5. Development Roadmap
Phase 0 — Project Definition

Objectives:

Define project scope
Define English → Dravidian translation goal
Select Tamil as primary fine-tuning target
Define supported languages
Define model strategy
Define evaluation strategy
Define application architecture

Deliverables:

ROADMAP.md
README.md
SETUP.md
Project architecture

Status:

Completed.

Phase 1 — Environment

Objectives:

Verify Python
Create virtual environment
Install PyTorch
Verify CPU support
Install Hugging Face Transformers
Install SentencePiece
Install Accelerate
Install SacreBLEU
Install Streamlit
Verify the ML environment

Current environment:

Python        3.12
PyTorch       2.14.0+cpu
Transformers  5.17.0
Streamlit     1.64.0
SentencePiece 0.2.2
Accelerate    1.15.0
SacreBLEU     2.6.0

Deliverable:

A working ML development environment.

Status:

Completed.

Phase 2 — Baseline Model

Objectives:

Load tokenizer
Load pretrained multilingual model
Run English → Tamil inference
Run English → Telugu inference
Run English → Kannada inference
Run English → Malayalam inference
Validate target-language tokens
Verify CPU inference

Baseline scripts:

experiments/baseline/baseline_translation.py
experiments/baseline/test_multilingual.py

Deliverable:

Working multilingual baseline translator.

Status:

Completed.

Phase 3 — Dataset Preparation

Objectives:

Obtain English–Tamil parallel data
Inspect data
Clean data
Remove empty records
Validate language pairs
Validate Tamil Unicode
Remove corrupted samples
Maintain train, validation, and test separation

Main file:

training/prepare_dataset.py

Deliverable:

Clean English–Tamil parallel dataset.

Status:

Completed.

Phase 4 — Dataset Analysis

Objectives:

Analyze:

Dataset size
Sentence lengths
Token lengths
Language consistency
Alignment quality
Corrupted examples
Long sentences
Short sentences
Data quality issues
Potential annotation contamination

Deliverable:

Dataset quality analysis.

Status:

Completed as part of the dataset preparation and evaluation process.

Phase 5 — Dataset Splitting

Create:

train
validation
test

Requirements:

Keep test data isolated
Keep validation data separate
Prevent test leakage
Preserve source-target alignment

Deliverable:

Versioned dataset splits.

Status:

Completed.

Phase 6 — Tokenization

Objectives:

Load pretrained tokenizer
Add target-language token to source text
Tokenize English inputs
Tokenize Tamil targets
Create labels
Set sequence length
Validate tokenization
Save tokenized Parquet data

Main file:

training/tokenize_dataset.py

Current configuration:

Maximum source length : 128
Maximum target length : 128
Batch size            : 256

Example:

>>tam<< How are you?

Deliverable:

Training-ready tokenized dataset.

Status:

Completed.

Phase 7 — Fine-Tuning

Objectives:

Load pretrained multilingual model
Load tokenized English–Tamil dataset
Configure Seq2Seq training
Train on Tamil data
Validate during training
Save checkpoints
Select final model
Save tokenizer with the final model

Main file:

training/train.py

Training configuration:

Maximum steps            : 500
Train batch size         : 2
Evaluation batch size    : 2
Gradient accumulation    : 1
Learning rate            : 3e-5
Warmup steps             : 50
Logging steps            : 25
Evaluation steps         : 100
Save steps               : 100
Device                   : CPU

Final local model:

models/finetuned/final/

Deliverable:

Fine-tuned English → Tamil model.

Status:

Completed.

Phase 8 — Model Evaluation

Evaluate:

Validation loss
BLEU
chrF
Base model output
Fine-tuned model output
Sentence-level translation quality
Translation regressions
Translation improvements

Main file:

training/evaluate_metrics.py

Evaluation flow:

Test Data
   ↓
Base Model Predictions
   ↓
Fine-tuned Model Predictions
   ↓
BLEU
   ↓
chrF
   ↓
Comparison

Deliverable:

Evaluation results.

Status:

Completed.

Phase 9 — Error Analysis

Analyze:

Incorrect word order
Missing words
Extra words
Incorrect meaning
Grammar problems
Named entities
Technical terms
Long sentence failures
Baseline regressions
Fine-tuning improvements

The project must record both improvements and regressions.

The pretrained baseline must remain part of the comparison.

Deliverable:

Translation error analysis.

Status:

Completed for the current fine-tuning experiments.

Phase 10 — Base vs Fine-Tuned Comparison

Compare:

                 BASE MODEL
                      vs
              FINE-TUNED MODEL

Measurements:

BLEU
chrF
Translation quality
Sentence-level results
Translation regressions
Translation improvements

Deliverable:

Model comparison report.

Status:

Completed.

Phase 11 — Model Packaging

Package the final Tamil model with:

Model weights
Configuration
Tokenizer
Tokenizer configuration
Model version

Local output:

models/finetuned/final/

The model directory is not uploaded to GitHub because the model files are large.

Deliverable:

Locally packaged Tamil translation model.

Status:

Completed.

Phase 12 — Translation Service

Build a reusable translation service.

Main file:

backend/app/services/translation_service.py

Responsibilities:

Request
   ↓
Target Language Validation
   ↓
Model Selection
   ↓
Tokenizer
   ↓
Model
   ↓
Generation
   ↓
Decoded Translation

Supported languages:

Tamil
Telugu
Kannada
Malayalam

Deliverable:

Working multilingual translation service.

Status:

Completed.

Phase 13 — Streamlit Frontend

Build:

English Input
      ↓
Target Language
      ↓
Translate
      ↓
Translation Output

Technology:

Streamlit
Python

Main file:

frontend/app.py

Components:

Application title
Target-language selector
English text box
Translate button
Loading state
Translation result
Warning state
Error state

Deliverable:

Working multilingual translation UI.

Status:

Completed.

Phase 14 — Frontend + Translation Service Integration

Connect:

Streamlit
   ↓
Translation Service
   ↓
Selected Language
   ↓
Appropriate Model
   ↓
Translation
   ↓
Streamlit Output

Current application command:

streamlit run frontend/app.py

Deliverable:

Working local translation application.

Status:

Completed.

Phase 15 — Translation History

Store:

English text
Translation
Target language
Timestamp
Model version

Initial implementation:

Local application storage.

Future implementation:

PostgreSQL.

Deliverable:

Persistent translation history.

Status:

Planned.

Phase 16 — PostgreSQL Integration

PostgreSQL is planned for future persistent application features.

Potential uses:

Translation history
Saved translations
User data
Usage analytics
Model usage statistics

Future architecture:

Streamlit
    │
    ├───────────────► Translation Service
    │
    └───────────────► PostgreSQL
                          │
                          ├── Translation History
                          ├── Users
                          └── Analytics

Objectives:

Create database schema
Configure secure PostgreSQL connection
Store translation history
Store user/application data
Add analytics
Manage database configuration through environment variables

Deliverable:

Database-backed application storage.

Status:

Planned.

Phase 17 — Testing
Backend
Translation service tests
Language selection tests
Model loading tests
Input validation tests
Error handling tests

Main test:

backend/test_translation_service.py
Multilingual
Tamil translation
Telugu translation
Kannada translation
Malayalam translation

Main test:

experiments/baseline/test_multilingual.py
ML
Dataset tests
Tokenization tests
Inference tests
Evaluation tests

Deliverable:

Tested translation application.

Status:

Completed for the current implementation.

Phase 18 — Docker

Create a containerized application for:

Streamlit Frontend
Translation Service
Model

Possible architecture:

User
  ↓
Streamlit
  ↓
Translation Service
  ↓
Translation Model

Docker support will be introduced only after the local application and model-loading behavior are stable.

Deliverable:

Docker-based local deployment.

Status:

Planned.

Phase 19 — Deployment

Deploy:

User
  ↓
Streamlit Application
  ↓
Translation Service
  ↓
Translation Model

Deployment requirements:

Correct Python environment
Required dependencies
Model availability
CPU memory planning
Environment variables
Application startup configuration
Secure configuration

Possible deployment targets will be selected after local validation.

Deliverable:

Deployable translation application.

Status:

Planned.

Phase 20 — Documentation

Document:

Project overview
Current architecture
Supported languages
Dataset
Dataset preparation
Tokenization
Model
Fine-tuning
Evaluation
Translation service
Streamlit frontend
Testing
Installation
Local usage
GitHub structure
PostgreSQL plans
Deployment
Limitations
Future improvements

Documentation files:

README.md
ROADMAP.md
SETUP.md

Deliverable:

Complete and up-to-date project documentation.

Status:

In progress.

Phase 21 — Advanced Features

Potential future features:

English speech input
Tamil text-to-speech
Telugu text-to-speech
Kannada text-to-speech
Malayalam text-to-speech
PDF translation
DOCX translation
Batch translation
Translation memory
Domain-specific translation
Model comparison
User accounts
PostgreSQL database
Analytics dashboard
Translation history
Model information
Monitoring
6. Final Product Architecture
                           USER
                            │
                            ▼
                   ┌─────────────────┐
                   │ Streamlit UI    │
                   │ frontend/app.py │
                   └────────┬────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │ Translation     │
                   │ Service         │
                   └────────┬────────┘
                            │
               ┌────────────┴────────────┐
               │                         │
               ▼                         ▼
       Fine-tuned Tamil          Multilingual Base
            Model                     Model
               │                         │
               └────────────┬────────────┘
                            │
                            ▼
                     Target Translation
                            │
                            ▼
                           USER

Future persistent architecture:

                           USER
                            │
                            ▼
                   ┌─────────────────┐
                   │ Streamlit UI    │
                   └────────┬────────┘
                            │
                  ┌─────────┴─────────┐
                  │                   │
                  ▼                   ▼
         Translation Service     PostgreSQL
                  │                   │
                  ▼                   ├── Users
               Model                 ├── History
                                      └── Analytics
7. Final Success Criteria

The project is considered complete when:

 Dataset is prepared
 Baseline model is evaluated
 Multilingual baseline is tested
 Dataset is cleaned
 Dataset is tokenized
 Tamil model is fine-tuned
 Fine-tuned model is evaluated
 Base and fine-tuned models are compared
 Model is packaged locally
 Translation service works
 Tamil translation works
 Telugu translation works
 Kannada translation works
 Malayalam translation works
 Streamlit frontend works
 Frontend communicates with translation service
 Backend tests work
 Large datasets are excluded from GitHub
 Large model files are excluded from GitHub
 Project documentation is maintained

Future completion targets:

 Translation history
 PostgreSQL integration
 Docker support
 Deployment
 Production monitoring
 Advanced translation features
8. Development Principle

The project should be developed incrementally.

Order:

Environment
    ↓
Baseline
    ↓
Dataset
    ↓
Dataset Analysis
    ↓
Dataset Splitting
    ↓
Tokenization
    ↓
Fine-Tuning
    ↓
Evaluation
    ↓
Error Analysis
    ↓
Translation Service
    ↓
Streamlit Frontend
    ↓
Integration
    ↓
Testing
    ↓
Documentation
    ↓
PostgreSQL
    ↓
Docker
    ↓
Deployment
    ↓
Monitoring

Do not build all components simultaneously.

Each phase must be verified before moving to the next major phase.

The pretrained multilingual model remains the baseline for future model improvements.

Every major model change should be evaluated before it is adopted.

The project prioritizes:

Correctness
    ↓
Evaluation
    ↓
Reproducibility
    ↓
Practicality
    ↓
Deployment
