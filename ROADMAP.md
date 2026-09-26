# TamilNLP

## English → Tamil Translation SLM

---

# 1. Project Vision

Build an end-to-end English → Tamil machine translation system using a pretrained Hugging Face Seq2Seq model, fine-tuned on high-quality English–Tamil parallel data.

The final product will contain:

* Fine-tuned translation model
* Dataset preparation pipeline
* Training pipeline
* Evaluation pipeline
* FastAPI backend
* React frontend
* Translation API
* Model information API
* Translation history
* Testing
* Docker support
* Deployment

---

# 2. Core Objective

The project will investigate whether fine-tuning a pretrained English/Dravidian translation model on curated English–Tamil parallel data improves translation quality.

Initial baseline model:

`Helsinki-NLP/opus-mt-en-dra`

The baseline will be evaluated before fine-tuning.

---

# 3. High-Level Architecture

```text
                         TamilNLP
                            │
              ┌─────────────┴─────────────┐
              │                           │
              ▼                           ▼
        React Frontend              FastAPI Backend
              │                           │
              │                    Translation Service
              │                           │
              │                           ▼
              │                    Fine-tuned SLM
              │                           │
              └───────────────┬───────────┘
                              │
                              ▼
                       English → Tamil
```

---

# 4. Project Components

## 4.1 Frontend

Technology:

* React
* Vite
* JavaScript
* CSS

Main responsibilities:

* English text input
* Translation button
* Tamil output
* Loading state
* Error state
* Copy translation
* Clear input
* Translation history
* Model information

Future features:

* Speech input
* Text-to-speech
* Document translation
* Dark mode
* Batch translation

---

# 4.2 Backend

Technology:

* Python
* FastAPI
* Uvicorn
* Pydantic

Responsibilities:

* API request validation
* Translation requests
* Model loading
* Tokenization
* Model inference
* Output decoding
* Error handling
* Health checks
* Model information

Main endpoints:

```text
POST /api/v1/translate

GET /api/v1/health

GET /api/v1/model-info
```

---

# 4.3 Translation Model

Initial model:

`Helsinki-NLP/opus-mt-en-dra`

Model architecture:

* Encoder-decoder Transformer
* Seq2Seq
* MarianMT-compatible architecture

Pipeline:

```text
English
   ↓
Tokenizer
   ↓
Encoder
   ↓
Decoder
   ↓
Tamil Tokens
   ↓
Tokenizer
   ↓
Tamil
```

The initial model is the baseline.

It will not automatically be assumed to be the final model.

---

# 4.4 Dataset

Create an English–Tamil parallel dataset.

Example:

```text
English:
I am learning artificial intelligence.

Tamil:
நான் செயற்கை நுண்ணறிவைக் கற்றுக்கொண்டு இருக்கிறேன்.
```

Dataset requirements:

* English sentence
* Tamil translation
* Correct alignment
* No empty samples
* No duplicate samples
* Minimal corrupted data
* Proper Tamil Unicode
* Appropriate sentence length

---

# 4.5 Dataset Splitting

Dataset:

```text
Training
Validation
Test
```

Initial split:

```text
80% Training
10% Validation
10% Test
```

The test set must remain isolated from training.

---

# 5. Development Roadmap

## Phase 0 — Project Definition

Objectives:

* Define project scope
* Define architecture
* Define technology stack
* Define model strategy
* Define evaluation strategy

Deliverables:

* `ROADMAP.md`
* Project architecture

---

# Phase 1 — Environment

Objectives:

* Verify Python
* Verify PyTorch
* Verify GPU
* Verify CUDA
* Install ML dependencies
* Install backend dependencies
* Verify Hugging Face Transformers

Deliverable:

A working ML development environment.

---

# Phase 2 — Baseline Model

Objectives:

* Load tokenizer
* Load pretrained model
* Run English → Tamil inference
* Test simple sentences
* Test paragraphs
* Record inference time
* Save baseline outputs

Deliverable:

Working baseline translator.

---

# Phase 3 — Dataset Preparation

Objectives:

* Obtain English–Tamil parallel data
* Inspect data
* Clean data
* Remove duplicates
* Remove empty records
* Validate language pairs
* Normalize text

Deliverable:

Clean parallel dataset.

---

# Phase 4 — Dataset Analysis

Objectives:

Analyze:

* Dataset size
* Sentence lengths
* Token lengths
* Duplicate ratio
* Empty records
* English/Tamil distribution
* Long sentences
* Short sentences

Deliverable:

Dataset analysis report.

---

# Phase 5 — Dataset Splitting

Create:

```text
train
validation
test
```

Ensure no test samples leak into training.

Deliverable:

Versioned dataset splits.

---

# Phase 6 — Tokenization

Objectives:

* Load pretrained tokenizer
* Tokenize English inputs
* Tokenize Tamil targets
* Create labels
* Set sequence length
* Validate tokenization

Deliverable:

Training-ready dataset.

---

# Phase 7 — Fine-Tuning

Objectives:

* Load pretrained model
* Configure Seq2Seq training
* Train on English–Tamil data
* Validate during training
* Save checkpoints
* Select final model

Initial training strategy:

```text
Small batch size
Gradient accumulation
FP16 where supported
Short initial sequence length
Low learning rate
Limited initial epochs
```

Training configuration will be adjusted according to hardware and results.

Deliverable:

Fine-tuned English → Tamil model.

---

# Phase 8 — Model Evaluation

Evaluate:

* Validation loss
* BLEU
* chrF
* COMET where appropriate
* Translation latency
* Human quality

Deliverable:

Evaluation results.

---

# Phase 9 — Error Analysis

Analyze:

* Incorrect word order
* Missing words
* Extra words
* Incorrect meaning
* Grammar
* Named entities
* Technical terms
* Tamil fluency
* Long sentence failures

Deliverable:

Translation error analysis.

---

# Phase 10 — Base vs Fine-Tuned Comparison

Compare:

```text
                 BASE MODEL
                      vs
               FINE-TUNED MODEL
```

Measurements:

* BLEU
* chrF
* COMET where appropriate
* Human evaluation
* Inference latency
* Model size

Deliverable:

Model comparison report.

---

# Phase 11 — Model Packaging

Package the final model with:

* Model weights
* Configuration
* Tokenizer
* Tokenizer configuration
* Model version

Deliverable:

Versioned translation model.

---

# Phase 12 — FastAPI Backend

Build:

```text
POST /api/v1/translate
GET  /api/v1/health
GET  /api/v1/model-info
```

Responsibilities:

```text
Request
 ↓
Validation
 ↓
Tokenizer
 ↓
Model
 ↓
Decoder
 ↓
Response
```

Deliverable:

Working translation API.

---

# Phase 13 — React Frontend

Build:

```text
English Input
      ↓
Translate
      ↓
Tamil Output
```

Components:

* Header
* Translation box
* Translation result
* History
* Loading state
* Error state

Deliverable:

Working translation UI.

---

# Phase 14 — Frontend + Backend Integration

Connect:

```text
React
  ↓
HTTP
  ↓
FastAPI
  ↓
Translation Model
  ↓
FastAPI
  ↓
React
```

Deliverable:

Full-stack translation application.

---

# Phase 15 — Translation History

Store:

* English text
* Tamil translation
* Timestamp
* Model version

Initial implementation:

Browser/local storage.

Future implementation:

Database.

---

# Phase 16 — Testing

## Backend

* API tests
* Validation tests
* Model tests
* Error tests

## Frontend

* Component tests
* Integration tests

## ML

* Dataset tests
* Inference tests
* Evaluation tests

Deliverable:

Tested application.

---

# Phase 17 — Docker

Create containerized services:

```text
Frontend
Backend
Model
```

Deliverable:

Docker-based local deployment.

---

# Phase 18 — Deployment

Deploy:

```text
Frontend
      ↓
Backend API
      ↓
Translation Model
```

Possible deployment targets will be selected after local validation.

---

# Phase 19 — Documentation

Document:

* Project overview
* Architecture
* Dataset
* Model
* Training
* Evaluation
* API
* Frontend
* Installation
* Usage
* Results
* Limitations
* Future improvements

Deliverable:

Complete project documentation.

---

# Phase 20 — Advanced Features

Potential future features:

* English speech input
* Tamil text-to-speech
* PDF translation
* DOCX translation
* Batch translation
* Translation memory
* Domain-specific models
* Model comparison
* User accounts
* Database
* Analytics dashboard

---

# 6. Final Product Architecture

```text
                           USER
                            │
                            ▼
                    ┌───────────────┐
                    │ React Web App │
                    └───────┬───────┘
                            │
                         REST API
                            │
                            ▼
                    ┌───────────────┐
                    │    FastAPI    │
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │ Translation   │
                    │   Service     │
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │ Fine-tuned    │
                    │ Seq2Seq SLM   │
                    └───────┬───────┘
                            │
                            ▼
                     Tamil Translation
```

---

# 7. Final Success Criteria

The project is considered complete when:

* [ ] Dataset is prepared
* [ ] Baseline model is evaluated
* [ ] Model is fine-tuned
* [ ] Fine-tuned model is evaluated
* [ ] Base and fine-tuned models are compared
* [ ] Model is packaged
* [ ] FastAPI backend works
* [ ] React frontend works
* [ ] Frontend communicates with backend
* [ ] Translation history works
* [ ] Tests pass
* [ ] Docker setup works
* [ ] Documentation is complete
* [ ] Application is deployable

---

# 8. Development Principle

The project should be developed incrementally.

Order:

```text
Environment
    ↓
Baseline
    ↓
Dataset
    ↓
Evaluation
    ↓
Fine-tuning
    ↓
Model packaging
    ↓
Backend
    ↓
Frontend
    ↓
Integration
    ↓
Testing
    ↓
Docker
    ↓
Deployment
```

Do not build all components simultaneously.

Each phase must be verified before moving to the next phase.
