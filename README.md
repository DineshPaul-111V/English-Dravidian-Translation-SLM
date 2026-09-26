# TamilNLP — English → Dravidian Translation SLM

TamilNLP is a multilingual machine translation project for translating English text into major Dravidian languages.

The current application focuses primarily on **Tamil**, with support for:

- Tamil
- Telugu
- Kannada
- Malayalam

The project uses the pretrained Hugging Face model:

`Helsinki-NLP/opus-mt-en-dra`

and a locally fine-tuned Tamil model for Tamil translation.

---

## Project Status

### Current working features

- English → Tamil translation
- English → Telugu translation
- English → Kannada translation
- English → Malayalam translation
- Target-language selection
- Streamlit web interface
- CPU-based inference
- Fine-tuned Tamil model support
- Multilingual base-model support
- Dynamic model loading
- Dataset preparation pipeline
- Dataset tokenization pipeline
- Fine-tuning pipeline
- BLEU evaluation
- chrF evaluation
- Backend translation service testing

---

## Current Application Architecture

```text
                         USER
                           │
                           ▼
                  ┌─────────────────┐
                  │ Streamlit UI    │
                  │ frontend/app.py │
                  └────────┬────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ Translation Service  │
                │ translation_service │
                └──────────┬───────────┘
                           │
              ┌────────────┴────────────┐
              │                         │
              ▼                         ▼
     Fine-tuned Tamil Model    Multilingual Base Model
     models/finetuned/final     Helsinki-NLP/opus-mt-en-dra
              │                         │
              ▼                         ▼
            Tamil             Telugu / Kannada / Malayalam
