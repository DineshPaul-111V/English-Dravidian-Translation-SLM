import sys
from pathlib import Path

import streamlit as st


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from backend.app.services.translation_service import (
    translation_service,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="TamilNLP",
    page_icon="🌐",
    layout="centered",
)


# ============================================================
# TITLE
# ============================================================

st.title("TamilNLP")
st.subheader("English → Dravidian Translation SLM")

st.write(
    "Translate English text into Tamil, Telugu, Kannada, "
    "or Malayalam."
)


# ============================================================
# TARGET LANGUAGE
# ============================================================

target_language = st.selectbox(
    "Target Language",
    [
        "Tamil",
        "Telugu",
        "Kannada",
        "Malayalam",
    ],
)


# ============================================================
# INPUT
# ============================================================

english_text = st.text_area(
    "Enter English text",
    placeholder="Example: How are you?",
    height=180,
)


# ============================================================
# TRANSLATE
# ============================================================

if st.button(
    "Translate",
    type="primary",
    use_container_width=True,
):

    if not english_text.strip():

        st.warning(
            "Please enter some English text."
        )

    else:

        try:

            with st.spinner(
                f"Translating to {target_language}..."
            ):

                translated_text = (
                    translation_service.translate(
                        english_text,
                        target_language,
                    )
                )

            st.success(
                "Translation completed."
            )

            st.text_area(
                f"{target_language} Translation",
                value=translated_text,
                height=180,
            )

        except Exception as error:

            st.error(
                f"Translation failed: {error}"
            )


