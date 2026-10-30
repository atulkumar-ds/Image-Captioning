
"""Image Captioning V3 - BLIP fine-tuned on Flickr30k."""

import hashlib
import io
import os
import time
from contextlib import nullcontext
from pathlib import Path

import streamlit as st
import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from transformers import BlipForConditionalGeneration, BlipProcessor


ROOT = Path(__file__).resolve().parent
LOCAL_MODEL_DIR = ROOT / "checkpoints" / "flickr30k_blip"

HF_MODEL_ID = "Atul0/flickr30k-blip-captioning-v3"

st.set_page_config(
    page_title="Image Captioning V3",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def get_hf_token():
    """Read the Hugging Face token from Streamlit secrets or environment."""
    token = os.getenv("HF_TOKEN")

    try:
        token = token or st.secrets.get("HF_TOKEN")
    except Exception:
        # Local development can work without secrets when
        # the local checkpoint is available.
        pass

    return token


@st.cache_resource(show_spinner=False)
def load_model(model_source: str, hf_token: str = ""):
    """Load and cache the selected BLIP model."""

    token = hf_token or None

    processor = BlipProcessor.from_pretrained(
        model_source,
        token=token,
        use_fast=False,
    )

    model = BlipForConditionalGeneration.from_pretrained(
        model_source,
        token=token,
        use_safetensors=True,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model.to(device)
    model.eval()

    return processor, model, device


def get_model_source():
    """Prefer a local checkpoint; otherwise use Hugging Face Hub."""

    if (LOCAL_MODEL_DIR / "config.json").is_file():
        return str(LOCAL_MODEL_DIR), None

    token = get_hf_token()

    if not token:
        raise RuntimeError(
            "The local model checkpoint is unavailable and the Hugging "
            "Face repository is private. Add HF_TOKEN to Streamlit "
            "secrets with read access to the model repository."
        )

    return HF_MODEL_ID, token


def get_autocast(device):
    """Select mixed precision when CUDA is available."""
    if device.type != "cuda":
        return nullcontext()

    dtype = (
        torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16
    )

    return torch.autocast(
        device_type="cuda",
        dtype=dtype,
    )


def generate_caption(
    image,
    processor,
    model,
    device,
    max_tokens,
    num_beams,
):
    """Generate a natural-language caption for an image."""

    inputs = processor(
        images=image,
        return_tensors="pt",
    ).to(device)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.inference_mode():
        with get_autocast(device):
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                num_beams=num_beams,
                do_sample=False,
            )

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    caption = processor.decode(
        output_ids[0],
        skip_special_tokens=True,
    ).strip()

    return caption, elapsed


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:
    st.title("⚙️ Generation Settings")

    max_tokens = st.slider(
        "Maximum caption length",
        min_value=10,
        max_value=60,
        value=30,
        step=5,
    )

    num_beams = st.slider(
        "Beam search width",
        min_value=1,
        max_value=5,
        value=3,
    )

    st.divider()

    st.subheader("Model Information")
    st.write("**Architecture:** BLIP")
    st.write("**Framework:** PyTorch")
    st.write("**Training dataset:** Flickr30k")
    st.write("**Fine-tuning:** 1 full epoch")
    st.write("**Model:** Flickr30k fine-tuned BLIP")

    st.divider()

    st.caption(
        "Version 3 is an independent PyTorch implementation "
        "of the BLIP image-captioning pipeline."
    )


# ==================================================
# MAIN PAGE
# ==================================================

st.title("🖼️ Image Captioning V3")

st.markdown(
    "### Transform images into natural-language descriptions"
)

st.write(
    "Upload any photograph and let the Flickr30k fine-tuned "
    "BLIP vision-language model generate its caption."
)

st.info(
    "The model can describe a wide variety of images, "
    "including photographs outside its training dataset. "
    "Generated captions may occasionally contain inaccuracies."
)

uploaded_file = st.file_uploader(
    "Upload your image",
    type=["jpg", "jpeg", "png", "webp", "bmp"],
)

if uploaded_file is None:

    st.markdown("### Get started")
    st.write("Upload an image to generate its caption.")

    st.markdown(
        """
        **Supported formats:** JPG, JPEG, PNG, WEBP, BMP

        **Model:** BLIP fine-tuned on Flickr30k

        **Inference:** Local GPU or cloud CPU
        """
    )

else:

    image_bytes = uploaded_file.getvalue()
    image_key = hashlib.sha256(image_bytes).hexdigest()

    try:
        with Image.open(io.BytesIO(image_bytes)) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")

    except (UnidentifiedImageError, OSError):
        st.error("Unable to read this image. Please upload another.")
        st.stop()

    preview_col, result_col = st.columns(
        [1.05, 1],
        gap="large",
    )

    with preview_col:

        st.subheader("Uploaded Image")

        st.image(
            image,
            caption=uploaded_file.name,
            width="stretch",
        )

        st.caption(
            f"Dimensions: {image.width} × {image.height} pixels"
        )

    with result_col:

        st.subheader("Generated Caption")

        generate_button = st.button(
            "✨ Generate Caption",
            type="primary",
            width="stretch",
        )

        if generate_button:

            try:
                model_source, token = get_model_source()

                with st.spinner(
                    "Loading BLIP model and generating caption..."
                ):

                    processor, model, device = load_model(
                        model_source,
                        token or "",
                    )

                    caption, elapsed = generate_caption(
                        image=image,
                        processor=processor,
                        model=model,
                        device=device,
                        max_tokens=max_tokens,
                        num_beams=num_beams,
                    )

                st.session_state["caption_result"] = {
                    "image_key": image_key,
                    "caption": caption,
                    "elapsed": elapsed,
                    "device": str(device),
                    "model_source": model_source,
                }

            except Exception as exc:
                st.error("Caption generation failed.")
                st.exception(exc)

        result = st.session_state.get("caption_result")

        if result and result["image_key"] == image_key:

            st.success("Caption generated successfully!")

            st.markdown("#### Caption")

            st.write(result["caption"])

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Inference time",
                    f"{result['elapsed']:.3f} seconds",
                )

            with col2:
                st.metric(
                    "Device",
                    result["device"].upper(),
                )

            st.download_button(
                "⬇️ Download Caption",
                data=result["caption"],
                file_name="generated_caption.txt",
                mime="text/plain",
                width="stretch",
            )

            st.caption(
                f"Model source: {result['model_source']}"
            )

        else:

            st.caption(
                "Your generated caption will appear here."
            )


# ==================================================
# PROJECT INFORMATION
# ==================================================

st.divider()

with st.expander("About Image Captioning V3"):

    st.markdown(
        """
        **Image Captioning V3** is an independent image-captioning
        implementation built using PyTorch and BLIP.

        **Technical details**

        - Vision-language architecture: BLIP
        - Model initialization: Pretrained BLIP base
        - Fine-tuning dataset: Flickr30k
        - Training: PyTorch with mixed precision
        - Evaluation: BLEU, METEOR, ROUGE-L and CIDEr
        - Interface: Streamlit

        The model was fine-tuned using all 29,000 training images
        in Flickr30k, with evaluation on its validation and
        held-out test splits.

        Fine-tuning improves agreement with Flickr30k reference
        captions in our reported evaluation, but does not
        guarantee perfect predictions for every image.
        """
    )

st.caption("Image Captioning V3 | PyTorch • BLIP • Flickr30k • Streamlit")
