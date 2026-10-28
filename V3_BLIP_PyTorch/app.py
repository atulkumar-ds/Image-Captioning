
"""Streamlit application for the Flickr30k fine-tuned BLIP model."""

import hashlib
import io
import time
from contextlib import nullcontext
from pathlib import Path

import streamlit as st
import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from transformers import BlipForConditionalGeneration, BlipProcessor


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "checkpoints" / "flickr30k_blip"

st.set_page_config(
    page_title="Image Captioning V3",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner=False)
def load_model():
    """Load and cache the fine-tuned BLIP model."""
    if not (MODEL_DIR / "config.json").is_file():
        raise FileNotFoundError(
            f"Fine-tuned model checkpoint not found: {MODEL_DIR}"
        )

    processor = BlipProcessor.from_pretrained(
        str(MODEL_DIR),
        use_fast=False,
    )

    model = BlipForConditionalGeneration.from_pretrained(
        str(MODEL_DIR)
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model.to(device)
    model.eval()

    return processor, model, device


def generate_caption(image, processor, model, device, max_tokens, num_beams):
    """Generate a caption for a PIL image."""
    inputs = processor(
        images=image,
        return_tensors="pt",
    ).to(device)

    if device.type == "cuda":
        amp_dtype = (
            torch.bfloat16
            if torch.cuda.is_bf16_supported()
            else torch.float16
        )
        autocast = torch.autocast(
            device_type="cuda",
            dtype=amp_dtype,
        )
    else:
        autocast = nullcontext()

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.inference_mode():
        with autocast:
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


# -------------------------
# Sidebar
# -------------------------

with st.sidebar:
    st.title("⚙️ Caption Settings")
    st.caption("Configure the generation process.")

    max_tokens = st.slider(
        "Maximum new tokens",
        min_value=10,
        max_value=60,
        value=30,
        step=5,
        help="Higher values allow longer captions.",
    )

    num_beams = st.slider(
        "Beam search width",
        min_value=1,
        max_value=5,
        value=3,
        help="Higher values explore more candidate captions but can take longer.",
    )

    st.divider()

    st.subheader("Model Information")
    st.write("**Architecture:** BLIP")
    st.write("**Framework:** PyTorch")
    st.write("**Fine-tuning dataset:** Flickr30k")
    st.write(
        "**Device:** "
        + ("CUDA GPU available" if torch.cuda.is_available() else "CPU")
    )

    st.caption(
        "This application uses the fine-tuned checkpoint stored "
        "inside the V3 project."
    )


# -------------------------
# Main application
# -------------------------

st.title("🖼️ Image Captioning V3")
st.markdown(
    "### Turn an image into a natural-language description"
)
st.write(
    "Upload a photograph, generate a caption using your fine-tuned "
    "BLIP model, and download the result."
)

st.info(
    "The model can describe a wide variety of images, but it may "
    "occasionally miss details or generate inaccurate descriptions."
)

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "webp", "bmp"],
    help="Supported formats: JPG, JPEG, PNG, WEBP and BMP.",
)

if uploaded_file is None:
    st.markdown("#### Get started")
    st.write(
        "Choose an image above to see a preview and generate its caption."
    )

else:
    image_bytes = uploaded_file.getvalue()
    image_key = hashlib.sha256(image_bytes).hexdigest()

    try:
        with Image.open(io.BytesIO(image_bytes)) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")

    except (UnidentifiedImageError, OSError):
        st.error(
            "The uploaded file could not be read as a valid image. "
            "Please try another file."
        )
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
            use_container_width=True,
        )
        st.caption(
            f"Image dimensions: {image.width} × {image.height} pixels"
        )

    with result_col:
        st.subheader("Generated Caption")

        generate_button = st.button(
            "✨ Generate Caption",
            type="primary",
            use_container_width=True,
        )

        if generate_button:
            try:
                with st.spinner(
                    "Loading model and generating caption..."
                ):
                    processor, model, device = load_model()

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
                    "model": "Flickr30k fine-tuned BLIP",
                }

            except Exception as exc:
                st.error("Caption generation failed.")
                st.exception(exc)

        result = st.session_state.get("caption_result")

        if result and result["image_key"] == image_key:
            st.success("Caption generated successfully!")
            st.markdown("#### Caption")
            st.write(result["caption"])

            metric_col1, metric_col2 = st.columns(2)

            with metric_col1:
                st.metric(
                    "Inference time",
                    f"{result['elapsed']:.3f} s",
                )

            with metric_col2:
                st.metric("Device", result["device"].upper())

            st.download_button(
                "⬇️ Download Caption",
                data=result["caption"],
                file_name="generated_caption.txt",
                mime="text/plain",
                use_container_width=True,
            )

            st.caption(
                f"Model: {result['model']}"
            )

        else:
            st.caption(
                "Your caption will appear here after you click "
                "'Generate Caption'."
            )

st.divider()

with st.expander("About this project"):
    st.write(
        "Version 3 uses a pretrained BLIP vision-language model "
        "fine-tuned on Flickr30k image-caption pairs. It was built "
        "as an independent PyTorch implementation with its own "
        "dataset pipeline, training scripts, checkpoint and evaluation."
    )
    st.write(
        "The model can generate captions for previously unseen images, "
        "but successful generation does not guarantee perfect accuracy."
    )

st.caption("Image Captioning V3 • PyTorch + BLIP + Streamlit")
