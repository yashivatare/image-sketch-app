import io
from typing import Optional

import streamlit as st
from PIL import Image

from sketch_generator import SketchGenerator


@st.cache_resource(show_spinner="Loading AI model... this can take a minute on first run.")
def load_sketch_generator() -> SketchGenerator:
    """Loads and caches the SketchGenerator model."""
    return SketchGenerator()


def read_uploaded_image(uploaded_file) -> Optional[Image.Image]:
    """Reads an uploaded file and converts it to a PIL Image."""
    if uploaded_file is None:
        return None
    bytes_data = uploaded_file.getvalue()
    return Image.open(io.BytesIO(bytes_data)).convert("RGB")


def main() -> None:
    st.set_page_config(page_title="Fast Image → Sketch (LCM)", layout="wide")
    st.title("⚡ Fast Image → Sketch with ControlNet + LCM")
    st.write(
        "This app uses a Latent Consistency Model (LCM) for extremely fast generation. "
        "High-quality sketches can be generated in just 4-8 steps!"
    )

    gen = load_sketch_generator()

    left_col, right_col = st.columns([1, 1])

    with left_col:
        uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp"])
        
        # --- CHANGE: Removed hardcoded 'value' and replaced with 'placeholder' ---
        prompt = st.text_input(
            "Style prompt",
            placeholder="e.g., A vibrant cartoon, Pixar style OR charcoal sketch, grainy",
        )
        negative_prompt = st.text_input(
            "Negative prompt (optional)",
            placeholder="e.g., photorealistic, blurry, noisy, grayscale",
        )
        
        with st.expander("Advanced Settings"):
            strength = st.slider(
                "Strength", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.6, 
                step=0.05,
                help="Controls how much the original image is changed. Higher values are more stylized."
            )
            steps = st.slider(
                "Steps", 
                min_value=1, 
                max_value=12, 
                value=4, 
                step=1,
                help="LCMs work best with very few steps (4-8)."
            )
            guidance = st.slider(
                "Guidance scale", 
                min_value=1.0, 
                max_value=3.0, 
                value=1.0, 
                step=0.1,
                help="LCMs work best with low guidance (1.0-2.0)."
            )
            conditioning = st.slider("ControlNet conditioning scale", min_value=0.1, max_value=2.0, value=1.0, step=0.05)
            seed = st.text_input("Seed (optional, integer)", value="")
            
        run = st.button("Generate Sketch", type="primary")

    with right_col:
        st.subheader("Input Image")
        if uploaded is not None:
            st.image(uploaded, use_container_width=True)
        else:
            st.info("Upload an image to get started.")

    if run:
        image = read_uploaded_image(uploaded)
        if image is None:
            st.warning("Please upload an image first.")
            return

        with st.spinner("Generating sketch... this will be quick!"):
            out_path = gen.generate_sketch(
                image,
                prompt,
                negative_prompt=negative_prompt.strip(),
                strength=float(strength),
                num_inference_steps=int(steps),
                guidance_scale=float(guidance),
                controlnet_conditioning_scale=float(conditioning),
                seed=int(seed) if seed.strip().isdigit() else None,
            )
        
        result_img = Image.open(out_path)
        
        right_col.subheader("Generated Sketch")
        right_col.image(result_img, use_container_width=True)
        right_col.caption(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()