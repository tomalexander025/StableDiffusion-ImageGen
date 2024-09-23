import streamlit as st
import time
import numpy as np
import torch
from diffusers import StableDiffusionPipeline

# Configuration class
class CFG:
    device = "cuda" if torch.cuda.is_available() else "cpu"  # Use CPU if CUDA is not available
    seed = 42
    generator = torch.Generator().manual_seed(seed)
    image_gen_steps = 35  # Number of steps for image generation
    image_gen_model_id = "stabilityai/stable-diffusion-2"  # Model ID from Hugging Face
    image_gen_size = (400, 400)  # Size of the generated image (width, height)
    image_gen_guidance_scale = 9.0  # Higher values give stronger adherence to the prompt
    prompt_dataset_size = 8  # Size of the prompt dataset
    prompt_max_length = 88  # Max length for the prompt

# Initialize session state for token
if 'token' not in st.session_state:
    st.session_state.token = ""

# Load model
@st.cache_resource
def load_model(token):
    image_gen_model = StableDiffusionPipeline.from_pretrained(
        CFG.image_gen_model_id, variant='fp16', use_auth_token=token
    )
    return image_gen_model.to(CFG.device)

# Streamlit app
st.title("Image Generation with Stable Diffusion")
st.markdown("**LinkedIn:** [Ayush Sharma](https://www.linkedin.com/in/ayush-sharma-a086552b8/)")
st.markdown("**Email:** sharmaayushajay025@gmail.com")

# Input for Hugging Face token
st.session_state.token = st.text_input("Enter your Hugging Face Token:", type="password", value=st.session_state.token)

# Load the model with the token
image_gen_model = load_model(st.session_state.token)

# Prompt and settings
prompt = st.text_input("Enter prompt:", "astronaut in space")
image_gen_steps = st.slider("Image Generation Steps:", min_value=1, max_value=100, value=CFG.image_gen_steps)
guidance_scale = st.slider("Guidance Scale:", min_value=1.0, max_value=20.0, value=CFG.image_gen_guidance_scale)

# Select GPT model
gpt_model = st.selectbox("Select GPT Model:", options=["gpt2", "gpt-3.5"])
if gpt_model == "gpt-3.5":
    prompt_gen_model_id = "gpt-3.5"
else:
    prompt_gen_model_id = "gpt2"

# Generate image button
if st.button("Generate Image"):
    with st.spinner("Generating image..."):
        start_time = time.time()
        image = image_gen_model(
            prompt, num_inference_steps=image_gen_steps,
            generator=CFG.generator,
            guidance_scale=guidance_scale
        ).images[0]

        # Resize the image
        image = image.resize(CFG.image_gen_size)

        # Convert image to NumPy array for display
        image_array = np.array(image)

        # Show image and processing time
        st.image(image_array, caption="Generated Image")
        processing_time = time.time() - start_time
        st.write(f"Processing time: {processing_time:.2f} seconds")

# Notes
st.markdown("""
### <span style='color: blue;'>Notes:</span>
- <span style='color: blue;'><strong>Image Generation Steps:</strong> Controls the number of steps taken to generate the image. Higher values usually result in better quality but take longer to process.</span>
- <span style='color: blue;'><strong>Guidance Scale:</strong> Adjusts how closely the generated image adheres to the prompt. Higher values (e.g., above 7) typically lead to more relevant images but may reduce diversity.</span>
- <span style='color: blue;'><strong>Model Used:</strong> <strong>Stable Diffusion 2:</strong> A state-of-the-art model from Stability AI, suitable for generating high-quality images from text prompts.</span>
- <span style='color: blue;'><strong>Token Input:</strong> Please provide your Hugging Face token to access the models. This token should be kept confidential.</span>
""", unsafe_allow_html=True)
