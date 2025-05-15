import os
import gradio as gr
import torch
import cv2
from PIL import Image
import numpy as np
from transformers import pipeline, AutoProcessor, AutoModelForVision2Seq
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import time
import nltk
import io

from transformers import BlipProcessor, BlipForConditionalGeneration
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoFeatureExtractor, set_seed
from transformers.models.speecht5.number_normalizer import EnglishNumberNormalizer
from string import punctuation
import re

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def initialize_vision_model():
    # Using BLIP for image captioning - lightweight but effective
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    
    return {
        "processor": processor,
        "model": model
    }

def analyze_image(image, vision_components):
    processor = vision_components["processor"]
    model = vision_components["model"]
    
    # Convert to RGB if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    inputs = processor(image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=30)
    
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    return caption

def initialize_llm():
    model_id = "meta-llama/Llama-3.2-1B-Instruct"
    hf_token = os.environ.get("HF_TOKEN")

    # Load and patch config
    config = AutoConfig.from_pretrained(model_id, token=hf_token)
    if hasattr(config, "rope_scaling"):
        rope_scaling = config.rope_scaling
        if isinstance(rope_scaling, dict):
            config.rope_scaling = {
                "type": rope_scaling.get("type", "linear"),
                "factor": rope_scaling.get("factor", 1.0)
            }

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=hf_token
    )
    return {
        "model": model,
        "tokenizer": tokenizer
    }

def generate_roast(caption, llm_components):
    model = llm_components["model"]
    tokenizer = llm_components["tokenizer"]
    
    prompt = f"""[INST] You are AsianMOM, a stereotypical Asian mother who always has high expectations. \nYou just observed your child doing this: \"{caption}\"\n    \nRespond with a short, humorous roast (maximum 2-3 sentences) in the style of a stereotypical Asian mother. \nInclude at least one of these elements:\n- Comparison to more successful relatives/cousins\n- High expectations about academic success\n- Mild threats about using slippers\n- Questioning life choices\n- Asking when they'll get married or have kids\n- Commenting on appearance\n- Saying \"back in my day\" and describing hardship\n\nBe funny but not hurtful. Keep it brief. [/INST]"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=300,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract just the response part, not the prompt
    response = response.split("[/INST]")[1].strip()
    
    return response

# Parler-TTS setup
parler_device = "cuda:0" if torch.cuda.is_available() else "cpu"
parler_repo_id = "parler-tts/parler-tts-mini-expresso"
parler_model = ParlerTTSForConditionalGeneration.from_pretrained(parler_repo_id).to(parler_device)
parler_tokenizer = AutoTokenizer.from_pretrained(parler_repo_id)
parler_feature_extractor = AutoFeatureExtractor.from_pretrained(parler_repo_id)
PARLER_SAMPLE_RATE = parler_feature_extractor.sampling_rate
PARLER_SEED = 42
parler_number_normalizer = EnglishNumberNormalizer()

def parler_preprocess(text):
    text = parler_number_normalizer(text).strip()
    if text and text[-1] not in punctuation:
        text = f"{text}."
    abbreviations_pattern = r'\b[A-Z][A-Z\.]+\b'
    def separate_abb(chunk):
        chunk = chunk.replace(".", "")
        return " ".join(chunk)
    abbreviations = re.findall(abbreviations_pattern, text)
    for abv in abbreviations:
        if abv in text:
            text = text.replace(abv, separate_abb(abv))
    return text

def text_to_speech(text):
    # Asian mom nagging style description
    description = ("Elisabeth speaks in a mature, strict, nagging, and slightly disappointed tone, "
                   "with a hint of love and high expectations, at a moderate pace with high quality audio. "
                   "She sounds like a stereotypical Asian mother who compares you to your cousins, "
                   "questions your life choices, and threatens you with a slipper, but ultimately wants the best for you.")
    inputs = parler_tokenizer(description, return_tensors="pt").to(parler_device)
    prompt = parler_tokenizer(parler_preprocess(text), return_tensors="pt").to(parler_device)
    set_seed(PARLER_SEED)
    generation = parler_model.generate(input_ids=inputs.input_ids, prompt_input_ids=prompt.input_ids)
    audio_arr = generation.cpu().numpy().squeeze()
    return (PARLER_SAMPLE_RATE, audio_arr)

def process_frame(image, vision_components, llm_components):
    caption = analyze_image(image, vision_components)
    roast = generate_roast(caption, llm_components)
    audio = text_to_speech(roast)
    return caption, roast, audio

def setup_processing_chain(video_feed, analysis_output, roast_output, audio_output):
    vision_components = initialize_vision_model()
    llm_components = initialize_llm()
    last_process_time = time.time() - 10
    processing_interval = 5
    def process_webcam(image):
        nonlocal last_process_time
        current_time = time.time()
        if current_time - last_process_time >= processing_interval and image is not None:
            last_process_time = current_time
            caption, roast, audio = process_frame(
                image,
                vision_components,
                llm_components
            )
            return image, caption, roast, audio
        return image, None, None, None
    video_feed.change(
        process_webcam,
        inputs=[video_feed],
        outputs=[video_feed, analysis_output, roast_output, audio_output]
    )

def create_app():
    with gr.Blocks(theme=gr.themes.Soft()) as app:
        gr.Markdown("# AsianMOM: Asian Mother Observer & Mocker")
        gr.Markdown("### Camera captures what you're doing and your Asian mom responds appropriately")
        
        with gr.Row():
            with gr.Column():
                video_feed = gr.Image(sources=["webcam"], streaming=True, label="Camera Feed")
            
            with gr.Column():
                analysis_output = gr.Textbox(label="What AsianMOM Sees", lines=2)
                roast_output = gr.Textbox(label="AsianMOM's Thoughts", lines=4)
                audio_output = gr.Audio(label="AsianMOM Says", autoplay=True)
        
        # Setup the processing chain
        setup_processing_chain(video_feed, analysis_output, roast_output, audio_output)
                
    return app

if __name__ == "__main__":
    os.system('python -m unidic download')
    nltk.download('averaged_perceptron_tagger_eng')
    app = create_app()
    app.launch() 