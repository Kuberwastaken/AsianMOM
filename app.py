import os
import gradio as gr
import torch
import cv2
from PIL import Image
import numpy as np
from transformers import AutoProcessor, AutoModelForVision2Seq
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
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    try:
        inputs = processor(image, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=30)
        caption = processor.decode(outputs[0], skip_special_tokens=True)
        return caption if isinstance(caption, str) else ""
    except Exception as e:
        print(f"Error in analyze_image: {str(e)}")
        return "" # Return empty string on error

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
    try:
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
        response = response.split("[/INST]")[1].strip()
        return response if isinstance(response, str) else ""
    except Exception as e:
        print(f"Error in generate_roast: {str(e)}")
        return "" # Return empty string on error

# Parler-TTS setup
def setup_tts():
    try:
        parler_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        parler_repo_id = "parler-tts/parler-tts-mini-expresso"
        parler_model = ParlerTTSForConditionalGeneration.from_pretrained(parler_repo_id).to(parler_device)
        parler_tokenizer = AutoTokenizer.from_pretrained(parler_repo_id)
        parler_feature_extractor = AutoFeatureExtractor.from_pretrained(parler_repo_id)
        PARLER_SAMPLE_RATE = parler_feature_extractor.sampling_rate
        PARLER_SEED = 42
        parler_number_normalizer = EnglishNumberNormalizer()
        
        return {
            "model": parler_model,
            "tokenizer": parler_tokenizer,
            "feature_extractor": parler_feature_extractor,
            "sample_rate": PARLER_SAMPLE_RATE,
            "seed": PARLER_SEED,
            "number_normalizer": parler_number_normalizer,
            "device": parler_device
        }
    except Exception as e:
        print(f"Error setting up TTS: {str(e)}")
        return None

def parler_preprocess(text, number_normalizer):
    text = number_normalizer(text).strip()
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

def text_to_speech(text, tts_components):
    if tts_components is None:
        return (16000, np.zeros(1))  # Default sample rate if components failed to load
    
    model = tts_components["model"]
    tokenizer = tts_components["tokenizer"]
    device = tts_components["device"]
    sample_rate = tts_components["sample_rate"]
    seed = tts_components["seed"]
    number_normalizer = tts_components["number_normalizer"]
    
    description = ("Elisabeth speaks in a mature, strict, nagging, and slightly disappointed tone, "
                  "with a hint of love and high expectations, at a moderate pace with high quality audio. "
                  "She sounds like a stereotypical Asian mother who compares you to your cousins, "
                  "questions your life choices, and threatens you with a slipper, but ultimately wants the best for you.")
    if not text or not isinstance(text, str):
        return (sample_rate, np.zeros(1))
    try:
        inputs = tokenizer(description, return_tensors="pt").to(device)
        prompt = tokenizer(parler_preprocess(text, number_normalizer), return_tensors="pt").to(device)
        set_seed(seed)
        generation = model.generate(input_ids=inputs.input_ids, prompt_input_ids=prompt.input_ids)
        audio_arr = generation.cpu().numpy().squeeze()
        return (sample_rate, audio_arr)
    except Exception as e:
        print(f"Error in text_to_speech: {str(e)}")
        return (sample_rate, np.zeros(1))

def process_frame(image, vision_components, llm_components, tts_components):
    try:
        caption = analyze_image(image, vision_components)
        roast = generate_roast(caption, llm_components)
        
        default_sample_rate = 16000
        if tts_components is not None:
            default_sample_rate = tts_components["sample_rate"]
            
        if not roast or not isinstance(roast, str):
            audio = (default_sample_rate, np.zeros(1))
        else:
            audio = text_to_speech(roast, tts_components)
        return caption, roast, audio
    except Exception as e:
        print(f"Error in process_frame: {str(e)}")
        return "", "", (default_sample_rate, np.zeros(1))

def create_app():
    try:
        # Initialize components before creating the app
        vision_components = initialize_vision_model()
        tts_components = setup_tts()
        
        # Try to initialize LLM with Hugging Face token
        hf_token = os.environ.get("HF_TOKEN")
        llm_components = None
        if hf_token:
            try:
                llm_components = initialize_llm()
            except Exception as e:
                print(f"Error initializing LLM: {str(e)}. Will use fallback.")
                
        # Fallback if LLM initialization failed
        if llm_components is None:
            def fallback_generate_roast(caption, _):
                return f"I see you {caption}. Why you not doctor yet? Your cousin studying at Harvard!"
            
            llm_components = {"generate_roast": fallback_generate_roast}
        
        # Set initial values and processing parameters
        last_process_time = time.time() - 10
        processing_interval = 5
        
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
            
            # Define processing function
            def process_webcam(image):
                nonlocal last_process_time
                current_time = time.time()
                default_caption = "" 
                default_roast = ""
                default_sample_rate = 16000
                if tts_components is not None:
                    default_sample_rate = tts_components["sample_rate"]
                default_audio = (default_sample_rate, np.zeros(1))
                
                if current_time - last_process_time >= processing_interval and image is not None:
                    last_process_time = current_time
                    try:
                        caption, roast, audio = process_frame(
                            image,
                            vision_components,
                            llm_components,
                            tts_components
                        )
                        final_caption = caption if isinstance(caption, str) else default_caption
                        final_roast = roast if isinstance(roast, str) else default_roast
                        final_audio = audio if isinstance(audio, tuple) and len(audio) == 2 and isinstance(audio[1], np.ndarray) else default_audio
                        return image, final_caption, final_roast, final_audio
                    except Exception as e:
                        print(f"Error in process_webcam: {str(e)}")
                return image, default_caption, default_roast, default_audio
            
            # Setup the processing chain
            video_feed.change(
                process_webcam,
                inputs=[video_feed],
                outputs=[video_feed, analysis_output, roast_output, audio_output]
            )
            
        return app
    except Exception as e:
        print(f"Error creating app: {str(e)}")
        # Create a fallback simple app that reports the error
        with gr.Blocks() as fallback_app:
            gr.Markdown("# AsianMOM: Error Initializing")
            gr.Markdown(f"Error: {str(e)}")
            gr.Markdown("Please check your environment setup and try again.")
        return fallback_app

if __name__ == "__main__":
    try:
        # Download required resources
        os.system('python -m unidic download')
        nltk.download('averaged_perceptron_tagger_eng')
        
        # Create and launch app
        app = create_app()
        app.launch(share=True, debug=True)
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        # If all else fails, create a minimal app
        with gr.Blocks() as minimal_app:
            gr.Markdown("# AsianMOM: Fatal Error")
            gr.Markdown(f"Fatal error: {str(e)}")
        minimal_app.launch(share=True)