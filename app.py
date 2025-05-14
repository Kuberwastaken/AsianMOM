import os
import gradio as gr
import torch
import cv2
from PIL import Image
import numpy as np
from transformers import pipeline, AutoProcessor, AutoModelForVision2Seq
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import time

from transformers import BlipProcessor, BlipForConditionalGeneration

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

def initialize_tts_model():
    tts_pipeline = pipeline(
        "text-to-speech", 
        model="parler-tts/parler-tts-mini-expresso"
    )
    return tts_pipeline

def text_to_speech(text, tts_pipeline):
    # Additional prompt to guide the voice style
    styled_text = f"[[voice:female_mature]] [[speed:0.9]] [[precision:0.8]] {text}"
    
    speech = tts_pipeline(styled_text)
    return (speech["sampling_rate"], speech["audio"])

def process_frame(image, vision_components, llm_components, tts_pipeline):
    # Step 1: Analyze what's in the image
    caption = analyze_image(image, vision_components)
    
    # Step 2: Generate roast based on the caption
    roast = generate_roast(caption, llm_components)
    
    # Step 3: Convert roast to speech
    audio = text_to_speech(roast, tts_pipeline)
    
    return caption, roast, audio

def setup_processing_chain(video_feed, analysis_output, roast_output, audio_output):
    # Initialize all models
    vision_components = initialize_vision_model()
    llm_components = initialize_llm()
    tts_pipeline = initialize_tts_model()
    
    last_process_time = time.time() - 10  # Initialize with an offset
    processing_interval = 5  # Process every 5 seconds
    
    def process_webcam(image):
        nonlocal last_process_time
        
        current_time = time.time()
        if current_time - last_process_time >= processing_interval and image is not None:
            last_process_time = current_time
            
            caption, roast, audio = process_frame(
                image, 
                vision_components, 
                llm_components, 
                tts_pipeline
            )
            
            return image, caption, roast, audio
        
        # Return None for outputs that shouldn't update
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
    app = create_app()
    app.launch() 