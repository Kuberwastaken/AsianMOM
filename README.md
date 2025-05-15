<h1 align="center">AsianMOM: Artificial Surveillance with Interactive Analysis and Nagging Maternal Oversight Model</h1>

<p align="center">
  <img src="Media/Assets/Readme-image.jpg" alt="AsianMOM Demo" style="max-width: 100%; height: auto;" />
</p>

<p align="center">
  <em>"Aiyah! You still staring at screen? Go study WebGPU, lah!"</em>
</p>

<p align="center">
  <img src="https://img.shields.io/static/v1?label=Kuberwastaken&message=AsianMOM&color=000000&logo=github" alt="Project AsianMOM WebML">
  <img src="https://img.shields.io/badge/Version-21.0-000000" alt="Version 21">
  <img src="https://img.shields.io/badge/License-Apache%202.0-000000" alt="License Apache 2.0">
  <br>
  <a href="https://huggingface.co/docs/transformers.js/index" target="_blank">
    <img src="https://img.shields.io/badge/Transformers.js-%23FFD21F?logo=huggingface&logoColor=black" alt="Transformers.js Documentation">
  </a>
  <a href="https://www.w3.org/TR/webgpu/" target="_blank">
    <img src="https://img.shields.io/badge/WebGPU-%23F34B7D?logo=webgpu&logoColor=white" alt="WebGPU Specification">
  </a>
  <a href="https://huggingface.co/onnx-community" target="_blank">
    <img src="https://img.shields.io/badge/ONNX%20Community-%23FFD21F?logo=huggingface&logoColor=black" alt="ONNX Community on HuggingFace">
  </a>
</p>

## 📖 Project Overview

**AsianMOM** is the result of a **2-day long study** I did to understand WebGPU and Vision models. It is an application of **Web-based Machine Learning (WebML)** to (try to) reacrate the average asian mom level comebacks in a fun little project - on just your web browser alone. This project offers a glimpse into a future where AI is more accessible, private, and democratized than ever before.

(less) hefty backend load and no API calls; AsianMOM processes your camera feed, understands what she "sees," and delivers witty (and brutally honest) roasts, all on your modern web browser.
---

## ✨ Features

*   **💻 Fully Client-Side AI:** Vision, language model processing, and text-to-speech (TTS) all execute directly in your browser. No data is sent to external servers for AI inference.

*   **🔒 Privacy First:** Your camera feed is processed locally, ensuring your data stays with you.

*   **⚡ Real-Time Interaction:** Leverages your device's camera and the Web Speech API for an immediate and engaging experience. 
*(well, kinda, depends on your specs)*

*   **🚀 WebGPU Accelerated:** Harnesses the power of your GPU through WebGPU for efficient and speedy model inference, making complex AI models viable in the browser.

*   **🌍 Zero Installation:** Runs seamlessly in modern web browsers that support WebGPU (e.g., Chrome, Edge). Just open the page and meet AsianMOM!

*   **🎙️ Customizable TTS:** Choose from available system voices on your platform to hear AsianMOM's wisdom in different tones.

*   **😂 The "AsianMOM" Persona:** Experience AI with a personality! hilarious, culturally-infused roasts and "encouragement."

---

## 🛠️ The Development Journey

This project was a rapid exploration, a 2-day deep dive into the evolving landscape of WebML. Here's how it was:

### Day 0: The Spark & Initial Prototyping (`Legacy-Gradio`)

The vision was clear: create an interactive AI persona that could "see" and "speak" entirely on the user's device. At first, I actually wanted it to run on Hugging Face Spaces, since I didn't even know it was possible to run these kinds of models directly in the browser! So, the initial explorations began here

*   **Gradio & Python:** Early prototypes were built using Gradio, a fantastic tool for quickly creating UI for machine learning models. This phase involved experimenting with:

    *   Image captioning models like **BLIP** (evidenced by artifacts in `/Old-Implementations/Legacy-Gradio/BLIP-Image-Captions/`).

    *   TTS solutions like **MeloTTS** (artifacts in `/Old-Implementations/Legacy-Gradio/MeloTTS/`).

*   **Limitations:** While Gradio was excellent for validation, the goal was true client-side, serverless AI. This meant moving beyond Python backends and, eventually, realizing that running everything in the browser was actually possible.

### Day 1: Venturing into the Browser - Vision & Early LLM Hurdles

The challenge shifted to bringing AI models directly into the browser

*   **Client-Side Vision:** The first step was to get a vision model running in the browser. This involved researching lightweight models and ONNX (Open Neural Network Exchange) for browser compatibility.

*   **The LLM Quest - API Detours & ONNX Challenges:**
    *   **API Considerations:** Briefly considered using LLM APIs, but this defeated the core "fully client-side" and "privacy-first" principles of the study.
    
    *   **ONNX LLMs:** Explored various ONNX-compatible LLMs. This phase was fraught with challenges:
        *   **Model Size & Performance:** Many LLMs, even quantized, were too large or slow for a good browser experience.
        *   **Compatibility Issues:** Ensuring seamless operation with ONNX Runtime Web and browser environments wasn't always straightforward.
        *   Several attempts with different small LLMs either failed to load, performed poorly, or lacked the desired conversational ability.

### Day 2: The Breakthrough - Transformers.js, WebGPU, and the Birth of AsianMOM

Things really started coming together once I found the right tools.

*   **Transformers.js - The Game Changer:** The `@huggingface/transformers` library (the browser-friendly version) made it way easier to load and run big models in JavaScript. No more wrestling with complicated setups—just import and go.

*   **WebGPU - Finally, Some Speed:** WebGPU was the missing puzzle piece for performance. With it, the models could actually use the GPU, so things ran fast enough to feel snappy and interactive.

*   **Picking the Right Models:**
    *   **Vision:** I went with `HuggingFaceTB/SmolVLM-500M-Instruct` because it’s small, gives good descriptions, and works well with both Transformers.js and WebGPU. The quantized setup (`embed_tokens: "fp16"`, `vision_encoder: "q4"`, `decoder_model_merged: "q4"`) really helped keep things lightweight.
    *   **Language Model:** For the "roasting" part, I used `onnx-community/Llama-3.2-1B-Instruct-q4f16`. It’s a quantized ONNX model that runs well in the browser and gives decent, funny responses without being too slow.

*   **Making AsianMOM a Character:** Once the tech was sorted, I spent some time tweaking the prompts to give AsianMOM her own unique, naggy personality.

*   **Giving Her a Voice:** I hooked up the Web Speech API for text-to-speech, so AsianMOM could actually talk back.

All in all, after two days of hacking, I had a working prototype: a fun, interactive AI that runs completely in your browser—no servers, no waiting, just instant roasts from AsianMOM.

---

## 💡 How It Works: Under the Hood of AsianMOM

The magic of AsianMOM unfolds through a carefully orchestrated sequence of client-side operations:

```mermaid
graph LR
    A[User Clicks "Roast Me!"] --> B{Camera Access};
    B -- Stream --> C[Capture Image Frame];
    C -- RawImage Data --> D[Image Preprocessing via AutoProcessor];
    D --> E[**SmolVLM-500M-Instruct** <br>(Vision Model on WebGPU) <br><em>Generates Image Description</em>];
    E -- Description Text --> F[Prompt Engineering <br><em>(Combines description with AsianMOM persona)</em>];
    F --> G[**Llama-3.2-1B-Instruct-q4f16** <br>(LLM on WebGPU) <br><em>Generates Roast</em>];
    G -- Roast Text --> H[**Web Speech API** <br>(Text-to-Speech Synthesis)];
    H -- Audio Output --> I[User Hears AsianMOM's Wisdom];
```

**Core Technologies & Components:**

*   **HTML, CSS, JavaScript:** The foundational structure, styling, and client-side logic of the web application.

*   **`@huggingface/transformers` (Transformers.js):** This library is the cornerstone for running both the vision and language models. It handles model loading, tokenization, pre/post-processing, and inference orchestration.

*   **ONNX Runtime Web:** The underlying inference engine that Transformers.js utilizes. It enables execution of ONNX models (like the ones used here) in the browser, with WebGPU as a backend for hardware acceleration.

*   **WebGPU:** The modern graphics and compute API that provides low-level access to the GPU. This is *critical* for running the AI models at an acceptable speed, making the application interactive rather than sluggish.

*   **WebRTC (`navigator.mediaDevices.getUserMedia`):** Used to access the live video feed from the user's camera.

*   **Canvas API:** Temporarily used to capture a still frame from the video feed for the vision model.

*   **Web Speech API (`SpeechSynthesisUtterance`):** Provides native browser capabilities for converting the generated text (the roast) into speech.

**Models in Detail:**

1.  **Vision Model: `HuggingFaceTB/SmolVLM-500M-Instruct`**
    *   **Architecture:** A compact Vision Language Model.
    *   **Quantization:** Utilizes mixed precision/quantization for efficiency:
        *   `embed_tokens: "fp16"`
        *   `vision_encoder: "q4"`
        *   `decoder_model_merged: "q4"`
    *   **Role:** Analyzes the captured image from the camera and generates a textual description of what it "sees" (e.g., "Person sitting in front of a computer").

2.  **Language Model: `onnx-community/Llama-3.2-1B-Instruct-q4f16`**
    *   **Architecture:** A 1-billion parameter instruction-tuned Llama model, quantized for ONNX.
    *   **Quantization:** `q4f16` (4-bit quantization with fp16 for certain parts) provides a significant reduction in model size and computational load while retaining good conversational quality.
    *   **Role:** Takes the description from the vision model as input, along with a carefully crafted system prompt defining the "AsianMOM" persona. It then generates a humorous, nagging, and often brutally honest roast based on the visual context.

---

## 🌍 WebML and WebGPU – What This Means for the World (and Why AsianMOM Is More Than a Meme)

Sure, AsianMOM is a fun project, but what’s really exciting is what it shows about the future of AI and the web. Thanks to new tech like WebML and WebGPU, we’re seeing a big shift in how people everywhere can use and build with artificial intelligence—right in their browsers.

### AI in Your Browser

Not long ago, running advanced AI meant you needed a beefy graphics card or had to send your data off to some company’s cloud. Now, with WebML and WebGPU, all you need is a modern browser. That means anyone with a laptop, tablet, or even a phone can try out cool AI apps—no downloads, no tech headaches. If you can open a website, you can use AI.

### Your Data, Your Device

A lot of AI tools send your info to remote servers for processing. With WebML and WebGPU, everything happens right on your device. In AsianMOM, for example, your camera feed and “roast” never leave your computer. That’s a big win for privacy and peace of mind.

### Cheaper, Greener, and More Open

Running AI in the browser means less need for expensive servers and less energy use. Developers (even solo ones!) can build and share creative tools without breaking the bank. It’s not just for tech giants anymore—anyone can join in.

### Fast, Responsive, and Even Works Offline

Because all the magic happens locally, you get instant feedback and smooth experiences. Sometimes, you can even use these tools without internet. Imagine playing with AI-powered apps on a plane or somewhere with spotty Wi-Fi—no problem.

### Lowering the Barriers for Developers

Before, building something like AsianMOM took deep AI knowledge and lots of backend work. Now, web developers can add smart features to their sites with just a bit of extra code. This means more creativity and more voices in the AI space.

### A Sneak Peek at the Future

When you use AsianMOM, you’re not just getting roasted—you’re seeing a preview of a world where powerful, private, and fun AI is available to everyone, everywhere. The same tech could power language learning, accessibility tools, games, art, and way more.

In short: WebML and WebGPU are making AI more accessible, private, and open than ever before. AsianMOM is just one example of what’s possible—and there’s a lot more to come.

---

## 🚀 Getting Started with AsianMOM

Want to experience AsianMOM's wisdom firsthand or peek under the hood?

**Prerequisites:**

*   A **modern web browser** with **WebGPU enabled**.

    *   **Google Chrome:** Version 113 or later (WebGPU is enabled by default).
    *   **Microsoft Edge:** Version 113 or later (WebGPU is enabled by default).
    *   You can verify WebGPU support at [webgpureport.org](https://webgpureport.org/).
*   A **webcam** connected to your device.

*   **Patience:** The models are loaded on-demand the first time and might take a moment depending on your internet speed and device.

**Running the App:**

Just download the index.html... really, it's that simple.

---

## Future Todos & Potential Enhancements

This 2-day study laid a strong foundation. Future iterations could explore:

*   **Better TTS Model:** Integrate a more natural-sounding, client-side TTS engine if memory and performance constraints allow. Web Speech API is convenient but voice quality varies by platform.

*   **Better Prompt Engineering:** Further refine and expand the prompt set for AsianMOM to increase the variety and comedic impact of her roasts. Perhaps even allow user-customizable persona elements.

*   **Smaller/Faster Models:** Continuously evaluate emerging, even more efficient quantized vision and language models optimized for WebGPU.

*   **⚙️ Performance Optimizations:** Dive deeper into WebGPU specifics to squeeze out more performance, perhaps through custom shaders or more fine-grained model loading.

*   **🤳 Expanded Interactivity:** Introduce more complex scenarios or allow users to "talk back" (though AsianMOM always gets the last word!).

*   **📊 Benchmarking:** Conduct formal performance benchmarks across different devices and browsers.

---

## 📜 License

This project is licensed under the **Apache License 2.0**.  
See the full license [here](https://github.com/Kuberwastaken/AsianMOM/blob/main/LICENSE).

---

## 🙏 Acknowledgements & Credits

*   **Hugging Face:** For their incredible `Transformers.js` library and for hosting a vast array of models.
*   **ONNX Community & Microsoft:** For the ONNX format and ONNX Runtime, enabling cross-platform model deployment.
*   **The WebGPU Working Group & Browser Vendors:** For developing and implementing the WebGPU API, making this class of application possible.
*   **All the researchers and developers** who create and open-source the underlying AI models.
*   And, of course, to all the **AsianMOMs** out there for their timeless wisdom and inspiration!

---

Made with <3 by [Kuber Mehta](https://x.com/Kuberwastaken)

