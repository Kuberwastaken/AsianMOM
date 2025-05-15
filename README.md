# AsianMOM: Artificial Surveillance with Interactive Analysis and Nagging Maternal Oversight Model

<p align="center">
  <img src="https://img.shields.io/badge/No%20Backend-100%25%20Browser-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/ONNX%20Runtime%20Web-%F0%9F%9A%80%20Edge%20AI-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Transformers.js-%F0%9F%A7%AA%20HuggingFace-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/WebGPU%20or%20WASM-Accelerated%20Inference-purple?style=for-the-badge" />
  <img src="https://img.shields.io/badge/AI%20MOM-Nagging%20Guaranteed-red?style=for-the-badge" />
</p>

---
## 👀 What is AsianMOM?

**AsianMOM** is a next-generation, fully client-side AI app that watches you through your webcam and delivers hilarious, culturally-inspired "mom roasts"—all in real time, with zero server or Python dependencies. It leverages the latest in browser-based machine learning: ONNX Runtime Web, Transformers.js, and WebGPU/WebAssembly acceleration. 

---

## ✨ Features

- 🎥 **Live Webcam Feed**: Real-time video capture, processed entirely in your browser.
- 🧠 **Vision Model (SmolVLM)**: Describes your actions using a compact vision-language model.
- 🤖 **Roast Generation (distilgpt2 ONNX)**: Generates witty, context-aware "mom roasts" using a quantized LLM running in-browser.
- 🔊 **Text-to-Speech**: Delivers the roast in a motherly voice using your browser's TTS engine.
- ⚡ **No Backend, No Python, No Gradio**: 100% browser-based. All AI runs locally—your data never leaves your device.
- 🛡️ **Privacy First**: No uploads, no tracking, no cloud inference.

---

## 🧩 Architecture & Smart Browser ML Hacks

### 🖼️ Vision: SmolVLM (HuggingFaceTB/SmolVLM-500M-Instruct)
- **Runs via [Transformers.js](https://github.com/xenova/transformers.js)**, a pure JS port of Hugging Face models.
- Uses WebGPU (if available) or WebAssembly for fast inference.
- Captures webcam frames, processes them as tensors, and generates a one-sentence description.

### 📝 Roast Generation: Xenova/distilgpt2 (ONNX, Quantized)
- **ONNX Runtime Web** loads a quantized GPT-2 model (83MB!) directly in the browser.
- **Tokenization** is handled by Transformers.js for full compatibility.
- **Token-by-token generation**: The ONNX model is called in a loop, each time feeding the growing prompt, to simulate autoregressive text generation.
- **Greedy decoding**: For speed and simplicity, the next token is always the one with the highest probability.
- **WebGPU/WASM fallback**: ONNX Runtime Web will use GPU if available, otherwise falls back to fast WASM.

### 🔥 How WebML/ONNX Runtime Web Works
- **ONNX** (Open Neural Network Exchange) is a portable format for ML models. Quantized models are much smaller and faster, perfect for browsers.
- **ONNX Runtime Web** is a JS runtime that can execute ONNX models using WebAssembly or WebGPU for acceleration.
- **WebGPU** is the next-gen browser graphics API, enabling near-native GPU compute in Chrome/Edge.
- **WebAssembly (WASM)** is a fast, portable binary format for running code in the browser, used as a fallback if WebGPU is not available.

### 🦾 Smart Browser AI Engineering
- **Model Quantization**: Reduces model size and memory, making LLMs feasible in-browser.
- **Streaming Token Generation**: Instead of generating all text at once, the app generates one token at a time, mimicking how LLMs work on the server.
- **Efficient Memory Use**: Only the necessary model weights are loaded, and all computation is done in the browser's memory space.
- **No External Calls**: Once models are loaded, all inference is local—no API keys, no cloud, no privacy risk.

---

## 🛠️ How It Works (Step-by-Step)

```mermaid
flowchart TD
    A[Webcam Frame] --> B[SmolVLM Vision Model (Transformers.js)]
    B --> C[Image Description]
    C --> D[Xenova/distilgpt2 ONNX LLM (ONNX Runtime Web)]
    D --> E[Roast Text]
    E --> F[Browser TTS]
    F --> G[Audio Output]
```

1. **Webcam Capture**: Streams your webcam feed in-browser.
2. **Image Captioning**: SmolVLM generates a description of what you're doing.
3. **Roast Generation**: Xenova/distilgpt2 ONNX model crafts a humorous, mom-style roast based on the caption.
4. **Voice Output**: The browser's TTS reads the roast aloud in a fitting voice.

---

## 🚦 Quickstart

```bash
# 1. Clone or Download this Repository
# 2. Open Test5.html in a modern browser (Chrome/Edge with WebGPU/WebAssembly support recommended)
# 3. Allow webcam access when prompted
# 4. Enjoy being roasted by AsianMOM!
```

---

## 🧬 Models Used

| Purpose         | Model Name & Link                                                                 | Format         | Size   |
|-----------------|-----------------------------------------------------------------------------------|---------------|--------|
| Vision/Caption  | [SmolVLM-500M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct) | Transformers.js| ~1GB   |
| Text/LLM Roast  | [Xenova/distilgpt2 (ONNX, quantized)](https://huggingface.co/Xenova/distilgpt2)      | ONNX (quant)   | ~83MB  |
| TTS             | Browser Built-in SpeechSynthesis API                                              | Native         | -      |

---

## 🧠 Technical Deep Dive

### ONNX Runtime Web Example
```js
const session = await ort.InferenceSession.create('model.onnx', { executionProviders: ['webgpu', 'wasm'] });
const inputTensor = new ort.Tensor('int64', inputIds, [1, inputIds.length]);
const feeds = { input_ids: inputTensor };
const results = await session.run(feeds);
```

### Transformers.js Example
```js
import { AutoProcessor, AutoModelForVision2Seq } from '@huggingface/transformers';
const processor = await AutoProcessor.from_pretrained('HuggingFaceTB/SmolVLM-500M-Instruct');
const model = await AutoModelForVision2Seq.from_pretrained('HuggingFaceTB/SmolVLM-500M-Instruct');
```

### Why This is Cool
- **No server costs**
- **No privacy risk**
- **Works offline after model load**
- **Cutting-edge browser ML**
- **Fun, interactive, and educational!**

---

## 🙏 Credits
- Inspired by classic Asian mom humor and memes
- Powered by [Hugging Face](https://huggingface.co/) models, [ONNX Runtime Web](https://onnxruntime.ai/docs/execution-providers/Web.html), and [Transformers.js](https://github.com/xenova/transformers.js)

---

## ⚠️ Disclaimer
> This app is for entertainment purposes only. Stereotypes are used in a lighthearted, humorous way—please use responsibly and respectfully.

