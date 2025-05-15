---
title: AsianMOM
emoji: 💢
colorFrom: gray
colorTo: red
sdk: gradio
sdk_version: 5.29.1
app_file: app.py
pinned: true
---

# AsianMOM 💢

**AsianMOM** is a fun, interactive Gradio Space that uses your webcam to observe what you're doing and then roasts you like a stereotypical Asian mom—complete with high expectations, cousin comparisons, and slipper threats! 

## 🚀 Features
- **Live Webcam Feed**: Observes your actions in real time.
- **Vision Model**: Describes what it sees using BLIP image captioning.
- **Roast Generation**: Uses Meta's Llama-3.2-1B-Instruct to generate witty, culturally-inspired "mom roasts".
- **Text-to-Speech**: Delivers the roast in a mature, motherly voice using Parler-TTS.
- **Fully Automated**: No button presses needed—just let AsianMOM do her thing!

## 🛠️ How It Works
1. **Webcam Capture**: The app streams your webcam feed.
2. **Image Captioning**: BLIP model generates a description of what you're doing.
3. **Roast Generation**: Llama-3.2-1B-Instruct crafts a humorous, mom-style roast based on the caption.
4. **Voice Output**: Parler-TTS reads the roast aloud in a fitting voice.

## 📦 Setup & Usage
1. **Clone or Fork this Space**
2. Ensure your hardware supports GPU (T4 or better recommended)
3. All dependencies are managed via `requirements.txt`
4. Launch the Space and allow webcam access
5. Enjoy being roasted by AsianMOM!

## 🧩 Models Used
- [BLIP Image Captioning](https://huggingface.co/Salesforce/blip-image-captioning-base)
- [Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)
- [Parler-TTS Mini Expresso](https://huggingface.co/parler-tts/parler-tts-mini-expresso)

## 🙏 Credits
- Inspired by classic Asian mom humor and memes
- Built with [Gradio](https://gradio.app/)
- Powered by Hugging Face models

## ⚠️ Disclaimer
This app is for entertainment purposes only. Stereotypes are used in a lighthearted, humorous way—please use responsibly and respectfully.

