# 🎙️ Qwen3-TTS — NVIDIA GPU Demo

A simple, local voice-cloning, voice-design, and custom-voice generator powered by [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS), running entirely on your own NVIDIA GPU. No coding experience needed — just follow the steps below.

▶️ **Watch the full setup walkthrough on YouTube:** [Insert your video link here]

<!-- Add a screenshot or GIF of the app here once you have one, e.g.: -->
<!-- ![App screenshot](screenshot.png) -->

> 🍎 On a Mac with Apple Silicon instead of an NVIDIA GPU? Use the companion repo: [qwen3-tts-apple-silicon-demo](#https://github.com/samiu11ah/qwen3-tts-apple-silicon-demo.git)

---

## What can this do?

This app gives you a simple web interface (opens in your browser) with three tools:

- **Clone** — clone any voice from a short reference audio clip (plus its transcript) and make it say anything.
- **VoiceDesign** — describe a voice in plain English ("a warm, older male voice with a British accent") and the model builds it from scratch.
- **CustomVoice** — pick from 9 built-in premium voices, with optional style control (emotion, pacing, tone).

All processing happens **locally on your machine** — nothing is uploaded anywhere.

---

## ✅ Before you start

Open a terminal (Command Prompt) and check each of these:

| Check | Command | If missing |
|---|---|---|
| NVIDIA driver | `nvidia-smi` | Install from [nvidia.com/Download](https://www.nvidia.com/Download/index.aspx), then restart your PC |
| Python 3.10–3.12 | `python --version` | Install from [python.org/downloads](https://www.python.org/downloads/) — tick **"Add Python to PATH"** during install |
| Git | `git --version` | Install from [git-scm.com/downloads](https://git-scm.com/downloads) |
| FFmpeg *(recommended)* | `ffmpeg -version` | `winget install --id Gyan.FFmpeg -e`, then reopen your terminal |

`nvidia-smi` should print a table with your GPU name and driver version — if it doesn't, install the driver first before continuing.

**About FFmpeg:** the app itself doesn't require it directly, but Gradio (the interface library) uses it behind the scenes to handle non-WAV audio uploads (MP3, M4A, etc.) for the Clone tab. If you'll only ever upload `.wav` files you can skip it — otherwise it's worth the one-line install above.

---

## 🚀 Installation

Open a terminal and run these commands one at a time:

```bash
git clone https://github.com/YOUR-USERNAME/qwen3-tts-nvidia-demo.git
cd qwen3-tts-nvidia-demo
python -m venv venv
venv\Scripts\activate
pip install -r requirements_nvidia.txt
python nvidia_app.py
```

*(On Linux/Mac, swap the activate line for `source venv/bin/activate`.)*

Your terminal will print a local address, usually:
```
http://127.0.0.1:7860
```
Open that in your browser — the app loads with three tabs: Clone, VoiceDesign, CustomVoice.

> ⏳ **First run only:** each tab downloads its AI model automatically the first time you use it (roughly 2.5 GB for 0.6B models, 4.5 GB for 1.7B models). This takes a few minutes depending on your internet speed — after that, it's instant.

---

## 🧯 Troubleshooting

- **"⚠️ No CUDA GPU detected"** — your driver may be installed but PyTorch isn't seeing it. Visit [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/), pick your CUDA version, and run the install command it gives you inside your activated `venv`.
- **"Ran out of GPU memory"** — switch to the **0.6B** model size, or shorten your text.
- **Repeated/garbled words when cloning** — double-check the **Reference transcript** field matches your uploaded audio exactly.
- **Model download fails or hangs** — check your internet connection; a restrictive firewall can block Hugging Face downloads.
- **Port 7860 already in use** — close any other Gradio app that's running, or edit the last line of `nvidia_app.py` to use `server_port=7861`.

---

## 📁 What's in this repo

| File | Purpose |
|---|---|
| `nvidia_app.py` | The Gradio app — run this to start |
| `requirements_nvidia.txt` | Python packages needed to run the app |
| `README.md` | This guide |
| `LICENSE` | This repo's open-source license |

---

## 🙏 Credits & Licenses

- Built on [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) by the Qwen team at Alibaba, released under the **Apache License 2.0**. Model weights download automatically from Hugging Face under that same license — see the [model cards](https://huggingface.co/collections/Qwen/qwen3-tts) for details.
- The code in **this repository** is released under the **MIT License** — see [`LICENSE`](LICENSE).

## ⚠️ Responsible use

Voice cloning is powerful — only clone voices you have permission to use (your own voice, or someone who has explicitly consented). Don't use this to impersonate real people without their consent, or to create misleading or deceptive content.
