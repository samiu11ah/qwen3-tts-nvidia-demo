"""
Qwen3-TTS Gradio App — NVIDIA / CUDA backend (transformers + qwen-tts package)

Setup:
    python -m venv venv
    source venv/bin/activate        # Windows: venv\\Scripts\\activate
    pip install -r requirements_nvidia.txt
    python nvidia_app.py

Notes:
    - Requires an NVIDIA GPU with CUDA for reasonable performance (CPU will
      technically run but will be very slow).
    - Model weights are downloaded automatically on first use via the
      `qwen_tts` package and cached locally afterwards.
"""

import tempfile

import gradio as gr
import soundfile as sf
import torch

from qwen_tts import Qwen3TTSModel

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LANGUAGES = [
    "Chinese", "English", "Japanese", "Korean", "German",
    "French", "Russian", "Portuguese", "Spanish", "Italian",
]

# Display name (shown in the UI) -> speaker id expected by the model
CUSTOM_VOICE_SPEAKERS = {
    "Serena": "Serena",
    "Uncle Fu": "Uncle_Fu",
    "Vivian": "Vivian",
    "Aiden": "Aiden",
    "Ryan": "Ryan",
    "Ono Anna": "Ono_Anna",
    "Sohee": "Sohee",
    "Dylan": "Dylan",
    "Eric": "Eric",
}

# Repo ids, keyed by model size, for each model family
BASE_MODEL_IDS = {
    "0.6B": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "1.7B": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
}
CUSTOM_VOICE_MODEL_IDS = {
    "0.6B": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "1.7B": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
}
# VoiceDesign is only released at 1.7B, so there is no size dropdown for it.
VOICE_DESIGN_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"

CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = "cuda:0" if CUDA_AVAILABLE else "cpu"
DTYPE = torch.bfloat16 if CUDA_AVAILABLE else torch.float32


# ---------------------------------------------------------------------------
# Model loading — cached so each model is loaded from disk/HF hub only once
# ---------------------------------------------------------------------------

_MODEL_CACHE = {}


def get_model(model_id: str) -> Qwen3TTSModel:
    """Load (or fetch from cache) a Qwen3-TTS model by its repo id."""
    if model_id in _MODEL_CACHE:
        return _MODEL_CACHE[model_id]

    try:
        # Prefer FlashAttention 2 when available for lower memory / faster inference.
        model = Qwen3TTSModel.from_pretrained(
            model_id,
            device_map=DEVICE,
            dtype=DTYPE,
            attn_implementation="flash_attention_2",
        )
    except Exception:
        # flash-attn may not be installed, or the GPU may not support it.
        # Fall back to the default attention implementation.
        model = Qwen3TTSModel.from_pretrained(
            model_id,
            device_map=DEVICE,
            dtype=DTYPE,
        )

    _MODEL_CACHE[model_id] = model
    return model


def _save_wav(wav, sr: int) -> str:
    """Write a generated waveform to a temp .wav file and return its path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, wav, sr)
    return tmp.name


def _friendly_error(e: Exception) -> str:
    """Turn an exception into a short, user-facing message (no traceback)."""
    msg = str(e)
    if "out of memory" in msg.lower():
        return "Ran out of GPU memory. Try the 0.6B model, or shorten the text."
    if "404" in msg or "not a local folder" in msg or "does not appear to have a file" in msg:
        return ("Model weights couldn't be found or downloaded. Check your internet "
                "connection, or that the model name is correct/accessible.")
    if not CUDA_AVAILABLE:
        return "No CUDA GPU was detected — this app is much slower (or may fail) on CPU."
    return f"Generation failed: {msg[:300]}"


# ---------------------------------------------------------------------------
# Tab-specific generation functions
# ---------------------------------------------------------------------------

def generate_clone(model_size, ref_audio, ref_text, text, language):
    """Clone tab: 3-second voice clone from an uploaded reference audio clip.

    A reference transcript is optional here. If provided, it's used for
    higher-fidelity transcript-guided cloning. If left blank, this falls
    back to the officially-supported speaker-embedding-only mode
    (x_vector_only_mode=True) — no transcript needed, but the docs note
    this may reduce quality somewhat.
    """
    if not ref_audio:
        raise gr.Error("Please upload a reference audio clip first.")
    if not text or not text.strip():
        raise gr.Error("Please enter some text for the cloned voice to say.")

    try:
        model = get_model(BASE_MODEL_IDS[model_size])
        if ref_text and ref_text.strip():
            wavs, sr = model.generate_voice_clone(
                text=text,
                language=language,
                ref_audio=ref_audio,
                ref_text=ref_text,
                x_vector_only_mode=False,
            )
        else:
            wavs, sr = model.generate_voice_clone(
                text=text,
                language=language,
                ref_audio=ref_audio,
                x_vector_only_mode=True,
            )
        path = _save_wav(wavs[0], sr)
        return path, path
    except gr.Error:
        raise
    except Exception as e:
        raise gr.Error(_friendly_error(e))


def generate_voice_design(description, text, language):
    """VoiceDesign tab: synthesize speech in a voice described in natural language."""
    if not description or not description.strip():
        raise gr.Error("Please describe the voice you want (age, tone, accent, personality...).")
    if not text or not text.strip():
        raise gr.Error("Please enter some text for the designed voice to say.")

    try:
        model = get_model(VOICE_DESIGN_MODEL_ID)
        wavs, sr = model.generate_voice_design(
            text=text,
            language=language,
            instruct=description,
        )
        path = _save_wav(wavs[0], sr)
        return path, path
    except gr.Error:
        raise
    except Exception as e:
        raise gr.Error(_friendly_error(e))


def generate_custom_voice(model_size, speaker_display, instruct, text, language):
    """CustomVoice tab: one of 9 built-in timbres, with 1.7B-only style control."""
    if not text or not text.strip():
        raise gr.Error("Please enter some text for the voice to say.")

    speaker = CUSTOM_VOICE_SPEAKERS.get(speaker_display, speaker_display)
    # The 0.6B CustomVoice model doesn't support instruction control.
    instruct = instruct if model_size == "1.7B" else None

    try:
        model = get_model(CUSTOM_VOICE_MODEL_IDS[model_size])
        kwargs = dict(text=text, language=language, speaker=speaker)
        if instruct:
            kwargs["instruct"] = instruct
        wavs, sr = model.generate_custom_voice(**kwargs)
        path = _save_wav(wavs[0], sr)
        return path, path
    except gr.Error:
        raise
    except Exception as e:
        raise gr.Error(_friendly_error(e))


# ---------------------------------------------------------------------------
# UI helpers — keep tab layout consistent and easy to extend with new tabs
# ---------------------------------------------------------------------------

def make_output_row():
    """Standard audio-output + download-button pair, shared by every tab."""
    audio_out = gr.Audio(label="Generated speech", type="filepath")
    download_btn = gr.DownloadButton(label="Download .wav", visible=True)
    return audio_out, download_btn


def toggle_instruct(model_size):
    """Show/enable the instruct box only for the 1.7B CustomVoice model."""
    is_large = model_size == "1.7B"
    return (
        gr.update(visible=is_large, interactive=is_large),
        gr.update(visible=not is_large),
    )


# ---------------------------------------------------------------------------
# Gradio UI layout
# ---------------------------------------------------------------------------

with gr.Blocks(title="Qwen3-TTS (NVIDIA / CUDA)") as demo:
    gr.Markdown("# Qwen3-TTS — NVIDIA / CUDA backend")
    gr.Markdown(
        "Voice cloning, voice design, and custom-voice generation, "
        "powered by `transformers` + CUDA."
    )
    if not CUDA_AVAILABLE:
        gr.Markdown(
            "⚠️ **No CUDA GPU detected.** This app will attempt to run on CPU, "
            "which is likely to be extremely slow."
        )

    with gr.Tabs():
        # -------------------------------------------------------------
        # Tab 1: Clone
        # -------------------------------------------------------------
        with gr.Tab("Clone"):
            with gr.Row():
                with gr.Column():
                    clone_size = gr.Dropdown(["0.6B", "1.7B"], value="0.6B", label="Model size")
                    clone_ref_audio = gr.Audio(label="Reference audio (~3 seconds)", type="filepath")
                    clone_ref_text = gr.Textbox(
                        label="Reference transcript (optional)",
                        lines=2,
                        placeholder="Exact transcript of the reference audio — leave blank to clone from timbre only.",
                    )
                    clone_text = gr.Textbox(
                        label="Text to speak", lines=3,
                        placeholder="What should the cloned voice say?",
                    )
                    clone_lang = gr.Dropdown(LANGUAGES, value="English", label="Language")
                    gr.Markdown(
                        "_Tip: adding an accurate transcript improves clone fidelity. Leaving "
                        "it blank still works (speaker-embedding-only mode), but may sound "
                        "slightly less accurate._"
                    )
                    clone_btn = gr.Button("Generate", variant="primary")
                with gr.Column():
                    clone_audio_out, clone_download = make_output_row()

            clone_btn.click(
                fn=generate_clone,
                inputs=[clone_size, clone_ref_audio, clone_ref_text, clone_text, clone_lang],
                outputs=[clone_audio_out, clone_download],
            )

        # -------------------------------------------------------------
        # Tab 2: VoiceDesign
        # -------------------------------------------------------------
        with gr.Tab("VoiceDesign"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("Model: `Qwen3-TTS-12Hz-1.7B-VoiceDesign` (only released at 1.7B).")
                    vd_description = gr.Textbox(
                        label="Voice description",
                        lines=4,
                        placeholder=(
                            "e.g. A warm, mellow older male voice, calm and reassuring, "
                            "slight British accent."
                        ),
                    )
                    vd_text = gr.Textbox(
                        label="Text to speak", lines=3,
                        placeholder="What should the designed voice say?",
                    )
                    vd_lang = gr.Dropdown(LANGUAGES, value="English", label="Language")
                    vd_btn = gr.Button("Generate", variant="primary")
                with gr.Column():
                    vd_audio_out, vd_download = make_output_row()

            vd_btn.click(
                fn=generate_voice_design,
                inputs=[vd_description, vd_text, vd_lang],
                outputs=[vd_audio_out, vd_download],
            )

        # -------------------------------------------------------------
        # Tab 3: CustomVoice
        # -------------------------------------------------------------
        with gr.Tab("CustomVoice"):
            with gr.Row():
                with gr.Column():
                    cv_size = gr.Dropdown(["0.6B", "1.7B"], value="1.7B", label="Model size")
                    cv_speaker = gr.Dropdown(
                        list(CUSTOM_VOICE_SPEAKERS.keys()), value="Serena", label="Timbre"
                    )
                    cv_instruct = gr.Textbox(
                        label="Style instruction (1.7B only)",
                        lines=2,
                        placeholder='e.g. "speak angrily", "whisper", "very slow"',
                        visible=True,
                        interactive=True,
                    )
                    cv_instruct_note = gr.Markdown(
                        "_The 0.6B CustomVoice model doesn't support style instructions "
                        "— switch to 1.7B to use this._",
                        visible=False,
                    )
                    cv_text = gr.Textbox(
                        label="Text to speak", lines=3,
                        placeholder="What should the voice say?",
                    )
                    cv_lang = gr.Dropdown(LANGUAGES, value="English", label="Language")
                    cv_btn = gr.Button("Generate", variant="primary")
                with gr.Column():
                    cv_audio_out, cv_download = make_output_row()

            cv_size.change(
                fn=toggle_instruct,
                inputs=[cv_size],
                outputs=[cv_instruct, cv_instruct_note],
            )

            cv_btn.click(
                fn=generate_custom_voice,
                inputs=[cv_size, cv_speaker, cv_instruct, cv_text, cv_lang],
                outputs=[cv_audio_out, cv_download],
            )

        # -------------------------------------------------------------
        # Add future tabs here, following the same pattern as above:
        #   with gr.Tab("Long-form"): ...
        #   with gr.Tab("Transcribe"): ...
        #   with gr.Tab("Enhance"): ...
        # -------------------------------------------------------------

if __name__ == "__main__":
    demo.queue().launch()
