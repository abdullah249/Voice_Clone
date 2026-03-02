# Real-Time Voice Cloning Chatbot
# Optimized for multi-GPU server (56 cores, 62GB RAM, 4x GPUs)

import os
import sys
import tempfile
import threading
import numpy as np

# --- Single-instance lock (prevent VS Code terminals from racing) ---
_LOCK_FILE = "/tmp/voice_chat.lock"
def _acquire_lock():
    import fcntl
    try:
        _lock_fd = open(_LOCK_FILE, "w")
        fcntl.flock(_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _lock_fd.write(str(os.getpid()))
        _lock_fd.flush()
        return _lock_fd
    except (IOError, OSError):
        print(f"⚠️ Another instance is already running (lock: {_LOCK_FILE}). Exiting.", flush=True)
        sys.exit(0)
_lock_fd = _acquire_lock()

import torch
import gradio as gr
from flask import Flask, request, jsonify, send_file

# Optimize CPU inference: leverage available cores
torch.set_num_threads(24)
torch.set_num_interop_threads(4)

# Pipeline log file for debugging
import logging
_pipeline_log = logging.getLogger("pipeline")
_pipeline_log.setLevel(logging.DEBUG)
_plh = logging.FileHandler("/tmp/pipeline.log")
_plh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
_pipeline_log.addHandler(_plh)

# Set SoX path (local installation)
SOX_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sox-14.4.2")
os.environ["PATH"] = SOX_PATH + os.pathsep + os.environ.get("PATH", "")

# Groq API key — set via environment variable GROQ_API_KEY
if not os.getenv("GROQ_API_KEY"):
    print("⚠️ GROQ_API_KEY not set — Groq STT/LLM will fail. Export it or add to systemd service.")
    os.environ["GROQ_API_KEY"] = ""

# TTS lock — prevent concurrent GPU access (causes queue backup and OOM)
_tts_lock = threading.Lock()

# ElevenLabs API (cloud TTS for phone pipeline)
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "cjVigY5qzO86Huf0OWal")  # Eric - Smooth, Trustworthy
ELEVENLABS_MODEL = "eleven_turbo_v2_5"

# Global model holders
tts_model = None
whisper_model = None
voice_clone_prompt = None
conversation_history = []

# Path to persist voice clone prompt across restarts
VOICE_PROMPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voice_prompt.pt")

# Directory for all saved voice clones
VOICES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voices")
os.makedirs(VOICES_DIR, exist_ok=True)

# Track active voice by dir_name
ACTIVE_VOICE_FILE = os.path.join(VOICES_DIR, ".active")

def _get_active_voice_name():
    """Get the dir_name of the currently active voice."""
    if os.path.exists(ACTIVE_VOICE_FILE):
        try:
            return open(ACTIVE_VOICE_FILE).read().strip()
        except Exception:
            pass
    return None

def _set_active_voice_name(dir_name):
    """Record the dir_name of the active voice."""
    try:
        with open(ACTIVE_VOICE_FILE, "w") as f:
            f.write(dir_name)
    except Exception:
        pass

import json as _json
import time as _time_mod
import shutil

def _save_voice_to_library(prompt, name=None, source_audio=None):
    """Save a voice clone to the voices library with metadata."""
    ts = _time_mod.strftime("%Y%m%d_%H%M%S")
    if not name:
        name = f"voice_{ts}"
    # Sanitize name for filesystem
    safe_name = "".join(c if c.isalnum() or c in ('-', '_', ' ') else '_' for c in name).strip()
    safe_name = safe_name.replace(' ', '_')
    if not safe_name:
        safe_name = f"voice_{ts}"

    voice_dir = os.path.join(VOICES_DIR, safe_name)
    os.makedirs(voice_dir, exist_ok=True)

    # Save the .pt prompt
    pt_path = os.path.join(voice_dir, "prompt.pt")
    torch.save(prompt, pt_path)

    # Copy source audio if available
    audio_path_saved = None
    if source_audio and os.path.exists(source_audio):
        ext = os.path.splitext(source_audio)[1] or ".wav"
        audio_dest = os.path.join(voice_dir, f"sample{ext}")
        shutil.copy2(source_audio, audio_dest)
        audio_path_saved = f"sample{ext}"

    # Save metadata
    meta = {
        "name": name,
        "created": ts,
        "created_epoch": _time_mod.time(),
        "source_audio": audio_path_saved,
        "model": "Qwen3-TTS-12Hz-1.7B-Base",
    }
    with open(os.path.join(voice_dir, "meta.json"), "w") as f:
        _json.dump(meta, f, indent=2)

    print(f"💾 Voice '{name}' saved to {voice_dir}")
    return safe_name

def _list_voices():
    """List all saved voices from the library."""
    voices = []
    if not os.path.isdir(VOICES_DIR):
        return voices
    for entry in sorted(os.listdir(VOICES_DIR)):
        voice_dir = os.path.join(VOICES_DIR, entry)
        meta_path = os.path.join(voice_dir, "meta.json")
        pt_path = os.path.join(voice_dir, "prompt.pt")
        if os.path.isdir(voice_dir) and os.path.exists(pt_path):
            meta = {"name": entry, "created": "unknown"}
            if os.path.exists(meta_path):
                try:
                    with open(meta_path) as f:
                        meta = _json.load(f)
                except Exception:
                    pass
            meta["dir_name"] = entry
            meta["pt_path"] = pt_path
            meta["has_sample"] = any(
                os.path.exists(os.path.join(voice_dir, f"sample{ext}"))
                for ext in [".wav", ".mp3", ".flac", ".ogg"]
            )
            # Check if this is the active voice
            meta["is_active"] = (_get_active_voice_name() == entry)
            voices.append(meta)
    return voices

def _activate_voice(dir_name):
    """Set a saved voice as the active voice for the pipeline."""
    global voice_clone_prompt
    voice_dir = os.path.join(VOICES_DIR, dir_name)
    pt_path = os.path.join(voice_dir, "prompt.pt")
    if not os.path.exists(pt_path):
        return False, "Voice prompt file not found"
    try:
        voice_clone_prompt = torch.load(pt_path, map_location=_tts_device, weights_only=False)
        # Also update the main voice_prompt.pt (copy, not symlink, for robustness)
        shutil.copy2(pt_path, VOICE_PROMPT_PATH)
        _set_active_voice_name(dir_name)
        print(f"✅ Activated voice: {dir_name}")
        return True, f"Voice '{dir_name}' activated"
    except Exception as e:
        return False, str(e)


# --- Pre-load models at startup ---
import torch as _torch
import gc as _gc
from qwen_tts import Qwen3TTSModel as _Qwen3TTSModel
import whisper as _whisper

# Determine best devices
_has_cuda = _torch.cuda.is_available()
_gpu_count = _torch.cuda.device_count() if _has_cuda else 0
# GPU layout: cuda:0,1 = RTX 4000 (fast, Tensor Cores), cuda:2,3 = P4000 (slow)
# Put TTS on fastest GPU (RTX 4000) for minimum latency
_tts_device = "cuda:0" if _gpu_count >= 1 else "cpu"
_whisper_device = "cuda:1" if _gpu_count >= 2 else ("cuda:0" if _gpu_count >= 1 else "cpu")
print(f"🖥️ Server: {os.cpu_count()} CPU cores, {_gpu_count} GPUs")
print(f"📍 TTS → {_tts_device} (RTX 4000), Whisper → {_whisper_device}")

# Load TTS on fastest GPU — 1.7B model, SDPA attention (Turing-compatible)
print("⏳ Pre-loading TTS 1.7B on RTX 4000 (float16 + SDPA)...")
try:
    tts_model = _Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        device_map=_tts_device,
        dtype=_torch.float16,
        attn_implementation="sdpa",
        local_files_only=True,
    )
    print("✅ TTS 1.7B loaded (float16 + SDPA) on RTX 4000!")
except Exception as _e:
    print(f"❌ TTS load failed: {_e}")
    tts_model = None
_gc.collect()
if _has_cuda:
    _torch.cuda.empty_cache()

# Auto-load saved voice clone prompt (1.7B compatible)
if os.path.exists(VOICE_PROMPT_PATH):
    try:
        voice_clone_prompt = torch.load(VOICE_PROMPT_PATH, map_location=_tts_device, weights_only=False)
        print(f"✅ Voice clone prompt loaded from {VOICE_PROMPT_PATH} → {_tts_device}")
    except Exception as e:
        print(f"⚠️ Could not load saved voice prompt: {e}")

print("⏳ Pre-loading Whisper base on GPU...")
whisper_model = _whisper.load_model("base", device=_whisper_device)
print(f"✅ Whisper base loaded on {_whisper_device}!")

# Warmup TTS with a short phrase (first call has overhead from CUDA kernel caching)
if tts_model is not None and voice_clone_prompt is not None:
    print("⏳ Warming up TTS...")
    try:
        import time as _warmup_time
        _t0 = _warmup_time.time()
        _warmup_wavs, _warmup_sr = tts_model.generate_voice_clone(
            text="Hello.",
            language="english",
            voice_clone_prompt=voice_clone_prompt,
            max_new_tokens=12,
            non_streaming_mode=True,
        )
        del _warmup_wavs
        print(f"✅ TTS warmup done in {_warmup_time.time()-_t0:.1f}s")
    except Exception as _we:
        print(f"⚠️ TTS warmup failed: {_we}")

print("🚀 Models ready — starting Gradio UI...")


def load_whisper(model_size="base"):
    """Load Whisper model on GPU."""
    global whisper_model
    import whisper
    import gc
    gc.collect()
    device = "cuda:1" if torch.cuda.device_count() >= 2 else ("cuda:0" if torch.cuda.is_available() else "cpu")
    whisper_model = whisper.load_model(model_size, device=device)
    return f"✅ Whisper '{model_size}' loaded on {device}!"


def load_tts(model_path="Qwen/Qwen3-TTS-12Hz-1.7B-Base"):
    """Load Qwen3-TTS model on GPU."""
    global tts_model
    import torch
    import gc
    from qwen_tts import Qwen3TTSModel
    
    gc.collect()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16
    
    try:
        tts_model = Qwen3TTSModel.from_pretrained(
            model_path,
            device_map=device,
            dtype=dtype,
            attn_implementation="sdpa",
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return f"✅ TTS '{model_path.split('/')[-1]}' loaded on {device}!"
    except Exception as e:
        gc.collect()
        return f"❌ TTS load failed: {e}"


def _trim_audio(audio_path, max_seconds=15):
    """Trim audio to max_seconds. More reference audio = better cloning quality."""
    import soundfile as sf
    data, sr = sf.read(audio_path)
    duration = len(data) / sr
    if duration <= max_seconds:
        return audio_path, duration
    # Take from the start (usually cleaner speech)
    end_sample = int(max_seconds * sr)
    trimmed = data[:end_sample]
    trimmed_path = audio_path + ".trimmed.wav"
    sf.write(trimmed_path, trimmed, sr)
    return trimmed_path, duration


def create_voice_clone(audio_path, audio_text):
    """Create voice clone prompt."""
    global voice_clone_prompt, tts_model
    import torch
    import gc
    
    if tts_model is None:
        return "❌ Load TTS model first!"
    
    if audio_path is None:
        return "❌ Please upload an audio file first!"
    
    # Clear memory before cloning
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    try:
        # Auto-trim long audio — use up to 15s for better cloning quality
        trimmed_path, original_duration = _trim_audio(audio_path, max_seconds=15)
        trim_msg = ""
        if original_duration > 15:
            trim_msg = f" (auto-trimmed {original_duration:.0f}s → 15s)"
        
        if audio_text and audio_text.strip():
            voice_clone_prompt = tts_model.create_voice_clone_prompt(
                ref_audio=trimmed_path,
                ref_text=audio_text.strip(),
                x_vector_only_mode=False,
            )
        else:
            voice_clone_prompt = tts_model.create_voice_clone_prompt(
                ref_audio=trimmed_path,
                x_vector_only_mode=True,
            )
        
        # Clean up trimmed file
        import os
        if trimmed_path != audio_path and os.path.exists(trimmed_path):
            os.unlink(trimmed_path)
        
        # Clear cache after operation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # Save voice prompt to disk for persistence across restarts
        try:
            torch.save(voice_clone_prompt, VOICE_PROMPT_PATH)
            print(f"💾 Voice prompt saved to {VOICE_PROMPT_PATH}")
        except Exception as save_err:
            print(f"⚠️ Could not save voice prompt: {save_err}")

        # Auto-save to voices library
        try:
            saved_name = _save_voice_to_library(voice_clone_prompt, source_audio=audio_path)
            if saved_name:
                _set_active_voice_name(saved_name)
        except Exception as lib_err:
            print(f"⚠️ Could not save to voice library: {lib_err}")
            saved_name = None

        extra = f" (saved as '{saved_name}')" if saved_name else ""
        return f"✅ Voice cloned!{trim_msg}{extra}"
    except (torch.cuda.OutOfMemoryError, MemoryError):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return "❌ Out of memory! Try using the smaller 0.6B model."
    except Exception as e:
        gc.collect()
        return f"❌ Voice clone failed: {e}"


def speech_to_text(audio_path):
    """Convert speech to text."""
    global whisper_model
    if whisper_model is None:
        return ""
    result = whisper_model.transcribe(audio_path)
    return result["text"].strip()


def get_llm_response(user_message, provider="groq", model="llama-3.1-8b-instant", api_key=""):
    """Get response from LLM API (Groq, OpenAI, or Ollama)."""
    global conversation_history
    import requests
    
    conversation_history.append({"role": "user", "content": user_message})
    
    system = "You are a helpful voice assistant. Keep responses SHORT — 1-2 sentences, under 25 words. Be clear, natural, and conversational. Never use lists or bullet points."
    # Send last 8 messages (4 exchanges) for good conversational context
    recent = conversation_history[-8:]
    messages = [{"role": "system", "content": system}] + recent
    
    try:
        if provider == "groq":
            # Groq API (free tier available: https://console.groq.com/keys)
            if not api_key:
                raise ValueError("Missing Groq API key")
            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": model, "messages": messages, "max_tokens": 60},
                timeout=30
            )
            resp.raise_for_status()
            answer = resp.json()["choices"][0]["message"]["content"]
        
        elif provider == "openai":
            # OpenAI API
            if not api_key:
                raise ValueError("Missing OpenAI API key")
            resp = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": model, "messages": messages, "max_tokens": 60},
                timeout=30
            )
            resp.raise_for_status()
            answer = resp.json()["choices"][0]["message"]["content"]
        
        else:  # ollama
            resp = requests.post(
                "http://localhost:11434/api/chat",
                json={"model": model, "messages": messages, "stream": False},
                timeout=60
            )
            resp.raise_for_status()
            answer = resp.json()["message"]["content"]
    
    except Exception as e:
        answer = f"Error: {e}"
    
    conversation_history.append({"role": "assistant", "content": answer})
    if len(conversation_history) > 20:
        conversation_history = conversation_history[-20:]
    
    return answer


def text_to_speech_cloned(text):
    """Convert text to speech with cloned voice."""
    global tts_model, voice_clone_prompt
    import gc
    
    if tts_model is None or voice_clone_prompt is None:
        return None
    
    gc.collect()
    # Cap text for voice output — 30 words is ~10-15s of speech
    words = text.split()
    if len(words) > 30:
        text = ' '.join(words[:30]) + '...'
    
    # Dynamic max_new_tokens: 12Hz codec = 12 tokens/sec of audio
    # ~15 tokens per word. Min 50, max 400.
    word_count = len(text.split())
    max_tokens = min(max(50, word_count * 15), 400)
    print(f"[TTS] text={text!r} words={word_count} max_tokens={max_tokens}")
    
    wavs, sr = tts_model.generate_voice_clone(
        text=text,
        language="Auto",
        voice_clone_prompt=voice_clone_prompt,
        max_new_tokens=max_tokens,
        non_streaming_mode=True,
    )
    gc.collect()
    return (sr, wavs[0])


def process_voice(audio, provider, model, api_key):
    """Full pipeline: voice -> text -> LLM -> cloned voice."""
    global voice_clone_prompt
    
    if whisper_model is None:
        return None, "❌ Load Whisper first!", "", ""
    if tts_model is None:
        return None, "❌ Load TTS first!", "", ""
    if voice_clone_prompt is None:
        return None, "❌ Clone a voice first!", "", ""
    
    if audio is None:
        return None, "❌ No audio!", "", ""
    
    try:
        import soundfile as sf
        sr, data = audio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, data, sr)
            user_text = speech_to_text(f.name)
            os.unlink(f.name)
        
        if not user_text:
            return None, "❌ Couldn't transcribe", "", ""
        
        ai_response = get_llm_response(user_text, provider=provider, model=model, api_key=api_key)
        audio_out = text_to_speech_cloned(ai_response)
        
        history = "\n".join([f"{'👤 You' if m['role']=='user' else '🤖 AI'}: {m['content']}" for m in conversation_history])
        
        return audio_out, "✅ Done!", user_text, history
    except Exception as e:
        return None, f"❌ {e}", "", ""


def process_text(text, provider, model, api_key):
    """Process text input."""
    global voice_clone_prompt
    
    if tts_model is None:
        return None, "❌ Load TTS first!", ""
    if voice_clone_prompt is None:
        return None, "❌ Clone a voice first!", ""
    
    if not text or not text.strip():
        return None, "❌ Enter text!", ""
    
    try:
        ai_response = get_llm_response(text.strip(), provider=provider, model=model, api_key=api_key)
        audio_out = text_to_speech_cloned(ai_response)
        history = "\n".join([f"{'👤 You' if m['role']=='user' else '🤖 AI'}: {m['content']}" for m in conversation_history])
        return audio_out, "✅ Done!", history
    except Exception as e:
        return None, f"❌ {e}", ""


def clear_history():
    global conversation_history
    conversation_history = []
    return "", "🧹 Cleared!"


# --- Voice dropdown helpers ---
def _get_voice_choices():
    """Return (label, dir_name) tuples for the voice dropdown."""
    voices = _list_voices()
    choices = []
    for v in voices:
        name = v.get("name", v["dir_name"])
        choices.append((name, v["dir_name"]))
    return choices

def _on_voice_selected(dir_name):
    """Activate a voice when user picks it from the dropdown."""
    if not dir_name:
        return "⚠️ No voice selected"
    ok, msg = _activate_voice(dir_name)
    if ok:
        return f"✅ Switched to voice: {dir_name}"
    return f"❌ {msg}"

def _refresh_voices_dropdown():
    """Return updated dropdown choices + current active value."""
    choices = _get_voice_choices()
    active = _get_active_voice_name()
    return gr.Dropdown(choices=choices, value=active)


# Build UI
with gr.Blocks(title="Voice Clone Chat") as demo:
    gr.Markdown("# 🎤 Real-Time Voice Cloning Chatbot")
    
    with gr.Tab("⚙️ Setup"):
        gr.Markdown("### Step 1: Models (auto-loaded at startup)")
        with gr.Row():
            whisper_size = gr.Dropdown(["tiny", "base", "small", "medium"], value="base", label="Whisper Size")
            load_whisper_btn = gr.Button("Reload Whisper")
            whisper_status = gr.Textbox(label="Status", value="✅ Whisper 'base' loaded on GPU!", interactive=False)
        
        with gr.Row():
            tts_path = gr.Dropdown(
                ["Qwen/Qwen3-TTS-12Hz-0.6B-Base", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"],
                value="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                label="TTS Model"
            )
            load_tts_btn = gr.Button("Reload TTS")
            tts_status = gr.Textbox(label="Status", value="✅ TTS '1.7B' loaded on GPU!", interactive=False)
        
        gr.Markdown("### Step 2: Clone Voice")
        clone_audio = gr.Audio(label="Upload Voice Sample (5-10 sec recommended)", type="filepath")
        clone_text = gr.Textbox(label="Transcript (optional)", placeholder="What is said in the audio...")
        clone_btn = gr.Button("Clone Voice")
        gr.Markdown("ℹ️ *Long audio is auto-trimmed to 15s. GPU-accelerated — cloning takes ~5-10s.*")
        clone_status = gr.Textbox(label="Status", interactive=False)
        
        gr.Markdown("### Step 3: LLM Settings")
        gr.Markdown("Get free Groq API key: https://console.groq.com/keys")
        llm_provider = gr.Dropdown(
            ["groq", "openai", "ollama"],
            value="groq",
            label="LLM Provider"
        )
        llm_model = gr.Textbox(
            value="llama-3.1-8b-instant",
            label="Model Name",
            info="Groq: llama-3.1-8b-instant, mixtral-8x7b-32768 | OpenAI: gpt-4o-mini | Ollama: qwen2.5:7b"
        )
        api_key = gr.Textbox(
            value=os.environ.get("GROQ_API_KEY", ""),
            label="API Key (required for Groq/OpenAI)",
            type="password",
            placeholder="Paste your API key here..."
        )
    
    with gr.Tab("🎙️ Chat"):
        with gr.Row():
            voice_selector = gr.Dropdown(
                choices=_get_voice_choices(),
                value=_get_active_voice_name(),
                label="🎭 Select Voice",
                interactive=True,
                scale=4,
            )
            refresh_voices_btn = gr.Button("🔄 Refresh", scale=1)
        with gr.Row():
            with gr.Column():
                voice_input = gr.Audio(label="🎤 Speak", sources=["microphone"], type="numpy")
                voice_btn = gr.Button("Send Voice", variant="primary")
                
                text_input = gr.Textbox(label="💬 Or Type", placeholder="Type here...")
                text_btn = gr.Button("Send Text")
            
            with gr.Column():
                audio_output = gr.Audio(label="🔊 Response", type="numpy")
                status = gr.Textbox(label="Status", interactive=False)
                transcription = gr.Textbox(label="Transcribed", interactive=False)
        
        conversation = gr.Textbox(label="Conversation", lines=10, interactive=False)
        clear_btn = gr.Button("Clear History")
    
    # Events
    load_whisper_btn.click(load_whisper, inputs=[whisper_size], outputs=[whisper_status])
    load_tts_btn.click(load_tts, inputs=[tts_path], outputs=[tts_status])
    clone_btn.click(create_voice_clone, inputs=[clone_audio, clone_text], outputs=[clone_status]).then(
        _refresh_voices_dropdown, outputs=[voice_selector]
    )
    voice_selector.change(_on_voice_selected, inputs=[voice_selector], outputs=[status])
    refresh_voices_btn.click(_refresh_voices_dropdown, outputs=[voice_selector])
    voice_btn.click(process_voice, inputs=[voice_input, llm_provider, llm_model, api_key], outputs=[audio_output, status, transcription, conversation])
    text_btn.click(process_text, inputs=[text_input, llm_provider, llm_model, api_key], outputs=[audio_output, status, conversation])
    clear_btn.click(clear_history, outputs=[conversation, status])


# ============================================================
# Flask API for Asterisk AGI integration (phone call pipeline)
# ============================================================
# Per-call conversation histories keyed by call ID
phone_conversations = {}

flask_app = Flask(__name__)


@flask_app.route("/api/health", methods=["GET"])
def api_health():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "whisper_loaded": whisper_model is not None,
        "tts_loaded": tts_model is not None,
        "voice_cloned": voice_clone_prompt is not None,
    })


@flask_app.route("/api/stt", methods=["POST"])
def api_stt():
    """Speech-to-text: accepts a WAV file, returns transcription."""
    if whisper_model is None:
        return jsonify({"error": "Whisper model not loaded"}), 503

    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["audio"]
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        audio_file.save(f.name)
        tmp_path = f.name

    try:
        text = speech_to_text(tmp_path)
        return jsonify({"text": text})
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@flask_app.route("/api/llm", methods=["POST"])
def api_llm():
    """Get LLM response for a user message. Supports per-call history."""
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    user_text = data["text"]
    call_id = data.get("call_id", "default")

    # Get or create per-call conversation history
    if call_id not in phone_conversations:
        phone_conversations[call_id] = []
    call_history = phone_conversations[call_id]

    call_history.append({"role": "user", "content": user_text})

    system = "You are a helpful voice assistant on a phone call. Keep responses SHORT — 1-2 sentences, under 25 words. Be clear and natural."
    recent = call_history[-8:]
    messages = [{"role": "system", "content": system}] + recent

    import requests as req
    api_key = os.environ.get("GROQ_API_KEY", "")
    try:
        resp = req.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": "llama-3.1-8b-instant", "messages": messages, "max_tokens": 60},
            timeout=30,
        )
        resp.raise_for_status()
        answer = resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        answer = "Sorry, I couldn't process that. Please try again."

    call_history.append({"role": "assistant", "content": answer})
    if len(call_history) > 20:
        phone_conversations[call_id] = call_history[-20:]

    return jsonify({"response": answer, "call_id": call_id})


@flask_app.route("/api/tts", methods=["POST"])
def api_tts():
    """Text-to-speech with cloned voice. Returns WAV file (16kHz)."""
    import gc
    import soundfile as sf

    if tts_model is None:
        return jsonify({"error": "TTS model not loaded"}), 503
    if voice_clone_prompt is None:
        return jsonify({"error": "Voice not cloned yet — go to the Setup tab first"}), 503

    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    text = data["text"]
    # Cap for voice output
    words = text.split()
    if len(words) > 30:
        text = " ".join(words[:30]) + "..."

    word_count = len(text.split())
    max_tokens = min(max(50, word_count * 15), 400)

    try:
        with _tts_lock:
            wavs, sr = tts_model.generate_voice_clone(
                text=text,
                language="Auto",
                voice_clone_prompt=voice_clone_prompt,
                max_new_tokens=max_tokens,
                non_streaming_mode=True,
            )
            gc.collect()

        # Save to temp file and return
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, wavs[0], sr)
            return send_file(f.name, mimetype="audio/wav", as_attachment=True,
                             download_name="response.wav")
    except Exception as e:
        gc.collect()
        return jsonify({"error": str(e)}), 500


@flask_app.route("/api/pipeline", methods=["POST"])
def api_pipeline():
    """Full pipeline: audio file in → voice audio file out.
    Uses cloud APIs (Groq Whisper + Groq LLM + Edge TTS) for speed.
    Falls back to local models only if cloud fails."""
    import gc
    import time as _time
    import subprocess
    import asyncio
    import sys

    t0 = _time.time()
    call_id = request.form.get("call_id", "default")
    print(f"[PIPELINE] ===== START call_id={call_id} =====", flush=True)

    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    # 1. Save uploaded audio
    audio_file = request.files["audio"]
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        audio_file.save(f.name)
        input_path = f.name

    try:
        import requests as req
        api_key = os.environ.get("GROQ_API_KEY", "")

        # 2. STT via Groq Whisper API (cloud — fast, no local CPU)
        t1 = _time.time()
        try:
            with open(input_path, "rb") as af:
                stt_resp = req.post(
                    "https://api.groq.com/openai/v1/audio/transcriptions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    files={"file": ("audio.wav", af, "audio/wav")},
                    data={"model": "whisper-large-v3-turbo"},
                    timeout=30,
                )
                stt_resp.raise_for_status()
                user_text = stt_resp.json()["text"].strip()
        except Exception as e:
            print(f"[PIPELINE] Groq STT failed ({e}), falling back to local Whisper", flush=True)
            # Fallback to local Whisper
            converted_path = input_path + ".16k.wav"
            subprocess.run(
                ["sox", input_path, "-r", "16000", "-c", "1", converted_path],
                check=True, capture_output=True,
            )
            user_text = speech_to_text(converted_path)
            if os.path.exists(converted_path):
                os.unlink(converted_path)

        print(f"[PIPELINE] STT: {_time.time()-t1:.1f}s → '{user_text}'", flush=True)
        if not user_text:
            return jsonify({"error": "Could not transcribe audio"}), 400

        # 3. LLM response via Groq (using per-call history)
        t2 = _time.time()
        if call_id not in phone_conversations:
            phone_conversations[call_id] = []
        call_history = phone_conversations[call_id]
        call_history.append({"role": "user", "content": user_text})

        system = "You are a friendly phone voice assistant having a natural conversation. Keep replies to 1-2 short sentences (around 10-15 words). Be warm, helpful, and conversational. Ask follow-up questions to keep the conversation going. Never give one-word or two-word replies."
        recent = call_history[-8:]
        messages = [{"role": "system", "content": system}] + recent

        try:
            resp = req.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": "llama-3.1-8b-instant", "messages": messages, "max_tokens": 60},
                timeout=30,
            )
            resp.raise_for_status()
            ai_response = resp.json()["choices"][0]["message"]["content"]
        except Exception:
            ai_response = "Sorry, I couldn't process that."

        print(f"[PIPELINE] LLM: {_time.time()-t2:.1f}s → '{ai_response}'", flush=True)

        call_history.append({"role": "assistant", "content": ai_response})
        if len(call_history) > 20:
            phone_conversations[call_id] = call_history[-20:]

        # 4. TTS — use local cloned voice if available, else ElevenLabs/Edge TTS
        t3 = _time.time()
        asterisk_path = tempfile.mktemp(suffix=".wav")
        tts_used = None

        # --- Try local cloned voice first (serialized via lock to prevent GPU contention) ---
        if voice_clone_prompt is not None and tts_model is not None:
            acquired = _tts_lock.acquire(timeout=15)
            if not acquired:
                print(f"[PIPELINE] TTS lock timeout — another request is using GPU, falling back to cloud", flush=True)
            else:
                try:
                    import soundfile as _sf
                    tts_text = ai_response
                    words = tts_text.split()
                    if len(words) > 15:
                        tts_text = " ".join(words[:15])
                    word_count = len(tts_text.split())
                    # 12Hz codec: ~6 tokens per word, cap at 100 for natural phone responses
                    max_tokens = min(max(36, word_count * 6), 100)

                    _tts_t0 = _time.time()
                    wavs, sr = tts_model.generate_voice_clone(
                        text=tts_text,
                        language="english",
                        voice_clone_prompt=voice_clone_prompt,
                        max_new_tokens=max_tokens,
                        non_streaming_mode=True,
                    )
                    _tts_gen_time = _time.time() - _tts_t0
                    # Convert to 8kHz mono for Asterisk
                    _conv_t0 = _time.time()
                    hq_path = tempfile.mktemp(suffix=".wav")
                    _sf.write(hq_path, wavs[0], sr)
                    subprocess.run(
                        ["ffmpeg", "-y", "-i", hq_path,
                         "-ar", "8000", "-ac", "1", "-sample_fmt", "s16", asterisk_path],
                        check=True, capture_output=True,
                    )
                    if os.path.exists(hq_path):
                        os.unlink(hq_path)
                    _conv_time = _time.time() - _conv_t0
                    tts_used = "cloned-voice"
                    print(f"[PIPELINE] Cloned Voice TTS: {_time.time()-t3:.1f}s (gen={_tts_gen_time:.1f}s conv={_conv_time:.1f}s)", flush=True)
                    gc.collect()
                except Exception as e:
                    print(f"[PIPELINE] Cloned voice TTS failed ({e}), falling back to cloud TTS", flush=True)
                    tts_used = None
                finally:
                    _tts_lock.release()

        # --- Fallback: ElevenLabs ---
        if tts_used is None:
            el_mp3_path = tempfile.mktemp(suffix=".mp3")
            try:
                el_resp = req.post(
                    f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}",
                    headers={
                        "xi-api-key": ELEVENLABS_API_KEY,
                        "Content-Type": "application/json",
                        "Accept": "audio/mpeg",
                    },
                    json={
                        "text": ai_response,
                        "model_id": ELEVENLABS_MODEL,
                        "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
                    },
                    timeout=15,
                )
                el_resp.raise_for_status()
                with open(el_mp3_path, "wb") as mp3f:
                    mp3f.write(el_resp.content)
                tts_used = "elevenlabs"
                print(f"[PIPELINE] ElevenLabs TTS: {_time.time()-t3:.1f}s ({len(el_resp.content)} bytes)", flush=True)
            except Exception as e:
                print(f"[PIPELINE] ElevenLabs TTS failed ({e}), falling back to Edge TTS", flush=True)

            # --- Fallback: Edge TTS ---
            if tts_used is None:
                try:
                    import edge_tts as _edge_tts
                    async def _gen():
                        communicate = _edge_tts.Communicate(ai_response, "en-US-GuyNeural")
                        await communicate.save(el_mp3_path)
                    asyncio.run(_gen())
                    tts_used = "edge-tts"
                    print(f"[PIPELINE] Edge TTS fallback: {_time.time()-t3:.1f}s", flush=True)
                except Exception as e2:
                    print(f"[PIPELINE] Edge TTS also failed ({e2})", flush=True)
                    return jsonify({"error": "TTS unavailable"}), 503

            # Convert mp3 → 8kHz 16-bit WAV for Asterisk
            t4 = _time.time()
            subprocess.run(
                ["ffmpeg", "-y", "-i", el_mp3_path,
                 "-ar", "8000", "-ac", "1", "-sample_fmt", "s16", asterisk_path],
                check=True, capture_output=True,
            )
            print(f"[PIPELINE] ffmpeg convert: {_time.time()-t4:.1f}s", flush=True)
            if os.path.exists(el_mp3_path):
                os.unlink(el_mp3_path)

        total = _time.time() - t0
        print(f"[PIPELINE] ===== DONE total={total:.1f}s tts={tts_used} =====", flush=True)

        # Cleanup input file
        if os.path.exists(input_path):
            os.unlink(input_path)

        response = send_file(asterisk_path, mimetype="audio/wav", as_attachment=True,
                         download_name="response.wav")
        response.headers["X-TTS-Engine"] = tts_used or "unknown"
        response.headers["X-AI-Response"] = ai_response[:200]
        return response

    except Exception as e:
        gc.collect()
        print(f"[PIPELINE] ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()
        # Cleanup on error
        for p in [input_path]:
            if os.path.exists(p):
                os.unlink(p)
        return jsonify({"error": str(e)}), 500


@flask_app.route("/api/end_call", methods=["POST"])
def api_end_call():
    """Clean up per-call conversation history."""
    data = request.get_json()
    call_id = data.get("call_id", "default") if data else "default"
    if call_id in phone_conversations:
        del phone_conversations[call_id]
    return jsonify({"status": "ok"})


@flask_app.route("/api/make_call", methods=["POST"])
def api_make_call():
    """Originate an outbound call via ARI.
    POST JSON: {"number": "18573948674"}
    The call goes out via SIP trunk; when answered, ARI voice agent takes over.
    """
    import re as _re
    data = request.get_json()
    if not data or "number" not in data:
        return jsonify({"error": "Missing 'number' field"}), 400
    number = _re.sub(r"[^\d]", "", data["number"])
    if not number:
        return jsonify({"error": "Invalid phone number"}), 400

    # Ensure US number has leading 1
    if len(number) == 10:
        number = "1" + number

    # Provider requires 9871 prefix for US outbound calls
    dial_str = f"9871{number}"

    try:
        resp = req.post(
            "http://127.0.0.1:8088/ari/channels",
            auth=("voiceagent", "voiceagent123"),
            params={
                "endpoint": f"PJSIP/{dial_str}@trunk",
                "app": "voiceagent",
                "callerId": "Voice Agent <1760990923>",
                "timeout": 60,
            },
            timeout=10,
        )
        if resp.status_code in (200, 201):
            ch = resp.json()
            return jsonify({"status": "calling", "channel": ch.get("id"), "number": number})
        else:
            return jsonify({"error": f"ARI error {resp.status_code}", "detail": resp.text[:300]}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@flask_app.route("/api/voices", methods=["GET"])
def api_voices_list():
    """List all saved voice clones."""
    voices = _list_voices()
    return jsonify({"voices": voices})


@flask_app.route("/api/voices/active", methods=["GET"])
def api_voices_active():
    """Get the currently active voice."""
    active = None
    voices = _list_voices()
    for v in voices:
        if v.get("is_active"):
            active = v["dir_name"]
            break
    return jsonify({
        "active": active,
        "voice_loaded": voice_clone_prompt is not None,
    })


@flask_app.route("/api/voices/activate", methods=["POST"])
def api_voices_activate():
    """Set a voice as active for the pipeline."""
    data = request.get_json()
    if not data or "dir_name" not in data:
        return jsonify({"error": "dir_name required"}), 400
    ok, msg = _activate_voice(data["dir_name"])
    if ok:
        return jsonify({"status": "ok", "message": msg})
    return jsonify({"error": msg}), 404


@flask_app.route("/api/voices/rename", methods=["POST"])
def api_voices_rename():
    """Rename a saved voice clone."""
    data = request.get_json()
    if not data or "dir_name" not in data or "new_name" not in data:
        return jsonify({"error": "dir_name and new_name required"}), 400
    old_dir = os.path.join(VOICES_DIR, data["dir_name"])
    if not os.path.isdir(old_dir):
        return jsonify({"error": "Voice not found"}), 404
    safe = "".join(c if c.isalnum() or c in ('-', '_', ' ') else '_' for c in data["new_name"]).strip().replace(' ', '_')
    if not safe:
        return jsonify({"error": "Invalid name"}), 400
    new_dir = os.path.join(VOICES_DIR, safe)
    if os.path.exists(new_dir):
        return jsonify({"error": "Name already exists"}), 409
    os.rename(old_dir, new_dir)
    # Update metadata
    meta_path = os.path.join(new_dir, "meta.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path) as f:
                meta = _json.load(f)
            meta["name"] = data["new_name"]
            with open(meta_path, "w") as f:
                _json.dump(meta, f, indent=2)
        except Exception:
            pass
    return jsonify({"status": "ok", "new_dir_name": safe})


@flask_app.route("/api/voices/delete", methods=["POST"])
def api_voices_delete():
    """Delete a saved voice clone."""
    data = request.get_json()
    if not data or "dir_name" not in data:
        return jsonify({"error": "dir_name required"}), 400
    voice_dir = os.path.join(VOICES_DIR, data["dir_name"])
    if not os.path.isdir(voice_dir):
        return jsonify({"error": "Voice not found"}), 404
    shutil.rmtree(voice_dir)
    return jsonify({"status": "ok"})


@flask_app.route("/api/voices/sample/<dir_name>", methods=["GET"])
def api_voices_sample(dir_name):
    """Serve the voice sample audio file."""
    voice_dir = os.path.join(VOICES_DIR, dir_name)
    if not os.path.isdir(voice_dir):
        return jsonify({"error": "Voice not found"}), 404
    # Find sample file (could be .wav, .mp3, etc.)
    for ext in [".wav", ".mp3", ".flac", ".ogg"]:
        sample_path = os.path.join(voice_dir, f"sample{ext}")
        if os.path.exists(sample_path):
            return send_file(sample_path, mimetype=f"audio/{ext[1:]}")
    return jsonify({"error": "No sample audio"}), 404


@flask_app.route("/api/voices/preview", methods=["POST"])
def api_voices_preview():
    """Generate a TTS preview with a specific voice. Returns WAV audio."""
    import gc
    import soundfile as _sf

    data = request.get_json()
    if not data or "dir_name" not in data:
        return jsonify({"error": "dir_name required"}), 400
    text = data.get("text", "Hello, this is a preview of my cloned voice.")
    voice_dir = os.path.join(VOICES_DIR, data["dir_name"])
    pt_path = os.path.join(voice_dir, "prompt.pt")
    if not os.path.exists(pt_path):
        return jsonify({"error": "Voice prompt not found"}), 404
    if tts_model is None:
        return jsonify({"error": "TTS model not loaded"}), 503

    try:
        prompt = torch.load(pt_path, map_location=_tts_device, weights_only=False)
        words = text.split()
        if len(words) > 20:
            text = " ".join(words[:20])
        word_count = len(text.split())
        max_tokens = min(max(30, word_count * 6), 120)

        wavs, sr = tts_model.generate_voice_clone(
            text=text,
            language="english",
            voice_clone_prompt=prompt,
            max_new_tokens=max_tokens,
            non_streaming_mode=True,
        )
        gc.collect()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            _sf.write(f.name, wavs[0], sr)
            return send_file(f.name, mimetype="audio/wav", as_attachment=True,
                             download_name="preview.wav")
    except Exception as e:
        gc.collect()
        return jsonify({"error": str(e)}), 500


def start_flask_api():
    """Run Flask API server in background thread (threaded for concurrent requests)."""
    flask_app.run(host="127.0.0.1", port=5050, debug=False, use_reloader=False, threaded=True)


if __name__ == "__main__":
    # Start Flask API in background thread (for Asterisk AGI)
    api_thread = threading.Thread(target=start_flask_api, daemon=True)
    api_thread.start()
    print("🌐 Flask API started on http://127.0.0.1:5050")

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
