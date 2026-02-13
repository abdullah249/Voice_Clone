# Simple Real-Time Voice Cloning Chatbot
# Lighter version - loads models on demand

import os
import tempfile
import threading
import numpy as np
import torch
import gradio as gr
from flask import Flask, request, jsonify, send_file

# Optimize CPU inference: use both vCPUs
torch.set_num_threads(2)
torch.set_num_interop_threads(1)

# Set SoX path (local installation)
SOX_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sox-14.4.2")
os.environ["PATH"] = SOX_PATH + os.pathsep + os.environ.get("PATH", "")

# API keys — set via environment variables (never hardcode secrets)
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")

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


# --- Pre-load models at startup ---
print("⏳ Pre-loading Whisper tiny...")
import whisper as _whisper
whisper_model = _whisper.load_model("tiny")
print("✅ Whisper tiny loaded!")

print("⏳ Pre-loading TTS 0.6B (this takes a few minutes on CPU)...")
import torch as _torch
from qwen_tts import Qwen3TTSModel as _Qwen3TTSModel
_device = "cuda:0" if _torch.cuda.is_available() else "cpu"
tts_model = _Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    device_map=_device,
    dtype=_torch.bfloat16,
    attn_implementation="sdpa",
)
import gc; gc.collect()
print("✅ TTS 0.6B loaded!")

# Auto-load saved voice clone prompt if available
if os.path.exists(VOICE_PROMPT_PATH):
    try:
        voice_clone_prompt = torch.load(VOICE_PROMPT_PATH, map_location="cpu", weights_only=False)
        print(f"✅ Voice clone prompt loaded from {VOICE_PROMPT_PATH}")
    except Exception as e:
        print(f"⚠️ Could not load saved voice prompt: {e}")

print("🚀 Models ready — starting Gradio UI...")


def load_whisper(model_size="tiny"):
    """Load Whisper model."""
    global whisper_model
    import whisper
    import gc
    gc.collect()
    whisper_model = whisper.load_model(model_size)
    return f"✅ Whisper '{model_size}' loaded!"


def load_tts(model_path="Qwen/Qwen3-TTS-12Hz-0.6B-Base"):
    """Load Qwen3-TTS model."""
    global tts_model
    import torch
    import gc
    from qwen_tts import Qwen3TTSModel
    
    gc.collect()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16  # Always use bfloat16 to save memory (works on CPU too)
    
    try:
        tts_model = Qwen3TTSModel.from_pretrained(
            model_path,
            device_map=device,
            dtype=dtype,
            attn_implementation="sdpa",
        )
        gc.collect()
        return f"✅ TTS '{model_path.split('/')[-1]}' loaded!"
    except Exception as e:
        gc.collect()
        return f"❌ TTS load failed: {e}"


def _trim_audio(audio_path, max_seconds=5):
    """Trim audio to max_seconds. Voice cloning only needs 5s of clear speech."""
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
        # Auto-trim long audio — voice cloning only needs ~5 seconds
        trimmed_path, original_duration = _trim_audio(audio_path, max_seconds=5)
        trim_msg = ""
        if original_duration > 5:
            trim_msg = f" (auto-trimmed {original_duration:.0f}s → 5s)"
        
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

        return f"✅ Voice cloned!{trim_msg}"
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
    
    system = "You are a helpful voice assistant. Keep responses VERY short - maximum 1-2 sentences, under 20 words. Be brief."
    # Only send last 4 messages (2 exchanges) to keep LLM context small & responses fresh
    recent = conversation_history[-4:]
    messages = [{"role": "system", "content": system}] + recent
    
    try:
        if provider == "groq":
            # Groq API (free tier available: https://console.groq.com/keys)
            if not api_key:
                raise ValueError("Missing Groq API key")
            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": model, "messages": messages, "max_tokens": 40},
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
                json={"model": model, "messages": messages, "max_tokens": 40},
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
    # Truncate text to ~20 words max to keep TTS generation fast on CPU
    words = text.split()
    if len(words) > 20:
        text = ' '.join(words[:20]) + '...'
    
    # Dynamic max_new_tokens: 12Hz codec = 12 tokens/sec of audio
    # ~20 tokens per word is generous. Min 50, max 200.
    word_count = len(text.split())
    max_tokens = min(max(50, word_count * 20), 200)
    print(f"[TTS] text={text!r} words={word_count} max_tokens={max_tokens}")
    
    wavs, sr = tts_model.generate_voice_clone(
        text=text,
        language="Auto",
        voice_clone_prompt=voice_clone_prompt,
        max_new_tokens=max_tokens,
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


# Build UI
with gr.Blocks(title="Voice Clone Chat") as demo:
    gr.Markdown("# 🎤 Real-Time Voice Cloning Chatbot")
    
    with gr.Tab("⚙️ Setup"):
        gr.Markdown("### Step 1: Models (auto-loaded at startup)")
        with gr.Row():
            whisper_size = gr.Dropdown(["tiny", "base", "small"], value="tiny", label="Whisper Size")
            load_whisper_btn = gr.Button("Reload Whisper")
            whisper_status = gr.Textbox(label="Status", value="✅ Whisper 'tiny' loaded!", interactive=False)
        
        with gr.Row():
            tts_path = gr.Dropdown(
                ["Qwen/Qwen3-TTS-12Hz-0.6B-Base", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"],
                value="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
                label="TTS Model"
            )
            load_tts_btn = gr.Button("Reload TTS")
            tts_status = gr.Textbox(label="Status", value="✅ TTS 'Qwen3-TTS-12Hz-0.6B-Base' loaded!", interactive=False)
        
        gr.Markdown("### Step 2: Clone Voice")
        clone_audio = gr.Audio(label="Upload Voice Sample (5-10 sec recommended)", type="filepath")
        clone_text = gr.Textbox(label="Transcript (optional)", placeholder="What is said in the audio...")
        clone_btn = gr.Button("Clone Voice")
        gr.Markdown("⚠️ *Long audio is auto-trimmed to 10s. On CPU servers, cloning takes ~60s. Please wait.*")
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
    clone_btn.click(create_voice_clone, inputs=[clone_audio, clone_text], outputs=[clone_status])
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

    system = "You are a helpful voice assistant on a phone call. Keep responses VERY short - maximum 1-2 sentences, under 20 words. Be brief and natural."
    recent = call_history[-4:]
    messages = [{"role": "system", "content": system}] + recent

    import requests as req
    api_key = os.environ.get("GROQ_API_KEY", "")
    try:
        resp = req.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": "llama-3.1-8b-instant", "messages": messages, "max_tokens": 40},
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
    # Truncate to keep TTS fast on CPU
    words = text.split()
    if len(words) > 20:
        text = " ".join(words[:20]) + "..."

    word_count = len(text.split())
    max_tokens = min(max(50, word_count * 20), 200)

    try:
        wavs, sr = tts_model.generate_voice_clone(
            text=text,
            language="Auto",
            voice_clone_prompt=voice_clone_prompt,
            max_new_tokens=max_tokens,
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

        system = "You are a helpful voice assistant on a phone call. Keep responses VERY short - maximum 1-2 sentences, under 15 words. Be brief and natural."
        recent = call_history[-4:]
        messages = [{"role": "system", "content": system}] + recent

        try:
            resp = req.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": "llama-3.1-8b-instant", "messages": messages, "max_tokens": 30},
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

        # 4. TTS via ElevenLabs API (cloud — fast, high quality)
        t3 = _time.time()
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
            print(f"[PIPELINE] ElevenLabs TTS: {_time.time()-t3:.1f}s ({len(el_resp.content)} bytes)", flush=True)
        except Exception as e:
            print(f"[PIPELINE] ElevenLabs TTS failed ({e}), falling back to Edge TTS", flush=True)
            # Fallback to Edge TTS
            try:
                import edge_tts as _edge_tts
                async def _gen():
                    communicate = _edge_tts.Communicate(ai_response, "en-US-GuyNeural")
                    await communicate.save(el_mp3_path)
                asyncio.run(_gen())
                print(f"[PIPELINE] Edge TTS fallback: {_time.time()-t3:.1f}s", flush=True)
            except Exception as e2:
                print(f"[PIPELINE] Edge TTS also failed ({e2})", flush=True)
                return jsonify({"error": "TTS unavailable"}), 503

        # 5. Convert mp3 → 8kHz 16-bit WAV for Asterisk
        t4 = _time.time()
        asterisk_path = tempfile.mktemp(suffix=".wav")
        subprocess.run(
            ["ffmpeg", "-y", "-i", el_mp3_path,
             "-ar", "8000", "-ac", "1", "-sample_fmt", "s16", asterisk_path],
            check=True, capture_output=True,
        )
        print(f"[PIPELINE] ffmpeg convert: {_time.time()-t4:.1f}s", flush=True)

        total = _time.time() - t0
        print(f"[PIPELINE] ===== DONE total={total:.1f}s =====", flush=True)

        # Cleanup intermediate files
        for p in [input_path, el_mp3_path]:
            if os.path.exists(p):
                os.unlink(p)

        return send_file(asterisk_path, mimetype="audio/wav", as_attachment=True,
                         download_name="response.wav")

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
