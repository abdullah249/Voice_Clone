# Simple Real-Time Voice Cloning Chatbot
# Lighter version - loads models on demand

import os
import tempfile
import numpy as np
import gradio as gr

# Set SoX path (local installation)
SOX_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sox-14.4.2")
os.environ["PATH"] = SOX_PATH + os.pathsep + os.environ.get("PATH", "")

# Global model holders
tts_model = None
whisper_model = None
voice_clone_prompt = None
conversation_history = []


def load_whisper(model_size="base"):
    """Load Whisper model."""
    global whisper_model
    import whisper
    whisper_model = whisper.load_model(model_size)
    return "✅ Whisper loaded!"


def load_tts(model_path="Qwen/Qwen3-TTS-12Hz-1.7B-Base"):
    """Load Qwen3-TTS model."""
    global tts_model
    import torch
    from qwen_tts import Qwen3TTSModel
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    tts_model = Qwen3TTSModel.from_pretrained(
        model_path,
        device_map=device,
        dtype=dtype,
        attn_implementation="sdpa",  # Windows-compatible default
    )
    return "✅ TTS loaded!"


def create_voice_clone(audio_path, audio_text):
    """Create voice clone prompt."""
    global voice_clone_prompt, tts_model
    import torch
    import gc
    
    if tts_model is None:
        return "❌ Load TTS model first!"
    
    # Clear GPU memory before cloning
    torch.cuda.empty_cache()
    gc.collect()
    
    try:
        if audio_text and audio_text.strip():
            voice_clone_prompt = tts_model.create_voice_clone_prompt(
                ref_audio=audio_path,
                ref_text=audio_text.strip(),
                x_vector_only_mode=False,
            )
        else:
            voice_clone_prompt = tts_model.create_voice_clone_prompt(
                ref_audio=audio_path,
                x_vector_only_mode=True,
            )
        
        # Clear cache after operation
        torch.cuda.empty_cache()
        return "✅ Voice cloned!"
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        gc.collect()
        return "❌ Out of GPU memory! Try using the smaller 0.6B model."


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
    
    system = "You are a helpful voice assistant. Keep responses concise and conversational, under 100 words."
    messages = [{"role": "system", "content": system}] + conversation_history
    
    try:
        if provider == "groq":
            # Groq API (free tier available: https://console.groq.com/keys)
            if not api_key:
                raise ValueError("Missing Groq API key")
            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": model, "messages": messages, "max_tokens": 200},
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
                json={"model": model, "messages": messages, "max_tokens": 200},
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
    
    if tts_model is None or voice_clone_prompt is None:
        return None
    
    wavs, sr = tts_model.generate_voice_clone(
        text=text,
        language="Auto",
        voice_clone_prompt=voice_clone_prompt,
        max_new_tokens=4096,
    )
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
        gr.Markdown("### Step 1: Load Models")
        with gr.Row():
            whisper_size = gr.Dropdown(["tiny", "base", "small"], value="base", label="Whisper Size")
            load_whisper_btn = gr.Button("Load Whisper")
            whisper_status = gr.Textbox(label="Status", interactive=False)
        
        with gr.Row():
            tts_path = gr.Dropdown(
                ["Qwen/Qwen3-TTS-12Hz-0.6B-Base", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"],
                value="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
                label="TTS Model (0.6B recommended for 6GB GPU)"
            )
            load_tts_btn = gr.Button("Load TTS")
            tts_status = gr.Textbox(label="Status", interactive=False)
        
        gr.Markdown("### Step 2: Clone Voice")
        clone_audio = gr.Audio(label="Upload Voice Sample", type="filepath")
        clone_text = gr.Textbox(label="Transcript (optional)", placeholder="What is said in the audio...")
        clone_btn = gr.Button("Clone Voice")
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
            value=os.getenv("GROQ_API_KEY", ""),
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

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
