# Real-Time Voice Cloning Chatbot with Gradio Web UI
# Beautiful web interface for voice conversation with cloned voice

import os
import time
import tempfile
import numpy as np
import torch
import soundfile as sf
import gradio as gr
from pathlib import Path

# For Speech-to-Text
import whisper

# For AI Response
import requests

# For Text-to-Speech with Voice Cloning
from qwen_tts import Qwen3TTSModel


class VoiceChatUI:
    def __init__(self):
        self.tts = None
        self.whisper_model = None
        self.voice_clone_prompt = None
        self.conversation_history = []
        self.is_loaded = False
        
    def load_models(self, tts_model_path: str, whisper_size: str, progress=gr.Progress()):
        """Load all required models."""
        try:
            progress(0.1, desc="Loading Whisper STT...")
            if self.whisper_model is None:
                self.whisper_model = whisper.load_model(whisper_size)
            
            progress(0.5, desc="Loading Qwen3-TTS...")
            if self.tts is None:
                tts_device = "cuda:0" if torch.cuda.is_available() else "cpu"
                tts_dtype = torch.bfloat16 if tts_device != "cpu" else torch.float32
                try:
                    self.tts = Qwen3TTSModel.from_pretrained(
                        tts_model_path,
                        device_map=tts_device,
                        dtype=tts_dtype,
                        attn_implementation="flash_attention_2",
                    )
                except Exception:
                    self.tts = Qwen3TTSModel.from_pretrained(
                        tts_model_path,
                        device_map=tts_device,
                        dtype=tts_dtype,
                        attn_implementation="sdpa",
                    )
            
            progress(1.0, desc="Models loaded!")
            self.is_loaded = True
            return "✅ Models loaded successfully!"
        except Exception as e:
            return f"❌ Error loading models: {str(e)}"
    
    def create_voice_clone(self, audio_path: str, audio_text: str, progress=gr.Progress()):
        """Create voice clone from uploaded audio."""
        if not self.is_loaded:
            return "❌ Please load models first!"
        
        try:
            progress(0.5, desc="Creating voice clone...")
            
            if audio_text and audio_text.strip():
                # ICL mode - better quality
                self.voice_clone_prompt = self.tts.create_voice_clone_prompt(
                    ref_audio=audio_path,
                    ref_text=audio_text.strip(),
                    x_vector_only_mode=False,
                )
                mode = "ICL mode (with transcript)"
            else:
                # X-vector only mode
                self.voice_clone_prompt = self.tts.create_voice_clone_prompt(
                    ref_audio=audio_path,
                    x_vector_only_mode=True,
                )
                mode = "X-vector mode (no transcript)"
            
            progress(1.0, desc="Voice cloned!")
            return f"✅ Voice cloned successfully using {mode}!"
        except Exception as e:
            return f"❌ Error creating voice clone: {str(e)}"
    
    def speech_to_text(self, audio_path: str) -> str:
        """Convert speech to text."""
        if self.whisper_model is None:
            return ""
        result = self.whisper_model.transcribe(audio_path)
        return result["text"].strip()
    
    def get_llm_response(self, user_message: str, llm_provider: str, llm_model: str) -> str:
        """Get response from LLM."""
        self.conversation_history.append({"role": "user", "content": user_message})
        
        system_prompt = """You are a helpful voice assistant. Keep your responses concise and conversational, 
suitable for spoken dialogue. Respond naturally as if having a real conversation. 
Keep responses under 100 words unless asked for more detail."""
        
        try:
            if llm_provider == "Ollama":
                response = self._ollama_chat(system_prompt, llm_model)
            elif llm_provider == "OpenAI":
                response = self._openai_chat(system_prompt, llm_model)
            else:
                response = "Please select an LLM provider."
        except Exception as e:
            response = f"Error getting response: {str(e)}"
        
        self.conversation_history.append({"role": "assistant", "content": response})
        
        # Keep history manageable
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]
        
        return response
    
    def _ollama_chat(self, system_prompt: str, model: str) -> str:
        """Chat with Ollama."""
        messages = [{"role": "system", "content": system_prompt}] + self.conversation_history
        response = requests.post(
            "http://localhost:11434/api/chat",
            json={"model": model, "messages": messages, "stream": False},
            timeout=60,
        )
        response.raise_for_status()
        return response.json()["message"]["content"]
    
    def _openai_chat(self, system_prompt: str, model: str) -> str:
        """Chat with OpenAI."""
        from openai import OpenAI
        client = OpenAI()
        messages = [{"role": "system", "content": system_prompt}] + self.conversation_history
        response = client.chat.completions.create(model=model, messages=messages)
        return response.choices[0].message.content
    
    def text_to_speech_cloned(self, text: str) -> tuple:
        """Convert text to speech with cloned voice."""
        if self.voice_clone_prompt is None:
            return None
        
        wavs, sr = self.tts.generate_voice_clone(
            text=text,
            language="Auto",
            voice_clone_prompt=self.voice_clone_prompt,
            max_new_tokens=4096,
        )
        return (sr, wavs[0])
    
    def process_voice_input(self, audio, llm_provider: str, llm_model: str):
        """Process voice input and generate cloned voice response."""
        if not self.is_loaded:
            return None, "❌ Please load models first!", "", ""
        
        if self.voice_clone_prompt is None:
            return None, "❌ Please create voice clone first!", "", ""
        
        if audio is None:
            return None, "❌ No audio input received!", "", ""
        
        try:
            # Save audio to temp file
            sr, audio_data = audio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                sf.write(f.name, audio_data, sr)
                temp_path = f.name
            
            # 1. Speech to Text
            user_text = self.speech_to_text(temp_path)
            os.unlink(temp_path)
            
            if not user_text:
                return None, "❌ Couldn't understand audio", "", ""
            
            # 2. Get LLM Response
            ai_response = self.get_llm_response(user_text, llm_provider, llm_model)
            
            # 3. Text to Speech with cloned voice
            audio_response = self.text_to_speech_cloned(ai_response)
            
            # Format conversation
            conversation = self._format_conversation()
            
            return audio_response, "✅ Response generated!", user_text, conversation
            
        except Exception as e:
            return None, f"❌ Error: {str(e)}", "", ""
    
    def process_text_input(self, text: str, llm_provider: str, llm_model: str):
        """Process text input and generate cloned voice response."""
        if not self.is_loaded:
            return None, "❌ Please load models first!", ""
        
        if self.voice_clone_prompt is None:
            return None, "❌ Please create voice clone first!", ""
        
        if not text or not text.strip():
            return None, "❌ Please enter some text!", ""
        
        try:
            # 1. Get LLM Response
            ai_response = self.get_llm_response(text.strip(), llm_provider, llm_model)
            
            # 2. Text to Speech with cloned voice
            audio_response = self.text_to_speech_cloned(ai_response)
            
            # Format conversation
            conversation = self._format_conversation()
            
            return audio_response, "✅ Response generated!", conversation
            
        except Exception as e:
            return None, f"❌ Error: {str(e)}", ""
    
    def _format_conversation(self) -> str:
        """Format conversation history for display."""
        formatted = ""
        for msg in self.conversation_history:
            role = "👤 You" if msg["role"] == "user" else "🤖 AI"
            formatted += f"{role}: {msg['content']}\n\n"
        return formatted
    
    def clear_conversation(self):
        """Clear conversation history."""
        self.conversation_history = []
        return "", "🧹 Conversation cleared!"


def create_ui():
    chat = VoiceChatUI()
    
    with gr.Blocks(title="Real-Time Voice Cloning Chatbot", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🎤 Real-Time Voice Cloning Chatbot
        
        **Talk to an AI that responds in any voice you clone!**
        
        1. **Load Models** → Load the TTS and STT models
        2. **Clone Voice** → Upload an audio sample to clone
        3. **Chat** → Speak or type to chat with the AI in the cloned voice
        """)
        
        with gr.Tab("⚙️ Setup"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 1️⃣ Load Models")
                    tts_model = gr.Dropdown(
                        label="TTS Model",
                        choices=[
                            "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                            "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
                        ],
                        value="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                    )
                    whisper_size = gr.Dropdown(
                        label="Whisper Model Size",
                        choices=["tiny", "base", "small", "medium", "large"],
                        value="base",
                    )
                    load_btn = gr.Button("🚀 Load Models", variant="primary")
                    load_status = gr.Textbox(label="Status", interactive=False)
                
                with gr.Column():
                    gr.Markdown("### 2️⃣ Clone Voice")
                    clone_audio = gr.Audio(label="Upload Voice Sample", type="filepath")
                    clone_text = gr.Textbox(
                        label="Transcript (Optional - improves quality)",
                        placeholder="Enter what is said in the audio sample...",
                        lines=2,
                    )
                    clone_btn = gr.Button("🎭 Create Voice Clone", variant="primary")
                    clone_status = gr.Textbox(label="Status", interactive=False)
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 3️⃣ LLM Settings")
                    llm_provider = gr.Radio(
                        label="LLM Provider",
                        choices=["Ollama", "OpenAI"],
                        value="Ollama",
                    )
                    llm_model = gr.Textbox(
                        label="Model Name",
                        value="qwen2.5:7b",
                        placeholder="e.g., qwen2.5:7b, llama3, gpt-4",
                    )
        
        with gr.Tab("🎙️ Voice Chat"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Speak to the AI")
                    voice_input = gr.Audio(
                        label="🎤 Record Your Voice",
                        sources=["microphone"],
                        type="numpy",
                    )
                    voice_btn = gr.Button("🚀 Send Voice", variant="primary", size="lg")
                    
                    gr.Markdown("### Or Type")
                    text_input = gr.Textbox(
                        label="💬 Type Your Message",
                        placeholder="Type your message here...",
                        lines=2,
                    )
                    text_btn = gr.Button("📤 Send Text", variant="primary")
                
                with gr.Column(scale=1):
                    gr.Markdown("### AI Response (Cloned Voice)")
                    audio_output = gr.Audio(label="🔊 Response", type="numpy")
                    status = gr.Textbox(label="Status", interactive=False)
                    transcription = gr.Textbox(label="Your Speech (Transcribed)", interactive=False)
            
            gr.Markdown("### 💬 Conversation History")
            conversation = gr.Textbox(
                label="",
                lines=10,
                interactive=False,
            )
            clear_btn = gr.Button("🧹 Clear Conversation")
        
        # Event handlers
        load_btn.click(
            chat.load_models,
            inputs=[tts_model, whisper_size],
            outputs=[load_status],
        )
        
        clone_btn.click(
            chat.create_voice_clone,
            inputs=[clone_audio, clone_text],
            outputs=[clone_status],
        )
        
        voice_btn.click(
            chat.process_voice_input,
            inputs=[voice_input, llm_provider, llm_model],
            outputs=[audio_output, status, transcription, conversation],
        )
        
        text_btn.click(
            chat.process_text_input,
            inputs=[text_input, llm_provider, llm_model],
            outputs=[audio_output, status, conversation],
        )
        
        clear_btn.click(
            chat.clear_conversation,
            outputs=[conversation, status],
        )
    
    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
