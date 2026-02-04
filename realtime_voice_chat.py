# Real-Time Voice Cloning Chatbot
# You speak → AI responds → Cloned voice speaks back

import os
import time
import tempfile
import threading
import queue
import numpy as np
import torch
import soundfile as sf
import sounddevice as sd
from pathlib import Path

# For Speech-to-Text
import whisper

# For AI Response (you can use OpenAI, Ollama, or any LLM)
# Option 1: OpenAI
# from openai import OpenAI
# Option 2: Ollama (local)
import requests

# For Text-to-Speech with Voice Cloning
from qwen_tts import Qwen3TTSModel


class RealtimeVoiceChatBot:
    def __init__(
        self,
        clone_audio_path: str,
        clone_audio_text: str = None,
        tts_model_path: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        whisper_model: str = "base",  # tiny, base, small, medium, large
        llm_provider: str = "ollama",  # "ollama" or "openai"
        llm_model: str = "qwen2.5:7b",  # or "gpt-4" for OpenAI
        device: str = "cuda:0",
        sample_rate: int = 16000,
    ):
        """
        Initialize the Real-Time Voice Cloning Chatbot.
        
        Args:
            clone_audio_path: Path to the audio file to clone voice from
            clone_audio_text: Transcript of the clone audio (for better quality)
            tts_model_path: Qwen3-TTS model path
            whisper_model: Whisper model size for STT
            llm_provider: "ollama" or "openai"
            llm_model: Model name for the LLM
            device: CUDA device
            sample_rate: Audio sample rate
        """
        self.device = device
        self.sample_rate = sample_rate
        self.clone_audio_path = clone_audio_path
        self.clone_audio_text = clone_audio_text
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.conversation_history = []
        
        print("🔄 Loading models... This may take a few minutes.")
        
        # Load Whisper for Speech-to-Text
        print("📝 Loading Whisper STT model...")
        whisper_device = "cuda" if torch.cuda.is_available() and "cuda" in self.device else "cpu"
        self.whisper_model = whisper.load_model(whisper_model, device=whisper_device)
        
        # Load Qwen3-TTS for Voice Cloning
        print("🎤 Loading Qwen3-TTS model...")
        tts_device = device if torch.cuda.is_available() and "cuda" in device else "cpu"
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
        
        # Pre-compute voice clone prompt for faster generation
        print("🎭 Creating voice clone prompt...")
        self.voice_clone_prompt = self._create_voice_prompt()
        
        print("✅ All models loaded! Ready to chat.")
    
    def _create_voice_prompt(self):
        """Create voice clone prompt from reference audio."""
        if self.clone_audio_text:
            # ICL mode - better quality
            prompt = self.tts.create_voice_clone_prompt(
                ref_audio=self.clone_audio_path,
                ref_text=self.clone_audio_text,
                x_vector_only_mode=False,
            )
        else:
            # X-vector only mode - no transcript needed
            prompt = self.tts.create_voice_clone_prompt(
                ref_audio=self.clone_audio_path,
                x_vector_only_mode=True,
            )
        return prompt
    
    def speech_to_text(self, audio_data: np.ndarray) -> str:
        """Convert speech to text using Whisper."""
        # Save temp file for Whisper
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio_data, self.sample_rate)
            result = self.whisper_model.transcribe(f.name)
            os.unlink(f.name)
        return result["text"].strip()
    
    def get_ai_response(self, user_message: str) -> str:
        """Get AI response from LLM."""
        # Add to conversation history
        self.conversation_history.append({"role": "user", "content": user_message})
        
        system_prompt = """You are a helpful voice assistant. Keep your responses concise and conversational, 
suitable for spoken dialogue. Respond naturally as if having a real conversation."""
        
        if self.llm_provider == "ollama":
            response = self._ollama_chat(system_prompt, self.conversation_history)
        elif self.llm_provider == "openai":
            response = self._openai_chat(system_prompt, self.conversation_history)
        else:
            response = "I don't understand. Please configure an LLM provider."
        
        # Add assistant response to history
        self.conversation_history.append({"role": "assistant", "content": response})
        
        # Keep conversation history manageable
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]
        
        return response
    
    def _ollama_chat(self, system_prompt: str, messages: list) -> str:
        """Chat with Ollama local LLM."""
        try:
            full_messages = [{"role": "system", "content": system_prompt}] + messages
            response = requests.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": self.llm_model,
                    "messages": full_messages,
                    "stream": False,
                },
                timeout=60,
            )
            response.raise_for_status()
            return response.json()["message"]["content"]
        except Exception as e:
            print(f"❌ Ollama error: {e}")
            return "Sorry, I couldn't generate a response. Please check if Ollama is running."
    
    def _openai_chat(self, system_prompt: str, messages: list) -> str:
        """Chat with OpenAI API."""
        try:
            from openai import OpenAI
            client = OpenAI()  # Uses OPENAI_API_KEY env var
            full_messages = [{"role": "system", "content": system_prompt}] + messages
            response = client.chat.completions.create(
                model=self.llm_model,
                messages=full_messages,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"❌ OpenAI error: {e}")
            return "Sorry, I couldn't generate a response. Please check your OpenAI API key."
    
    def text_to_speech_cloned(self, text: str) -> tuple:
        """Convert text to speech using cloned voice."""
        wavs, sr = self.tts.generate_voice_clone(
            text=text,
            language="Auto",
            voice_clone_prompt=self.voice_clone_prompt,
            max_new_tokens=4096,
        )
        return wavs[0], sr
    
    def play_audio(self, audio_data: np.ndarray, sample_rate: int):
        """Play audio through speakers."""
        sd.play(audio_data, sample_rate)
        sd.wait()
    
    def record_audio(self, duration: float = None, silence_threshold: float = 0.01, 
                     silence_duration: float = 1.5) -> np.ndarray:
        """
        Record audio from microphone.
        
        Args:
            duration: Fixed duration in seconds (if None, uses silence detection)
            silence_threshold: Amplitude threshold for silence detection
            silence_duration: Seconds of silence before stopping
        """
        print("🎙️ Listening... (speak now)")
        
        if duration:
            # Fixed duration recording
            audio = sd.rec(int(duration * self.sample_rate), 
                          samplerate=self.sample_rate, channels=1, dtype='float32')
            sd.wait()
            return audio.flatten()
        else:
            # Voice activity detection
            audio_chunks = []
            silence_start = None
            
            def callback(indata, frames, time, status):
                audio_chunks.append(indata.copy())
            
            with sd.InputStream(samplerate=self.sample_rate, channels=1, 
                               dtype='float32', callback=callback):
                # Wait for speech to start
                while True:
                    if len(audio_chunks) > 0:
                        chunk = audio_chunks[-1]
                        if np.abs(chunk).max() > silence_threshold:
                            break
                    time.sleep(0.05)
                
                print("📢 Speech detected, recording...")
                
                # Record until silence
                while True:
                    if len(audio_chunks) > 0:
                        chunk = audio_chunks[-1]
                        if np.abs(chunk).max() < silence_threshold:
                            if silence_start is None:
                                silence_start = time.time()
                            elif time.time() - silence_start > silence_duration:
                                break
                        else:
                            silence_start = None
                    time.sleep(0.05)
            
            return np.concatenate(audio_chunks).flatten()
    
    def chat_turn(self) -> bool:
        """
        Perform one turn of conversation.
        Returns False if user wants to quit.
        """
        try:
            # 1. Record user's speech
            audio = self.record_audio()
            
            # 2. Convert speech to text
            print("🔄 Transcribing...")
            user_text = self.speech_to_text(audio)
            print(f"👤 You said: {user_text}")
            
            # Check for exit commands
            if any(word in user_text.lower() for word in ["quit", "exit", "bye", "goodbye", "stop"]):
                print("👋 Goodbye!")
                return False
            
            if not user_text:
                print("❌ Couldn't understand. Please try again.")
                return True
            
            # 3. Get AI response
            print("🤖 Thinking...")
            ai_response = self.get_ai_response(user_text)
            print(f"🤖 AI says: {ai_response}")
            
            # 4. Convert response to speech with cloned voice
            print("🎭 Generating cloned voice response...")
            audio_response, sr = self.text_to_speech_cloned(ai_response)
            
            # 5. Play the response
            print("🔊 Playing response...")
            self.play_audio(audio_response, sr)
            
            return True
            
        except KeyboardInterrupt:
            print("\n👋 Interrupted. Goodbye!")
            return False
        except Exception as e:
            print(f"❌ Error: {e}")
            return True
    
    def run(self):
        """Run the chatbot in a loop."""
        print("\n" + "="*60)
        print("🎤 REAL-TIME VOICE CLONING CHATBOT 🎤")
        print("="*60)
        print(f"Voice cloned from: {self.clone_audio_path}")
        print("Say 'quit', 'exit', or 'bye' to stop")
        print("="*60 + "\n")
        
        while self.chat_turn():
            print("\n" + "-"*40 + "\n")
        
        print("Chat ended. Thank you!")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Real-Time Voice Cloning Chatbot")
    parser.add_argument("--clone-audio", "-c", required=True, 
                       help="Path to audio file for voice cloning")
    parser.add_argument("--clone-text", "-t", default=None,
                       help="Transcript of clone audio (optional, improves quality)")
    parser.add_argument("--llm", default="ollama", choices=["ollama", "openai"],
                       help="LLM provider (default: ollama)")
    parser.add_argument("--llm-model", default="qwen2.5:7b",
                       help="LLM model name (default: qwen2.5:7b for ollama)")
    parser.add_argument("--whisper-model", default="base",
                       choices=["tiny", "base", "small", "medium", "large"],
                       help="Whisper model size (default: base)")
    parser.add_argument("--tts-model", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                       help="Qwen3-TTS model path")
    parser.add_argument("--device", default="cuda:0",
                       help="CUDA device (default: cuda:0)")
    
    args = parser.parse_args()
    
    bot = RealtimeVoiceChatBot(
        clone_audio_path=args.clone_audio,
        clone_audio_text=args.clone_text,
        tts_model_path=args.tts_model,
        whisper_model=args.whisper_model,
        llm_provider=args.llm,
        llm_model=args.llm_model,
        device=args.device,
    )
    
    bot.run()


if __name__ == "__main__":
    main()
