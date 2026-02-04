# 🎤 Real-Time Voice Cloning Chatbot

Talk to an AI that responds in **any voice you clone**!

## How It Works

```
You Speak → Whisper STT → Your Text → LLM (AI) → Response Text → Qwen3-TTS (Cloned Voice) → AI Speaks Back
```

1. **You speak** into the microphone
2. **Whisper** transcribes your speech to text
3. **LLM (Ollama/OpenAI)** generates a response
4. **Qwen3-TTS** converts the response to speech **in the cloned voice**
5. **You hear** the AI response in the voice you uploaded!

## Installation

### 1. Install PyTorch (with CUDA)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 2. Install Qwen3-TTS
```bash
pip install git+https://github.com/QwenLM/Qwen3-TTS.git
```

### 3. Install Other Dependencies
```bash
pip install openai-whisper sounddevice soundfile numpy librosa requests gradio openai
```

### 4. (Optional) Install Flash Attention for faster inference
```bash
pip install flash-attn --no-build-isolation
```

### 5. Install Ollama (for local LLM)
Download from: https://ollama.ai/download

Then pull a model:
```bash
ollama pull qwen2.5:7b
```

## Usage

### Option 1: Web UI (Recommended)

```bash
python voice_chat_ui.py
```

Then open http://localhost:7860 in your browser.

**Steps:**
1. Click "Load Models" and wait
2. Upload a voice sample to clone (e.g., a 5-10 second audio clip)
3. Optionally enter the transcript for better quality
4. Click "Create Voice Clone"
5. Go to "Voice Chat" tab
6. Record your voice or type a message
7. The AI will respond in the cloned voice!

### Option 2: Command Line

```bash
# Basic usage (requires Ollama running)
python realtime_voice_chat.py --clone-audio "path/to/voice_sample.wav"

# With transcript (better quality)
python realtime_voice_chat.py --clone-audio "voice.wav" --clone-text "Hello, this is a test recording."

# Using OpenAI instead of Ollama
python realtime_voice_chat.py --clone-audio "voice.wav" --llm openai --llm-model gpt-4

# With smaller Whisper model (faster but less accurate)
python realtime_voice_chat.py --clone-audio "voice.wav" --whisper-model tiny
```

## Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--clone-audio, -c` | Path to voice sample for cloning | (required) |
| `--clone-text, -t` | Transcript of clone audio | None |
| `--llm` | LLM provider: "ollama" or "openai" | ollama |
| `--llm-model` | Model name | qwen2.5:7b |
| `--whisper-model` | Whisper size: tiny/base/small/medium/large | base |
| `--tts-model` | Qwen3-TTS model path | Qwen/Qwen3-TTS-12Hz-1.7B-Base |
| `--device` | CUDA device | cuda:0 |

## Tips for Best Results

### Voice Sample Tips:
- Use a **5-15 second** clear audio clip
- **Less background noise** = better clone
- Include **varied speech** (questions, statements)
- Providing the **transcript** significantly improves quality

### LLM Tips:
- **Ollama** is free and runs locally
- **OpenAI** may give better responses but costs money
- Set `OPENAI_API_KEY` environment variable for OpenAI
- For Groq in the simple UI, set `GROQ_API_KEY` or paste the key in the UI

### Performance Tips:
- Use **GPU** for much faster processing
- Smaller Whisper models (`tiny`, `base`) are faster
- The 0.6B TTS model is faster than 1.7B but lower quality

## Troubleshooting

### "Ollama error: Connection refused"
Make sure Ollama is running:
```bash
ollama serve
```

### "CUDA out of memory"
- Use smaller models:
  - `--whisper-model tiny`
  - `--tts-model Qwen/Qwen3-TTS-12Hz-0.6B-Base`

### "No audio input"
- Check microphone permissions
- Try a different `sounddevice` input device

## Example Voice Samples

You can use any audio file. For testing, try:
- A YouTube clip (download with yt-dlp)
- A podcast snippet
- Your own voice recording

## License

This project uses:
- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) - Apache 2.0
- [Whisper](https://github.com/openai/whisper) - MIT
- [Ollama](https://ollama.ai/) - MIT
