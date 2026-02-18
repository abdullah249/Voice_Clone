# Voice Clone - Complete Server Deployment Guide

> End-to-end guide: from a fresh server to a running Voice Cloning Chatbot.

---

## Table of Contents

1. [Server Requirements](#1-server-requirements)
2. [Server Setup (Fresh Ubuntu)](#2-server-setup-fresh-ubuntu)
3. [Clone the Repository](#3-clone-the-repository)
4. [Python Environment Setup](#4-python-environment-setup)
5. [Install Dependencies](#5-install-dependencies)
6. [Configure Environment Variables](#6-configure-environment-variables)
7. [Install & Configure Ollama (Optional)](#7-install--configure-ollama-optional)
8. [Test the Application](#8-test-the-application)
9. [Production Deployment with systemd](#9-production-deployment-with-systemd)
10. [Firewall & Port Configuration](#10-firewall--port-configuration)
11. [Reverse Proxy with Nginx (Optional)](#11-reverse-proxy-with-nginx-optional)
12. [SSL with Let's Encrypt (Optional)](#12-ssl-with-lets-encrypt-optional)
13. [Monitoring & Logs](#13-monitoring--logs)
14. [Troubleshooting](#14-troubleshooting)

---

## 1. Server Requirements

### Minimum (CPU-only, slower inference):
| Resource | Minimum |
|----------|---------|
| CPU | 2 vCPUs |
| RAM | 4 GB (8 GB recommended) |
| Swap | 8 GB |
| Disk | 30 GB |
| OS | Ubuntu 22.04 / 24.04 LTS |

### Recommended (GPU, fast inference):
| Resource | Recommended |
|----------|-------------|
| GPU | NVIDIA with 8+ GB VRAM (T4, A10G, RTX 3060+) |
| CPU | 4+ vCPUs |
| RAM | 16 GB |
| Disk | 50 GB |
| OS | Ubuntu 22.04 / 24.04 LTS |

### Ports Used:
| Port | Service | Access |
|------|---------|--------|
| `7860` | Gradio Web UI (main app) | Public (or via reverse proxy) |
| `5050` | Flask API (Asterisk/phone pipeline) | Internal only (127.0.0.1) |
| `11434` | Ollama LLM (if using local LLM) | Internal only |
| `22` | SSH | Admin only |
| `80/443` | Nginx reverse proxy (optional) | Public |

---

## 2. Server Setup (Fresh Ubuntu)

### 2.1 SSH into your new server
```bash
ssh root@YOUR_SERVER_IP
```

### 2.2 Create a non-root user
```bash
adduser clone
usermod -aG sudo clone
su - clone
```

### 2.3 Update system packages
```bash
sudo apt update && sudo apt upgrade -y
```

### 2.4 Install system dependencies
```bash
sudo apt install -y \
    python3 python3-pip python3-venv \
    git curl wget \
    ffmpeg sox libsox-dev \
    build-essential \
    libsndfile1 \
    portaudio19-dev
```

> **Note:** `ffmpeg` is required by Whisper. `sox` is required for audio processing (replaces the Windows sox-14.4.2 folder). `portaudio19-dev` is needed for `sounddevice`.

### 2.5 (GPU Only) Install NVIDIA drivers & CUDA
```bash
# Check if GPU is available
nvidia-smi

# If not installed, install NVIDIA drivers:
sudo apt install -y nvidia-driver-535
sudo reboot

# After reboot, verify:
nvidia-smi
```

### 2.6 Set up swap (if RAM < 8 GB)
```bash
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# Verify
free -h
```

---

## 3. Clone the Repository

```bash
cd ~
git clone https://github.com/abdullah249/Voice_Clone.git
cd Voice_Clone
```

---

## 4. Python Environment Setup

### 4.1 Create virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4.2 Upgrade pip
```bash
pip install --upgrade pip setuptools wheel
```

---

## 5. Install Dependencies

### 5.1 Install PyTorch

**For GPU (CUDA 12.1):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**For CPU only:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 5.2 Install Qwen3-TTS
```bash
pip install git+https://github.com/QwenLM/Qwen3-TTS.git
```

### 5.3 Install remaining Python packages
```bash
pip install openai-whisper sounddevice soundfile numpy librosa gradio requests openai flask
```

### 5.4 (GPU Only, Optional) Install Flash Attention for faster inference
```bash
pip install flash-attn --no-build-isolation
```

### 5.5 Verify installation
```bash
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
import whisper
print('Whisper: OK')
from qwen_tts import Qwen3TTSModel
print('Qwen3-TTS: OK')
import gradio
print(f'Gradio: {gradio.__version__}')
print('All dependencies OK!')
"
```

---

## 6. Configure Environment Variables

### 6.1 Create your `.env` file from the template
```bash
cp .env.example .env
nano .env
```

### 6.2 Fill in your API keys
```env
# REQUIRED: Get from https://console.groq.com/keys
GROQ_API_KEY=gsk_your_actual_key_here

# REQUIRED for phone pipeline: Get from https://elevenlabs.io/app/settings/api-keys
ELEVENLABS_API_KEY=sk_your_actual_key_here

# OPTIONAL: ElevenLabs Voice ID
ELEVENLABS_VOICE_ID=cjVigY5qzO86Huf0OWal

# OPTIONAL: If using OpenAI as LLM provider
# OPENAI_API_KEY=sk-your_key_here
```

### 6.3 Load environment variables
Add to your shell profile so they persist:
```bash
# Add to ~/.bashrc
echo 'set -a; source ~/Voice_Clone/.env; set +a' >> ~/.bashrc
source ~/.bashrc
```

Or load manually before running:
```bash
set -a; source .env; set +a
```

---

## 7. Install & Configure Ollama (Optional)

> Only needed if you want to use a **local LLM** instead of Groq cloud API.
> Skip this if using Groq (default in `simple_voice_chat.py`).

### 7.1 Install Ollama
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

### 7.2 Pull a model
```bash
ollama pull qwen2.5:7b
# Or a smaller model for low-resource servers:
ollama pull qwen2.5:1.5b
```

### 7.3 Start Ollama (runs on port 11434)
```bash
# Ollama usually starts automatically as a service
systemctl status ollama

# If not running:
ollama serve &
```

---

## 8. Test the Application

### 8.1 Quick test (foreground)
```bash
cd ~/Voice_Clone
source .venv/bin/activate
set -a; source .env; set +a

python3 simple_voice_chat.py
```

You should see:
```
⏳ Pre-loading Whisper tiny...
✅ Whisper tiny loaded!
⏳ Pre-loading TTS 0.6B (this takes a few minutes on CPU)...
✅ TTS 0.6B loaded!
🚀 Models ready — starting Gradio UI...
🌐 Flask API started on http://127.0.0.1:5050
Running on local URL:  http://0.0.0.0:7860
```

### 8.2 Access from browser
Open in your browser:
```
http://YOUR_SERVER_IP:7860
```

> **Note:** First startup downloads the TTS model (~3.5 GB) and Whisper model. This can take 5-15 minutes depending on your internet speed.

### 8.3 Test the workflow
1. Open `http://YOUR_SERVER_IP:7860` in your browser
2. Upload a 5-10 second voice sample audio file
3. (Optional) Enter the transcript of the audio
4. Click **"Create Voice Clone"**
5. Go to **"Voice Chat"** tab
6. Type a message or record audio
7. The AI responds in the cloned voice!

Press `Ctrl+C` to stop the test.

---

## 9. Production Deployment with systemd

### 9.1 Create a systemd service file
```bash
sudo nano /etc/systemd/system/voice-clone.service
```

Paste this content:
```ini
[Unit]
Description=Voice Clone Chatbot (Gradio + Flask)
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=clone
Group=clone
WorkingDirectory=/home/clone/Voice_Clone
EnvironmentFile=/home/clone/Voice_Clone/.env
ExecStart=/home/clone/Voice_Clone/.venv/bin/python3 simple_voice_chat.py
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

# Resource limits
LimitNOFILE=65536
TimeoutStartSec=300

[Install]
WantedBy=multi-user.target
```

### 9.2 Enable and start the service
```bash
sudo systemctl daemon-reload
sudo systemctl enable voice-clone
sudo systemctl start voice-clone
```

### 9.3 Check status
```bash
sudo systemctl status voice-clone
```

### 9.4 View logs
```bash
# Live logs
sudo journalctl -u voice-clone -f

# Last 100 lines
sudo journalctl -u voice-clone -n 100
```

### 9.5 Restart / Stop
```bash
sudo systemctl restart voice-clone
sudo systemctl stop voice-clone
```

---

## 10. Firewall & Port Configuration

### 10.1 Using UFW (Ubuntu Firewall)
```bash
# Allow SSH (always do this first!)
sudo ufw allow 22/tcp

# Allow Gradio Web UI
sudo ufw allow 7860/tcp

# (Optional) Allow HTTP/HTTPS for Nginx reverse proxy
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# Enable firewall
sudo ufw enable

# Check status
sudo ufw status verbose
```

### 10.2 Cloud Provider Firewall
If using a cloud provider (DigitalOcean, AWS, GCP, etc.), also open ports in their firewall/security group:

**DigitalOcean:**
- Go to Networking → Firewalls → Create Firewall
- Add Inbound Rule: TCP, Port 7860, All IPv4/IPv6

**AWS (Security Group):**
- Add Inbound Rule: Custom TCP, Port 7860, Source 0.0.0.0/0

**GCP:**
- VPC Network → Firewall → Create Rule → tcp:7860

### 10.3 Port Summary
```
┌─────────────────────────────────────────────────┐
│  Internet                                       │
│    │                                            │
│    ▼ Port 7860 (or 80/443 via Nginx)            │
│  ┌─────────────────────────────────────┐        │
│  │  Gradio Web UI  (0.0.0.0:7860)     │        │
│  │  ┌─────────────────────────────┐    │        │
│  │  │ Flask API (127.0.0.1:5050)  │    │        │
│  │  │ (internal only, for phone)  │    │        │
│  │  └─────────────────────────────┘    │        │
│  └─────────────────────────────────────┘        │
│    │                                            │
│    ▼ Port 11434 (local only)                    │
│  ┌─────────────────────────────────────┐        │
│  │  Ollama LLM (optional)             │        │
│  └─────────────────────────────────────┘        │
└─────────────────────────────────────────────────┘
```

---

## 11. Reverse Proxy with Nginx (Optional)

> Use Nginx if you want to serve on port 80/443 with a domain name.

### 11.1 Install Nginx
```bash
sudo apt install -y nginx
```

### 11.2 Create Nginx config
```bash
sudo nano /etc/nginx/sites-available/voice-clone
```

Paste:
```nginx
server {
    listen 80;
    server_name your-domain.com;  # Replace with your domain or server IP

    # Increase timeouts for long TTS generation
    proxy_read_timeout 300s;
    proxy_send_timeout 300s;
    proxy_connect_timeout 60s;

    # Max upload size (for voice samples)
    client_max_body_size 50M;

    location / {
        proxy_pass http://127.0.0.1:7860;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### 11.3 Enable the site
```bash
sudo ln -s /etc/nginx/sites-available/voice-clone /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl restart nginx
```

Now access via `http://your-domain.com` (port 80).

---

## 12. SSL with Let's Encrypt (Optional)

```bash
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

Follow the prompts. Certbot auto-renews. Now access via `https://your-domain.com`.

---

## 13. Monitoring & Logs

### 13.1 Check if the app is running
```bash
sudo systemctl status voice-clone
```

### 13.2 View live logs
```bash
sudo journalctl -u voice-clone -f
```

### 13.3 Check resource usage
```bash
# Memory
free -h

# CPU and processes
htop

# Disk
df -h

# GPU (if available)
nvidia-smi
```

### 13.4 Auto-restart on crash
The systemd service already has `Restart=on-failure`. To also restart on reboot, the `WantedBy=multi-user.target` handles that.

---

## 14. Troubleshooting

### "Module not found" errors
```bash
# Make sure venv is activated
source ~/Voice_Clone/.venv/bin/activate
pip list | grep -i "whisper\|qwen\|gradio\|torch"
```

### "CUDA out of memory"
Use smaller models by editing `simple_voice_chat.py`:
- Change Whisper model from `"base"` to `"tiny"`
- Change TTS model from `1.7B` to `0.6B`

### "Connection refused" on port 7860
```bash
# Check if the process is running
sudo systemctl status voice-clone

# Check if port is listening
sudo ss -tlnp | grep 7860

# Check logs for errors
sudo journalctl -u voice-clone -n 50
```

### "Ollama connection refused" (port 11434)
```bash
systemctl status ollama
# Or start it:
ollama serve &
```

### "SoX not found"
```bash
# On Linux, install via apt (don't use the Windows DLLs in sox-14.4.2/)
sudo apt install -y sox libsox-dev
which sox
```

### Slow inference on CPU
- This is expected. TTS generation takes 30-60 seconds per response on CPU.
- Consider using a GPU server for real-time performance.
- Use the 0.6B TTS model (default) instead of 1.7B.

### Application won't start (takes too long)
- First startup downloads models (~3-4 GB total). Be patient.
- Check disk space: `df -h`
- The systemd service has `TimeoutStartSec=300` (5 minutes) to allow for model loading.

### Update the application
```bash
cd ~/Voice_Clone
git pull origin main
sudo systemctl restart voice-clone
```

---

## Quick Start Cheat Sheet

```bash
# === On a fresh Ubuntu server ===

# 1. System setup
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3 python3-pip python3-venv git curl wget ffmpeg sox libsox-dev build-essential libsndfile1 portaudio19-dev

# 2. Setup swap (if RAM < 8GB)
sudo fallocate -l 8G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# 3. Clone repo
cd ~ && git clone https://github.com/abdullah249/Voice_Clone.git && cd Voice_Clone

# 4. Python setup
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip setuptools wheel

# 5. Install PyTorch (CPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 6. Install app dependencies
pip install git+https://github.com/QwenLM/Qwen3-TTS.git
pip install openai-whisper sounddevice soundfile numpy librosa gradio requests openai flask

# 7. Configure API keys
cp .env.example .env
nano .env  # Fill in GROQ_API_KEY

# 8. Run!
set -a; source .env; set +a
python3 simple_voice_chat.py

# App available at http://YOUR_SERVER_IP:7860
```

---

**That's it! Your Voice Clone Chatbot should now be running on port 7860.**
