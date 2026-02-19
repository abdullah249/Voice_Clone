#!/usr/bin/env python3
"""
ARI WebSocket Voice Agent for Asterisk
=======================================
Connects to Asterisk ARI via WebSocket and handles incoming phone calls.
Each call is answered, recorded (with silence detection), sent through the
voice-clone pipeline (Flask API on port 5050), and the AI response is
played back.  The loop repeats until the caller hangs up.

Flow per call:
  1. StasisStart → answer channel
  2. Short beep → start recording (dynamic silence detection)
  3. RecordingFinished → beep → send to pipeline
  4. Play TTS response WAV
  5. PlaybackFinished → go to step 2

Designed for real-time conversational feel: fast silence detection,
no hold music, minimal latency between turns.

Requirements:
  pip install aiohttp requests

Usage:
  python3 voice_agent_ari.py          # foreground
  nohup python3 voice_agent_ari.py &  # background
"""

import asyncio
import json
import logging
import os
import sys
import time
import uuid

import aiohttp
import requests

# ─── Configuration ──────────────────────────────────────────────────────────
ARI_HOST = "127.0.0.1"
ARI_PORT = 8088
ARI_USER = "voiceagent"
ARI_PASS = "voiceagent123"
ARI_APP  = "voiceagent"

ARI_WS   = f"ws://{ARI_HOST}:{ARI_PORT}/ari/events"
ARI_REST = f"http://{ARI_HOST}:{ARI_PORT}/ari"

FLASK_API = "http://127.0.0.1:5050"

# Where TTS response WAVs are saved for Asterisk to play back
PLAYBACK_DIR = "/tmp/ari_voice_agent"

# Recording parameters — tuned for real-time conversational feel
MAX_SILENCE_SEC = 2    # stop after 2s silence (ARI requires integer)
MAX_RECORD_SEC  = 10   # keep turns short for faster responses
DTMF_TERMINATE  = "#"  # caller can press # to finish early

# ─── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s  %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("ari-agent")

# ─── Call Session Tracker ───────────────────────────────────────────────────

class CallSession:
    """State for one active phone call."""
    def __init__(self, channel_id: str, caller_id: str):
        self.channel_id = channel_id
        self.caller_id  = caller_id
        self.call_id    = uuid.uuid4().hex[:8]
        self.turn       = 0
        self.state      = "new"   # new → listening → processing → speaking → …
        self.active     = True    # set False on hangup
        self.files      = []      # temp files to clean up

    def __repr__(self):
        return f"Call({self.call_id} from={self.caller_id} state={self.state})"


# channel_id → CallSession
active_calls: dict[str, CallSession] = {}

# ─── ARI REST Helpers ───────────────────────────────────────────────────────

def ari(method: str, path: str, **kwargs):
    """Send an ARI REST request.  Returns Response or None on error."""
    url = f"{ARI_REST}{path}"
    try:
        r = requests.request(
            method, url,
            auth=(ARI_USER, ARI_PASS),
            timeout=10,
            **kwargs,
        )
        if r.status_code >= 400:
            log.warning("ARI %s %s → %s: %s", method, path, r.status_code, r.text[:200])
        return r
    except Exception as e:
        log.error("ARI %s %s failed: %s", method, path, e)
        return None


def answer_channel(channel_id: str):
    return ari("POST", f"/channels/{channel_id}/answer")


def hangup_channel(channel_id: str, reason: str = "normal"):
    return ari("DELETE", f"/channels/{channel_id}", params={"reason": reason})


def play_media(channel_id: str, media: str, playback_id: str | None = None):
    """Play audio on a channel.
    media examples:
      sound:hello-world          → /var/lib/asterisk/sounds/hello-world
      sound:/tmp/ari_voice_agent/resp_abc_12345  → absolute path (no ext)
      recording:rec_name         → ARI stored recording
    """
    params = {"media": media}
    if playback_id:
        params["playbackId"] = playback_id
    return ari("POST", f"/channels/{channel_id}/play", params=params)


def start_recording(channel_id: str, name: str):
    return ari("POST", f"/channels/{channel_id}/record", params={
        "name":               name,
        "format":             "wav",
        "maxDurationSeconds":  int(MAX_RECORD_SEC),
        "maxSilenceSeconds":   int(MAX_SILENCE_SEC),
        "beep":               "false",   # we play our own beep for tighter control
        "terminateOn":         DTMF_TERMINATE,
        "ifExists":           "overwrite",
    })


def download_recording(name: str) -> bytes | None:
    """Download a stored recording file via ARI."""
    r = ari("GET", f"/recordings/stored/{name}/file")
    if r and r.status_code == 200:
        return r.content
    return None


def delete_recording(name: str):
    ari("DELETE", f"/recordings/stored/{name}")


def start_moh(channel_id: str):
    ari("POST", f"/channels/{channel_id}/moh", params={"mohClass": "default"})


def stop_moh(channel_id: str):
    ari("DELETE", f"/channels/{channel_id}/moh")


# ─── Pipeline (blocking – always run in executor) ──────────────────────────

def run_pipeline(audio_bytes: bytes, call_id: str) -> str | None:
    """Send recorded audio to Flask /api/pipeline.
    Returns the file-stem (no extension) of the saved WAV, or None."""
    try:
        resp = requests.post(
            f"{FLASK_API}/api/pipeline",
            files={"audio": ("recording.wav", audio_bytes, "audio/wav")},
            data={"call_id": call_id},
            timeout=60,
        )
        if resp.status_code == 200:
            os.makedirs(PLAYBACK_DIR, exist_ok=True)
            stem = os.path.join(PLAYBACK_DIR, f"resp_{call_id}_{int(time.time())}")
            wav  = stem + ".wav"
            with open(wav, "wb") as f:
                f.write(resp.content)
            log.info("Pipeline OK → %s (%d bytes)", wav, len(resp.content))
            return stem   # Asterisk wants path without extension
        else:
            log.error("Pipeline HTTP %s: %s", resp.status_code, resp.text[:300])
            return None
    except Exception as e:
        log.error("Pipeline request failed: %s", e)
        return None


def cleanup_call(call_id: str, files: list[str]):
    """Tell Flask to drop conversation history; delete temp files."""
    try:
        requests.post(
            f"{FLASK_API}/api/end_call",
            json={"call_id": call_id},
            timeout=5,
        )
    except Exception:
        pass
    for f in files:
        for path in (f, f + ".wav"):
            try:
                if os.path.exists(path):
                    os.unlink(path)
            except OSError:
                pass


# ─── ARI Event Handlers ────────────────────────────────────────────────────

async def on_stasis_start(event: dict):
    ch = event["channel"]
    channel_id = ch["id"]
    caller = ch.get("caller", {}).get("number", "unknown")

    sess = CallSession(channel_id, caller)
    active_calls[channel_id] = sess
    log.info("📞  New call: %s", sess)

    # Answer the channel
    answer_channel(channel_id)
    await asyncio.sleep(0.3)

    # Short beep to signal "speak now" then immediately start recording
    play_media(channel_id, "sound:custom/processing-beep")
    await asyncio.sleep(0.3)  # let beep play

    sess.state = "listening"
    sess.turn += 1
    rec_name = f"vc_{sess.call_id}_{sess.turn}"
    start_recording(channel_id, rec_name)
    log.info("🎙️   Listening: %s", rec_name)


async def on_recording_finished(event: dict):
    rec = event.get("recording", {})
    rec_name   = rec.get("name", "")
    target_uri = rec.get("target_uri", "")
    duration   = rec.get("duration", 0)
    talk_dur   = rec.get("talking_duration", 0)

    # Resolve channel from target_uri  "channel:<id>"
    channel_id = target_uri.split(":", 1)[1] if ":" in target_uri else None
    if not channel_id or channel_id not in active_calls:
        return
    sess = active_calls[channel_id]
    if not sess.active:
        return

    log.info("⏹️   Recording done: %s  dur=%ss talk=%ss", rec_name, duration, talk_dur)

    # If no speech detected, re-prompt
    if talk_dur is not None and talk_dur <= 0:
        log.info("   No speech – re-listening")
        delete_recording(rec_name)
        sess.turn += 1
        new_name = f"vc_{sess.call_id}_{sess.turn}"
        start_recording(channel_id, new_name)
        return

    sess.state = "processing"

    # Download the recording
    audio_bytes = download_recording(rec_name)
    if not audio_bytes:
        log.error("   Failed to download recording %s", rec_name)
        hangup_channel(channel_id)
        return

    # Quick beep to acknowledge we heard them — processing starts
    play_media(channel_id, "sound:custom/processing-beep")

    # Run pipeline in thread pool (blocking I/O)
    loop = asyncio.get_running_loop()
    response_stem = await loop.run_in_executor(
        None, run_pipeline, audio_bytes, sess.call_id,
    )

    # Clean up the ARI recording
    delete_recording(rec_name)

    # Caller may have hung up during processing
    if not sess.active or channel_id not in active_calls:
        log.info("   Call ended during processing")
        return

    if response_stem is None:
        log.error("   Pipeline failed – hanging up")
        play_media(channel_id, "sound:an-error-has-occurred")
        await asyncio.sleep(2)
        hangup_channel(channel_id)
        return

    # Track temp file for later cleanup
    sess.files.append(response_stem)

    # Play the TTS response
    sess.state = "speaking"
    pb_id = f"pb_{sess.call_id}_{sess.turn}"
    play_media(channel_id, f"sound:{response_stem}", playback_id=pb_id)
    log.info("🔊  Playing response: %s", response_stem)


async def on_playback_finished(event: dict):
    pb = event.get("playback", {})
    target_uri = pb.get("target_uri", "")
    channel_id = target_uri.split(":", 1)[1] if ":" in target_uri else None
    if not channel_id or channel_id not in active_calls:
        return

    sess = active_calls[channel_id]
    if sess.state != "speaking" or not sess.active:
        return

    # Quick beep then start next recording turn — no delay
    play_media(channel_id, "sound:custom/processing-beep")
    await asyncio.sleep(0.3)

    sess.state = "listening"
    sess.turn += 1
    rec_name = f"vc_{sess.call_id}_{sess.turn}"
    start_recording(channel_id, rec_name)
    log.info("🎙️   Listening: %s", rec_name)


async def on_stasis_end(event: dict):
    ch = event.get("channel", {})
    channel_id = ch.get("id", "")
    if channel_id not in active_calls:
        return

    sess = active_calls.pop(channel_id)
    sess.active = False
    log.info("📴  Call ended: %s", sess)

    # Background cleanup
    loop = asyncio.get_running_loop()
    loop.run_in_executor(None, cleanup_call, sess.call_id, sess.files)


# ─── WebSocket Event Loop ──────────────────────────────────────────────────

HANDLERS = {
    "StasisStart":       on_stasis_start,
    "RecordingFinished": on_recording_finished,
    "PlaybackFinished":  on_playback_finished,
    "StasisEnd":         on_stasis_end,
}


async def ws_loop():
    url = (
        f"{ARI_WS}"
        f"?api_key={ARI_USER}:{ARI_PASS}"
        f"&app={ARI_APP}"
        f"&subscribeAll=true"
    )

    while True:
        try:
            log.info("Connecting to ARI WebSocket …")
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(
                    url,
                    heartbeat=20,       # send ping every 20 s (aiohttp handles pong)
                    max_msg_size=2**20,
                    timeout=aiohttp.ClientWSTimeout(ws_close=5),
                ) as ws:
                    log.info("✅  Connected to ARI WebSocket")
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            try:
                                ev = json.loads(msg.data)
                                ev_type = ev.get("type", "")
                                handler = HANDLERS.get(ev_type)
                                if handler:
                                    asyncio.create_task(handler(ev))
                                elif ev_type not in ("Dial", "ChannelStateChange",
                                                      "ChannelVarset", "ChannelDialplan",
                                                      "ChannelConnectedLine"):
                                    log.debug("Unhandled event: %s", ev_type)
                            except Exception:
                                log.exception("Error handling ARI event")
                        elif msg.type in (aiohttp.WSMsgType.CLOSED,
                                          aiohttp.WSMsgType.ERROR):
                            log.warning("WebSocket msg type: %s", msg.type)
                            break
        except (aiohttp.ClientError, ConnectionRefusedError, OSError) as e:
            log.warning("WebSocket lost (%s) – reconnecting in 3 s …", e)
        except Exception:
            log.exception("Unexpected error in WS loop")
        await asyncio.sleep(3)


# ─── Entrypoint ─────────────────────────────────────────────────────────────

def main():
    os.makedirs(PLAYBACK_DIR, exist_ok=True)

    # Quick health-check against Flask API
    try:
        r = requests.get(f"{FLASK_API}/api/health", timeout=5)
        health = r.json()
        log.info("Flask API health: %s", health)
        if not health.get("voice_cloned"):
            log.info("Voice not cloned — phone pipeline will use Edge TTS (cloud).")
    except Exception as e:
        log.warning("Flask API not reachable (%s) – make sure simple_voice_chat.py is running", e)

    log.info("🚀  Voice Agent ARI starting …")
    log.info("   ARI WS:    %s  app=%s", ARI_WS, ARI_APP)
    log.info("   Flask API:  %s", FLASK_API)
    log.info("   Playback:   %s", PLAYBACK_DIR)
    asyncio.run(ws_loop())


if __name__ == "__main__":
    main()
