#!/usr/bin/env python3
"""
Voice Admin Panel — Manage cloned voices for the AI phone agent.
Runs on port 9090 with password protection.
Talks to the Flask API on port 5050 for voice operations.
"""

import os
import json
import time
import requests
from functools import wraps
from flask import (
    Flask, render_template_string, request, redirect,
    url_for, session, flash, Response, jsonify,
)

app = Flask(__name__)
app.secret_key = os.urandom(32).hex()

# Configuration
ADMIN_PASSWORD = os.getenv("VOICE_ADMIN_PASS", "changeme")
API_BASE = os.getenv("VOICE_API_BASE", "http://127.0.0.1:5050")
ADMIN_PORT = int(os.getenv("VOICE_ADMIN_PORT", "9090"))

# ============================================================
# Authentication
# ============================================================

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("authenticated"):
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


# ============================================================
# Routes
# ============================================================

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        if request.form.get("password") == ADMIN_PASSWORD:
            session["authenticated"] = True
            return redirect(url_for("index"))
        flash("Invalid password", "error")
    return render_template_string(LOGIN_HTML)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/")
@login_required
def index():
    """Main admin dashboard — list all cloned voices."""
    try:
        voices_resp = requests.get(f"{API_BASE}/api/voices", timeout=10)
        voices = voices_resp.json().get("voices", []) if voices_resp.ok else []
    except Exception:
        voices = []

    try:
        active_resp = requests.get(f"{API_BASE}/api/voices/active", timeout=5)
        active_data = active_resp.json() if active_resp.ok else {}
    except Exception:
        active_data = {}

    try:
        health_resp = requests.get(f"{API_BASE}/api/health", timeout=5)
        health = health_resp.json() if health_resp.ok else {"status": "offline"}
    except Exception:
        health = {"status": "offline"}

    return render_template_string(
        DASHBOARD_HTML,
        voices=voices,
        active=active_data.get("active"),
        voice_loaded=active_data.get("voice_loaded", False),
        health=health,
    )


@app.route("/activate/<dir_name>", methods=["POST"])
@login_required
def activate(dir_name):
    try:
        r = requests.post(
            f"{API_BASE}/api/voices/activate",
            json={"dir_name": dir_name},
            timeout=10,
        )
        if r.ok:
            flash(f"Voice '{dir_name}' activated!", "success")
        else:
            flash(f"Error: {r.json().get('error', 'Unknown')}", "error")
    except Exception as e:
        flash(f"Failed: {e}", "error")
    return redirect(url_for("index"))


@app.route("/rename/<dir_name>", methods=["POST"])
@login_required
def rename(dir_name):
    new_name = request.form.get("new_name", "").strip()
    if not new_name:
        flash("Name cannot be empty", "error")
        return redirect(url_for("index"))
    try:
        r = requests.post(
            f"{API_BASE}/api/voices/rename",
            json={"dir_name": dir_name, "new_name": new_name},
            timeout=10,
        )
        if r.ok:
            flash(f"Renamed to '{new_name}'", "success")
        else:
            flash(f"Error: {r.json().get('error', 'Unknown')}", "error")
    except Exception as e:
        flash(f"Failed: {e}", "error")
    return redirect(url_for("index"))


@app.route("/delete/<dir_name>", methods=["POST"])
@login_required
def delete(dir_name):
    try:
        r = requests.post(
            f"{API_BASE}/api/voices/delete",
            json={"dir_name": dir_name},
            timeout=10,
        )
        if r.ok:
            flash(f"Deleted '{dir_name}'", "success")
        else:
            flash(f"Error: {r.json().get('error', 'Unknown')}", "error")
    except Exception as e:
        flash(f"Failed: {e}", "error")
    return redirect(url_for("index"))


@app.route("/preview/<dir_name>", methods=["POST"])
@login_required
def preview(dir_name):
    """Generate and return a TTS preview for a voice."""
    text = request.form.get("text", "Hello, this is a preview of my cloned voice.")
    try:
        r = requests.post(
            f"{API_BASE}/api/voices/preview",
            json={"dir_name": dir_name, "text": text},
            timeout=120,
        )
        if r.ok:
            return Response(r.content, mimetype="audio/wav",
                            headers={"Content-Disposition": "inline"})
        flash(f"Preview error: {r.json().get('error', 'Unknown')}", "error")
    except Exception as e:
        flash(f"Preview failed: {e}", "error")
    return redirect(url_for("index"))


@app.route("/sample/<dir_name>")
@login_required
def sample(dir_name):
    """Serve the original voice sample audio."""
    try:
        r = requests.get(f"{API_BASE}/api/voices/sample/{dir_name}", timeout=10)
        if r.ok:
            return Response(r.content, mimetype=r.headers.get("Content-Type", "audio/wav"))
        return "No sample available", 404
    except Exception:
        return "Error fetching sample", 500


# ============================================================
# HTML Templates
# ============================================================

LOGIN_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Voice Admin - Login</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: #0f172a; color: #e2e8f0; min-height: 100vh;
         display: flex; align-items: center; justify-content: center; }
  .login-card { background: #1e293b; border-radius: 16px; padding: 48px;
                box-shadow: 0 25px 50px rgba(0,0,0,0.5); width: 400px; text-align: center; }
  .login-card h1 { font-size: 28px; margin-bottom: 8px; color: #f8fafc; }
  .login-card p { color: #94a3b8; margin-bottom: 32px; }
  .login-card input[type="password"] {
    width: 100%; padding: 14px 16px; border-radius: 8px; border: 1px solid #334155;
    background: #0f172a; color: #f8fafc; font-size: 16px; margin-bottom: 16px;
    outline: none; transition: border-color 0.2s; }
  .login-card input:focus { border-color: #3b82f6; }
  .login-card button {
    width: 100%; padding: 14px; border-radius: 8px; border: none;
    background: #3b82f6; color: white; font-size: 16px; font-weight: 600;
    cursor: pointer; transition: background 0.2s; }
  .login-card button:hover { background: #2563eb; }
  .flash-error { background: #7f1d1d; padding: 10px; border-radius: 8px;
                 margin-bottom: 16px; color: #fca5a5; }
</style>
</head>
<body>
<div class="login-card">
  <h1>🔒 Voice Admin</h1>
  <p>Enter password to manage voice clones</p>
  {% with messages = get_flashed_messages(with_categories=true) %}
    {% for cat, msg in messages %}
      <div class="flash-error">{{ msg }}</div>
    {% endfor %}
  {% endwith %}
  <form method="POST">
    <input type="password" name="password" placeholder="Admin password" autofocus required>
    <button type="submit">Login</button>
  </form>
</div>
</body>
</html>
"""

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Voice Admin Panel</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: #0f172a; color: #e2e8f0; min-height: 100vh; }
  .header { background: #1e293b; border-bottom: 1px solid #334155;
            padding: 16px 32px; display: flex; align-items: center; justify-content: space-between; }
  .header h1 { font-size: 22px; }
  .header h1 span { color: #3b82f6; }
  .header-right { display: flex; align-items: center; gap: 16px; }
  .status-badge { padding: 6px 14px; border-radius: 20px; font-size: 12px; font-weight: 600; }
  .status-ok { background: #064e3b; color: #6ee7b7; }
  .status-off { background: #7f1d1d; color: #fca5a5; }
  .logout-btn { color: #94a3b8; text-decoration: none; font-size: 14px; }
  .logout-btn:hover { color: #f8fafc; }
  .container { max-width: 1200px; margin: 0 auto; padding: 32px; }
  .flash { padding: 12px 20px; border-radius: 8px; margin-bottom: 20px; font-size: 14px; }
  .flash-success { background: #064e3b; color: #6ee7b7; }
  .flash-error { background: #7f1d1d; color: #fca5a5; }
  .info-bar { display: flex; gap: 16px; margin-bottom: 24px; flex-wrap: wrap; }
  .info-card { background: #1e293b; border-radius: 12px; padding: 20px;
               flex: 1; min-width: 200px; }
  .info-card .label { color: #94a3b8; font-size: 12px; text-transform: uppercase;
                      letter-spacing: 0.5px; margin-bottom: 4px; }
  .info-card .value { font-size: 24px; font-weight: 700; }
  .info-card .value.active { color: #6ee7b7; }
  h2 { font-size: 20px; margin-bottom: 16px; }
  .voice-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
                gap: 16px; }
  .voice-card { background: #1e293b; border-radius: 12px; padding: 24px;
                border: 2px solid transparent; transition: border-color 0.2s; position: relative; }
  .voice-card.is-active { border-color: #22c55e; }
  .voice-card .active-tag { position: absolute; top: 12px; right: 12px;
                            background: #064e3b; color: #6ee7b7; padding: 4px 12px;
                            border-radius: 12px; font-size: 11px; font-weight: 600; }
  .voice-name { font-size: 18px; font-weight: 600; margin-bottom: 8px; color: #f8fafc; }
  .voice-meta { color: #94a3b8; font-size: 13px; margin-bottom: 16px; }
  .voice-meta span { margin-right: 16px; }
  .voice-actions { display: flex; gap: 8px; flex-wrap: wrap; }
  .btn { padding: 8px 16px; border-radius: 6px; border: none; font-size: 13px;
         font-weight: 500; cursor: pointer; transition: all 0.2s; text-decoration: none;
         display: inline-flex; align-items: center; gap: 4px; }
  .btn-activate { background: #065f46; color: #6ee7b7; }
  .btn-activate:hover { background: #047857; }
  .btn-preview { background: #1e3a5f; color: #93c5fd; }
  .btn-preview:hover { background: #1e40af; }
  .btn-rename { background: #44403c; color: #fbbf24; }
  .btn-rename:hover { background: #57534e; }
  .btn-delete { background: #450a0a; color: #fca5a5; }
  .btn-delete:hover { background: #7f1d1d; }
  .btn-sample { background: #312e81; color: #a5b4fc; }
  .btn-sample:hover { background: #3730a3; }
  .btn:disabled { opacity: 0.5; cursor: not-allowed; }
  .empty-state { text-align: center; padding: 60px; color: #64748b; }
  .empty-state h3 { font-size: 20px; margin-bottom: 8px; color: #94a3b8; }
  .modal-overlay { display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%;
                   background: rgba(0,0,0,0.7); z-index: 100; align-items: center; justify-content: center; }
  .modal-overlay.active { display: flex; }
  .modal { background: #1e293b; border-radius: 16px; padding: 32px; width: 440px;
           box-shadow: 0 25px 50px rgba(0,0,0,0.5); }
  .modal h3 { margin-bottom: 16px; }
  .modal input[type="text"], .modal input[type="text"] {
    width: 100%; padding: 12px; border-radius: 8px; border: 1px solid #334155;
    background: #0f172a; color: #f8fafc; font-size: 14px; margin-bottom: 12px; }
  .modal-actions { display: flex; gap: 8px; justify-content: flex-end; }
  .btn-cancel { background: #334155; color: #94a3b8; }
  .btn-confirm { background: #3b82f6; color: white; }
  audio { width: 100%; margin-top: 8px; border-radius: 8px; }
  .preview-section { margin-top: 12px; padding-top: 12px; border-top: 1px solid #334155; }
  .preview-text { width: 100%; padding: 8px 12px; border-radius: 6px; border: 1px solid #334155;
                  background: #0f172a; color: #f8fafc; font-size: 13px; margin-bottom: 8px; }
  .spinner { display: inline-block; width: 14px; height: 14px; border: 2px solid #94a3b8;
             border-top-color: transparent; border-radius: 50%; animation: spin 0.6s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .cloning-link { color: #3b82f6; text-decoration: none; font-size: 14px; }
  .cloning-link:hover { text-decoration: underline; }
</style>
</head>
<body>

<div class="header">
  <h1>🎙️ <span>Voice</span> Admin Panel</h1>
  <div class="header-right">
    {% if health.status == 'ok' %}
      <span class="status-badge status-ok">● API Online</span>
    {% else %}
      <span class="status-badge status-off">● API Offline</span>
    {% endif %}
    <a href="{{ url_for('logout') }}" class="logout-btn">Logout</a>
  </div>
</div>

<div class="container">
  {% with messages = get_flashed_messages(with_categories=true) %}
    {% for cat, msg in messages %}
      <div class="flash flash-{{ cat }}">{{ msg }}</div>
    {% endfor %}
  {% endwith %}

  <div class="info-bar">
    <div class="info-card">
      <div class="label">Total Voices</div>
      <div class="value">{{ voices|length }}</div>
    </div>
    <div class="info-card">
      <div class="label">Active Voice</div>
      <div class="value active">{{ active or 'None' }}</div>
    </div>
    <div class="info-card">
      <div class="label">TTS Model</div>
      <div class="value" style="font-size:16px;">{{ health.get('tts_loaded', False) and '1.7B ✅' or 'Not Loaded ❌' }}</div>
    </div>
    <div class="info-card">
      <div class="label">Clone New Voice</div>
      <div class="value" style="font-size:14px;"><a href="http://{{ request.host.split(':')[0] }}:7860" target="_blank" class="cloning-link">Open Gradio UI →</a></div>
    </div>
  </div>

  <h2>Cloned Voices</h2>

  {% if voices %}
  <div class="voice-grid">
    {% for v in voices %}
    <div class="voice-card {{ 'is-active' if v.dir_name == active else '' }}">
      {% if v.dir_name == active %}
        <span class="active-tag">● ACTIVE</span>
      {% endif %}
      <div class="voice-name">{{ v.name or v.dir_name }}</div>
      <div class="voice-meta">
        <span>📅 {{ v.created }}</span>
        <span>🧠 {{ v.get('model', 'unknown') }}</span>
      </div>
      <div class="voice-actions">
        {% if v.dir_name != active %}
        <form method="POST" action="{{ url_for('activate', dir_name=v.dir_name) }}" style="display:inline;">
          <button class="btn btn-activate" type="submit">✅ Activate</button>
        </form>
        {% endif %}

        {% if v.has_sample %}
        <button class="btn btn-sample" onclick="playSample('{{ v.dir_name }}')">🔈 Sample</button>
        {% endif %}

        <button class="btn btn-preview" onclick="showPreview('{{ v.dir_name }}', '{{ v.name or v.dir_name }}')">🎵 Preview TTS</button>

        <button class="btn btn-rename" onclick="showRename('{{ v.dir_name }}', '{{ v.name or v.dir_name }}')">✏️ Rename</button>

        <form method="POST" action="{{ url_for('delete', dir_name=v.dir_name) }}" style="display:inline;"
              onsubmit="return confirm('Delete voice \\'{{ v.name or v.dir_name }}\\'? This cannot be undone.')">
          <button class="btn btn-delete" type="submit">🗑️ Delete</button>
        </form>
      </div>

      <!-- Audio player for sample/preview -->
      <div id="audio-{{ v.dir_name }}" class="preview-section" style="display:none;">
        <audio id="player-{{ v.dir_name }}" controls></audio>
      </div>
    </div>
    {% endfor %}
  </div>
  {% else %}
  <div class="empty-state">
    <h3>No Cloned Voices Yet</h3>
    <p>Go to the <a href="http://{{ request.host.split(':')[0] }}:7860" target="_blank" class="cloning-link">Gradio UI</a> to clone voices. They'll appear here automatically.</p>
  </div>
  {% endif %}
</div>

<!-- Rename Modal -->
<div class="modal-overlay" id="renameModal">
  <div class="modal">
    <h3>✏️ Rename Voice</h3>
    <form id="renameForm" method="POST">
      <input type="text" name="new_name" id="renameInput" placeholder="New name..." required>
      <div class="modal-actions">
        <button type="button" class="btn btn-cancel" onclick="closeModal('renameModal')">Cancel</button>
        <button type="submit" class="btn btn-confirm">Rename</button>
      </div>
    </form>
  </div>
</div>

<!-- Preview Modal -->
<div class="modal-overlay" id="previewModal">
  <div class="modal">
    <h3>🎵 Preview Voice: <span id="previewVoiceName"></span></h3>
    <form id="previewForm" method="POST" target="previewFrame">
      <input type="text" name="text" value="Hello, this is a preview of my cloned voice." class="preview-text">
      <div class="modal-actions">
        <button type="button" class="btn btn-cancel" onclick="closeModal('previewModal')">Cancel</button>
        <button type="submit" class="btn btn-confirm" id="previewBtn">
          Generate Preview
        </button>
      </div>
    </form>
    <div id="previewAudioArea" style="margin-top: 12px; display: none;">
      <audio id="previewPlayer" controls style="width: 100%;"></audio>
    </div>
    <iframe name="previewFrame" style="display:none;"></iframe>
  </div>
</div>

<script>
function showRename(dirName, currentName) {
  document.getElementById('renameForm').action = '/rename/' + dirName;
  document.getElementById('renameInput').value = currentName;
  document.getElementById('renameModal').classList.add('active');
  document.getElementById('renameInput').focus();
}

function showPreview(dirName, voiceName) {
  document.getElementById('previewForm').action = '/preview/' + dirName;
  document.getElementById('previewVoiceName').textContent = voiceName;
  document.getElementById('previewAudioArea').style.display = 'none';
  document.getElementById('previewModal').classList.add('active');

  // Override form submit to fetch audio via JS
  const form = document.getElementById('previewForm');
  form.onsubmit = async function(e) {
    e.preventDefault();
    const btn = document.getElementById('previewBtn');
    btn.innerHTML = '<span class="spinner"></span> Generating...';
    btn.disabled = true;

    try {
      const formData = new FormData(form);
      const resp = await fetch('/preview/' + dirName, { method: 'POST', body: formData });
      if (resp.ok) {
        const blob = await resp.blob();
        const url = URL.createObjectURL(blob);
        const player = document.getElementById('previewPlayer');
        player.src = url;
        document.getElementById('previewAudioArea').style.display = 'block';
        player.play();
      } else {
        const data = await resp.json().catch(() => ({}));
        alert('Preview failed: ' + (data.error || 'Unknown error'));
      }
    } catch(err) {
      alert('Preview failed: ' + err);
    } finally {
      btn.innerHTML = 'Generate Preview';
      btn.disabled = false;
    }
  };
}

function playSample(dirName) {
  const container = document.getElementById('audio-' + dirName);
  const player = document.getElementById('player-' + dirName);
  if (container.style.display === 'none') {
    container.style.display = 'block';
    player.src = '/sample/' + dirName;
    player.play();
  } else {
    container.style.display = 'none';
    player.pause();
  }
}

function closeModal(id) {
  document.getElementById(id).classList.remove('active');
}

// Close modals on overlay click
document.querySelectorAll('.modal-overlay').forEach(el => {
  el.addEventListener('click', function(e) {
    if (e.target === el) el.classList.remove('active');
  });
});

// Close modals on Escape
document.addEventListener('keydown', function(e) {
  if (e.key === 'Escape') {
    document.querySelectorAll('.modal-overlay.active').forEach(el => el.classList.remove('active'));
  }
});
</script>

</body>
</html>
"""

if __name__ == "__main__":
    print(f"🔒 Voice Admin Panel starting on port {ADMIN_PORT}")
    print(f"   Password: {'*' * len(ADMIN_PASSWORD)}")
    print(f"   API: {API_BASE}")
    app.run(host="0.0.0.0", port=ADMIN_PORT, debug=False)
