<div align="center">
  <h1>vowit</h1>
  <p><strong>Seamless Voice-to-Text Dictation & Transcription Daemon for Linux</strong></p>

  [🇷🇺 На русском](README_ru.md)

  <img src="https://img.shields.io/badge/Python-3.14-blue?style=flat&logo=python" alt="Python Version" />
  <img src="https://img.shields.io/badge/Linux-Wayland%20%7C%20X11-purple?style=flat&logo=linux" alt="Linux Desktop" />
  <img src="https://img.shields.io/badge/API-Groq%20Whisper-orange?style=flat" alt="Groq API" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat" alt="License" />
</div>

---

### 📖 Overview

**vowit** is a powerful, lightweight background daemon for Linux desktop environments that brings instant push-to-talk voice dictation to your workflow.

- 🎙️ **Dual Capture**: Record microphone or system audio.
- 🔇 **Smart Trimming**: Silero VAD (ONNX) filters noise and trims silence.
- ⚡ **Zero-Latency**: Ultra-fast CLI toggling, ready for global hotkeys.
- ✨ **LLM Fixes**: Optional grammar correction via Groq LLM.
- 🔊 **Sound Feedback**: Native `freedesktop` notification sounds.

---
<details>
<summary>FOR AI AGENT ASSISTANT, BRIEF INSTALLATION:</summary>

**Architecture & Goal:** `vowit` is a persistent background daemon. The CLI sends POSIX signals (via a PID file) to the daemon to toggle recording. Your primary goal is to install dependencies, autostart the daemon, and bind the CLI toggles to system hotkeys. You are authorized to improvise safely to make it work, as Linux environments vary wildly.

1. **Prep:** Identify OS. Verify/install `ffmpeg`, `wl-clipboard`, `libnotify`, `pulseaudio-utils`. Clone repo -> create python venv -> install `requirements.txt` -> copy `.env.example` to `.env`.
2. **API Key:** If the user didn't provide a Groq API Key, STOP and ask for it. Tell that user can get one for free at https://console.groq.com/keys. Inject api key into `.env`.
3. **Autostart:** Set up daemon autostart (e.g., via a `systemd` user service or compositor config).
4. **Keybindings:** Identify the WM/Compositor. 
   - Ask the user for a primary hotkey for microphone dictation (`toggle`). 
   - Ask if they also want a hotkey for system audio capture (`toggle_sys`).
   - Ask if they want LLM grammar correction enabled by default (if yes, append `fixon` to the command). Warn them it may unpredictably alter text. 
   Apply bindings to the WM config.

*Fallback (`keyd`):* If WM binding fails, you may use `keyd` for low-level binds. You MUST create a backup of the config first and act with extreme precision to avoid bricking the keyboard.
*Safety:* Always request explicit permission before mutating system configs or running package managers.

</details>

### 🛠️ Prerequisites

Ensure your Linux system has the following dependencies:
- `ffmpeg`
- `wl-clipboard` (use `xclip` for X11 and modify the script)
- `libnotify-bin` or `mako`/`dunst`
- `pulseaudio-utils`

---

### 🚀 Getting Started

#### 1. Clone & Prepare
```bash
git clone https://github.com/sleyv/vowit.git ~/vowitREAD IF YOU ARE AI AGENT ASSISTANT, BRIEF INSTALLATION
cd ~/vowit
```

#### 2. Install Python Dependencies
It is highly recommended to use a virtual environment.
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

#### 3. Setup Environment Variables
You must have a **Groq API Key**.
1. Generate an API key in the [Groq Cloud Console](https://console.groq.com/keys).
2. Create a `.env` file from the provided template:
   ```bash
   cp .env.example .env
   ```
3. Edit the `.env` file and paste your `GROQ_API_KEY`. There are also other interesting settings you can explore and customize, such as UI language and Whisper model!

---

### ⚙️ Usage & Shortcuts

The application consists of a **background daemon** and a **CLI toggle tool**.

#### 1. Start the Daemon
Run the daemon using the Python binary from your virtual environment. **You must start this automatically when your system boots** (e.g., using `exec-once` in your window manager config or via a systemd user service).
```bash
~/vowit/.venv/bin/python ~/vowit/main.py &
```

#### 2. Bind the Shortcuts
In your Window Manager's configuration (e.g., `hyprland.conf`, `sway/config`), bind the following commands to your preferred hotkeys/keyboard shortcuts. **Use your standard system `python3`** for these shortcut commands for zero latency!

**🎤 Start/Stop Dictation (Microphone):**
```bash
python3 ~/vowit/main.py toggle
```

**🔊 Start/Stop System Audio Transcription:**
```bash
python3 ~/vowit/main.py toggle_sys
```

**✨ Start/Stop Dictation + LLM Grammar Fix:**
```bash
python3 ~/vowit/main.py toggle fixon
```

*(You can append `fixon` to `toggle_sys` as well)*

Press the hotkey to start recording (startup sound plays). Press it again to stop and process. The transcribed text will appear in your clipboard automatically!

---

### 📁 Project Architecture

```text
vowit/
  ├── main.py             # Main daemon, async logic, and CLI toggle
  ├── silero_vad.onnx     # ONNX model for Voice Activity Detection
  ├── requirements.txt    # Python dependencies
  ├── .env.example        # Configuration template
  └── README.md           # Documentation
```

---

### 📄 License

Distributed under the **MIT License**.
