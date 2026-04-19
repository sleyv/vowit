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

**vowit** is a powerful, lightweight background daemon for Linux desktop environments (like Hyprland, Sway, KDE, GNOME) that brings instant push-to-talk voice dictation to your workflow.

It captures audio directly from your microphone or system output, automatically trims silence using an ONNX Voice Activity Detection (VAD) model, and transcribes the speech using the lightning-fast Groq Whisper API. The transcribed text is instantly copied to your clipboard.

With built-in LLM grammar correction capabilities, desktop notifications, and zero-latency CLI toggling, it acts as the ultimate voice-typing assistant for power users.

---

### ✨ Key Features

- **🚀 Ultra-Fast Toggling**: Designed to be bound to global keyboard shortcuts. Sending signals to the daemon takes milliseconds and doesn't require loading a Python virtual environment.
- **🎙️ Dual Capture Modes**: Record from your microphone for dictation, or capture internal system audio (e.g., YouTube, Discord) for transcription.
- **🔇 Smart Silence Trimming**: Uses a lightweight Silero VAD (ONNX) pipeline and TrueRMS limiter to filter out background noise and trim silence locally before sending data, saving bandwidth and improving accuracy.
- **✨ LLM Grammar Fixing (Optional)**: Automatically pass your transcribed text through a Groq LLM to gently fix grammar, phonetic miswordings, and add paragraphs.
- **🔊 Rich Sound Design**: Utilizes native `freedesktop` sounds to give auditory feedback on 4 distinct states: Start, Stop/Processing, Success, and Error.
- **⏱️ Auto-Timeout**: A built-in 10-minute safety limit automatically stops recording and processes your audio if you forget to turn it off.

---

### 🛠️ Prerequisites

Before installing, ensure your Linux system has the following dependencies installed via your package manager (`apt`, `pacman`, `dnf`):
- `ffmpeg` (for audio capture and format conversion)
- `wl-clipboard` (specifically `wl-copy` for Wayland clipboard support. Use `xclip` if on X11 and modify the script accordingly)
- `libnotify-bin` or `mako`/`dunst` (for `notify-send`)
- `pulseaudio-utils` (for `paplay` sound playback)

---

### 🚀 Getting Started

#### 1. Clone & Prepare
```bash
git clone https://github.com/yourusername/vowit.git
cd vowit
```

#### 2. Install Python Dependencies
It is highly recommended to use a virtual environment.
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

#### 3. Setup Environment Variables
To use this tool, you must have a **Groq API Key**. It's free.
1. Go to the [Groq Cloud Console](https://console.groq.com/keys) and generate a new API key.
2. Create a `.env` file from the provided template:
   ```bash
   cp .env.example .env
# Edit .env and paste your GROQ_API_KEY
```

---

### ⚙️ Usage & Shortcuts

The application is split into a **background daemon** and a **CLI toggle tool**.

#### 1. Start the Daemon (On System Boot)
Run the daemon using the Python binary from your virtual environment. It will start silently in the background and wait for signals.
```bash
/path/to/vowit/.venv/bin/python /path/to/vowit/main.py &
```

#### 2. Bind the Shortcuts
In your Window Manager's configuration file (e.g., `hyprland.conf`, `sway/config`), bind the following commands to your preferred keys.

**Important:** Because we wrote a smart "fast-path" into the script, you **do not** need to use the virtual environment binary for these shortcut commands! Using your standard system `python3` is recommended, as it bypasses loading heavy dependencies, making your push-to-talk hotkeys react instantly (zero latency).

**🎤 Toggle Dictation (Microphone):**
```bash
python3 /path/to/vowit/main.py toggle
```

**🔊 Toggle System Audio Transcription:**
```bash
python3 /path/to/vowit/main.py toggle_sys
```

**✨ Toggle Dictation + LLM Grammar Fix:**
```bash
python3 /path/to/vowit/main.py toggle fixon
```

*(You can append `fixon` to `toggle_sys` as well!)*

Press the hotkey once to start recording (you will hear a startup sound). Press it again to stop and process the audio. The transcribed text will appear in your clipboard automatically!

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
