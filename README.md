# Interactive YouTube Transcript Generator

A Streamlit-based application that automatically transcribes YouTube videos into interactive, timestamped transcripts with speaker detection. Click any timestamp to jump the video player to that exact moment.

## Features

- **Automatic Transcription:** Uses OpenAI's Whisper model for accurate speech-to-text conversion
- **Speaker Detection:** Identifies different speakers and labels them in the transcript
- **GPU Acceleration:** Automatically detects and uses CUDA (Nvidia) or MPS (Apple Silicon) when available
- **Interactive Navigation:** Click any timestamp to jump the video to that exact point
- **Multiple Input Methods:** Process from YouTube URLs or upload local MP3 files
- **Two-step Alignment:** Uses WhisperX's alignment model for precise timestamps

## How It Works

1. **Audio Download:** Downloads audio from YouTube using `yt-dlp`
2. **Transcription:** Whisper model converts audio to text with word-level timestamps
3. **Alignment:** WhisperX aligns transcription with original audio for accuracy
4. **Speaker Detection:** Identifies different speakers in multi-speaker audio
5. **Interactive UI:** Streamlit renders an interactive transcript with clickable timestamps

## Prerequisites

Before installing, ensure you have:

- **Python 3.12+:** The programming language
- **FFmpeg:** For audio/video processing
- **Git:** To clone this repository

---

## Installation

### Windows

#### 1. Install Python 3.12

- Download the Windows Installer (64-bit) from [python.org](https://www.python.org/downloads/windows/)
- **Important:** Check "Add Python.exe to PATH" before clicking Install

#### 2. Install FFmpeg

- Open PowerShell as Administrator (search "PowerShell" → right-click → Run as Administrator)
- Run: `winget install ffmpeg`
- Restart your computer

#### 3. Clone and Install Dependencies

```cmd
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
pip install -r requirements.txt
```

That's it! All dependencies are automatically resolved.

### macOS

#### 1. Install Homebrew

Open Terminal and run:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

#### 2. Install Dependencies

```bash
brew install python@3.12 ffmpeg git
```

#### 3. Clone and Install

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
pip3 install -r requirements.txt
```

**MPS Support (Apple Silicon M1/M2/M3):**

MPS is enabled by default on Apple Silicon Macs. No additional installation needed—the app will automatically detect and use your GPU.

### Linux (Ubuntu/Debian)

```bash
sudo apt-get install python3.12 ffmpeg git
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
pip install -r requirements.txt
```

---

## Quick Start

1. Open Terminal/Command Prompt and navigate to the repository:
```bash
cd path/to/your/repo
```

2. Run the application:
```bash
streamlit run app.py
```

3. The app will open in your browser at `http://localhost:8501`

---

## Usage

### YouTube URL

1. Select "YouTube URL" from the sidebar
2. Paste a YouTube link
3. (Optional) Check "Delete audio after processing" to save storage
4. Click "Process Video"
5. Wait for processing (first run downloads ~500MB of models)
6. Click timestamps to jump the video player

### Upload MP3

1. Select "Upload MP3" from the sidebar
2. Upload an MP3 file
3. (Optional) Paste a YouTube URL to display the video player
4. Click "Process Audio"
5. Navigate using the interactive transcript

---

## Performance Notes

- **CUDA (Nvidia GPU):** Fastest processing (~1-2 minutes for 1 hour video)
- **MPS (Apple Silicon):** Fast processing (~2-3 minutes for 1 hour video)
- **CPU:** Slower (~10-15 minutes for 1 hour video)

The app automatically detects and uses the best available device.

## Performance Tuning

You can optimize performance by adjusting the batch size in `app.py`:

**Line 91:** `result = model.transcribe(audio, batch_size=96, language="en")`

- **Higher batch size (128, 256):** Faster processing but uses more RAM
- **Default, Lower batch size (32, 48):** Slower but more memory-efficient
- **(96):** Optimized for 32GB+ RAM systems

**Recommendations:**
- **32GB+ RAM:** `batch_size=96-128` (recommended default)
- **16GB RAM:** `batch_size=64`
- **8GB RAM:** `batch_size=32-48`
- **M1 MacBook Pro (8GB unified memory):** `batch_size=16-32` max

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "Python not found" | Reinstall Python and check "Add Python.exe to PATH" |
| "FFmpeg not found" | Restart your computer after installing FFmpeg |
| YouTube download fails | Update yt-dlp: `pip install -U yt-dlp` |
| Out of memory | Reduce batch size in `app.py` or close other applications |
| Models won't download | Check internet connection. Models cache at `~/.cache/huggingface/` |
| Slow transcription | Increase batch size in `app.py` if you have RAM available |

---

## Project Structure

```
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

---

## Dependencies

All dependencies are automatically installed via `pip install -r requirements.txt`:

- **whisperx:** Speech-to-text transcription with alignment
- **yt-dlp:** YouTube audio downloader
- **streamlit:** Web application framework

PyTorch and other required packages are automatically resolved by pip.