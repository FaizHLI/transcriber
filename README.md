# Interactive YouTube Transcript Generator Kalshi

A Streamlit-based application that automatically transcribes YouTube videos into interactive, timestamped transcripts with speaker detection. Click any timestamp to jump the video player to that exact moment.

## Features

- **Automatic Transcription:** Uses OpenAI's Whisper model for accurate speech-to-text conversion
- **GPU Acceleration:** Automatically detects and uses CUDA (Nvidia) when available, with full CUDA 12.0+ support
- **Apple Silicon Support:** MLX Whisper for native GPU acceleration on M1/M2/M3 Macs (~3-5x faster than CPU)
- **Performance Timing:** Prints transcription duration and device info (GPU/CPU) to the terminal for benchmarking
- **Interactive Navigation:** Click any timestamp to jump the video to that exact point
- **Multiple Input Methods:** Process from YouTube URLs or upload local MP3 files
- **Voice Activity Detection:** Built-in VAD for better transcription quality

## Limitations

⚠️ **Important:** This application runs entirely on your local machine without cloud services.

- **CPU Processing:** Without a GPU, transcription is slow (~10-15 minutes per hour of video)
- **GPU Support:** GPU acceleration requires a compatible Nvidia GPU with CUDA. Full CUDA 12.0+ support via faster-whisper
- **Apple Silicon (M1/M2/M3):** Use MLX Whisper option for native GPU acceleration (~3-5x faster than CPU). MLX is automatically enabled if available.
- **Memory Usage:** Requires 8GB+ RAM (16GB+ recommended)
- **Model Size:** First run downloads ~500MB-1GB of models to `~/.cache/huggingface/`
- **Audio Quality:** Transcription accuracy depends on audio quality and clarity

**Recommended for:** Videos up to 2-3 hours on CPU, unlimited with a supported Nvidia GPU or Apple Silicon Mac

## How It Works

1. **Audio Download:** Downloads audio from YouTube using `yt-dlp`
2. **Transcription:** faster-whisper (or MLX on Apple Silicon) converts audio to text with segment-level timestamps
3. **Voice Activity Detection:** Built-in VAD filters out silence for better accuracy
4. **Interactive UI:** Streamlit renders an interactive transcript with clickable timestamps

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
git clone https://github.com/FaizHLI/transcriber.git
cd transcriber
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
git clone https://github.com/FaizHLI/transcriber.git
cd transcriber
pip3 install -r requirements.txt
```

**Note about Apple Silicon (M1/M2/M3):**

The app automatically detects Apple Silicon and offers MLX Whisper for native GPU acceleration. MLX provides ~3-5x faster transcription than CPU. If `mlx-whisper` is installed, you'll see a checkbox to enable it in the sidebar. Install with: `pip install mlx-whisper`

### Linux (Ubuntu/Debian)

```bash
sudo apt-get install python3.12 ffmpeg git
git clone https://github.com/FaizHLI/transcriber.git
cd transcriber
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

- **CUDA (Nvidia GPU):** Fastest processing (~3-4 minutes for 1 hour video with "small" model). Full CUDA 12.0+ support via faster-whisper
- **MLX (Apple Silicon GPU):** Native GPU acceleration (~3-5x faster than CPU, ~6-8 minutes for 1 hour video)
- **CPU (Intel/AMD/Apple Silicon without MLX):** Slower (~10-15 minutes for 1 hour video)

The app automatically detects and uses the best available device. On startup, it prints to the terminal whether GPU or CPU is being used.

## Performance Tuning

faster-whisper automatically optimizes batch processing internally. You can adjust performance by:

- **Model Size:** Smaller models (tiny, base) are faster but less accurate. "small" is the default balance
- **Compute Type:** Automatically set (float16 for GPU, int8 for CPU)
- **VAD Filter:** Enabled by default for better accuracy (can be disabled in code if needed)

**Model Recommendations:**
- **Speed priority:** Use "tiny" or "base" models
- **Accuracy priority:** Use "medium" or "large" models
- **Balanced (default):** "small" model

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "Python not found" | Reinstall Python and check "Add Python.exe to PATH" |
| "FFmpeg not found" | Restart your computer after installing FFmpeg |
| YouTube download fails | Update yt-dlp: `pip install -U yt-dlp` |
| Out of memory | Close other applications or use a smaller model (tiny/base) |
| Models won't download | Check internet connection. Models cache at `~/.cache/huggingface/` |
| Slow transcription | Use GPU if available, or enable MLX on Apple Silicon Macs |
| "module 'torchaudio' has no attribute 'AudioMetaData'" | This shouldn't occur with faster-whisper. If it does, reinstall: `pip install --upgrade faster-whisper` |
| GPU not detected | Check CUDA installation. The app will print device info on startup |

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

- **faster-whisper:** Fast Whisper implementation with CUDA 12.0+ support (CTranslate2 backend)
- **mlx-whisper:** (Optional) Apple Silicon GPU acceleration for M1/M2/M3 Macs
- **yt-dlp:** YouTube audio downloader
- **streamlit:** Web application framework
- **torch:** PyTorch for device detection (CUDA support)

PyTorch and other required packages are automatically resolved by pip.
