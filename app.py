import subprocess
import os
import streamlit as st
import warnings
import torch
import whisperx
from urllib.parse import urlparse, parse_qs
import re

warnings.filterwarnings("ignore")
original_load = torch.load
def patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = patched_load

st.set_page_config(layout="wide", page_title="YouTube Transcript", initial_sidebar_state="expanded")

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    .stDeployButton {display:none;}
    .sticky-video-wrapper {
        display: flex;
        justify-content: center;
    }
    .youtube-embed {
        aspect-ratio: 9 / 16;
        width: 100%;
        max-width: 350px;
    }
</style>
""", unsafe_allow_html=True)

AUDIO_FILE = "temp_audio.mp3"

def get_youtube_id(url):
    patterns = [
        r'youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})',
        r'youtu\.be/([a-zA-Z0-9_-]{11})',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def embed_youtube(url, start_time=0, autoplay=False):
    video_id = get_youtube_id(url)
    if video_id:
        autoplay_param = 1 if autoplay else 0
        embed_url = f"https://www.youtube.com/embed/{video_id}?start={int(start_time)}&autoplay={autoplay_param}"
        st.markdown(f'<div class="youtube-embed"><iframe width="100%" height="100%" src="{embed_url}" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen style="position: absolute; top: 0; left: 0;"></iframe></div>', unsafe_allow_html=True)
    else:
        st.error("Could not extract YouTube video ID")

def download_audio(url):
    try:
        urlparse(url)
    except Exception:
        raise ValueError("Invalid URL format")
    
    result = subprocess.run([
        "yt-dlp", "-x", "--audio-format", "mp3",
        "--audio-quality", "9",
        "-o", AUDIO_FILE, url
    ], capture_output=True, text=True, shell=False)
    
    if result.returncode != 0:
        st.error(f"yt-dlp failed: {result.stderr}")
        raise Exception(f"Download failed: {result.stderr}")

def transcribe():
    # Device detection with debug info
    print("=" * 50)
    print("DEVICE DETECTION:")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"MPS built: {torch.backends.mps.is_built()}")
    
    # WhisperX/faster-whisper doesn't support MPS yet, only CUDA and CPU
    if torch.cuda.is_available():
        device = "cuda"
        compute_type = "float16"
    else:
        device = "cpu"
        compute_type = "int8"
        print("Note: WhisperX doesn't support MPS (Apple Silicon GPU) yet.")
        print("Using CPU instead. For GPU acceleration, faster-whisper needs CUDA.")
    
    print(f"Selected device: {device}")
    print(f"Compute type: {compute_type}")
    
    try:
        model = whisperx.load_model("small", device, compute_type=compute_type)
        print(f"Model loaded successfully on device: {device}")
        print("=" * 50)
    except Exception as e:
        print(f"Failed to load on {device}: {e}")
        device = "cpu"
        compute_type = "int8"
        model = whisperx.load_model("small", device, compute_type=compute_type)
        print(f"Model loaded on fallback device: {device}")
        print("=" * 50)
    
    audio = whisperx.load_audio(AUDIO_FILE)
    result = model.transcribe(audio, batch_size=24, language="en")
    
    model_a, metadata = whisperx.load_align_model(language_code="en", device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device)
    
    return result.get("segments", [])

st.sidebar.title("Settings")
input_method = st.sidebar.radio("Input Method", ["YouTube URL", "Upload MP3"])

if input_method == "YouTube URL":
    url = st.sidebar.text_input("YouTube URL")
    delete_after = st.sidebar.checkbox("Delete audio after processing", value=False)
    
    if st.sidebar.button("Process Video"):
        if url:
            with st.spinner("Processing..."):
                download_audio(url)
                segments = transcribe()
                if delete_after and os.path.exists(AUDIO_FILE):
                    os.remove(AUDIO_FILE)
                st.session_state["segments"] = segments
                st.session_state["url"] = url
                st.session_state["time"] = 0
            st.success("Done!")
else:
    uploaded_file = st.sidebar.file_uploader("Upload MP3", type=["mp3"])
    url_for_video = st.sidebar.text_input("YouTube URL (for video player)")
    
    if st.sidebar.button("Process Audio"):
        if uploaded_file:
            with st.spinner("Processing..."):
                with open(AUDIO_FILE, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                segments = transcribe()
                st.session_state["segments"] = segments
                st.session_state["url"] = url_for_video if url_for_video else None
                st.session_state["time"] = 0
            st.success("Done!")

st.title("Interactive Transcript")

if "segments" in st.session_state:
    st.markdown('<div class="sticky-video-wrapper">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.session_state.get("url"):
            should_autoplay = st.session_state.get("autoplay", False)
            embed_youtube(st.session_state["url"], st.session_state.get("time", 0), autoplay=should_autoplay)
        else:
            st.info("No video URL provided - audio only mode")
    st.markdown('</div>', unsafe_allow_html=True)

    st.subheader("Transcript")
    with st.container(height=500):
        for i, seg in enumerate(st.session_state["segments"]):
            speaker = seg.get("speaker", f"Speaker {i%3+1}")
            text = seg.get("text", "").strip()
            start = seg.get("start", 0)
            
            mins = int(start // 60)
            secs = int(start % 60)
            
            cols = st.columns([0.15, 0.85])
            with cols[0]:
                if st.button(f"{mins:02d}:{secs:02d}", key=f"t_{i}", use_container_width=True):
                    st.session_state["time"] = start
                    st.session_state["autoplay"] = True
                    st.rerun()
            with cols[1]:
                st.write(f"**{speaker}:** {text}")
else:
    st.info("👈 Choose input method and click Process to get started")