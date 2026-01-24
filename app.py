import subprocess
import os
import streamlit as st
import warnings
import torch
import whisperx
import platform
from urllib.parse import urlparse
import re
import time

# Check if we're on Apple Silicon Mac
IS_APPLE_SILICON = platform.machine() == "arm64" and platform.system() == "Darwin"

# Try to import MLX Whisper (only available on Mac)
MLX_AVAILABLE = False
if IS_APPLE_SILICON:
    try:
        import mlx_whisper
        MLX_AVAILABLE = True
    except ImportError:
        MLX_AVAILABLE = False

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
        position: relative;
        width: 100%;
        max-width: 560px;
        padding-bottom: 56.25%; /* 16:9 aspect ratio */
    }
    .youtube-embed iframe {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
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
        st.markdown(f'<div class="youtube-embed"><iframe src="{embed_url}" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe></div>', unsafe_allow_html=True)
    else:
        st.error("Could not extract YouTube video ID")

def download_audio(url, progress_bar=None, status_text=None):
    try:
        urlparse(url)
    except Exception:
        raise ValueError("Invalid URL format")
    
    # DELETE OLD AUDIO FILE FIRST
    if os.path.exists(AUDIO_FILE):
        os.remove(AUDIO_FILE)
    
    if status_text:
        status_text.text("📥 Downloading audio from YouTube...")
    if progress_bar:
        progress_bar.progress(0.1)
    
    result = subprocess.run([
        "yt-dlp", "-x", "--audio-format", "mp3",
        "--audio-quality", "9",
        "--force-overwrites",
        "-o", AUDIO_FILE, url
    ], capture_output=True, text=True, shell=False)
    
    if result.returncode != 0:
        st.error(f"yt-dlp failed: {result.stderr}")
        raise Exception(f"Download failed: {result.stderr}")
    
    if not os.path.exists(AUDIO_FILE):
        raise Exception("Audio file was not created")
    
    if status_text:
        status_text.text("✅ Audio downloaded successfully")
    if progress_bar:
        progress_bar.progress(0.2)

def transcribe_whisperx(model_name="small", batch_size=24, progress_bar=None, status_text=None):
    """Transcribe using WhisperX"""
    if status_text:
        status_text.text("🔍 Detecting device...")
    if progress_bar:
        progress_bar.progress(0.25)
    
    if torch.cuda.is_available():
        device = "cuda"
        compute_type = "float16"
    else:
        device = "cpu"
        compute_type = "int8"
    
    if status_text:
        status_text.text(f"📦 Loading WhisperX model '{model_name}'...")
    if progress_bar:
        progress_bar.progress(0.3)
    
    model = whisperx.load_model(model_name, device, compute_type=compute_type)
    
    if status_text:
        status_text.text("🎤 Transcribing audio...")
    if progress_bar:
        progress_bar.progress(0.4)
    
    audio = whisperx.load_audio(AUDIO_FILE)
    result = model.transcribe(audio, batch_size=batch_size, language="en")
    
    if status_text:
        status_text.text("🔗 Aligning transcription...")
    if progress_bar:
        progress_bar.progress(0.7)
    
    model_a, metadata = whisperx.load_align_model(language_code="en", device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device)
    
    if status_text:
        status_text.text("✅ Transcription complete!")
    if progress_bar:
        progress_bar.progress(1.0)
    
    return result.get("segments", [])

def transcribe_mlx(model_name="small", progress_bar=None, status_text=None):
    """Transcribe using MLX Whisper (Apple Silicon)"""
    if status_text:
        status_text.text("🔍 Using MLX Whisper (Apple Silicon GPU)...")
    if progress_bar:
        progress_bar.progress(0.25)
    
    mlx_model_name = f"mlx-community/whisper-{model_name}-mlx"
    
    if status_text:
        status_text.text("📦 Loading MLX model...")
    if progress_bar:
        progress_bar.progress(0.3)
    
    if status_text:
        status_text.text("🎤 Transcribing with GPU acceleration...")
    if progress_bar:
        progress_bar.progress(0.4)
    
    try:
        result = mlx_whisper.transcribe(AUDIO_FILE, path_or_hf_repo=mlx_model_name)
    except:
        result = mlx_whisper.transcribe(AUDIO_FILE)
    
    if status_text:
        status_text.text("✅ Transcription complete!")
    if progress_bar:
        progress_bar.progress(1.0)
    
    # Convert to pure Python types to avoid MLX array issues with Streamlit
    segments = result.get("segments", [])
    clean_segments = []
    for seg in segments:
        clean_seg = {
            "start": float(seg.get("start", 0)),
            "end": float(seg.get("end", 0)),
            "text": str(seg.get("text", ""))
        }
        clean_segments.append(clean_seg)
    
    return clean_segments

# ============ SIDEBAR ============
st.sidebar.title("Settings")
input_method = st.sidebar.radio("Input Method", ["YouTube URL", "Upload MP3"])

model_name = st.sidebar.selectbox(
    "Model Size",
    ["tiny", "base", "small", "medium", "large"],
    index=2,
    help="Larger models are more accurate but slower"
)

# MLX Whisper option (only on Apple Silicon)
use_mlx = False
if IS_APPLE_SILICON:
    if MLX_AVAILABLE:
        use_mlx = st.sidebar.checkbox(
            "Use MLX Whisper (Apple Silicon GPU)",
            value=False,
            help="Use MLX for GPU acceleration on Apple Silicon"
        )
    else:
        st.sidebar.info("Install mlx-whisper for GPU acceleration: `pip install mlx-whisper`")

# Advanced settings (only for WhisperX)
with st.sidebar.expander("Advanced Settings"):
    if use_mlx:
        st.info("Batch size not applicable for MLX Whisper")
        batch_size = 24
    else:
        batch_size = st.slider("Batch Size", 1, 64, 24, help="Higher = faster but more memory")

# ============ PROCESSING ============
if input_method == "YouTube URL":
    url = st.sidebar.text_input("YouTube URL")
    delete_after = st.sidebar.checkbox("Delete audio after processing", value=False)
    
    if st.sidebar.button("Process Video"):
        if url:
            st.session_state.pop("segments", None)
            st.session_state.pop("url", None)
            st.session_state.pop("time", None)
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                download_audio(url, progress_bar, status_text)
                
                transcribe_start = time.time()
                if use_mlx and MLX_AVAILABLE:
                    segments = transcribe_mlx(model_name, progress_bar, status_text)
                else:
                    segments = transcribe_whisperx(model_name, batch_size, progress_bar, status_text)
                transcribe_elapsed = time.time() - transcribe_start
                print(f"Transcription completed in {transcribe_elapsed:.2f} seconds")
                
                if delete_after and os.path.exists(AUDIO_FILE):
                    os.remove(AUDIO_FILE)
                
                st.session_state["segments"] = segments
                st.session_state["url"] = url
                st.session_state["time"] = 0
                
                time.sleep(0.5)
                progress_bar.empty()
                status_text.empty()
                st.success("Done!")
                st.rerun()
                
            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"Error: {str(e)}")
else:
    uploaded_file = st.sidebar.file_uploader("Upload MP3", type=["mp3"])
    url_for_video = st.sidebar.text_input("YouTube URL (for video player)")
    
    if st.sidebar.button("Process Audio"):
        if uploaded_file:
            st.session_state.pop("segments", None)
            st.session_state.pop("url", None)
            st.session_state.pop("time", None)
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                status_text.text("💾 Saving audio file...")
                progress_bar.progress(0.05)
                
                if os.path.exists(AUDIO_FILE):
                    os.remove(AUDIO_FILE)
                
                with open(AUDIO_FILE, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                transcribe_start = time.time()
                if use_mlx and MLX_AVAILABLE:
                    segments = transcribe_mlx(model_name, progress_bar, status_text)
                else:
                    segments = transcribe_whisperx(model_name, batch_size, progress_bar, status_text)
                transcribe_elapsed = time.time() - transcribe_start
                print(f"Transcription completed in {transcribe_elapsed:.2f} seconds")
                
                st.session_state["segments"] = segments
                st.session_state["url"] = url_for_video if url_for_video else None
                st.session_state["time"] = 0
                
                time.sleep(0.5)
                progress_bar.empty()
                status_text.empty()
                st.success("Done!")
                st.rerun()
                
            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"Error: {str(e)}")

def search_keywords_in_segments(segments, keywords):
    """Search for keywords in transcript segments and return matches with context"""
    results = {}
    
    for keyword in keywords:
        keyword_clean = keyword.strip()
        if not keyword_clean:
            continue
            
        matches = []
        # Create regex pattern for case-insensitive word boundary matching
        pattern = re.compile(r'\b' + re.escape(keyword_clean) + r'\b', re.IGNORECASE)
        
        for i, seg in enumerate(segments):
            text = seg.get("text", "").strip()
            if pattern.search(text):
                matches.append({
                    "segment_index": i,
                    "timestamp": seg.get("start", 0),
                    "text": text
                })
        
        if matches:
            results[keyword_clean] = matches
    
    return results

# ============ DISPLAY ============
st.title("Interactive Transcript")

if "segments" in st.session_state and st.session_state["segments"]:
    if st.button("🗑️ Clear Transcript"):
        st.session_state.pop("segments", None)
        st.session_state.pop("url", None)
        st.session_state.pop("time", None)
        st.rerun()
    
    st.markdown('<div class="sticky-video-wrapper">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.session_state.get("url"):
            should_autoplay = st.session_state.get("autoplay", False)
            embed_youtube(st.session_state["url"], st.session_state.get("time", 0), autoplay=should_autoplay)
            st.session_state["autoplay"] = False
        else:
            st.info("No video URL provided - audio only mode")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # ============ KEYWORD SEARCH ============
    st.subheader("🔍 Keyword Search")
    
    col_left, col_right = st.columns([1, 2])
    
    with col_left:
        keywords_input = st.text_area(
            "Paste keywords (one per line)",
            height=200,
            placeholder="Adopt / Adoption\nFaith\nPro Life / Pro-Life\nUnborn\nFamily",
            help="Enter keywords separated by newlines. Use / to separate variations (e.g., 'Pro Life / Pro-Life')"
        )
        
        if st.button("🔎 Search Keywords", use_container_width=True):
            if keywords_input.strip():
                # Parse keywords - handle variations separated by /
                raw_keywords = keywords_input.strip().split('\n')
                all_keywords = []
                for kw in raw_keywords:
                    # Split by / to get variations
                    variations = [v.strip() for v in kw.split('/') if v.strip()]
                    all_keywords.extend(variations)
                
                # Perform search
                results = search_keywords_in_segments(st.session_state["segments"], all_keywords)
                st.session_state["keyword_results"] = results
                st.session_state["searched_keywords"] = all_keywords
            else:
                st.warning("Please enter at least one keyword")
    
    with col_right:
        if "keyword_results" in st.session_state and st.session_state["keyword_results"]:
            results = st.session_state["keyword_results"]
            searched_keywords = st.session_state.get("searched_keywords", [])
            
            # Summary
            total_matches = sum(len(matches) for matches in results.values())
            found_count = len(results)
            searched_count = len(searched_keywords)
            
            st.markdown(f"**Found {found_count}/{searched_count} keywords with {total_matches} total occurrences**")
            
            # Display results with scrollable container
            with st.container(height=400):
                # Show found keywords
                for keyword, matches in sorted(results.items()):
                    with st.expander(f"**{keyword}** ({len(matches)} occurrence{'s' if len(matches) != 1 else ''})", expanded=False):
                        for idx, match in enumerate(matches):
                            timestamp = match["timestamp"]
                            mins = int(timestamp // 60)
                            secs = int(timestamp % 60)
                            text = match["text"]
                            
                            # Highlight the keyword in the text
                            pattern = re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)
                            highlighted_text = pattern.sub(f"**{keyword.upper()}**", text)
                            
                            col_time, col_text = st.columns([0.2, 0.8])
                            with col_time:
                                if st.button(f"⏩ {mins:02d}:{secs:02d}", key=f"kw_{keyword}_{idx}", use_container_width=True):
                                    st.session_state["time"] = timestamp
                                    st.session_state["autoplay"] = True
                                    st.rerun()
                            with col_text:
                                st.markdown(highlighted_text)
                
                # Show keywords not found
                not_found = [kw for kw in searched_keywords if kw not in results]
                if not_found:
                    with st.expander("❌ Keywords not found", expanded=False):
                        for kw in not_found:
                            st.text(f"• {kw}")
        elif "searched_keywords" in st.session_state:
            st.info("No matches found for the searched keywords")
    
    st.divider()
    
    # ============ FULL TRANSCRIPT ============
    st.subheader("📝 Full Transcript")
    with st.container(height=500):
        for i, seg in enumerate(st.session_state["segments"]):
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
                st.write(text)
else:
    st.info("👈 Choose input method and click Process to get started")
