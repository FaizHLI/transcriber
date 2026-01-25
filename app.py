import subprocess
import os
import streamlit as st
import warnings
import torch
from faster_whisper import WhisperModel
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
    """Extract YouTube video ID from various URL formats"""
    patterns = [
        r'(?:youtube\.com|m\.youtube\.com)/watch\?v=([a-zA-Z0-9_-]{11})',  # Standard watch URL
        r'(?:youtube\.com|m\.youtube\.com)/watch\?.*[&?]v=([a-zA-Z0-9_-]{11})',  # v= with other params
        r'youtu\.be/([a-zA-Z0-9_-]{11})',  # Shortened youtu.be
        r'(?:youtube\.com|m\.youtube\.com)/embed/([a-zA-Z0-9_-]{11})',  # Embed URL
        r'(?:youtube\.com|m\.youtube\.com)/v/([a-zA-Z0-9_-]{11})',  # /v/ format
        r'(?:youtube\.com|m\.youtube\.com)/shorts/([a-zA-Z0-9_-]{11})',  # YouTube Shorts
        r'(?:youtube\.com|m\.youtube\.com)/live/([a-zA-Z0-9_-]{11})',  # YouTube Live
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
        st.error(f"Could not extract YouTube video ID from URL: {url[:50]}... Please check the URL format.")

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

def transcribe_faster_whisper(model_name="small", progress_bar=None, status_text=None):
    """Transcribe using faster-whisper (better CUDA 12.0+ support)"""
    if status_text:
        status_text.text("🔍 Detecting device...")
    if progress_bar:
        progress_bar.progress(0.25)
    
    # faster-whisper uses "cuda" or "cpu" for device
    # For compute_type: "float16" for CUDA, "int8" for CPU
    if torch.cuda.is_available():
        device = "cuda"
        compute_type = "float16"
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✅ Using GPU: {gpu_name} (CUDA {torch.version.cuda})")
    else:
        device = "cpu"
        compute_type = "int8"
        print("⚠️  Using CPU (GPU not available)")
    
    if status_text:
        status_text.text(f"📦 Loading faster-whisper model '{model_name}'...")
    if progress_bar:
        progress_bar.progress(0.3)
    
    # Initialize faster-whisper model
    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    
    if status_text:
        status_text.text("🎤 Transcribing audio...")
    if progress_bar:
        progress_bar.progress(0.4)
    
    # Transcribe with faster-whisper
    # segments is an iterator, info contains language and other metadata
    segments, info = model.transcribe(
        AUDIO_FILE,
        language="en",
        beam_size=5,
        vad_filter=True,  # Voice activity detection
        vad_parameters=dict(min_silence_duration_ms=500)
    )
    
    # Convert segments iterator to list format compatible with existing code
    result_segments = []
    for segment in segments:
        result_segments.append({
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip()
        })
    
    if status_text:
        status_text.text("✅ Transcription complete!")
    if progress_bar:
        progress_bar.progress(1.0)
    
    return result_segments

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
    help="Larger models are more accurate but slower. 'small' is the default."
)

# MLX Whisper option (only on Apple Silicon)
use_mlx = False
if IS_APPLE_SILICON:
    if MLX_AVAILABLE:
        use_mlx = st.sidebar.checkbox(
            "🚀 Use MLX Whisper (Apple Silicon GPU - FASTER)",
            value=True,
            help="Use MLX for GPU acceleration on Apple Silicon. MUCH faster than CPU!"
        )
        if use_mlx:
            st.sidebar.info("⚡ MLX enabled: ~3-5x faster transcription!")
    else:
        st.sidebar.info("Install mlx-whisper for GPU acceleration: `pip install mlx-whisper`")

# Advanced settings
with st.sidebar.expander("Advanced Settings"):
    if use_mlx:
        st.info("Batch size not applicable for MLX Whisper")
    else:
        st.info("faster-whisper uses beam_size=5 by default. No batch size configuration needed.")

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
                    segments = transcribe_faster_whisper(model_name, progress_bar, status_text)
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
                    segments = transcribe_faster_whisper(model_name, progress_bar, status_text)
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

def create_flexible_pattern(keyword):
    """
    Create a regex pattern that handles:
    - Plurals and possessives (s, s', 's)
    - Compound words (hyphenated or adjacent: "fire truck" matches "firetruck")
    - But excludes closed compounds (searching "fire" won't match "firetruck")
    - Ordinal forms for numbers
    - Multi-word phrases with flexible spacing
    """
    # Remove leading/trailing whitespace
    keyword = keyword.strip()
    
    # Split into words to handle multi-word keywords
    words = keyword.split()
    
    if len(words) == 1:
        # Single word - handle plurals, possessives, and compounds
        word = re.escape(words[0])
        
        # Check if it ends with a number (for ordinal forms like "January 6")
        has_number = bool(re.search(r'\d$', words[0]))
        
        if has_number:
            # Allow ordinal suffixes (st, nd, rd, th)
            # Capture the prefix (space/hyphen) in group 0, word in group 1
            pattern = rf'(^|\s|-)({word}(?:st|nd|rd|th)?)(?=\s|[.,!?;:\-]|$|\'s|\')'
        else:
            # Allow plurals (s, es) and possessives ('s, s', ')
            # Allow hyphenated compounds (pro-Palestine)
            # Exclude closed compounds: require word boundary or hyphen/space before and after
            # Capture the prefix (space/hyphen) in group 1, word in group 2
            pattern = rf'(^|\s|-)({word}(?:es|s)?(?:\'s|s\'|\')?)(?=\s|[.,!?;:\-]|$)'
    else:
        # Multi-word keyword - handle spacing variations and plurals on each word
        word_patterns = []
        for i, word in enumerate(words):
            escaped_word = re.escape(word)
            
            # Check if this word ends with a number (for ordinal forms)
            has_number = bool(re.search(r'\d$', word))
            
            # Last word gets special handling
            if i == len(words) - 1:
                if has_number:
                    # Add ordinal suffix support
                    word_patterns.append(rf'{escaped_word}(?:st|nd|rd|th)?')
                else:
                    # Add plural/possessive support
                    word_patterns.append(rf'{escaped_word}(?:es|s)?(?:\'s|s\'|\')?')
            else:
                # Middle words can have plurals (e.g., "Laws and order")
                word_patterns.append(rf'{escaped_word}(?:s)?')
        
        # Join with flexible spacing:
        # - Allow spaces, hyphens, or NO space (for compound matching)
        # - "fire truck" search will match "fire truck", "fire-truck", or "firetruck"
        inner_pattern = r'[\s\-]?'.join(word_patterns)
        # Capture the prefix (space/hyphen) in group 1, word phrase in group 2
        pattern = rf'(^|\s|-)({inner_pattern})(?=\s|[.,!?;:]|$|\'s|\')'
    
    return re.compile(pattern, re.IGNORECASE)

def search_keywords_in_segments(segments, keywords):
    """Search for keywords in transcript segments and return matches with context"""
    results = {}
    
    for keyword in keywords:
        keyword_clean = keyword.strip()
        if not keyword_clean:
            continue
            
        matches = []
        
        # Create flexible pattern for this keyword
        pattern = create_flexible_pattern(keyword_clean)
        
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
    
    with st.expander("ℹ️ How keyword matching works", expanded=False):
        st.markdown("""
        **✅ MATCHES:**
        - Plurals & possessives: `Immigrant` → Immigrants, Immigrant's
        - Compound words: `fire truck` → fire truck, firetruck
        - Hyphenated: `Palestine` → pro-Palestine, Palestine-Israel
        - Ordinals: `January 6` → January 6th
        - Context: `Elon` → Elon University, Elon Musk
        
        **❌ DOES NOT MATCH:**
        - Closed compounds: `fire` won't match firetruck (but `fire truck` will)
        - Inflections: `Immigrant` won't match Immigration
        - Different words: Single letter differences won't match
        
        **💡 TIP:** Use `/` to add spelling variations: `Zelensky / Zelenskiy / Zelenski`
        """)
    
    col_left, col_right = st.columns([1, 2])
    
    with col_left:
        keywords_input = st.text_area(
            "Paste keywords (one per line)",
            height=200,
            placeholder="Democrat\nFaith\nPro Life\nJanuary 6\nZelensky / Zelenskiy / Zelenski\nICE\nfire truck",
            help="""Smart matching includes:
• Plurals & possessives (Democrat → Democrats, Democrat's, eggs')
• Compound words (fire truck → firetruck, fire truck)
• Hyphenated forms (Palestine → pro-Palestine, couch → couch potato)
• Ordinals with numbers (January 6 → January 6th)
• Use / for spelling variations (Zelensky / Zelenskiy / Zelenski)
• Excludes: inflections (Immigrant ≠ Immigration), closed compounds (fire ≠ firetruck)"""
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
                            
                            # Highlight the keyword in the text using the flexible pattern
                            pattern = create_flexible_pattern(keyword)
                            
                            # Find the matched text and highlight it, preserving spacing
                            def highlight_match(match_obj):
                                # Group 1 is the prefix (^, space, or hyphen)
                                # Group 2 is the actual matched word
                                prefix = match_obj.group(1)
                                matched_text = match_obj.group(2)
                                
                                # If prefix is start of string (^), don't include it
                                if prefix in ('^', ''):
                                    prefix = ""
                                
                                return f"{prefix}**{matched_text.upper()}**"
                            
                            highlighted_text = pattern.sub(highlight_match, text)
                            
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
