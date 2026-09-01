"""
Retake Detection Demo — Streamlit Application

A visual, interactive demo for the retake detection pipeline.
Upload a video (or select a sample), run the full pipeline,
and see the cleaned result with timeline visualization.

Launch:
    streamlit run app/demo.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Project path setup
# ---------------------------------------------------------------------------
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Retake Detection",
    page_icon="✂️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
VIDEOS_DIR = _project_root / "data" / "videos"
OUTPUTS_DIR = _project_root / "data" / "processed" / "outputs"
CUTLISTS_DIR = _project_root / "data" / "processed" / "cutlists"


# ---------------------------------------------------------------------------
# CSS Styling
# ---------------------------------------------------------------------------
def inject_styles():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Hero header */
    .hero-header {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .hero-header h1 {
        color: #ffffff;
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .hero-header p {
        color: rgba(255, 255, 255, 0.65);
        font-size: 1rem;
        margin: 0.4rem 0 0 0;
        font-weight: 300;
    }

    /* Stat cards */
    .stat-card {
        background: linear-gradient(145deg, #1e1e2e 0%, #2a2a3e 100%);
        padding: 1.25rem;
        border-radius: 12px;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.06);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .stat-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
    }
    .stat-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #a78bfa;
        margin: 0;
    }
    .stat-label {
        font-size: 0.8rem;
        color: rgba(255, 255, 255, 0.5);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin: 0.25rem 0 0 0;
    }

    /* Timeline */
    .timeline-container {
        background: #1a1a2e;
        border-radius: 12px;
        padding: 1.25rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.06);
    }
    .timeline-bar {
        height: 40px;
        border-radius: 6px;
        position: relative;
        overflow: hidden;
        background: #2a2a3e;
    }
    .timeline-segment {
        position: absolute;
        height: 100%;
        transition: opacity 0.2s ease;
    }
    .timeline-segment:hover {
        opacity: 0.85;
    }
    .timeline-keep {
        background: linear-gradient(180deg, #22c55e 0%, #16a34a 100%);
    }
    .timeline-discard {
        background: linear-gradient(180deg, #ef4444 0%, #dc2626 100%);
    }
    .timeline-labels {
        display: flex;
        justify-content: space-between;
        margin-top: 0.5rem;
        font-size: 0.75rem;
        color: rgba(255, 255, 255, 0.4);
    }

    /* Retake group cards */
    .retake-card {
        background: linear-gradient(145deg, #1e1e2e 0%, #252538 100%);
        padding: 1.25rem;
        border-radius: 12px;
        margin: 0.75rem 0;
        border-left: 4px solid #ef4444;
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-left: 4px solid #ef4444;
    }
    .retake-card .keep-badge {
        background: rgba(34, 197, 94, 0.15);
        color: #22c55e;
        padding: 0.15rem 0.6rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .retake-card .discard-badge {
        background: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        padding: 0.15rem 0.6rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    /* Legend */
    .legend {
        display: flex;
        gap: 1.5rem;
        margin-top: 0.75rem;
    }
    .legend-item {
        display: flex;
        align-items: center;
        gap: 0.4rem;
        font-size: 0.8rem;
        color: rgba(255, 255, 255, 0.6);
    }
    .legend-dot {
        width: 12px;
        height: 12px;
        border-radius: 3px;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Sidebar tweaks */
    .css-1d391kg {
        background: #0f0c29;
    }
    </style>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# UI Components
# ---------------------------------------------------------------------------

def render_hero():
    st.markdown("""
    <div class="hero-header">
        <h1>✂️ Retake Detection</h1>
        <p>Automatically detect and remove verbal retakes from your videos</p>
    </div>
    """, unsafe_allow_html=True)


def render_stats(result):
    cols = st.columns(5)
    stats = [
        (f"{result.source_duration:.1f}s", "Original"),
        (f"{result.output_duration:.1f}s", "Cleaned"),
        (f"{result.time_saved:.1f}s", "Time Saved"),
        (str(result.retake_groups_count), "Retakes Found"),
        (f"{result.likely_retakes}", "Confirmed Cuts"),
    ]
    for col, (value, label) in zip(cols, stats):
        with col:
            st.markdown(f"""
            <div class="stat-card">
                <p class="stat-value">{value}</p>
                <p class="stat-label">{label}</p>
            </div>
            """, unsafe_allow_html=True)


def render_timeline(cutlist_dict: dict):
    duration = cutlist_dict.get("source_duration", 1.0)
    keep_spans = cutlist_dict.get("keep_spans", [])
    discard_spans = cutlist_dict.get("discard_spans", [])

    segments_html = ""
    for span in keep_spans:
        left_pct = (span["start"] / duration) * 100
        width_pct = ((span["end"] - span["start"]) / duration) * 100
        segments_html += (
            f'<div class="timeline-segment timeline-keep" '
            f'style="left:{left_pct:.2f}%; width:{width_pct:.2f}%;" '
            f'title="Keep: {span["start"]:.1f}s → {span["end"]:.1f}s"></div>'
        )

    for span in discard_spans:
        left_pct = (span["start"] / duration) * 100
        width_pct = ((span["end"] - span["start"]) / duration) * 100
        segments_html += (
            f'<div class="timeline-segment timeline-discard" '
            f'style="left:{left_pct:.2f}%; width:{width_pct:.2f}%;" '
            f'title="Discard: {span["start"]:.1f}s → {span["end"]:.1f}s"></div>'
        )

    st.markdown(f"""
    <div class="timeline-container">
        <div style="font-size: 0.85rem; font-weight: 600; color: rgba(255,255,255,0.7); margin-bottom: 0.75rem;">
            Video Timeline
        </div>
        <div class="timeline-bar">
            {segments_html}
        </div>
        <div class="timeline-labels">
            <span>0:00</span>
            <span>{duration / 2:.0f}s</span>
            <span>{duration:.0f}s</span>
        </div>
        <div class="legend">
            <div class="legend-item">
                <div class="legend-dot" style="background: #22c55e;"></div>
                Kept
            </div>
            <div class="legend-item">
                <div class="legend-dot" style="background: #ef4444;"></div>
                Discarded (Retake)
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_retake_groups(cutlist_dict: dict):
    groups = cutlist_dict.get("retake_groups", [])
    if not groups:
        st.info("No retake groups detected.")
        return

    st.markdown("### 🔍 Retake Groups")

    for group in groups:
        take_a = group["take_a_span"]
        take_b = group["take_b_span"]
        kept = group["kept_take"]
        reason = group.get("reason", "")

        with st.expander(
            f"Group {group['group_id']} — "
            f"{group['pair_count']} pair(s) · "
            f"Discarded {take_a['end'] - take_a['start']:.1f}s",
            expanded=False,
        ):
            col1, col2 = st.columns(2)
            with col1:
                badge = "keep-badge" if kept == "a" else "discard-badge"
                badge_text = "✓ KEPT" if kept == "a" else "✗ DISCARDED"
                st.markdown(f"""
                <div class="retake-card">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <strong>Take A</strong>
                        <span class="{badge}">{badge_text}</span>
                    </div>
                    <p style="color:rgba(255,255,255,0.5); font-size:0.8rem; margin:0.5rem 0 0;">
                        {take_a['start']:.2f}s → {take_a['end']:.2f}s
                        ({take_a['end'] - take_a['start']:.1f}s)
                    </p>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                badge = "keep-badge" if kept == "b" else "discard-badge"
                badge_text = "✓ KEPT" if kept == "b" else "✗ DISCARDED"
                st.markdown(f"""
                <div class="retake-card">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <strong>Take B</strong>
                        <span class="{badge}">{badge_text}</span>
                    </div>
                    <p style="color:rgba(255,255,255,0.5); font-size:0.8rem; margin:0.5rem 0 0;">
                        {take_b['start']:.2f}s → {take_b['end']:.2f}s
                        ({take_b['end'] - take_b['start']:.1f}s)
                    </p>
                </div>
                """, unsafe_allow_html=True)

            if reason:
                st.caption(f"💡 {reason}")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar() -> tuple[Path | None, dict]:
    with st.sidebar:
        st.markdown("### ⚙️ Pipeline Settings")

        # Video selection
        st.markdown("#### 📁 Input Video")
        source_mode = st.radio(
            "Source",
            ["Select sample", "Upload file"],
            label_visibility="collapsed",
        )

        input_video: Path | None = None

        if source_mode == "Select sample":
            sample_videos = sorted(VIDEOS_DIR.glob("*.mp4")) if VIDEOS_DIR.exists() else []
            if sample_videos:
                selected = st.selectbox(
                    "Sample video",
                    sample_videos,
                    format_func=lambda p: f"{p.name} ({p.stat().st_size / 1024 / 1024:.1f} MB)",
                )
                input_video = selected
            else:
                st.warning("No sample videos found in `data/videos/`")
        else:
            uploaded = st.file_uploader("Upload video", type=["mp4", "mov", "mkv", "avi", "webm"])
            if uploaded:
                upload_path = VIDEOS_DIR / uploaded.name
                VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
                upload_path.write_bytes(uploaded.read())
                input_video = upload_path
                st.success(f"Uploaded: {uploaded.name}")

        st.markdown("---")

        # Pipeline config
        st.markdown("#### 🎛️ Configuration")
        config = {}

        config["provider"] = st.selectbox(
            "ASR Provider",
            ["whisperx", "whisper", "assemblyai"],
            index=0,
        )

        config["model_name"] = st.selectbox(
            "Model Size",
            ["base", "small", "medium", "large-v3"],
            index=0,
        )

        config["device"] = st.selectbox(
            "Device",
            ["cuda", "cpu"],
            index=0,
        )

        config["take_strategy"] = st.selectbox(
            "Take Selection",
            ["keep_later", "keep_earlier", "keep_higher_confidence"],
            index=0,
            help="Which take to keep when a retake is detected",
        )

        config["copy_mode"] = st.checkbox(
            "Fast mode (stream copy)",
            value=False,
            help="Faster but cuts may not be frame-exact",
        )

    return input_video, config


# ---------------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------------

def main():
    inject_styles()
    render_hero()

    input_video, config = render_sidebar()

    # Main content area
    if input_video is None:
        st.markdown(
            """
            <div style="text-align: center; padding: 4rem 2rem; color: rgba(255,255,255,0.4);">
                <p style="font-size: 3rem; margin: 0;">📹</p>
                <p style="font-size: 1.1rem; margin: 0.5rem 0;">
                    Select a sample video or upload your own to get started
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # Show selected video info
    st.markdown(f"**Selected:** `{input_video.name}` ({input_video.stat().st_size / 1024 / 1024:.1f} MB)")

    # Run pipeline button
    if st.button("🚀 Run Pipeline", type="primary", use_container_width=True):
        _run_pipeline_with_ui(input_video, config)

    # Check for existing results
    elif "pipeline_result" in st.session_state:
        _display_results(st.session_state["pipeline_result"], input_video)


def _run_pipeline_with_ui(input_video: Path, config: dict):
    """Run the pipeline with Streamlit progress indicators."""
    from pipeline.run_pipeline import run_full_pipeline

    status_container = st.container()
    progress_bar = st.progress(0, text="Initializing pipeline...")

    stages = [
        ("Transcription", 0.15),
        ("Token Matching", 0.35),
        ("Semantic Matching", 0.60),
        ("Cutlist Generation", 0.75),
        ("Video Editing", 0.95),
    ]
    stage_map = {name: frac for name, frac in stages}

    def progress_callback(stage_name: str, stage_frac: float):
        base_frac = stage_map.get(stage_name, 0.0)
        # Interpolate within the stage's portion
        progress_bar.progress(
            min(base_frac + 0.05, 1.0),
            text=f"⏳ {stage_name}...",
        )

    start_time = time.time()

    result = run_full_pipeline(
        input_video=input_video,
        provider=config.get("provider", "whisperx"),
        model_name=config.get("model_name", "base"),
        device=config.get("device", "cuda"),
        take_strategy=config.get("take_strategy", "keep_later"),
        copy_mode=config.get("copy_mode", False),
        progress_callback=progress_callback,
    )

    elapsed = time.time() - start_time
    progress_bar.progress(1.0, text="✅ Pipeline complete!")

    if result.success:
        st.session_state["pipeline_result"] = result
        st.toast(f"Pipeline completed in {elapsed:.1f}s!", icon="🎉")
        _display_results(result, input_video)
    else:
        st.error(f"Pipeline failed: {result.error}")


def _display_results(result, input_video: Path):
    """Display pipeline results with videos, timeline, and stats."""
    from pipeline.run_pipeline import PipelineResult

    if not result.success:
        st.error(f"Pipeline failed: {result.error}")
        return

    st.markdown("---")

    # Stats
    render_stats(result)
    st.markdown("")

    # Timeline
    if result.cutlist_dict:
        render_timeline(result.cutlist_dict)

    st.markdown("")

    # Side-by-side video comparison
    st.markdown("### 🎬 Video Comparison")
    col_orig, col_clean = st.columns(2)

    with col_orig:
        st.markdown("**Original**")
        if input_video.exists():
            st.video(str(input_video))

    with col_clean:
        st.markdown("**Cleaned**")
        output_path = Path(result.output_video)
        if output_path.exists():
            st.video(str(output_path))

            # Download button
            with open(output_path, "rb") as f:
                st.download_button(
                    "⬇️ Download Cleaned Video",
                    data=f,
                    file_name=output_path.name,
                    mime="video/mp4",
                    use_container_width=True,
                )
        else:
            st.warning("Cleaned video file not found")

    st.markdown("")

    # Retake groups details
    if result.cutlist_dict:
        render_retake_groups(result.cutlist_dict)

    # Stage timings
    with st.expander("⏱️ Stage Timings"):
        for stage, timing in result.stage_timings.items():
            st.markdown(f"- **{stage.replace('_', ' ').title()}**: {timing:.2f}s")
        total = sum(result.stage_timings.values())
        st.markdown(f"- **Total**: {total:.2f}s")

    # Raw JSON
    with st.expander("📋 Raw Cutlist JSON"):
        st.json(result.cutlist_dict)


if __name__ == "__main__":
    main()
