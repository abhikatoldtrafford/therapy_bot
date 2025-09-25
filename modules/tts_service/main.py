"""
TTS Service - Advanced Text-to-Speech
Streamlit app with latest OpenAI TTS models and voice instructions
"""
import streamlit as st
import sys
from pathlib import Path
import base64

# Add parent directory to path to import sibling modules
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Import modules
from tts_service import TTSService

st.set_page_config(page_title="TTS Service - Advanced Voice Synthesis", page_icon="🔊", layout="wide")

# Custom CSS for enhanced UI
st.markdown("""
<style>
    .stButton > button {
        width: 100%;
    }
    .voice-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1rem;
    }
    .audio-preview {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.title("🔊 Advanced Text-to-Speech Service")
st.markdown("Production-ready TTS with latest OpenAI models, voice instructions, and streaming support")

# Initialize TTS service
if "tts_service" not in st.session_state:
    try:
        st.session_state.tts_service = TTSService()
        st.session_state.service_initialized = True
    except Exception as e:
        st.error(f"Failed to initialize service: {str(e)}")
        st.session_state.service_initialized = False

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")

    api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")

    st.divider()
    st.subheader("🎛️ Voice Settings")

    # Model selection with new options
    model_types = st.session_state.tts_service.get_model_types() if st.session_state.get('service_initialized') else {}
    model_type = st.selectbox(
        "Model Type",
        options=["standard", "hd", "advanced"],
        format_func=lambda x: f"{x.title()} - {model_types.get(x, '')}" if model_types else x,
        index=2,
        help="Choose model based on quality needs"
    )

    # Voice selection with descriptions
    voices = st.session_state.tts_service.get_voices() if st.session_state.get('service_initialized') else {}
    voice = st.selectbox(
        "Voice",
        options=list(voices.keys()),
        format_func=lambda x: f"{x.title()} - {voices.get(x, '')}" if voices else x,
        index=0,
        help="Select voice personality"
    )

    # Speed control
    speed = st.slider("Speed", 0.25, 4.0, 1.05, 0.05,
                     help="1.0 is normal, <1 is slower, >1 is faster")

    # Output format
    output_format = st.selectbox("Output Format",
                                ["mp3", "opus", "aac", "flac", "wav", "pcm"],
                                help="Choose audio format")

    # Voice instructions (for advanced model)
    if model_type == "advanced":
        st.divider()
        st.subheader("🎭 Voice Instructions")
        instructions = st.text_area(
            "Custom Instructions",
            value="Speak with warm empathy, natural pauses, and therapeutic tone.",
            height=100,
            help="Only works with gpt-4o-mini-tts model"
        )
    else:
        instructions = None

    if st.button("🔄 Reinitialize Service"):
        try:
            st.session_state.tts_service = TTSService(api_key)
            st.session_state.service_initialized = True
            st.success("Service reinitialized!")
        except Exception as e:
            st.error(f"Failed: {str(e)}")
            st.session_state.service_initialized = False

if not st.session_state.get('service_initialized', False):
    st.error("⚠️ Service not initialized. Please configure API key in sidebar.")
    st.stop()

service = st.session_state.tts_service

# Main tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎵 Quick TTS", "🎙️ Voice Showcase", "📝 Long Text", "💾 Streaming", "💰 Cost Estimator"])

with tab1:
    st.header("Quick Text-to-Speech")

    col1, col2 = st.columns([2, 1])

    with col1:
        text_input = st.text_area("Enter text to convert",
                                 placeholder="Type or paste your text here...",
                                 height=200,
                                 max_chars=4096)

    with col2:
        if text_input:
            char_count = len(text_input)
            word_count = len(text_input.split())
            st.metric("Characters", char_count)
            st.metric("Words", word_count)
            st.metric("Est. Duration", f"{word_count / 150:.1f} min")

    if text_input:
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("🎵 Generate Speech", type="primary"):
                with st.spinner("Generating speech..."):
                    result = service.text_to_speech(
                        text=text_input,
                        voice=voice,
                        model_type=model_type,
                        speed=speed,
                        response_format=output_format,
                        instructions=instructions
                    )

                    if result["status"] == "success":
                        st.success("Speech generated successfully!")
                        st.audio(result["audio_data"], format=f"audio/{output_format}")

                        # Save to session
                        st.session_state.last_audio = result["audio_data"]
                        st.session_state.last_format = output_format

                        # Download button
                        b64 = base64.b64encode(result["audio_data"]).decode()
                        href = f'<a href="data:audio/{output_format};base64,{b64}" download="speech.{output_format}">📥 Download Audio</a>'
                        st.markdown(href, unsafe_allow_html=True)
                    else:
                        st.error(f"Generation failed: {result['error']}")

        with col2:
            if st.button("🔈 Preview Voice", type="secondary"):
                preview_text = "This is how your selected voice sounds."
                with st.spinner("Generating preview..."):
                    result = service.text_to_speech(
                        text=preview_text,
                        voice=voice,
                        model_type=model_type,
                        speed=speed,
                        instructions=instructions
                    )
                    if result["status"] == "success":
                        st.audio(result["audio_data"], format="audio/mp3")

        with col3:
            if st.button("🎲 Random Voice", type="secondary"):
                import random
                random_voice = random.choice(list(voices.keys()))
                st.info(f"Try voice: {random_voice}")
                st.experimental_rerun()

with tab2:
    st.header("Voice Showcase")
    st.info("Compare different voices with the same text")

    showcase_text = st.text_input(
        "Sample Text",
        value="Hello! This is a sample of my voice. I can speak with different emotions and styles.",
        key="showcase_text"
    )

    if st.button("🎭 Generate All Voices", key="gen_all"):
        cols = st.columns(3)
        voice_samples = {}

        for i, (voice_name, voice_desc) in enumerate(voices.items()):
            col_idx = i % 3
            with cols[col_idx]:
                st.markdown(f"**{voice_name.title()}**")
                st.caption(voice_desc)

                with st.spinner(f"Generating {voice_name}..."):
                    result = service.text_to_speech(
                        text=showcase_text,
                        voice=voice_name,
                        model_type=model_type,
                        speed=speed,
                        instructions=instructions
                    )

                    if result["status"] == "success":
                        st.audio(result["audio_data"], format="audio/mp3")
                        voice_samples[voice_name] = result["audio_data"]

        st.session_state.voice_samples = voice_samples

with tab3:
    st.header("Long Text Processing")
    st.info("Automatically splits long text into chunks for processing")

    long_text = st.text_area("Enter long text",
                            height=400,
                            key="long_text",
                            placeholder="Paste your article, story, or document here...")

    if long_text:
        chunks = service.chunk_text(long_text, max_chunk_size=1000)

        col1, col2 = st.columns([3, 1])
        with col1:
            st.info(f"Text will be processed in {len(chunks)} chunks")
        with col2:
            chunk_preview = st.checkbox("Preview chunks", value=False)

        if chunk_preview:
            for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
                with st.expander(f"Chunk {i+1}"):
                    st.text(chunk[:200] + "..." if len(chunk) > 200 else chunk)

        if st.button("🎵 Generate All Chunks", key="chunk_gen"):
            audio_chunks = []
            progress = st.progress(0)
            status = st.empty()

            for i, chunk in enumerate(chunks):
                status.text(f"Processing chunk {i+1}/{len(chunks)}...")
                progress.progress((i + 1) / len(chunks))

                result = service.text_to_speech(
                    text=chunk,
                    voice=voice,
                    model_type=model_type,
                    speed=speed,
                    instructions=instructions
                )

                if result["status"] == "success":
                    audio_chunks.append(result["audio_data"])
                else:
                    st.error(f"Failed on chunk {i+1}: {result['error']}")
                    break

            if audio_chunks:
                st.success(f"Generated {len(audio_chunks)} audio chunks!")

                # Combine audio chunks (simplified - in production use audio processing library)
                combined_audio = b''.join(audio_chunks)
                st.audio(combined_audio, format="audio/mp3")

                # Download combined audio
                b64 = base64.b64encode(combined_audio).decode()
                href = f'<a href="data:audio/mp3;base64,{b64}" download="long_speech.mp3">📥 Download Complete Audio</a>'
                st.markdown(href, unsafe_allow_html=True)

with tab4:
    st.header("Streaming TTS")
    st.info("Stream audio generation for real-time applications")

    streaming_text = st.text_area("Text for streaming",
                                 placeholder="Enter text to stream...",
                                 height=150,
                                 key="streaming_text")

    if streaming_text:
        if st.button("🌊 Start Streaming", key="stream_btn"):
            with st.spinner("Streaming audio..."):
                try:
                    # Collect streamed chunks
                    audio_data = b''
                    chunk_count = 0

                    for chunk in service.text_to_speech_streaming(
                        text=streaming_text,
                        voice=voice,
                        model_type=model_type,
                        speed=speed,
                        instructions=instructions
                    ):
                        if chunk:
                            audio_data += chunk
                            chunk_count += 1

                    if audio_data:
                        st.success(f"Streamed {chunk_count} chunks successfully!")
                        st.audio(audio_data, format="audio/mp3")
                    else:
                        st.error("No audio data received")

                except Exception as e:
                    st.error(f"Streaming error: {str(e)}")

    with st.expander("ℹ️ About Streaming"):
        st.markdown("""
        **Benefits of Streaming:**
        - Lower latency for first audio
        - Better for real-time applications
        - Memory efficient for long texts
        - Progressive playback support

        **Use Cases:**
        - Live chat applications
        - Interactive voice assistants
        - Real-time narration
        - Low-latency voice responses
        """)

with tab5:
    st.header("Cost Estimator")

    st.markdown("Estimate costs for TTS generation")

    col1, col2 = st.columns(2)

    with col1:
        est_text = st.text_area("Text for estimation",
                               placeholder="Paste text to estimate cost...",
                               height=200,
                               key="est_text")

        est_model = st.selectbox("Model for estimation",
                                ["tts-1", "tts-1-hd", "gpt-4o-mini-tts"],
                                key="est_model")

    with col2:
        if est_text:
            char_count = len(est_text)

            # Calculate costs
            if est_model == "tts-1":
                cost_per_1k = 0.015
            elif est_model == "tts-1-hd":
                cost_per_1k = 0.030
            else:  # gpt-4o-mini-tts (estimated)
                cost_per_1k = 0.025

            estimated_cost = (char_count / 1000) * cost_per_1k

            st.metric("Character Count", f"{char_count:,}")
            st.metric("Estimated Cost", f"${estimated_cost:.4f}")
            st.metric("Cost per 1K chars", f"${cost_per_1k:.3f}")

            # Bulk estimation
            st.divider()
            st.markdown("**Bulk Processing**")
            multiplier = st.number_input("Number of generations", min_value=1, value=1)
            total_cost = estimated_cost * multiplier
            st.metric("Total Cost", f"${total_cost:.2f}")

# Enhanced Status Bar
st.divider()
status_cols = st.columns(6)
with status_cols[0]:
    st.caption(f"🟢 Service: {'Connected' if st.session_state.get('service_initialized') else 'Not Connected'}")
with status_cols[1]:
    st.caption(f"🤖 Model: {model_type}")
with status_cols[2]:
    st.caption(f"🎭 Voice: {voice}")
with status_cols[3]:
    st.caption(f"⚡ Speed: {speed}x")
with status_cols[4]:
    st.caption(f"🎵 Format: {output_format}")
with status_cols[5]:
    st.caption(f"💎 Instructions: {'Yes' if instructions else 'No'}")

# Last audio in session
if st.session_state.get('last_audio'):
    with st.expander("📼 Last Generated Audio"):
        st.audio(st.session_state.last_audio, format=f"audio/{st.session_state.get('last_format', 'mp3')}")