
import streamlit as st
from stt_service import STTService
from audiorecorder import audiorecorder
import tempfile
import os

st.set_page_config(page_title="STT Service - Advanced Speech Recognition", page_icon="🎙️", layout="wide")

# Custom CSS for better UI
st.markdown("""
<style>
    .stButton > button {
        width: 100%;
    }
    .status-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("🎙️ Advanced Speech-to-Text Service")
st.markdown("Production-ready transcription with latest OpenAI models and streaming support")

# Initialize STT service
if "stt_service" not in st.session_state:
    try:
        st.session_state.stt_service = STTService()
        st.session_state.service_initialized = True
    except Exception as e:
        st.error(f"Failed to initialize service: {str(e)}")
        st.session_state.service_initialized = False

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")

    api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")

    st.divider()
    st.subheader("🎛️ Transcription Settings")

    # Model selection with new options
    st.markdown("**Model Selection**")
    model_options = st.session_state.stt_service.get_model_options() if st.session_state.get('service_initialized') else {}
    model_type = st.selectbox(
        "Choose Model",
        options=["fast", "accurate", "legacy"],
        format_func=lambda x: model_options.get(x, x) if model_options else x,
        index=1,
        help="Select model based on your needs"
    )

    # Streaming option
    use_streaming = st.checkbox(
        "Enable Streaming",
        value=True,
        help="Stream transcription results for faster response (only for GPT-4o models)"
    )

    temperature = st.slider("Temperature", 0.0, 1.0, 0.15, 0.05,
                           help="Lower values for more consistent results")
    response_format = st.selectbox("Response Format",
                                  ["text", "json", "srt", "verbose_json", "vtt"],
                                  help="Choose output format for transcription")

    if st.button("🔄 Reinitialize Service"):
        try:
            st.session_state.stt_service = STTService(api_key)
            st.session_state.service_initialized = True
            st.success("Service reinitialized!")
        except Exception as e:
            st.error(f"Failed: {str(e)}")
            st.session_state.service_initialized = False

if not st.session_state.get('service_initialized', False):
    st.error("⚠️ Service not initialized. Please configure API key in sidebar.")
    st.stop()

service = st.session_state.stt_service

# Main tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Record Audio", "Upload File", "Translate", "Settings", "Examples"])

with tab1:
    st.header("Record and Transcribe")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Audio recorder
        audio_data = audiorecorder("🎙️ Start Recording", "⏹️ Stop Recording")

        # Language selection
        languages = service.get_language_codes()
        selected_lang = st.selectbox("Language", list(languages.keys()))
        lang_code = languages[selected_lang]

        # Optional prompt
        custom_prompt = st.text_area("Guide Prompt (optional)",
                                    placeholder="Optional text to guide the transcription style...",
                                    height=100)

    with col2:
        st.markdown("**Recording Status**")
        if len(audio_data) > 0:
            st.success("✅ Audio recorded")
            # Export audio to bytes for playback
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_audio:
                audio_data.export(tmp_audio.name, format="mp3")
                tmp_audio_path = tmp_audio.name
            with open(tmp_audio_path, "rb") as f:
                audio_bytes = f.read()
            st.audio(audio_bytes)
            os.remove(tmp_audio_path)
        else:
            st.info("⏸️ Ready to record")

    if len(audio_data) > 0:
        if st.button("📝 Transcribe Recording", type="primary"):
            with st.spinner("Transcribing audio..."):
                # Save audio to temp file
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                    audio_data.export(tmp.name, format="mp3")
                    tmp_path = tmp.name

                try:
                    # Read file and transcribe
                    with open(tmp_path, "rb") as f:
                        audio_bytes = f.read()

                    result = service.transcribe_audio(
                        audio_data=audio_bytes,
                        filename="recording.mp3",
                        language=lang_code,
                        prompt=custom_prompt if custom_prompt else None,
                        response_format=response_format,
                        temperature=temperature,
                        model_type=model_type,
                        stream=use_streaming
                    )

                    if result["status"] == "success":
                        st.success("Transcription Complete!")

                        # Display result based on format
                        if response_format == "text":
                            st.markdown("**Transcription:**")
                            st.write(result["transcription"])
                        elif response_format in ["json", "verbose_json"]:
                            st.json(result["transcription"])
                        else:
                            st.code(result["transcription"], language="text")

                        # Save to session state
                        st.session_state.last_transcription = result["transcription"]
                    else:
                        st.error(f"Transcription failed: {result['error']}")

                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

with tab2:
    st.header("Upload Audio File")

    uploaded_file = st.file_uploader("Choose an audio file",
                                    type=['mp3', 'mp4', 'wav', 'webm', 'm4a'])

    if uploaded_file:
        st.audio(uploaded_file)

        col1, col2 = st.columns(2)
        with col1:
            languages = service.get_language_codes()
            upload_lang = st.selectbox("Language ", list(languages.keys()), key="upload_lang")
            upload_lang_code = languages[upload_lang]

        with col2:
            upload_prompt = st.text_input("Guide Prompt (optional)", key="upload_prompt")

        if st.button("📝 Transcribe File", type="primary"):
            with st.spinner("Transcribing audio file..."):
                try:
                    result = service.transcribe_audio(
                        audio_data=uploaded_file.read(),
                        filename=uploaded_file.name,
                        language=upload_lang_code,
                        prompt=upload_prompt if upload_prompt else None,
                        response_format=response_format,
                        temperature=temperature
                    )

                    if result["status"] == "success":
                        st.success("Transcription Complete!")

                        # Display result
                        if response_format == "text":
                            st.markdown("**Transcription:**")
                            st.write(result["transcription"])

                            # Word count
                            word_count = len(result["transcription"].split())
                            st.caption(f"Word count: {word_count}")
                        elif response_format in ["json", "verbose_json"]:
                            st.json(result["transcription"])
                        else:
                            st.code(result["transcription"], language="text")

                        # Copy button
                        st.code(result["transcription"], language="text")
                    else:
                        st.error(f"Transcription failed: {result['error']}")

                except Exception as e:
                    st.error(f"Error: {str(e)}")

with tab3:
    st.header("Translate Audio to English")

    st.info("Translation automatically converts any language to English")

    translate_method = st.radio("Input Method", ["Record", "Upload"])

    if translate_method == "Record":
        translate_audio = audiorecorder("🎙️ Record", "⏹️ Stop", key="translate_recorder")

        if len(translate_audio) > 0:
            # Export audio to bytes for playback
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_audio:
                translate_audio.export(tmp_audio.name, format="mp3")
                tmp_audio_path = tmp_audio.name
            with open(tmp_audio_path, "rb") as f:
                audio_bytes = f.read()
            st.audio(audio_bytes)
            os.remove(tmp_audio_path)

            if st.button("🌐 Translate to English", type="primary"):
                with st.spinner("Translating..."):
                    # Save audio to temp file
                    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                        translate_audio.export(tmp.name, format="mp3")
                        tmp_path = tmp.name

                    try:
                        with open(tmp_path, "rb") as f:
                            audio_bytes = f.read()

                        result = service.translate_audio(
                            audio_data=audio_bytes,
                            filename="recording.mp3",
                            response_format=response_format,
                            temperature=temperature
                        )

                        if result["status"] == "success":
                            st.success("Translation Complete!")
                            st.markdown("**English Translation:**")
                            st.write(result["translation"])
                        else:
                            st.error(f"Translation failed: {result['error']}")

                    finally:
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)

    else:  # Upload
        translate_file = st.file_uploader("Choose audio file to translate",
                                        type=['mp3', 'mp4', 'wav', 'webm', 'm4a'],
                                        key="translate_upload")

        if translate_file:
            st.audio(translate_file)

            if st.button("🌐 Translate to English", type="primary"):
                with st.spinner("Translating..."):
                    try:
                        result = service.translate_audio(
                            audio_data=translate_file.read(),
                            filename=translate_file.name,
                            response_format=response_format,
                            temperature=temperature
                        )

                        if result["status"] == "success":
                            st.success("Translation Complete!")
                            st.markdown("**English Translation:**")
                            st.write(result["translation"])
                        else:
                            st.error(f"Translation failed: {result['error']}")

                    except Exception as e:
                        st.error(f"Error: {str(e)}")

with tab4:
    st.header("Service Settings")

    st.subheader("Supported Formats")
    formats = service.get_supported_formats()
    cols = st.columns(4)
    for i, fmt in enumerate(formats):
        cols[i % 4].markdown(f"• `.{fmt}`")

    st.divider()

    st.subheader("Language Codes")
    languages = service.get_language_codes()
    lang_df = []
    for lang, code in languages.items():
        lang_df.append({"Language": lang, "Code": code or "auto"})

    st.dataframe(lang_df, use_container_width=True)

    st.divider()

    st.subheader("API Information")
    st.markdown("""
    - **Models Available:**
        - `gpt-4o-transcribe`: Most accurate, supports streaming
        - `gpt-4o-mini-transcribe`: Fastest response time
        - `whisper-1`: Stable fallback option
    - **Max File Size:** 25 MB
    - **Supported Languages:** 50+ languages
    - **Auto Language Detection:** Yes
    - **Streaming Support:** Yes (for GPT-4o models)
    """)

with tab5:
    st.header("Example Prompts")

    st.markdown("Guide prompts can improve transcription accuracy for specific domains:")

    examples = {
        "Medical": "The patient is a 45-year-old male presenting with acute chest pain.",
        "Legal": "The defendant pleads not guilty to all charges.",
        "Technical": "The API endpoint returns a JSON response with status code 200.",
        "Academic": "The hypothesis states that increased temperature affects reaction rate.",
        "NARM Therapy": "The client expresses feelings of disconnection and seeks emotional regulation."
    }

    for domain, prompt in examples.items():
        with st.expander(f"📝 {domain} Domain"):
            st.code(prompt, language="text")
            if st.button(f"Use {domain} Prompt", key=f"use_{domain}"):
                st.session_state.example_prompt = prompt
                st.info(f"Prompt copied to clipboard area below")

    if st.session_state.get('example_prompt'):
        st.text_area("Selected Prompt", value=st.session_state.example_prompt, height=100)

# Enhanced Status Bar
st.divider()
status_cols = st.columns(5)
with status_cols[0]:
    st.caption(f"🟢 Service: {'Connected' if st.session_state.get('service_initialized') else 'Not Connected'}")
with status_cols[1]:
    st.caption(f"🤖 Model: {model_type}")
with status_cols[2]:
    st.caption(f"🎯 Format: {response_format}")
with status_cols[3]:
    st.caption(f"⚡ Streaming: {'On' if use_streaming else 'Off'}")
with status_cols[4]:
    st.caption(f"🌡️ Temperature: {temperature}")

# Last transcription
if st.session_state.get('last_transcription'):
    with st.expander("📋 Last Transcription"):
        st.text(st.session_state.last_transcription)