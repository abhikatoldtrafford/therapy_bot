import streamlit as st
import time
import os
from tempfile import NamedTemporaryFile
from audiorecorder import audiorecorder
from openai import OpenAI

from backend import (
    initiate_chat,
    chat,
    image_analysis,
    voice_analysis,
    global_context,
    MODEL_NAME
)

###################################
# Configuration
###################################
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=OPENAI_API_KEY)  # For Whisper STT

###################################
# Session State Initialization
###################################
if "assistant_id" not in st.session_state:
    st.session_state["assistant_id"] = None
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = None
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
###################################
# Utility Functions
###################################
def send_message_stream(prompt: str):
    """Send text message with streaming response using backend.chat()"""
    if not st.session_state["assistant_id"]:
        st.error("⚠️ Please refress to state a new session.")
        return

    with st.chat_message("user"):
        st.markdown(prompt)

    assistant_message = st.chat_message("assistant")
    message_placeholder = assistant_message.empty()
    message_placeholder.markdown("_💭 Thinking..._")

    try:
        response = chat(prompt, stream=True)
        
        if response["status"] != "success":
            message_placeholder.markdown(f"❌ Error: {response.get('message', 'Unknown error')}")
            return

        full_response = ""
        for chunk in response["stream"]:
            full_response += chunk
            message_placeholder.markdown(full_response + "▌")
            time.sleep(0.01)
        
        message_placeholder.markdown(full_response)
        st.session_state["chat_history"].append({"role": "user", "content": prompt})
        st.session_state["chat_history"].append({"role": "assistant", "content": full_response})
        
    except Exception as e:
        message_placeholder.markdown(f"🚨 Error: {str(e)}")

###################################
# UI Layout
###################################
st.set_page_config(page_title="NARM Therapy Assistant", layout="wide")

# Custom CSS
st.markdown("""
<style>
.fixed-bottom {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    background: white;
    padding: 10px;
    z-index: 1000;
    box-shadow: 0 -2px 10px rgba(0, 0, 0, 0.1);
}
.login-card {
    max-width: 500px;
    margin: 2rem auto;
    padding: 2rem;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    background: white;
}
.stChatInput {
    width: 80% !important;
}
.tool-card {
    background: #f9f9f9;
    border-left: 5px solid #6366F1;
    border-radius: 10px;
    padding: 15px;
    margin-bottom: 15px;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
    transition: transform 0.2s, box-shadow 0.2s;
}
.tool-card:hover {
    transform: scale(1.02);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
}
</style>
""", unsafe_allow_html=True)

###################################
# Session Initiation (Login Screen)
###################################
if not global_context.get("assistant_id"):
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
            <div style="text-align: center; margin-bottom: 2rem;">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 150" width="200">
                    <rect width="300" height="150" fill="#ffffff"/>
                    <g transform="translate(115, 35)">
                        <circle cx="35" cy="35" r="25" fill="#6366F1"/>
                        <circle cx="80" cy="35" r="6" fill="#6366F1" opacity="0.8"/>
                        <circle cx="95" cy="35" r="4" fill="#6366F1" opacity="0.6"/>
                        <circle cx="105" cy="35" r="3" fill="#6366F1" opacity="0.4"/>
                    </g>
                    <text x="150" y="110" font-family="sans-serif" font-size="32" font-weight="500" text-anchor="middle" fill="#1E293B">NARM</text>
                    <text x="150" y="130" font-family="sans-serif" font-size="16" font-weight="400" text-anchor="middle" fill="#6366F1">whisper</text>
                </svg>
                <h1 style='color: #1E293B; margin-top: 0.5rem;'>Therapy Session Setup</h1>
            </div>
        """, unsafe_allow_html=True)
        with st.form("session_init_form"):
            name = st.text_input("Full Name")
            email = st.text_input("Email")
            focus_today = st.text_area("Today's Focus")
            desired_outcome = st.text_area("Desired Outcome")
            current_challenges = st.text_area("Current Challenges")

            if st.form_submit_button("🚀 Start Session"):
                response = initiate_chat(
                    name=name,
                    email=email,
                    focus_today=focus_today,
                    desired_outcome=desired_outcome,
                    current_challenges=current_challenges
                )
                
                if response["status"] == "success":
                    st.rerun()
                else:
                    st.error(f"❌ Failed to initialize session: {response.get('message', 'Unknown error')}")
    st.stop()

###################################
# Main Chat Interface (Post-Login)
###################################
else:
    # Sidebar Tools
    with st.sidebar:
        st.header("🛠️ Tools")

        # Image Analysis
        with st.expander("📸 Image Analysis"):
            st.markdown('<div class="tool-card">', unsafe_allow_html=True)
            img_prompt = st.text_input("Analysis Prompt", key="img_prompt")
            image_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"], key="image_file")
            if st.button("🔍 Analyze Image", key="analyze_image"):
                if image_file:
                    with st.spinner("📷 Analyzing image..."):
                        response = image_analysis(
                            prompt=img_prompt,
                            image_data=image_file.getvalue(),
                            filename=image_file.name
                        )
                        
                        if response["status"] == "success":
                            st.session_state["chat_history"].append({
                                "role": "user",
                                "content": f"Image Analysis Request: {img_prompt}",
                                "has_image": True
                            })
                            st.session_state["chat_history"].append({
                                "role": "assistant",
                                "content": response["response"][0]["content"]
                            })
                            st.rerun()
                        else:
                            st.error(f"❌ Image analysis failed: {response.get('message', 'Unknown error')}")
                else:
                    st.warning("⚠️ Please upload an image first")
            st.markdown('</div>', unsafe_allow_html=True)

        # Audio Recorder
        st.subheader("🎙️ Audio Recorder")
        audio_data = audiorecorder("⏺️ Record", "⏸️ Stop")

    # Chat Window Header
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 150" width="200">
            <rect width="300" height="150" fill="#ffffff"/>
            <g transform="translate(115, 35)">
                <circle cx="35" cy="35" r="25" fill="#6366F1"/>
                <circle cx="80" cy="35" r="6" fill="#6366F1" opacity="0.8"/>
                <circle cx="95" cy="35" r="4" fill="#6366F1" opacity="0.6"/>
                <circle cx="105" cy="35" r="3" fill="#6366F1" opacity="0.4"/>
            </g>
            <text x="150" y="110" font-family="sans-serif" font-size="32" font-weight="500" text-anchor="middle" fill="#1E293B">NARM</text>
            <text x="150" y="130" font-family="sans-serif" font-size="16" font-weight="400" text-anchor="middle" fill="#6366F1">whisper</text>
        </svg>
        <h1 style='color: #1E293B; margin-top: 0.5rem;'>Therapy Chatbot</h1>
    </div>
    """, unsafe_allow_html=True)

    # Display chat history
    for msg in st.session_state["chat_history"]:
        if msg["role"] == "assistant":
            with st.chat_message("assistant"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("user"):
                if msg.get("has_image"):
                    st.write(f"Image Analysis Request: {msg['content']}")
                    st.image(image_file, caption="Uploaded Image")
                else:
                    st.markdown(msg["content"])

    # Chat Input at Bottom
    user_input = st.chat_input("💬 Type your message and press Enter...")

    # Text message handling
    if user_input:
        send_message_stream(user_input)

    # Audio message handling
    if audio_data and len(audio_data) > 0:
        with NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            audio_data.export(tmp.name, format="mp3")
            tmp_filename = tmp.name

        try:
            with open(tmp_filename, "rb") as audio_file:
                transcription = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
            transcript_text = transcription.text.strip()
            
            # Ensure transcript is only pushed once until new audio is recorded
            if transcript_text and st.session_state.get("last_transcript") != transcript_text:
                st.session_state["chat_history"].append({
                    "role": "user",
                    "content": transcript_text,
                    "has_audio": True
                })
                send_message_stream(transcript_text)
                st.session_state["last_transcript"] = transcript_text  # Store last transcript to prevent duplicates
            
        finally:
            os.remove(tmp_filename)
