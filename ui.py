import streamlit as st
import requests
import time
from typing import Optional
import subprocess
import os
from tempfile import NamedTemporaryFile

from audiorecorder import audiorecorder
from openai import OpenAI

###################################
# Configuration
###################################
API_BASE_URL = "http://localhost:8080"  # Your existing FastAPI port

# Must be set in Streamlit secrets or replace with your own key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
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
# Placeholder for image analysis
###################################
def analyze_image(prompt: str, image_file):
    """Analyze image using AI"""
    if not st.session_state["assistant_id"]:
        st.error("⚠️ Please initiate a session first.")
        return
    if not image_file:
        st.warning("⚠️ No image selected.")
        return
    
    with st.spinner("📷 Analyzing image..."):
        try:
            with st.chat_message("user"):
                st.write(f"Image Analysis Request: {prompt}")
                st.image(image_file, caption="Uploaded Image")
            
            files = {"image": (image_file.name, image_file.getvalue(), "image/jpeg")}
            data = {"prompt": prompt}
            url = f"{API_BASE_URL}/image-analysis"
            response = requests.post(url, files=files, data=data)
            
            if response.status_code == 200:
                res_data = response.json()
                st.session_state["chat_history"].append({
                    "role": "user",
                    "content": f"Image Analysis Request: {prompt}",
                    "has_image": True
                })
                
                out_text = res_data["response"][0]["content"]
                with st.chat_message("assistant"):
                    st.markdown(out_text)
                st.session_state["chat_history"].append({
                    "role": "assistant",
                    "content": out_text
                })
            else:
                st.error(f"❌ Image analysis failed: {response.text}")
        except Exception as e:
            st.error(f"❌ Error during image analysis: {str(e)}")

###################################
# Utility Functions
###################################
def send_message_stream(prompt: str):
    """Send text message with streaming response to /chat endpoint."""
    if not st.session_state["assistant_id"]:
        st.error("⚠️ Please initiate a session first.")
        return

    with st.chat_message("user"):
        st.markdown(prompt)

    assistant_message = st.chat_message("assistant")
    message_placeholder = assistant_message.empty()
    message_placeholder.markdown("_💭 Thinking..._")

    url = f"{API_BASE_URL}/chat"
    try:
        response = requests.post(
            url,
            data={"prompt": prompt},
            params={"stream": True},
            stream=True
        )

        if response.status_code == 200:
            full_response = ""
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    chunk_text = chunk.decode("utf-8")
                    full_response += chunk_text
                    message_placeholder.markdown(full_response + "▌")
                    time.sleep(0.01)
            message_placeholder.markdown(full_response)

            st.session_state["chat_history"].append({"role": "user", "content": prompt})
            st.session_state["chat_history"].append({"role": "assistant", "content": full_response})
        else:
            message_placeholder.markdown(f"❌ Error: {response.text}")
    except Exception as e:
        message_placeholder.markdown(f"🚨 Connection error: {str(e)}")

def start_call():
    """Start call by launching call.py as a subprocess."""
    if "call_process" not in st.session_state or st.session_state.call_process is None:
        st.session_state.call_process = subprocess.Popen(["python3", "call.py"])
        st.success("Call started!")
    else:
        st.warning("Call is already running.")

def end_call():
    """End call if running."""
    if "call_process" in st.session_state and st.session_state.call_process is not None:
        st.session_state.call_process.terminate()
        st.session_state.call_process = None
        st.success("Call ended.")
    else:
        st.warning("No call is running.")

def initiate_session(name: str, email: str, focus_today: str, desired_outcome: str, current_challenges: str):
    """Initialize therapy session via /initiate-chat."""
    with st.spinner("🔄 Initializing therapy session..."):
        try:
            response = requests.post(
                f"{API_BASE_URL}/initiate-chat",
                data={
                    "name": name,
                    "email": email,
                    "focus_today": focus_today,
                    "desired_outcome": desired_outcome,
                    "current_challenges": current_challenges
                }
            )
            if response.status_code == 200:
                data = response.json()
                st.session_state["assistant_id"] = data["assistant_id"]
                st.session_state["chat_history"] = []
                st.rerun()
            else:
                st.error(f"❌ Failed to initialize session: {response.text}")
        except Exception as e:
            st.error(f"🚨 Connection error: {str(e)}")

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
if not st.session_state["assistant_id"]:
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
            st.text_input("Full Name", key="name")
            st.text_input("Email", key="email")
            st.text_area("Today's Focus", key="focus")
            st.text_area("Desired Outcome", key="desire")
            st.text_area("Current Challenges", key="challenges")

            if st.form_submit_button("🚀 Start Session"):
                initiate_session(
                    st.session_state.name,
                    st.session_state.email,
                    st.session_state.focus,
                    st.session_state.desire,
                    st.session_state.challenges
                )
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
                analyze_image(img_prompt, image_file)
            st.markdown('</div>', unsafe_allow_html=True)


        # Audio Recorder
        st.subheader("🎙️ Audio Recorder")
        audio_data = audiorecorder("⏺️ Record", "⏸️ Stop")


        # Call Handling
        st.subheader("📞 Call Handling")
        if st.button("📞 Start Call", key="call_button"):
            start_call()
        if st.button("❌ End Call", key="end_call"):
            end_call()

    # Chat Window Header
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
            if st.chat_message("user") and msg.get("has_audio"):
                st.markdown(msg["content"])
            else:
                st.markdown(msg["content"])

    # Chat Input at Bottom
    user_input = st.chat_input("💬 Type your message and press Enter...")

    # If user typed a text message
    if user_input:
        send_message_stream(user_input)

    # If user recorded an audio message
    if audio_data and len(audio_data) > 0:
        # Save recorded audio to a temp file
        with NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            audio_data.export(tmp.name, format="mp3")
            tmp_filename = tmp.name

        try:
            # Transcribe with OpenAI
            with open(tmp_filename, "rb") as audio_file:
                transcription = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
            transcript_text = transcription.text.strip()
            # Ensure transcript is only pushed once until new audio is recorded
            if transcript_text and st.session_state.get("last_transcript") != transcript_text:
                st.session_state["chat_history"].append({"role": "user", "content": transcript_text})
                send_message_stream(transcript_text)
                st.session_state["last_transcript"] = transcript_text  # Store last transcript to prevent duplicates

        finally:
            os.remove(tmp_filename)
