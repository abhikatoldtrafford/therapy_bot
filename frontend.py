import streamlit as st
import time
import os
import asyncio
from tempfile import NamedTemporaryFile
from audiorecorder import audiorecorder
from openai import OpenAI
import base64
from backend import (
    initiate_chat,
    chat,
    image_analysis,
    voice_analysis,
    # Removed: global_context  <-- Not used
)
from streamlit_extras.bottom_container import bottom
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
    st.session_state["thread_id"] = None
    st.session_state["chat_history"] = []
    st.session_state["last_transcript"] = None
if "session_id" not in st.session_state:
    st.session_state["session_id"] = None
# Add a place to store "Listen Mode" in session state.
if "listen_mode" not in st.session_state:
    st.session_state["listen_mode"] = False


###################################
# NEW FUNCTION: stt (Text to Speech)
###################################
def stt(text: str):
    """
    Convert text to speech and immediately play it.
    Uses OpenAI's TTS in streaming mode. Minimal changes:
    we just write the file, then st.audio it.
    """
    # Create a temporary audio file
    import uuid
    out_file = f"tts_{uuid.uuid4().hex}.mp3"
    
    # Create TTS stream
    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=text,
    )
    response.stream_to_file(out_file)

    # Streamlit to play the audio
    st.audio(out_file, format="audio/mp3")

def speak_tts_autoplay_chunk(text_chunk: str):
    """
    Convert a partial text chunk to speech and immediately autoplay it,
    without showing audio controls (hidden).
    """
    text_chunk = text_chunk.strip()
    if not text_chunk:
        return
    out_file = "tts_chunk.wav"
    # Make a TTS request specifying 'wav' so we can embed it easily
    with client.audio.speech.with_streaming_response.create(model="tts-1",voice="alloy",input=text_chunk) as  response:
        response.stream_to_file(out_file)
    # Convert WAV to base64
    with open(out_file, "rb") as f:
        audio_bytes = f.read()
    b64_audio = base64.b64encode(audio_bytes).decode("utf-8")

    # Create hidden autoplay <audio> element
    audio_html = f"""
    <audio autoplay style="display:none;">
        <source src="data:audio/wav;base64,{b64_audio}" type="audio/wav">
    </audio>
    """

    st.markdown(audio_html, unsafe_allow_html=True)

###################################
# Utility Functions
###################################
async def send_message_stream(prompt: str):
    """
    Send text message with streaming response using backend.chat().

    Converted to async so that it doesn't block the entire app run.
    """
    if not st.session_state["assistant_id"]:
        st.error("⚠️ Please refresh to start a session.")
        return

    with st.chat_message("user"):
        st.markdown(prompt)

    assistant_message = st.chat_message("assistant")
    message_placeholder = assistant_message.empty()
    # message_placeholder.markdown("_💭 Thinking..._")

    try:
        # The call to 'chat' is executed in a background thread, so it doesn't block.
        with st.spinner("Typing..."):
            response = await asyncio.to_thread(chat, st.session_state["session_id"], prompt, True)
            
            if response["status"] != "success":
                message_placeholder.markdown(f"❌ Error: {response.get('message', 'Unknown error')}")
                return

            full_response = ""
            for chunk in response["stream"]:
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")
                # (No logic change; only replaced time.sleep with asyncio.sleep)
                await asyncio.sleep(0.001)
            
        message_placeholder.markdown(full_response)
        st.session_state["chat_history"].append({"role": "user", "content": prompt})
        st.session_state["chat_history"].append({"role": "assistant", "content": full_response})
            
        # If "Listen Mode" is ON, speak the assistant's final text
        if st.session_state["listen_mode"]:
            speak_tts_autoplay_chunk(full_response)

    except Exception as e:
        message_placeholder.markdown(f"🚨 Error: {str(e)}")


###################################
# UI Layout
###################################
st.set_page_config(page_title="ARUNYA AI Therapybot")

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
            <div style="
                display: flex;
                align-items: center;
                justify-content: flex-start;
                margin-bottom: 2rem;
                margin-left: -150px;
                padding: 1rem 1.5rem;
                background: #F9FAFB;
                border-radius: 10px;
                box-shadow: 0 2px 6px rgba(0,0,0,0.05);
                max-width: 80%;
            ">
                <div style="flex-shrink: 0;">
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
                </div>
                <div style="margin-left: 10px; text-align: left;">
                    <h1 style="color: #1E293B; margin: 0; font-size: 36px; font-weight: bold; white-space: nowrap;">
                        Therapy Assistant
                    </h1>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with st.form("session_init_form"):
        st.markdown("""
            <h2 style='text-align: center; color: #1E293B; margin-bottom: 10px;'>
                🚀 Start Your Therapy Session
            </h2>
            <hr style="border: 1px solid #6366F1; margin-bottom: 20px;">
        """, unsafe_allow_html=True)
        
        # Two-column layout for Name & Email with better spacing
        col1, col2 = st.columns([1, 1])  
        with col1:
            name = st.text_input("🧑 **Full Name**", placeholder="Enter your full name")  
        with col2:
            email = st.text_input("📧 **Email**", placeholder="Enter your email")

        focus_today = st.text_input("🎯 **Today's Focus**", placeholder="What do you want to focus on?")
        desired_outcome = st.text_input("🌟 **Desired Outcome**", placeholder="What result are you hoping for?")
        current_challenges = st.text_input("⚠️ **Current Challenges**", placeholder="What are you struggling with?")

        st.markdown("<br>", unsafe_allow_html=True)  # Adds space before button

        # Handle Form Submission
        if st.form_submit_button(label = 'Initiate Session',use_container_width=True, type = 'primary'):
            response = initiate_chat(
                name=name,
                email=email,
                focus_today=focus_today,
                desired_outcome=desired_outcome,
                current_challenges=current_challenges
            )
            
            if response["status"] == "success":
                st.session_state["assistant_id"] = response['assistant_id']
                st.session_state["session_id"] = response['session_id']
                st.rerun()
            else:
                st.error(f"❌ Failed to initialize session: {response.get('message', 'Unknown error')}")

    st.stop()

###################################
# Main Chat Interface (Post-Login)
###################################
# Sidebar Tools
with st.sidebar:
    st.header("🛠️ Tools")
    with st.expander("📸 Image Analysis"):
        st.markdown('<div class="tool-card">', unsafe_allow_html=True)
        img_prompt = st.text_input("Analysis Prompt", key="img_prompt")
        image_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"], key="image_file")
        if st.button("🔍 Analyze Image", key="analyze_image"):
            if image_file:
                with st.spinner("📷 Analyzing image..."):
                    response = image_analysis(
                        session_id=st.session_state["session_id"],
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
    # st.subheader("🎙️ Audio Recorder")
    #audio_data = audiorecorder("⏺️ Record", "⏸️ Stop")

# Chat Window Header
st.markdown("""
<div style="display: flex; align-items: center; justify-content: flex-start; margin-bottom: 2rem;">
    <div style="flex-shrink: 0;">
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
    </div>
    <div style="margin-left: 20px; text-align: left;">
        <h1 style="color: #1E293B; margin: 0; font-size: 36px; font-weight: bold;">AI Therapeutic Assist</h1>
    </div>
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
                # Just re-displaying the last uploaded image; minimal approach
                st.image(image_file, caption="Uploaded Image")
            else:
                st.markdown(msg["content"])



with bottom():
    col1, col2, col3 = st.columns([1, 5, 2])  # Adjust column widths as needed

    with col2:
        user_input = st.chat_input("💬 Type your message and press Enter...")

    with col3:
        audio_data = audiorecorder("🎙️Record", "⏹️Stop")

    with col1:
        st.session_state["listen_mode"] = st.checkbox("📢", value=False)


# Text message handling
if user_input:
    asyncio.run(send_message_stream(user_input))

# Audio message handling
if audio_data and len(audio_data) > 0:
    original_listen_mode = st.session_state["listen_mode"]
    st.session_state["listen_mode"] = False
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
        st.session_state["listen_mode"] = original_listen_mode
        if transcript_text:
            # Only process if this transcript is new
            if st.session_state.get("last_transcript") != transcript_text:
                st.session_state["last_transcript"] = transcript_text
                st.session_state["chat_history"].append({
                    "role": "user",
                    "content": transcript_text,
                    "has_audio": True
                })
                asyncio.run(send_message_stream(transcript_text))
        else:
            st.error("Transcription returned no text.")
            st.session_state["last_transcript"] = ""  # Update state to avoid reprocessing

    except Exception as e:
        # Catch errors from the OpenAI transcription call
        st.error("Error transcribing audio: " + str(e))
        st.session_state["last_transcript"] = ""  # Update state so error won't cause repeat attempts

    finally:
        os.remove(tmp_filename)
        st.session_state["listen_mode"] = True