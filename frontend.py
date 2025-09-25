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
    create_vector_store_from_files,
    generate_custom_prompt,
    text_to_speech_stream,
    # Removed: global_context  <-- Not used
)

# Import for better async handling
import threading
from queue import Queue
from streamlit_extras.bottom_container import bottom
###################################
# Configuration
###################################
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
TTS_MODEL = "gpt-4o-mini-tts"  # Standard TTS model for compatibility
STT_MODEL = "gpt-4o-transcribe"  # Latest transcription with streaming
STT_MINI_MODEL = "gpt-4o-mini-transcribe"  # Faster for quick responses
TTS_VOICE_INSTRUCTIONS = "Speak with warm empathy, natural pauses, and therapeutic tone."
client = OpenAI(api_key=OPENAI_API_KEY)

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
    st.session_state["listen_mode"] = True
if "tts_voice" not in st.session_state:
    st.session_state["tts_voice"] = "sage"  # Default voice


###################################
# NEW FUNCTION: stt (Text to Speech)
###################################
def stt(text: str):
    """
    Convert text to speech and immediately play it.
    Uses OpenAI's TTS API with proper file writing.
    """
    import uuid
    out_file = f"tts_{uuid.uuid4().hex}.mp3"

    # Create TTS with proper model
    try:
        response = client.audio.speech.create(
            model="tts-1",  # Use standard model for compatibility
            voice="alloy",
            input=text,
            speed=1.1,
            response_format="mp3"
        )
        # Write the audio content to file
        with open(out_file, 'wb') as f:
            f.write(response.content)
    except Exception as e:
        st.error(f"TTS Error: {str(e)}")
        return

    # Streamlit to play the audio
    st.audio(out_file, format="audio/mp3")

    # Clean up the file after playing
    try:
        import os
        os.remove(out_file)
    except:
        pass

def speak_tts_autoplay_chunk(text_chunk: str, voice: str = "alloy"):
    """
    Convert text to speech with autoplay.
    Uses proper API based on documentation.
    """
    text_chunk = text_chunk.strip()
    if not text_chunk or len(text_chunk) < 10:  # Skip very short chunks
        return

    out_file = "tts_chunk.mp3"

    try:
        # Try advanced model with instructions first
        response = client.audio.speech.create(
            model="gpt-4o-mini-tts",  # Advanced model that supports instructions
            voice=voice,
            input=text_chunk,
            instructions="Speak warmly and naturally with good pacing.",
            speed=1.1,
            response_format="mp3"
        )
        # Write the audio content to file
        with open(out_file, 'wb') as f:
            f.write(response.content)
    except Exception as e:
        # Fallback to standard model without instructions
        try:
            response = client.audio.speech.create(
                model="tts-1-hd",  # HD model for better quality
                voice=voice,
                input=text_chunk,
                speed=1.05,
                response_format="mp3"
            )
            with open(out_file, 'wb') as f:
                f.write(response.content)
        except Exception as e2:
            print(f"TTS Error: {str(e2)}")
            return  # Skip TTS if both fail

    # Convert to base64 for embedding
    try:
        with open(out_file, "rb") as f:
            audio_bytes = f.read()
        b64_audio = base64.b64encode(audio_bytes).decode("utf-8")

        # Create hidden autoplay element with preload
        audio_html = f"""
        <audio autoplay preload="auto" style="display:none;">
            <source src="data:audio/mp3;base64,{b64_audio}" type="audio/mp3">
        </audio>
        """
        st.markdown(audio_html, unsafe_allow_html=True)

        # Clean up
        import os
        os.remove(out_file)
    except:
        pass

###################################
# Utility Functions
###################################
async def process_voice_stream(transcript_text: str):
    """
    Process voice input with streaming response for lower latency.
    Uses the optimized voice_analysis function with streaming.
    """
    if not st.session_state["assistant_id"]:
        st.error("⚠️ Please refresh to start a session.")
        return

    # Display user's transcribed message
    with st.chat_message("user"):
        st.markdown(transcript_text)

    # Process with streaming response
    assistant_message = st.chat_message("assistant")
    message_placeholder = assistant_message.empty()

    try:
        # Call voice_analysis with streaming enabled
        response = await asyncio.to_thread(
            voice_analysis,
            st.session_state["session_id"],
            transcript_text,
            "voice.mp3",
            stream_response=True
        )

        if response["status"] != "success":
            message_placeholder.markdown(f"❌ Error: {response.get('message', 'Unknown error')}")
            return

        # Display transcript
        st.session_state["chat_history"].append({
            "role": "user",
            "content": response["transcript"],
            "has_audio": True
        })

        # Stream the response
        full_response = ""

        for chunk in response.get("stream", []):
            full_response += chunk
            message_placeholder.markdown(full_response + "▌")

            await asyncio.sleep(0.001)

        message_placeholder.markdown(full_response)
        st.session_state["chat_history"].append({
            "role": "assistant",
            "content": full_response
        })

        # Speak the ENTIRE response ONCE at the end
        if st.session_state["listen_mode"] and full_response.strip():
            speak_tts_autoplay_chunk(full_response.strip(), st.session_state.get("tts_voice", "alloy"))

    except Exception as e:
        message_placeholder.markdown(f"🚨 Error: {str(e)}")

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
    
        # Don't speak again here - already handled during streaming

    except Exception as e:
        message_placeholder.markdown(f"🚨 Error: {str(e)}")


###################################
# UI Layout
###################################
st.set_page_config(page_title="AI Chatbot")

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
    with st.form("session_init_form"):
        st.markdown("""
            <style>
                @media (max-width: 768px) {
                    .form-header {
                        flex-direction: column; /* Stack logo and text on mobile */
                        align-items: center;
                        text-align: center;
                    }
                    .form-logo {
                        width: 80px !important; /* Smaller logo on mobile */
                    }
                    .form-title {
                        font-size: 24px !important;
                    }
                }
            </style>

            <div class="form-header" style="
                display: flex;
                align-items: flex-start;
                justify-content: center;
                margin-top: 0;
                margin-bottom: 5px;
            ">
                <div class="form-logo" style="flex-shrink: 0;">
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 150" width="300">
                        <rect width="300" height="150" fill="#ffffff"/>
                        <g transform="translate(115, 35)">
                            <circle cx="35" cy="35" r="25" fill="#6366F1"/>
                            <circle cx="80" cy="35" r="6" fill="#6366F1" opacity="0.8"/>
                            <circle cx="95" cy="35" r="4" fill="#6366F1" opacity="0.6"/>
                            <circle cx="105" cy="35" r="3" fill="#6366F1" opacity="0.4"/>
                        </g>
                        <text x="150" y="110" font-family="sans-serif" font-size="20" font-weight="500" text-anchor="middle" fill="#1E293B">NARM</text>
                        <text x="150" y="130" font-family="sans-serif" font-size="12" font-weight="400" text-anchor="middle" fill="#6366F1">whisper</text>
                    </svg>
                </div>
            </div>
            <h2 style="text-align:center; color:#6366F1;">Chatbot</h2>
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

        # Advanced configuration section
        with st.expander("⚙️ **Advanced Configuration** (Optional)"):
            st.markdown("### 📚 Upload Knowledge Base")
            st.info("Upload PDF, DOCX, or TXT files to create your custom knowledge base. Leave empty to use default NARM therapy knowledge.")
            uploaded_files = st.file_uploader(
                "Choose files",
                type=['pdf', 'docx', 'txt'],
                accept_multiple_files=True,
                help="Upload documents that will be used as the AI's knowledge base"
            )

            st.markdown("### 🤖 AI System Prompt Builder")

            # Show status if prompt is finalized
            if st.session_state.get("finalized_prompt"):
                st.success("✅ **Custom prompt is finalized and ready to use!**")
            else:
                st.info("Describe what kind of assistant you want, and AI will generate a production-ready prompt for you.")

            # Initialize session state for prompt generation
            if "generated_prompt" not in st.session_state:
                st.session_state.generated_prompt = ""
            if "prompt_instructions" not in st.session_state:
                st.session_state.prompt_instructions = ""
            if "pending_generation" not in st.session_state:
                st.session_state.pending_generation = False
            if "is_generating" not in st.session_state:
                st.session_state.is_generating = False
            if "edit_mode" not in st.session_state:
                st.session_state.edit_mode = False
            if "finalized_prompt" not in st.session_state:
                st.session_state.finalized_prompt = None

            # Instructions input
            prompt_instructions = st.text_area(
                "Describe Your Assistant",
                placeholder="""Example: Create a CBT therapist who specializes in anxiety and depression, uses evidence-based techniques, and provides homework exercises.

Or: Build a life coach focused on career development and goal-setting, with expertise in executive functioning and ADHD support.

Or: Design a mindfulness meditation guide with expertise in Buddhist psychology and somatic practices.""",
                height=120,
                value=st.session_state.prompt_instructions,
                help="Describe the type of assistant, expertise, approach, and style you want"
            )

            # Update instructions in session state
            if prompt_instructions:
                st.session_state.prompt_instructions = prompt_instructions

            # Checkbox to trigger AI prompt generation
            generate_prompt = st.checkbox(
                "🎯 Generate AI Prompt from instructions",
                value=st.session_state.get("generate_prompt_flag", False),
                help="Check this to have AI generate a production-ready prompt based on your instructions"
            )
            st.session_state.generate_prompt_flag = generate_prompt

            # Show generation status
            if st.session_state.get("is_generating", False):
                st.info("🧠 AI is crafting your custom prompt... This may take a moment.")

            # Show generated prompt preview first (read-only)
            if st.session_state.generated_prompt and not st.session_state.get("edit_mode", False):
                st.markdown("#### ✨ Generated Prompt Preview")
                st.success("AI has generated your custom prompt! Review it below:")

                # Show preview in a text area (no nested expander)
                st.text_area(
                    "📖 Generated Prompt",
                    value=st.session_state.generated_prompt,
                    height=200,
                    disabled=True
                )

                # Info about using the prompt
                st.info("🎯 To use this prompt, check the box below and submit the form.")

                # Single checkbox to confirm using the prompt
                use_this_prompt = st.checkbox(
                    "✅ **Use this AI-generated prompt**",
                    key="use_generated_prompt",
                    value=st.session_state.get("use_generated_prompt", False)
                )

                if use_this_prompt:
                    st.session_state.finalized_prompt = st.session_state.generated_prompt
                    custom_prompt = st.session_state.generated_prompt
                else:
                    custom_prompt = None

            # Allow direct editing of the generated prompt
            elif st.session_state.generated_prompt:
                st.markdown("#### ✏️ Edit Generated Prompt (Optional)")
                st.info("You can edit the AI-generated prompt below to fine-tune it.")

                edited_prompt = st.text_area(
                    "System Prompt",
                    value=st.session_state.generated_prompt,
                    height=300,
                    help="Edit the generated prompt to customize it further",
                    key="edited_prompt_text"
                )

                # Checkbox to use the edited version
                use_edited = st.checkbox(
                    "✅ **Use this edited version**",
                    key="use_edited_prompt",
                    value=st.session_state.get("use_edited_prompt", False)
                )

                if use_edited:
                    st.session_state.finalized_prompt = edited_prompt
                    custom_prompt = edited_prompt
                else:
                    custom_prompt = None

            # Default NARM prompt option
            else:
                if not st.session_state.generated_prompt:
                    st.markdown("#### 💡 Or Use Default NARM Therapy Prompt")
                    use_default = st.checkbox("Use default NARM therapy assistant (recommended for therapy sessions)")
                    custom_prompt = "" if use_default else None
                else:
                    custom_prompt = st.session_state.get("finalized_prompt", None)

        st.markdown("<br>", unsafe_allow_html=True)  # Adds space before button

        # Handle Form Submission
        submit_button = st.form_submit_button(label='Initiate Session', use_container_width=True, type='primary')

        if submit_button:
            # Check if we need to generate AI prompt first
            if st.session_state.get("generate_prompt_flag") and st.session_state.prompt_instructions and not st.session_state.generated_prompt:
                with st.spinner("🧠 AI is crafting your custom prompt..."):
                    user_info = {
                        "name": name,
                        "email": email,
                        "focus_today": focus_today,
                        "desired_outcome": desired_outcome,
                        "current_challenges": current_challenges
                    }
                    generated = generate_custom_prompt(st.session_state.prompt_instructions, user_info)
                    st.session_state.generated_prompt = generated
                    st.session_state.is_generating = False
                    st.session_state.edit_mode = False
                    st.success("✨ Prompt generated successfully! Please review and submit again.")
                    st.rerun()

            custom_vector_store_id = None

            # Create vector store from uploaded files if any
            if uploaded_files:
                with st.spinner("📚 Creating custom knowledge base from uploaded files..."):
                    custom_vector_store_id = create_vector_store_from_files(
                        uploaded_files,
                        f"CustomKB-{email[:8] if email else 'user'}"
                    )
                    if not custom_vector_store_id:
                        st.warning("⚠️ Could not create custom knowledge base. Using default.")

            # Use finalized prompt if available
            final_prompt_to_use = st.session_state.get("finalized_prompt", None)

            # Debug info to confirm what's being used
            if final_prompt_to_use:
                st.info(f"🎯 Using {'AI-generated' if 'sports' in final_prompt_to_use.lower() or 'analyst' in final_prompt_to_use.lower() else 'custom'} prompt")
            else:
                st.info("🎯 Using default NARM therapy prompt")

            # Initialize chat with custom or default settings
            with st.spinner("🚀 Initializing your personalized session..."):
                response = initiate_chat(
                    name=name,
                    email=email,
                    focus_today=focus_today,
                    desired_outcome=desired_outcome,
                    current_challenges=current_challenges,
                    custom_vector_store_id=custom_vector_store_id,
                    custom_prompt=final_prompt_to_use if final_prompt_to_use else None
                )

                if response["status"] == "success":
                    st.session_state["assistant_id"] = response['assistant_id']
                    st.session_state["session_id"] = response['session_id']
                    if custom_vector_store_id:
                        st.success("✅ Custom knowledge base loaded successfully!")
                    if final_prompt_to_use:
                        st.success("✅ Custom prompt configured and active!")
                    time.sleep(1)  # Brief pause to show success messages
                    st.rerun()
                else:
                    st.error(f"❌ Failed to initialize session: {response.get('message', 'Unknown error')}")

    # Clean up unused pending generation flag
    if "pending_generation" in st.session_state:
        st.session_state.pending_generation = False

    st.stop()


###################################
# Main Chat Interface (Post-Login)
###################################
# Sidebar Tools
with st.sidebar:
    st.header("🛠️ Tools")

    # Voice Settings
    with st.expander("🔊 Voice Settings"):
        st.markdown('<div class="tool-card">', unsafe_allow_html=True)
        st.session_state["tts_voice"] = st.selectbox(
            "Assistant Voice",
            ["alloy", "nova", "shimmer", "echo", "fable", "onyx"],
            index=0,
            help="Choose the voice for text-to-speech responses"
        )

        # Voice preview button
        if st.button("🔈 Preview Voice", key="preview_voice"):
            preview_text = f"Hello {st.session_state.get('name', 'there')}! This is how I sound."
            speak_tts_autoplay_chunk(preview_text, st.session_state["tts_voice"])
            st.success("Voice preview played!")
        st.markdown('</div>', unsafe_allow_html=True)

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
            <style>
                @media (max-width: 768px) {
                    .form-header {
                        flex-direction: column; /* Stack logo and text on mobile */
                        align-items: center;
                        text-align: center;
                    }
                    .form-logo {
                        width: 80px !important; /* Smaller logo on mobile */
                    }
                    .form-title {
                        font-size: 24px !important;
                    }
                }
            </style>

            <div class="form-header" style="
                display: flex;
                align-items: flex-start;
                justify-content: center;
                margin-top: 0;
                margin-bottom: 5px;
            ">
                <div class="form-logo" style="flex-shrink: 0;">
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 150" width="300">
                        <rect width="300" height="150" fill="#ffffff"/>
                        <g transform="translate(115, 35)">
                            <circle cx="35" cy="35" r="25" fill="#6366F1"/>
                            <circle cx="80" cy="35" r="6" fill="#6366F1" opacity="0.8"/>
                            <circle cx="95" cy="35" r="4" fill="#6366F1" opacity="0.6"/>
                            <circle cx="105" cy="35" r="3" fill="#6366F1" opacity="0.4"/>
                        </g>
                        <text x="150" y="110" font-family="sans-serif" font-size="20" font-weight="500" text-anchor="middle" fill="#1E293B">NARM</text>
                        <text x="150" y="130" font-family="sans-serif" font-size="12" font-weight="400" text-anchor="middle" fill="#6366F1">whisper</text>
                    </svg>
                </div>
            </div>
            <h2 style="text-align:center; color:#6366F1;">Therapybot</h2>
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
        audio_data = audiorecorder("🎙️Record", "⏹️Stop", custom_style={'color': 'red'})

    with col1:
        st.session_state["listen_mode"] = st.checkbox("📢Talk", value=True)


# Text message handling
if user_input:
    asyncio.run(send_message_stream(user_input))

# Audio message handling - Optimized pipeline
if audio_data and len(audio_data) > 0:
    original_listen_mode = st.session_state["listen_mode"]
    st.session_state["listen_mode"] = False

    with NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        audio_data.export(tmp.name, format="mp3")
        tmp_filename = tmp.name

    try:
        # Read audio file
        with open(tmp_filename, "rb") as f:
            audio_bytes = f.read()

        # Use voice_analysis with streaming for complete pipeline
        # Determine if we should use the mini model for faster response
        use_mini = len(audio_bytes) < 500000  # Use mini for short clips (<500KB)

        with st.spinner("🎙️ Processing voice..."):
            response = voice_analysis(
                session_id=st.session_state["session_id"],
                audio_data=audio_bytes,
                filename="voice_input.mp3",
                stream_response=True,
                use_mini_model=use_mini
            )

            if response["status"] == "success":
                transcript_text = response.get("transcript", "").strip()

                if transcript_text and st.session_state.get("last_transcript") != transcript_text:
                    st.session_state["last_transcript"] = transcript_text

                    # Display user message
                    st.session_state["chat_history"].append({
                        "role": "user",
                        "content": transcript_text,
                        "has_audio": True
                    })

                    # Display and stream assistant response
                    with st.chat_message("user"):
                        st.markdown(transcript_text)

                    assistant_message = st.chat_message("assistant")
                    message_placeholder = assistant_message.empty()
                    full_response = ""

                    # Stream the response WITHOUT TTS during streaming
                    for chunk in response.get("stream", []):
                        full_response += chunk
                        message_placeholder.markdown(full_response + "▌")
                        time.sleep(0.001)

                    message_placeholder.markdown(full_response)
                    st.session_state["chat_history"].append({
                        "role": "assistant",
                        "content": full_response
                    })

                    # Speak the ENTIRE response ONCE at the end
                    if original_listen_mode and full_response.strip():
                        speak_tts_autoplay_chunk(full_response.strip(), st.session_state.get("tts_voice", "alloy"))

                    # Don't rerun immediately - let the audio play
            else:
                st.error(f"❌ {response.get('message', 'Voice processing failed')}")
                st.session_state["last_transcript"] = ""

    except Exception as e:
        st.error(f"Error processing voice: {str(e)}")
        st.session_state["last_transcript"] = ""

    finally:
        os.remove(tmp_filename)
        st.session_state["listen_mode"] = original_listen_mode
