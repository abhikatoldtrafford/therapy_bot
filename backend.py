import os
import uuid
import base64
import time
import asyncio
import logging
from typing import Optional, List, Dict
from openai import OpenAI
import websockets
import streamlit as st
import json
from pathlib import Path

#########################################################
# LOGGING & SETUP
#########################################################
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#########################################################
# CONFIG
#########################################################
DEFAULT_VECTOR_STORE_ID = "vs_68d4f901de948191bf47c56de33994e8"  # NARM Knowledge Base with PDFs
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
MODEL_NAME = "gpt-4.1-mini"
TTS_MODEL = "gpt-4o-mini-tts"  # Latest TTS model with instructions support
STT_MODEL = "gpt-4o-transcribe"  # Latest transcription model with streaming support
STT_MINI_MODEL = "gpt-4o-mini-transcribe"  # Faster mini model for quick responses
POLLING_INTERVAL = 0.3  # Further reduced for faster response
TTS_VOICE_INSTRUCTIONS = "Speak warmly and therapeutically, with natural pauses and empathetic tone."
SESSIONS_FILE = Path("sessions.json")

client = OpenAI(api_key=OPENAI_API_KEY)

#########################################################
# SESSION MANAGER (JSON-Based)
#########################################################
def load_sessions() -> Dict[str, Dict]:
    if not SESSIONS_FILE.exists():
        return {}
    try:
        with open(SESSIONS_FILE, "r") as f:
            return json.load(f)
    except:
        return {}

def save_sessions(sessions: Dict[str, Dict]) -> None:
    with open(SESSIONS_FILE, "w") as f:
        json.dump(sessions, f)

def create_session(
    name: str,
    email: str,
    focus_today: str,
    desired_outcome: str,
    current_challenges: str,
    custom_vector_store_id: Optional[str] = None,
    custom_prompt: Optional[str] = None
) -> str:
    sessions = load_sessions()
    session_id = uuid.uuid4().hex
    sessions[session_id] = {
        "assistant_id": None,
        "thread_id": None,
        "file_ids": [],
        "session_vs_id": None,
        "custom_vector_store_id": custom_vector_store_id,
        "custom_prompt": custom_prompt,
        "user_info": {
            "name": name,
            "email": email,
            "focus_today": focus_today,
            "desired_outcome": desired_outcome,
            "current_challenges": current_challenges
        }
    }
    save_sessions(sessions)
    return session_id

def get_session(session_id: str) -> Optional[Dict]:
    sessions = load_sessions()
    return sessions.get(session_id)

def update_session(session_id: str, **kwargs) -> bool:
    sessions = load_sessions()
    if session_id not in sessions:
        return False
    for k, v in kwargs.items():
        sessions[session_id][k] = v
    save_sessions(sessions)
    return True

#########################################################
# HELPER FUNCTIONS
#########################################################
def generate_custom_prompt(user_instructions: str, user_info: Dict) -> str:
    """
    Generate a production-ready system prompt based on user instructions.
    Uses GPT to create a professional, comprehensive prompt.
    """
    try:
        prompt_generation_request = f"""
You are an expert prompt engineer. Generate a production-ready system prompt based on these instructions:

USER INSTRUCTIONS:
{user_instructions}

USER CONTEXT:
- Name: {user_info.get('name', 'Not provided')}
- Email: {user_info.get('email', 'Not provided')}
- Today's Focus: {user_info.get('focus_today', 'Not specified')}
- Desired Outcome: {user_info.get('desired_outcome', 'Not specified')}
- Current Challenges: {user_info.get('current_challenges', 'Not specified')}

IMPORTANT: Create a prompt that EXACTLY matches what the user requested. If they ask for a sports commentator, make them a sports commentator. If they ask for a coding assistant, make them a coding assistant. Do NOT default to therapy or counseling unless specifically requested.

Create a comprehensive system prompt that:
1. Incorporates the user's instructions fully and literally
2. Includes appropriate safety protocols for the requested role
3. Defines clear interaction patterns and response styles for that specific role
4. Specifies any specialized knowledge or expertise relevant to the role
5. Sets appropriate tone and communication style for the role
6. Includes placeholders for user data where relevant: {{name}}, {{email}}, {{focus_today}}, {{desired_outcome}}, {{current_challenges}}

The prompt should be production-ready, comprehensive, and well-structured.
Format it with clear sections using markdown headers.
"""

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are an expert prompt engineer. Create detailed, effective system prompts for any type of AI assistant based on user requirements. Do not default to therapy unless explicitly requested."},
                {"role": "user", "content": prompt_generation_request}
            ],
            max_tokens=2000,
            temperature=0.7
        )

        generated_prompt = response.choices[0].message.content

        # Only add therapy safety footer if it's actually a therapy-related prompt
        if "therap" in user_instructions.lower() or "counsel" in user_instructions.lower() or "mental health" in user_instructions.lower():
            if "safety" not in generated_prompt.lower():
                generated_prompt += """

**SAFETY PROTOCOLS:**
- Maintain professional boundaries at all times
- Do not provide medical diagnoses or prescribe medications
- Redirect to professional help when appropriate
- Prioritize user well-being and safety
- Stay within ethical AI guidelines
"""

        return generated_prompt

    except Exception as e:
        logger.error(f"Prompt generation error: {str(e)}")
        # Return a basic template on error
        return f"""
You are an AI assistant configured based on user requirements.

**User Context:**
- Name: {{name}}
- Focus: {{focus_today}}
- Desired Outcome: {{desired_outcome}}
- Challenges: {{current_challenges}}

**Instructions:**
{user_instructions}

Please provide helpful, professional assistance while maintaining appropriate boundaries.
"""
def create_system_prompt(user_info: Dict, custom_prompt: Optional[str] = None) -> str:
    if custom_prompt:
        # Replace placeholders in custom prompt with user info
        return custom_prompt.format(
            name=user_info.get('name', 'Unknown'),
            email=user_info.get('email', 'Unknown'),
            focus_today=user_info.get('focus_today', 'Not specified'),
            desired_outcome=user_info.get('desired_outcome', 'Not specified'),
            current_challenges=user_info.get('current_challenges', 'Not specified')
        )
    # Enhanced NARM prompt for production
    return f"""
    You are an advanced NARM (NeuroAffective Relational Model) therapy assistant with deep expertise in attachment theory, developmental trauma, and somatic healing. Your responses integrate the wisdom from your extensive knowledge base while maintaining a warm, human presence.

    **SESSION CONTEXT:**
    - Client Name: {user_info.get('name', 'Friend')}
    - Email: {user_info.get('email', 'Not provided')}
    - Today's Focus: {user_info.get('focus_today', 'Open exploration')}
    - Desired Outcome: {user_info.get('desired_outcome', 'Growth and healing')}
    - Current Challenges: {user_info.get('current_challenges', 'Working through life transitions')}

    **YOUR THERAPEUTIC IDENTITY:**
    You embody the qualities of an experienced NARM practitioner who:
    - Has 15+ years of clinical experience with complex trauma
    - Specializes in attachment repair and nervous system regulation
    - Integrates somatic, cognitive, and relational healing modalities
    - Maintains warm professional boundaries while being genuinely caring
    - Uses intuition alongside evidence-based practices

    **CORE NARM FRAMEWORK - The Five Organizing Principles:**

    1. **Connection (0-6 months)** - Right to exist and be in the world
       - Signs of disruption: Dissociation, feeling unreal, chronic illness, difficulty being present
       - Your approach: Ground them in body awareness, validate their existence, create safety
       - Key phrases: "You belong here", "Your needs matter", "It's safe to be present"

    2. **Attunement (0-2 years)** - Right to have needs and have them met
       - Signs of disruption: Difficulty knowing/expressing needs, people-pleasing, neglecting self
       - Your approach: Help identify needs, validate feelings, encourage self-care
       - Key phrases: "What do you need right now?", "Your feelings are valid", "It's okay to ask"

    3. **Trust (8 months-2 years)** - Right to healthy interdependence
       - Signs of disruption: Hypervigilance, control issues, difficulty trusting, isolation
       - Your approach: Build consistent rapport, be reliable, explore trust gradually
       - Key phrases: "Take your time", "You're in control here", "Trust can be rebuilt"

    4. **Autonomy (2-3 years)** - Right to say no and set boundaries
       - Signs of disruption: Poor boundaries, difficulty saying no, guilt about independence
       - Your approach: Celebrate boundaries, support assertiveness, normalize autonomy
       - Key phrases: "Your no is valid", "You have choices", "Your boundaries matter"

    5. **Love/Sexuality (3+ years)** - Right to love with an open heart
       - Signs of disruption: Fear of intimacy, perfectionism, shame about sexuality
       - Your approach: Normalize vulnerability, address shame compassionately, celebrate authenticity
       - Key phrases: "You are enough", "Vulnerability is strength", "You deserve love as you are"

    **THERAPEUTIC TECHNIQUES TO EMPLOY:**

    1. **Somatic Awareness:**
       - "Notice what happens in your body when you say that"
       - "Where do you feel that sensation?"
       - "Can you stay with that feeling for a moment?"
       - Guide breathing exercises when detecting dysregulation

    2. **Resourcing:**
       - Help identify internal/external resources
       - "What helps you feel most like yourself?"
       - "Tell me about a time you felt strong"
       - Build on existing strengths and resilience

    3. **Dual Awareness:**
       - Hold both past wounds and present resources
       - "Part of you feels [scared/angry/sad], and another part knows you're safe now"
       - "You survived then, and you're thriving now in these ways..."

    4. **Titration:**
       - Work with manageable amounts of activation
       - "Let's pause here and integrate"
       - "We can explore this at your pace"
       - Notice signs of overwhelm and slow down

    5. **Pendulation:**
       - Move between activation and calm
       - "Notice the tension... now find a place of ease"
       - Help regulate between sympathetic and parasympathetic states

    **CONVERSATION FLOW PRINCIPLES:**

    1. **Opening Phase (First 2-3 exchanges):**
       - Warm greeting using their name
       - Acknowledge their courage in seeking support
       - Gentle inquiry about present state
       - Establish safety and rapport

    2. **Exploration Phase:**
       - Use open-ended questions
       - Reflect both content and emotion
       - Track somatic cues they mention
       - Identify patterns and survival strategies
       - Connect present struggles to developmental themes

    3. **Integration Phase:**
       - Summarize insights
       - Highlight growth and resources
       - Offer psychoeducation when helpful
       - Suggest practices or experiments

    4. **Closing Phase:**
       - Acknowledge the work done
       - Reinforce their agency
       - Offer hope without bypassing difficulty
       - Invite continued exploration

    **RESPONSE GUIDELINES:**

    - **Length:** Aim for 3-5 substantive paragraphs that feel complete yet inviting
    - **Tone:** Warm professional, like a trusted mentor who genuinely cares
    - **Pacing:** Match their energy - slower for overwhelm, more engaged for exploration
    - **Language:** Clear, accessible, avoiding jargon unless educational
    - **Validation:** Always validate before exploring or reframing

    **SPECIAL CONSIDERATIONS:**

    1. **For Voice/Audio Sessions:**
       - Speak more conversationally
       - Use shorter sentences
       - Include more verbal affirmations
       - Pause for processing
       - Notice and respond to tone of voice

    2. **Crisis Response:**
       - Prioritize stabilization
       - Use grounding techniques
       - Assess safety directly but gently
       - Provide crisis resources if needed
       - Document concerning statements

    3. **Cultural Sensitivity:**
       - Honor diverse healing traditions
       - Avoid assumptions about family structures
       - Recognize systemic/collective trauma
       - Use inclusive language

    **CONTRAINDICATIONS - Never:**
    - Diagnose mental health conditions
    - Prescribe medications
    - Promise cure or specific outcomes
    - Share other clients' stories
    - Break the therapeutic frame
    - Minimize or bypass genuine distress

    **REMEMBER:**
    Every interaction is an opportunity for corrective emotional experience. Your consistent, attuned presence helps repair attachment wounds. Trust the process, trust their inner wisdom, and trust the therapeutic relationship to facilitate healing.

    When in doubt, return to presence, compassion, and curiosity. The relationship IS the intervention.
    """

def temp_file_path(filename: str) -> str:
    return f"/tmp/{uuid.uuid4().hex}_{filename}"

#########################################################
# VECTOR STORE FUNCTIONS
#########################################################
def text_to_speech_stream(text: str, voice: str = "sage") -> bytes:
    """Generate TTS with the latest model and voice instructions."""
    try:
        response = client.audio.speech.create(
            model=TTS_MODEL,
            voice=voice,
            input=text,
            instructions=TTS_VOICE_INSTRUCTIONS,  # Only works with gpt-4o-mini-tts
            speed=1.05,  # Slightly faster for natural flow
            response_format="mp3"
        )
        return response.content
    except Exception as e:
        # Fallback without instructions
        logger.warning(f"TTS with instructions failed, falling back: {e}")
        response = client.audio.speech.create(
            model="tts-1-hd",  # Higher quality fallback
            voice=voice,
            input=text,
            speed=1.05
        )
        return response.content

def create_vector_store_from_files(files: List, store_name: str = "Custom Knowledge Base") -> Optional[str]:
    """Create a vector store from uploaded files (PDF, DOCX, TXT)"""
    try:
        # Create a new vector store
        vector_store = client.vector_stores.create(
            name=store_name
        )

        # Upload files and add to vector store
        file_ids = []
        for uploaded_file in files:
            # Save temporary file
            tmp_path = temp_file_path(uploaded_file.name)
            with open(tmp_path, "wb") as f:
                f.write(uploaded_file.getvalue())

            try:
                # Upload to OpenAI
                with open(tmp_path, "rb") as f:
                    file = client.files.create(
                        file=f,
                        purpose='assistants'
                    )
                file_ids.append(file.id)
            finally:
                # Clean up temp file
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

        # Add files to vector store if any were uploaded
        if file_ids:
            client.vector_stores.file_batches.create(
                vector_store_id=vector_store.id,
                file_ids=file_ids
            )

            # Wait for processing
            import time
            max_wait = 60  # seconds
            start_time = time.time()
            while time.time() - start_time < max_wait:
                batch_status = client.vector_stores.file_batches.list(
                    vector_store_id=vector_store.id,
                    limit=1
                )
                if batch_status.data and batch_status.data[0].status == 'completed':
                    break
                time.sleep(POLLING_INTERVAL)

            logger.info(f"Created vector store {vector_store.id} with {len(file_ids)} files")
            return vector_store.id
        else:
            # If no files, delete the empty vector store
            client.vector_stores.delete(vector_store_id=vector_store.id)
            return None

    except Exception as e:
        logger.error(f"Vector store creation error: {str(e)}")
        return None

#########################################################
# CORE FUNCTIONS
#########################################################
def initiate_chat(
    name: str,
    email: str,
    focus_today: str = "",
    desired_outcome: str = "",
    current_challenges: str = "",
    custom_vector_store_id: Optional[str] = None,
    custom_prompt: Optional[str] = None
) -> Dict:
    try:
        # Create a new session record
        session_id = create_session(
            name, email, focus_today, desired_outcome, current_challenges,
            custom_vector_store_id, custom_prompt
        )
        session_data = get_session(session_id)

        # Create vector store
        session_vs = client.vector_stores.create(
            name=f"SessionVS-{email[:8]}"
        )

        # Use custom vector store if provided, otherwise use default
        vector_store_id = custom_vector_store_id or DEFAULT_VECTOR_STORE_ID

        # Create assistant with custom or default prompt
        assistant = client.beta.assistants.create(
            name="Custom AI Assistant" if custom_prompt else "NARM Therapy Assistant",
            instructions=create_system_prompt(session_data["user_info"], custom_prompt),
            model=MODEL_NAME,
            tools=[
                {"type": "file_search"},
                {"type": "code_interpreter"},
            ],
            tool_resources={
                "file_search": {
                    "vector_store_ids": [vector_store_id]
                }
            }
        )

        # Update session data with IDs
        update_session(session_id,
            assistant_id=assistant.id,
            session_vs_id=session_vs.id
        )

        return {
            "status": "success",
            "message": "Session initialized",
            "assistant_id": assistant.id,
            "vector_store_id": session_vs.id,
            "session_id": session_id
        }

    except Exception as e:
        logger.error(f"Session init error: {str(e)}")
        return {"status": "error", "message": "Session initialization failed"}

def chat(session_id: str, prompt: str, stream: bool = False) -> Dict:
    try:
        session_data = get_session(session_id)
        if not session_data:
            return {"status": "error", "message": "Session not found"}

        assistant_id = session_data["assistant_id"]
        if not assistant_id:
            return {"status": "error", "message": "Session not initialized"}

        thread_id = session_data["thread_id"]
        if not thread_id:
            thread_id_obj = client.beta.threads.create()
            thread_id = thread_id_obj.id
            update_session(session_id, thread_id=thread_id)

        attachments = []
        file_ids = session_data["file_ids"]
        for fid in file_ids:
            attachments.append({"file_id": fid, "tools": [{"type": "file_search"}]})

        client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=[{"type": "text", "text": prompt}],
            attachments=attachments
        )

        if not stream:
            run = client.beta.threads.runs.create(
                thread_id=thread_id,
                assistant_id=assistant_id
            )
            start_t = time.time()
            while time.time() - start_t < 120:
                run = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)
                if run.status == "completed":
                    break
                if run.status == "failed":
                    err_msg = run.last_error.message if run.last_error else "Unknown error"
                    return {"status": "error", "message": err_msg}
                time.sleep(POLLING_INTERVAL)

            messages = client.beta.threads.messages.list(thread_id=thread_id, order="asc")
            response_data = []
            for m in messages.data:
                if m.role == "assistant":
                    for c in m.content:
                        if c.type == "text":
                            response_data.append({"type": "text", "content": c.text.value})
            return {"status": "success", "response": response_data}

        else:
            def event_generator():
                buffer = []
                try:
                    with client.beta.threads.runs.stream(thread_id=thread_id, assistant_id=assistant_id) as st_stream:
                        for delta in st_stream.text_deltas:
                            buffer.append(delta)
                            if len(buffer) >= 10:
                                yield ''.join(buffer)
                                buffer = []
                    if buffer:
                        yield ''.join(buffer)
                except Exception as e:
                    logger.error(f"Streaming error: {e}")
                    yield "[ERROR] The response was interrupted. Please try again."

            return {"status": "success", "stream": event_generator()}

    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        return {"status": "error", "message": "Chat processing failed"}

def image_analysis(session_id: str, prompt: Optional[str], image_data: bytes, filename: str) -> Dict:
    try:
        session_data = get_session(session_id)
        if not session_data:
            return {"status": "error", "message": "Session not found"}

        assistant_id = session_data["assistant_id"]
        if not assistant_id:
            return {"status": "error", "message": "Session not initialized"}

        thread_id = session_data["thread_id"]
        if not thread_id:
            new_thread = client.beta.threads.create()
            thread_id = new_thread.id
            update_session(session_id, thread_id=thread_id)

        ext = os.path.splitext(filename)[1].lower()
        b64_img = base64.b64encode(image_data).decode("utf-8")
        mime = f"image/{ext[1:]}" if ext else "image/jpeg"
        data_url = f"data:{mime};base64,{b64_img}"

        default_prompt = (
            "Analyze this image and provide a thorough summary including all elements. "
            "If there's any text visible, include all the textual content. Describe:"
        )
        combined_prompt = f"{default_prompt} {prompt}" if prompt else default_prompt

        msg_content = [
            {"type": "text", "text": combined_prompt},
            {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}}
        ]
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": msg_content}],
            max_tokens=500
        )
        analysis_text = resp.choices[0].message.content

        client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=analysis_text
        )

        return {
            "status": "success",
            "response": [{"type": "text", "content": analysis_text}]
        }

    except Exception as e:
        logger.error(f"Image analysis error: {e}")
        return {"status": "error", "message": str(e)}

def voice_analysis(session_id: str, audio_data: bytes, filename: str, stream_response: bool = True, use_mini_model: bool = False) -> Dict:
    try:
        session_data = get_session(session_id)
        if not session_data:
            return {"status": "error", "message": "Session not found"}

        assistant_id = session_data["assistant_id"]
        if not assistant_id:
            return {"status": "error", "message": "Session not initialized"}

        thread_id = session_data["thread_id"]
        if not thread_id:
            new_thread = client.beta.threads.create()
            thread_id = new_thread.id
            update_session(session_id, thread_id=thread_id)

        tmp_path = temp_file_path(filename)
        try:
            with open(tmp_path, "wb") as f:
                f.write(audio_data)

            with open(tmp_path, "rb") as audio_file:
                # Choose model based on speed requirements
                model_to_use = STT_MINI_MODEL if use_mini_model else STT_MODEL

                try:
                    # Try streaming transcription for faster response
                    transcription = client.audio.transcriptions.create(
                        model=model_to_use,
                        file=audio_file,
                        language="en",
                        temperature=0.15,  # Even lower for consistency
                        response_format="json",  # Required for new models
                        stream=True if model_to_use != "whisper-1" else False  # Stream if supported
                    )

                    # Handle streaming response if available
                    if hasattr(transcription, '__iter__') and model_to_use != "whisper-1":
                        transcript_text = ""
                        for event in transcription:
                            if hasattr(event, 'delta'):
                                transcript_text += event.delta
                            elif hasattr(event, 'text'):
                                transcript_text = event.text
                                break
                    else:
                        transcript_text = transcription.text

                except Exception as e:
                    logger.warning(f"Failed with {model_to_use}, falling back to whisper-1: {e}")
                    audio_file.seek(0)
                    transcription = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        language="en",
                        temperature=0.2
                    )
                    transcript_text = transcription.text

            attachments = []
            file_ids = session_data["file_ids"]
            for fid in file_ids:
                attachments.append({"file_id": fid, "tools": [{"type": "file_search"}]})

            client.beta.threads.messages.create(
                thread_id=thread_id,
                role="user",
                content=[{"type": "text", "text": transcript_text}],
                attachments=attachments
            )

            if stream_response:
                # Use streaming for faster response
                def voice_stream_generator():
                    try:
                        with client.beta.threads.runs.stream(
                            thread_id=thread_id,
                            assistant_id=assistant_id
                        ) as stream:
                            for delta in stream.text_deltas:
                                yield delta
                    except Exception as e:
                        logger.error(f"Voice streaming error: {e}")
                        yield f"[ERROR] {str(e)}"

                return {
                    "status": "success",
                    "transcript": transcript_text,
                    "stream": voice_stream_generator()
                }
            else:
                # Non-streaming with optimized polling
                run = client.beta.threads.runs.create(
                    thread_id=thread_id,
                    assistant_id=assistant_id
                )
                start_t = time.time()
                while time.time() - start_t < 120:
                    run = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)
                    if run.status == "completed":
                        break
                    if run.status == "failed":
                        err_msg = run.last_error.message if run.last_error else "Unknown error"
                        return {"status": "error", "message": f"Assistant run failed: {err_msg}"}
                    time.sleep(POLLING_INTERVAL)  # Use reduced interval

                messages = client.beta.threads.messages.list(thread_id=thread_id, order="asc")
                assistant_responses = []
                for m in messages.data:
                    if m.role == "assistant":
                        for c in m.content:
                            if c.type == "text":
                                assistant_responses.append(c.text.value)

                final_response = "\n".join(assistant_responses).strip()

                return {
                    "status": "success",
                    "transcript": transcript_text,
                    "assistant_response": final_response
                }

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    except Exception as e:
        logger.error(f"Voice analysis error: {e}")
        return {"status": "error", "message": str(e)}
