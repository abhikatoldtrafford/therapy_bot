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
COMMON_VECTOR_STORE_ID = "vs_67a7a6bd68d48191a4f446ddeaec2e2b"
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
MODEL_NAME = "gpt-4o"
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
    current_challenges: str
) -> str:
    sessions = load_sessions()
    session_id = uuid.uuid4().hex
    sessions[session_id] = {
        "assistant_id": None,
        "thread_id": None,
        "file_ids": [],
        "session_vs_id": None,
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
def create_system_prompt(user_info: Dict) -> str:
    return f"""
    You are a compassionate and intuitive NARM therapy assistant, trained in the NeuroAffective Relational Model (NARM). Your goal is to provide thoughtful, human-like support to users while drawing from the **attached vector store files**, which contain NARM guidelines, methodologies, and real-world case studies.

    **How You Should Interact:**
    - Speak like a real therapist, not a rulebook. Be natural, warm, and understanding.
    - Never just recite guidelines. Instead, integrate NARM principles naturally into the conversation.
    - If a topic isn’t explicitly covered in the knowledge base, use common sense and logical reasoning.
    - Ask relevant, open-ended questions to help users explore their experiences.
    - Offer reflections, not rigid instructions. Help users arrive at insights organically.

    **Session Details:**
    - Name: {user_info.get('name', 'Unknown')}
    - Email: {user_info.get('email', 'Unknown')}
    - Focus Today: {user_info.get('focus_today', 'Not specified')}
    - Desired Outcome: {user_info.get('desired_outcome', 'Not specified')}
    - Current Challenges: {user_info.get('current_challenges', 'Not specified')}

    **Core NARM Principles (Your Foundation for Support):**
    - Recognize how present struggles connect to early developmental experiences.
    - Support users in self-regulation and emotional connection.
    - Stay present and focused on the here and now while acknowledging past influences.
    - Attune to the user's emotional state and nervous system regulation.
    - Help integrate cognitive, emotional, and somatic experiences.

    **Therapeutic Approach:**
    - Offer support in a way that empowers users rather than diagnosing them.
    - Guide users through self-exploration, survival patterns, and core needs.
    - Encourage curiosity, self-compassion, and personal agency.
    - Use strengths-based approaches to help users feel more grounded and resilient.

    **Engagement Style:**
    - When a user shares a challenge, respond with warmth and curiosity.
    - Ask meaningful follow-up questions instead of jumping to solutions.
    - If something isn’t covered in the knowledge base, respond thoughtfully based on logic and empathy.
    - Make sure users feel heard before offering guidance.

    **Available Features (Suggest These When Useful):**
    - **Voice Messages** – If a user struggles with written communication.
    - **Image Sharing** – If visuals might help in expressing feelings or experiences.
    - **Live Calls** – When deeper support is needed.

    **Safety Protocols:**
    - If a user is in distress, prioritize their well-being over advice.
    - Stay within ethical and AI boundaries – do not act as a replacement for a licensed therapist.
    - Direct users to professional human support when necessary.

    **Final Goal:** 
    Help users process their emotions, build self-awareness, and feel supported in their healing journey using NARM principles. Approach each conversation with care, curiosity, and a desire to help.
    """

def temp_file_path(filename: str) -> str:
    return f"/tmp/{uuid.uuid4().hex}_{filename}"

#########################################################
# CORE FUNCTIONS
#########################################################
def initiate_chat(
    name: str,
    email: str,
    focus_today: str = "",
    desired_outcome: str = "",
    current_challenges: str = ""
) -> Dict:
    try:
        # Create a new session record
        session_id = create_session(name, email, focus_today, desired_outcome, current_challenges)
        session_data = get_session(session_id)

        # Create vector store
        session_vs = client.beta.vector_stores.create(
            name=f"SessionVS-{email[:8]}"
        )

        # Create assistant
        assistant = client.beta.assistants.create(
            name="NARM Therapy Assistant",
            instructions=create_system_prompt(session_data["user_info"]),
            model=MODEL_NAME,
            tools=[
                {"type": "file_search"},
                {"type": "code_interpreter"},
            ],
            tool_resources={
                "file_search": {
                    "vector_store_ids": [COMMON_VECTOR_STORE_ID]
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
                time.sleep(2)

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

def voice_analysis(session_id: str, audio_data: bytes, filename: str) -> Dict:
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
                transcription = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
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
                time.sleep(2)

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
