import os
import uuid
import json
import base64
import time
import asyncio
import logging
from openai import OpenAI
from typing import Optional, List, Dict
from pydantic import BaseModel, EmailStr
import websockets
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect, Request, Query
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from starlette.websockets import WebSocketState
import streamlit as st
#########################################################
# LOGGING & FASTAPI SETUP
#########################################################
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI(title="NARM Therapy Assistant")

#########################################################
# CONFIG
#########################################################
COMMON_VECTOR_STORE_ID = "vs_67a7a6bd68d48191a4f446ddeaec2e2b"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "gpt-4o"
PORT = int(os.getenv("PORT", 8080))

client = OpenAI(api_key=OPENAI_API_KEY)

# Global context to store session data
global_context: Dict[str, Optional[object]] = {
    "assistant_id": None,
    "thread_id": None,
    "file_ids": [],
    "session_vs_id": None,
    "user_info": None
}
global_file_tools: Dict[str, str] = {}

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
# ENDPOINT: /initiate-chat
#########################################################
@app.post("/initiate-chat")
async def initiate_chat(
    name: str = Form(...),
    email: EmailStr = Form(...),
    focus_today: str = Form(""),
    desired_outcome: str = Form(""),
    current_challenges: str = Form("")
):
    try:
        # Store user info
        global_context["user_info"] = {
            "name": name,
            "email": email,
            "focus_today": focus_today,
            "desired_outcome": desired_outcome,
            "current_challenges": current_challenges
        }

        # Create session-specific vector store
        session_vs = client.beta.vector_stores.create(
            name=f"SessionVS-{email[:8]}"
        )

        # Create assistant with combined vector stores
        assistant = client.beta.assistants.create(
            name="NARM Therapy Assistant",
            instructions=create_system_prompt(global_context["user_info"]),
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

        # Update global context
        global_context["assistant_id"] = assistant.id
        global_context["session_vs_id"] = session_vs.id
        global_context["file_ids"] = []
        global_file_tools.clear()

        return JSONResponse({
            "message": "Session initialized",
            "assistant_id": assistant.id,
            "vector_store_id": session_vs.id
        })

    except Exception as e:
        logger.error(f"Session init error: {str(e)}")
        raise HTTPException(500, "Session initialization failed")



#########################################################
# ENDPOINT: /chat (Synchronous/Streaming)
#########################################################
@app.post("/chat")
async def chat(prompt: str = Form(...), stream: bool = Query(False)):
    if not global_context["assistant_id"]:
        raise HTTPException(400, "Session not initialized")

    # Create thread if it doesn't exist
    if not global_context["thread_id"]:
        global_context["thread_id"] = client.beta.threads.create().id

    # Prepare attachments
    attachments = []
    for fid in global_context["file_ids"]:
        tool_type = global_file_tools.get(fid)
        if tool_type in ["file_search", "code_interpreter"]:
            attachments.append({"file_id": fid, "tools": [{"type": tool_type}]})

    # Add user prompt to thread
    client.beta.threads.messages.create(
        thread_id=global_context["thread_id"],
        role="user",
        content=[{"type": "text", "text": prompt}],
        attachments=attachments
    )

    # Sync or streaming response
    if not stream:
        run = client.beta.threads.runs.create(
            thread_id=global_context["thread_id"],
            assistant_id=global_context["assistant_id"]
        )
        start_t = time.time()
        while time.time() - start_t < 120:
            run = client.beta.threads.runs.retrieve(
                thread_id=global_context["thread_id"],
                run_id=run.id
            )
            if run.status == "completed":
                break
            if run.status == "failed":
                err_msg = run.last_error.message if run.last_error else "Unknown error"
                raise HTTPException(500, detail=err_msg)
            time.sleep(2)

        # Gather assistant messages
        messages = client.beta.threads.messages.list(
            thread_id=global_context["thread_id"],
            order="asc"
        )
        response_data = []
        for m in messages.data:
            if m.role == "assistant":
                for c in m.content:
                    if c.type == "text":
                        response_data.append({"type": "text", "content": c.text.value})
        return JSONResponse({"response": response_data})

    else:
        def event_generator():
            buffer = []
            try:
                with client.beta.threads.runs.stream(
                    thread_id=global_context["thread_id"],
                    assistant_id=global_context["assistant_id"]
                ) as st:
                    for delta in st.text_deltas:
                        buffer.append(delta)
                        if len(buffer) >= 10:
                            yield ''.join(buffer)
                            buffer = []
                if buffer:
                    yield ''.join(buffer)
            except Exception as e:
                logger.error(f"Streaming error: {e}")
                yield "[ERROR] The response was interrupted. Please try again."

        return StreamingResponse(event_generator(), media_type="text/event-stream")
#########################################################
# ENDPOINT: /image-analysis
#########################################################
@app.post("/image-analysis")
async def image_analysis(prompt: Optional[str] = Form(None), image: UploadFile = File(...)):
    default_prompt = ("Analyze this image and provide a thorough summary including all elements in it. "
                      "If there's any text visible in the image, include all the textual content in your response. Describe:")
    combined_prompt = f"{default_prompt} {prompt}" if prompt else default_prompt
    if not global_context["assistant_id"]:
        raise HTTPException(400, "Session not initialized")
    try:
        # Create thread if it doesn't exist
        if not global_context["thread_id"]:
            thread = client.beta.threads.create()
            global_context["thread_id"] = thread.id

        # Process image
        ext = os.path.splitext(image.filename)[1].lower()
        raw = await image.read()
        b64_img = base64.b64encode(raw).decode("utf-8")
        mime = f"image/{ext[1:]}" if ext else "image/jpeg"
        data_url = f"data:{mime};base64,{b64_img}"

        # Combine default and user prompts
        default_prompt = "Analyze this image and provide a thorough summary including all elements in it. If there's any text visible in the image, include all the textual content in your response. Describe: "
        combined_prompt = f"{default_prompt} {prompt}" if prompt else default_prompt

        # Get image analysis from GPT
        msg_content = [
            {"type": "text", "text": combined_prompt},
            {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}}
        ]
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": msg_content}],
            max_tokens=300
        )
        analysis_text = resp.choices[0].message.content

        # Add analysis to thread if it exists
        if global_context["thread_id"]:
            client.beta.threads.messages.create(
                thread_id=global_context["thread_id"],
                role="user",
                content=analysis_text
            )
            

        return JSONResponse({
            "response": [
                {"type": "text", "content": analysis_text}
            ]
        })

    except Exception as e:
        logger.error(f"Image analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# #########################################################
# # ENDPOINT: /voice-analysis
# #########################################################
# @app.post("/voice-analysis")
# async def voice_analysis(audio: UploadFile = File(...)):
#     """Transcribe voice message and add to chat thread"""
#     if not global_context["assistant_id"]:
#         raise HTTPException(400, "Session not initialized")

#     try:
#         # Create thread if it doesn't exist
#         if not global_context["thread_id"]:
#             thread = client.beta.threads.create()
#             global_context["thread_id"] = thread.id

#         # Save audio file temporarily
#         tmp_path = temp_file_path(audio.filename)
#         try:
#             with open(tmp_path, "wb") as f:
#                 f.write(await audio.read())

#             # Transcribe audio using Whisper
#             with open(tmp_path, "rb") as audio_file:
#                 transcription = client.audio.transcriptions.create(
#                     model="whisper-1",
#                     file=audio_file
#                 )

#             transcript_text = transcription.text

#             # Add transcription to thread if it exists
#             if global_context["thread_id"]:
#                 client.beta.threads.messages.create(
#                     thread_id=global_context["thread_id"],
#                     role="user",
#                     content=f"Voice Message: {transcript_text}"
#                 )

#             return JSONResponse({
#                 "transcript": transcript_text,
#                 "response": [
#                     {"type": "text", "content": transcript_text}
#                 ]
#             })

#         finally:
#             if os.path.exists(tmp_path):
#                 os.remove(tmp_path)

#     except Exception as e:
#         logger.error(f"Voice analysis error: {e}")
#         raise HTTPException(status_code=500, detail=str(e))
#########################################################
# ENDPOINT: /voice-analysis
#########################################################
@app.post("/voice-analysis")
async def voice_analysis(audio: UploadFile = File(...)):
    """
    1) Transcribe the uploaded audio with Whisper
    2) Add that transcription as a user prompt into the same chat thread
    3) Invoke the assistant run
    4) Return the assistant's response along with the transcript
    """
    if not global_context["assistant_id"]:
        raise HTTPException(400, "Session not initialized")

    try:
        # Create thread if it doesn't exist
        if not global_context["thread_id"]:
            thread = client.beta.threads.create()
            global_context["thread_id"] = thread.id

        # Save audio file temporarily
        tmp_path = temp_file_path(audio.filename)
        try:
            with open(tmp_path, "wb") as f:
                f.write(await audio.read())

            # Step 1: Transcribe audio using Whisper
            with open(tmp_path, "rb") as audio_file:
                transcription = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
            transcript_text = transcription.text

            # Step 2: Add user's message to the chat thread
            # We'll store the transcript text as if the user typed it.
            # client.beta.threads.messages.create(
            #     thread_id=global_context["thread_id"],
            #     role="user",
            #     content=f"Voice Message: {transcript_text}"
            # )
            attachments = []
            for fid in global_context["file_ids"]:
                tool_type = global_file_tools.get(fid)
                if tool_type in ["file_search", "code_interpreter"]:
                    attachments.append({"file_id": fid, "tools": [{"type": tool_type}]})
            client.beta.threads.messages.create(
                thread_id=global_context["thread_id"],
                role="user",
                content=[{"type": "text", "text": transcript_text}],
                attachments=attachments
            )
            # Step 3: Invoke the assistant
            # Create a new "run" to get the assistant's response
            run = client.beta.threads.runs.create(
                thread_id=global_context["thread_id"],
                assistant_id=global_context["assistant_id"]
            )

            # Wait for run to complete (up to ~2 minutes)
            start_t = time.time()
            while time.time() - start_t < 120:
                run = client.beta.threads.runs.retrieve(
                    thread_id=global_context["thread_id"],
                    run_id=run.id
                )
                if run.status == "completed":
                    break
                elif run.status == "failed":
                    err_msg = run.last_error.message if run.last_error else "Unknown error"
                    raise HTTPException(500, detail=f"Assistant run failed: {err_msg}")
                time.sleep(2)

            # Step 4: Gather the assistant's final messages
            messages = client.beta.threads.messages.list(
                thread_id=global_context["thread_id"],
                order="asc"
            )
            assistant_responses = []
            for m in messages.data:
                if m.role == "assistant":
                    # Each message can have multiple content chunks
                    assistant_responses.extend([c.text.value for c in m.content if c.type == "text"])

            # Build a single string from all assistant messages (or just the last one)
            final_response = "\n".join(assistant_responses).strip()

            # Return JSON with the transcript and the assistant's final text
            return JSONResponse({
                "transcript": transcript_text,
                "assistant_response": final_response
            })

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    except Exception as e:
        logger.error(f"Voice analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

#########################################################
# WEBSOCKET: /call => Realtime Audio
#########################################################
@app.websocket("/call")
async def call_endpoint(websocket: WebSocket):
    """Handles persistent real-time two-way audio communication with OpenAI."""
    await websocket.accept()
    logger.info("Client connected to /call endpoint")

    realtime_url = "wss://api.openai.com/v1/realtime?model=gpt-4o-mini-realtime-preview"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "OpenAI-Beta": "realtime=v1"}

    # Keep the FastAPI connection open until the client disconnects.
    while websocket.application_state == WebSocketState.CONNECTED:
        try:
            # Establish an OpenAI WebSocket session
            async with websockets.connect(realtime_url, extra_headers=headers) as openai_ws:
                logger.info("Connected to OpenAI Realtime endpoint")
                session_update = {
                    "type": "session.update",
                    "session": {
                        "modalities": ["audio", "text"],
                        "input_audio_format": "pcm16",
                        "output_audio_format": "pcm16",
                        "voice": "verse",
                        "turn_detection": {"type": "server_vad"},
                        "instructions": (
                            "You are a helpful assistant helping depression patients in fixing appointments. "
                            "Greet the user, and gently tell them that appointments with doctors are available "
                            "from 9 AM to 9 PM PST from Monday to Friday. Then ask for a convenient date and fix the appointment. "
                            "Your task is also to listen to problems of users and try to solve them based on narm guidelines."
                            "Try to direct them towards appointment booking or chat service"
                            "Always speak in English and in 1.2x speed."
                        )
                    }
                }
                await openai_ws.send(json.dumps(session_update))

                # Start tasks for bidirectional forwarding
                forward_tasks = [
                    asyncio.create_task(forward_client_to_openai(websocket, openai_ws)),
                    asyncio.create_task(forward_openai_to_client(openai_ws, websocket))
                ]
                # Wait until one task errors out (or the session ends)
                done, pending = await asyncio.wait(forward_tasks, return_when=asyncio.FIRST_EXCEPTION)
                for task in pending:
                    task.cancel()
                logger.info("OpenAI session ended; restarting session...")
                # If the client is still connected, wait a moment and restart the OpenAI session.
                await asyncio.sleep(0.1)
        except Exception as e:
            logger.error(f"OpenAI WS error: {e}")
            # Wait a bit before trying to re‑establish the connection
            await asyncio.sleep(1)

    logger.info("Client disconnected from /call endpoint (outer loop exit).")
    # No need to explicitly close websocket here; FastAPI will clean up.

# --- Remove closing from within the forwarding tasks ---

async def forward_client_to_openai(client_ws: WebSocket, openai_ws: websockets.WebSocketClientProtocol):
    """
    Forwards messages from the client to OpenAI.
    (No longer closes websockets in the finally block.)
    """
    try:
        while True:
            msg_str = await client_ws.receive_text()
            data = json.loads(msg_str)
            event_name = data.get("event")

            if event_name == "start":
                logger.info("Client signaled start. No direct forward to OpenAI.")
                continue
            elif event_name == "text":
                item_create = {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": data["text"]}]
                    }
                }
                await openai_ws.send(json.dumps(item_create))
            elif event_name == "media":
                chunk_b64 = data["media"]["payload"]
                transform = {
                    "type": "input_audio_buffer.append",
                    "audio": chunk_b64
                }
                await openai_ws.send(json.dumps(transform))
                logger.debug("Forwarded PCM16 audio -> input_audio_buffer.append")
                if data.get("last_chunk", False):
                    await asyncio.sleep(0.2)  # Allow OpenAI time to process
                    commit_signal = {"type": "input_audio_buffer.commit"}
                    await openai_ws.send(json.dumps(commit_signal))
                    logger.info("✅ Sent input_audio_buffer.commit to OpenAI.")
                    response_create = {
                        "type": "response.create",
                        "response": {"modalities": ["audio", "text"]}
                    }
                    await openai_ws.send(json.dumps(response_create))
                    logger.info("✅ Sent response.create to OpenAI.")
            else:
                logger.warning(f"Unknown event={event_name}, skipping.")
    except WebSocketDisconnect:
        logger.info("Client disconnected from /call in forward_client_to_openai.")
    except Exception as e:
        logger.error(f"Error reading from client: {e}")

async def forward_openai_to_client(openai_ws, client_ws):
    """
    Forwards messages from OpenAI to the client.
    (No longer closes client_ws in the finally block.)
    """
    final_transcript = None
    try:
        async for msg_str in openai_ws:
            data = json.loads(msg_str)
            openai_type = data.get("type")
            if openai_type == "response.audio.delta":
                audio_b64 = data.get("delta", "")
                new_msg = {"event": "media", "media": {"payload": audio_b64}}
                await client_ws.send_text(json.dumps(new_msg))
            elif openai_type == "response.audio.done":
                new_msg = {"event": "mark", "mark": {"name": "audio_done"}}
                await client_ws.send_text(json.dumps(new_msg))
            elif openai_type == "response.done":
                new_msg = {"event": "mark", "mark": {"name": "response_end"}}
                await client_ws.send_text(json.dumps(new_msg))
            elif openai_type == "response.audio_transcript.delta":
                continue
            elif openai_type == "response.audio_transcript.done":
                final_transcript = data.get("transcript", "")
                logger.info(f"✅ Final transcript received: {final_transcript}")
            else:
                continue
        if final_transcript:
            transcript_msg = {"event": "text", "text": final_transcript}
            await client_ws.send_text(json.dumps(transcript_msg))
    except websockets.ConnectionClosed:
        logger.info("OpenAI Realtime WS closed in forward_openai_to_client.")
    except Exception as e:
        logger.error(f"Error reading from OpenAI: {e}")
#########################################################
# MAIN
#########################################################
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
