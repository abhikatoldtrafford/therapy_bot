"""
Chat Service Module
Handles chat functionality with streaming and history management
"""
import streamlit as st
from openai import OpenAI
import time
import logging
from typing import Dict, List, Optional, Generator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatService:
    def __init__(self, api_key: str = None):
        """Initialize chat service"""
        self.api_key = api_key or st.secrets.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided")
        self.client = OpenAI(api_key=self.api_key)

    def send_message(self, thread_id: str, assistant_id: str, message: str,
                    attachments: List[Dict] = None) -> Dict:
        """Send a message and get response"""
        try:
            # Add message to thread
            message_params = {
                "thread_id": thread_id,
                "role": "user",
                "content": [{"type": "text", "text": message}]
            }
            if attachments:
                message_params["attachments"] = attachments

            self.client.beta.threads.messages.create(**message_params)

            # Run assistant
            run = self.client.beta.threads.runs.create(
                thread_id=thread_id,
                assistant_id=assistant_id
            )

            # Wait for completion
            start_time = time.time()
            while time.time() - start_time < 120:
                run = self.client.beta.threads.runs.retrieve(
                    thread_id=thread_id,
                    run_id=run.id
                )
                if run.status == "completed":
                    break
                elif run.status == "failed":
                    return {
                        "status": "error",
                        "message": run.last_error.message if run.last_error else "Run failed"
                    }
                time.sleep(2)

            # Get response
            messages = self.client.beta.threads.messages.list(
                thread_id=thread_id,
                order="asc"
            )

            response_text = ""
            for msg in messages.data:
                if msg.role == "assistant":
                    for content in msg.content:
                        if content.type == "text":
                            response_text = content.text.value
                            break
                    if response_text:
                        break

            return {
                "status": "success",
                "response": response_text
            }

        except Exception as e:
            logger.error(f"Chat error: {str(e)}")
            return {"status": "error", "message": str(e)}

    def stream_message(self, thread_id: str, assistant_id: str, message: str,
                      attachments: List[Dict] = None) -> Generator:
        """Stream a message response"""
        try:
            # Add message to thread
            message_params = {
                "thread_id": thread_id,
                "role": "user",
                "content": [{"type": "text", "text": message}]
            }
            if attachments:
                message_params["attachments"] = attachments

            self.client.beta.threads.messages.create(**message_params)

            # Stream response
            buffer = []
            with self.client.beta.threads.runs.stream(
                thread_id=thread_id,
                assistant_id=assistant_id
            ) as stream:
                for delta in stream.text_deltas:
                    buffer.append(delta)
                    if len(buffer) >= 10:
                        yield ''.join(buffer)
                        buffer = []
                if buffer:
                    yield ''.join(buffer)

        except Exception as e:
            logger.error(f"Streaming error: {str(e)}")
            yield f"[ERROR] {str(e)}"

def get_chat_service(api_key: str = None) -> ChatService:
    """Factory function"""
    return ChatService(api_key)