"""
OpenAI Assistant Service
Handles creation and management of OpenAI assistants
"""
import streamlit as st
from openai import OpenAI
from typing import Dict, Optional, List, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AssistantService:
    def __init__(self, api_key: str = None, model: str = "gpt-4o"):
        """Initialize assistant service with configurable settings"""
        self.api_key = api_key or st.secrets.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided")
        self.client = OpenAI(api_key=self.api_key)

        # Model configurations with fallback
        # Note: gpt-5-mini is not yet available for Assistants API
        self.models = {
            "latest": "gpt-4.1-mini",           # Latest available for assistants
            "standard": "gpt-4.1",         # Standard model
            "legacy": "gpt-4o"       # Legacy model
        }
        self.default_model = model

    def create_assistant(self,
                        name: str = "Assistant",
                        instructions: str = "You are a helpful assistant.",
                        model: str = None,
                        tools: List[Dict] = None,
                        tool_resources: Dict = None,
                        temperature: float = 0.7,
                        metadata: Dict = None) -> str:
        """
        Create a new OpenAI assistant with configurable settings
        Returns assistant_id
        """
        try:
            if tools is None:
                tools = [{"type": "file_search"}, {"type": "code_interpreter"}]

            assistant_params = {
                "name": name,
                "instructions": instructions,
                "model": model or self.default_model,
                "tools": tools,
                "temperature": temperature
            }

            if tool_resources:
                assistant_params["tool_resources"] = tool_resources

            if metadata:
                assistant_params["metadata"] = metadata

            assistant = self.client.beta.assistants.create(**assistant_params)
            logger.info(f"Assistant created: {assistant.id}")
            return assistant.id

        except Exception as e:
            logger.error(f"Error creating assistant: {str(e)}")
            raise

    def update_assistant(self, assistant_id: str, **kwargs) -> bool:
        """Update an existing assistant"""
        try:
            self.client.beta.assistants.update(assistant_id=assistant_id, **kwargs)
            logger.info(f"Assistant updated: {assistant_id}")
            return True
        except Exception as e:
            logger.error(f"Error updating assistant: {str(e)}")
            return False

    def get_assistant(self, assistant_id: str) -> Optional[Any]:
        """Get assistant details"""
        try:
            return self.client.beta.assistants.retrieve(assistant_id=assistant_id)
        except Exception as e:
            logger.error(f"Error retrieving assistant: {str(e)}")
            return None

    def delete_assistant(self, assistant_id: str) -> bool:
        """Delete an assistant"""
        try:
            self.client.beta.assistants.delete(assistant_id=assistant_id)
            logger.info(f"Assistant deleted: {assistant_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting assistant: {str(e)}")
            return False

    def list_assistants(self, limit: int = 20) -> List:
        """List all assistants"""
        try:
            assistants = self.client.beta.assistants.list(limit=limit)
            return assistants.data
        except Exception as e:
            logger.error(f"Error listing assistants: {str(e)}")
            return []

    def create_thread(self, messages: List[Dict] = None) -> str:
        """Create a new conversation thread"""
        try:
            thread_params = {}
            if messages:
                thread_params["messages"] = messages

            thread = self.client.beta.threads.create(**thread_params)
            logger.info(f"Thread created: {thread.id}")
            return thread.id
        except Exception as e:
            logger.error(f"Error creating thread: {str(e)}")
            raise

    def add_message_to_thread(self,
                            thread_id: str,
                            content: str,
                            role: str = "user",
                            attachments: List[Dict] = None) -> Any:
        """Add a message to a thread"""
        try:
            message_params = {
                "thread_id": thread_id,
                "role": role,
                "content": [{"type": "text", "text": content}]
            }

            if attachments:
                message_params["attachments"] = attachments

            message = self.client.beta.threads.messages.create(**message_params)
            logger.info(f"Message added to thread {thread_id}")
            return message
        except Exception as e:
            logger.error(f"Error adding message: {str(e)}")
            raise

    def run_assistant(self,
                     thread_id: str,
                     assistant_id: str,
                     instructions: str = None,
                     tools: List[Dict] = None) -> Any:
        """Run assistant on a thread"""
        try:
            run_params = {
                "thread_id": thread_id,
                "assistant_id": assistant_id
            }

            if instructions:
                run_params["instructions"] = instructions

            if tools:
                run_params["tools"] = tools

            run = self.client.beta.threads.runs.create(**run_params)
            logger.info(f"Run created: {run.id}")
            return run
        except Exception as e:
            logger.error(f"Error running assistant: {str(e)}")
            raise

    def stream_assistant(self,
                        thread_id: str,
                        assistant_id: str,
                        instructions: str = None):
        """Stream assistant response"""
        try:
            stream_params = {
                "thread_id": thread_id,
                "assistant_id": assistant_id
            }

            if instructions:
                stream_params["instructions"] = instructions

            with self.client.beta.threads.runs.stream(**stream_params) as stream:
                for delta in stream.text_deltas:
                    yield delta

        except Exception as e:
            logger.error(f"Error streaming assistant: {str(e)}")
            yield f"[ERROR] {str(e)}"

    def get_run_status(self, thread_id: str, run_id: str) -> str:
        """Get the status of a run"""
        try:
            run = self.client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)
            return run.status
        except Exception as e:
            logger.error(f"Error getting run status: {str(e)}")
            return "error"

    def get_thread_messages(self, thread_id: str, order: str = "asc") -> List:
        """Get all messages from a thread"""
        try:
            messages = self.client.beta.threads.messages.list(thread_id=thread_id, order=order)
            return messages.data
        except Exception as e:
            logger.error(f"Error getting thread messages: {str(e)}")
            return []

# Default configurations
DEFAULT_SYSTEM_PROMPT = """
You are a helpful assistant. Please provide thoughtful and accurate responses.
"""

DEFAULT_NARM_PROMPT = """
You are a compassionate and intuitive NARM therapy assistant, trained in the NeuroAffective Relational Model (NARM). Your goal is to provide thoughtful, human-like support to users while drawing from the **attached vector store files**, which contain NARM guidelines, methodologies, and real-world case studies.

**How You Should Interact:**
- Speak like a real therapist, not a rulebook. Be natural, warm, and understanding.
- Never just recite guidelines. Instead, integrate NARM principles naturally into the conversation.
- If a topic isn't explicitly covered in the knowledge base, use common sense and logical reasoning.
- Ask relevant, open-ended questions to help users explore their experiences.
- Offer reflections, not rigid instructions. Help users arrive at insights organically.
"""

def get_assistant_service(api_key: str = None, model: str = "gpt-4o") -> AssistantService:
    """Factory function to get assistant service instance"""
    return AssistantService(api_key, model)