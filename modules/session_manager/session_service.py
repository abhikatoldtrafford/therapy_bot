"""
Session Management Service
Handles session creation, storage, and retrieval
"""
import json
import uuid
from pathlib import Path
from typing import Dict, Optional, Any
import streamlit as st

class SessionManager:
    def __init__(self, sessions_file: str = "sessions.json"):
        """Initialize session manager with configurable storage file"""
        self.sessions_file = Path(sessions_file)
        self._ensure_file_exists()

    def _ensure_file_exists(self):
        """Create sessions file if it doesn't exist"""
        if not self.sessions_file.exists():
            self.sessions_file.write_text("{}")

    def load_sessions(self) -> Dict[str, Dict]:
        """Load all sessions from storage"""
        try:
            with open(self.sessions_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {}

    def save_sessions(self, sessions: Dict[str, Dict]) -> None:
        """Save sessions to storage"""
        with open(self.sessions_file, "w") as f:
            json.dump(sessions, f, indent=2)

    def create_session(self, **user_data) -> str:
        """
        Create a new session with user data
        Returns session_id
        """
        sessions = self.load_sessions()
        session_id = uuid.uuid4().hex

        sessions[session_id] = {
            "assistant_id": None,
            "thread_id": None,
            "file_ids": [],
            "session_vs_id": None,
            "user_info": user_data,
            "created_at": str(uuid.uuid4()),
            "config": {}
        }

        self.save_sessions(sessions)
        return session_id

    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get a specific session by ID"""
        sessions = self.load_sessions()
        return sessions.get(session_id)

    def update_session(self, session_id: str, **kwargs) -> bool:
        """Update session data"""
        sessions = self.load_sessions()
        if session_id not in sessions:
            return False

        for key, value in kwargs.items():
            if key == "user_info" and isinstance(value, dict):
                sessions[session_id]["user_info"].update(value)
            elif key == "config" and isinstance(value, dict):
                sessions[session_id]["config"].update(value)
            else:
                sessions[session_id][key] = value

        self.save_sessions(sessions)
        return True

    def delete_session(self, session_id: str) -> bool:
        """Delete a session"""
        sessions = self.load_sessions()
        if session_id in sessions:
            del sessions[session_id]
            self.save_sessions(sessions)
            return True
        return False

    def list_sessions(self) -> Dict[str, Dict]:
        """List all sessions"""
        return self.load_sessions()

    def clear_all_sessions(self) -> None:
        """Clear all sessions"""
        self.save_sessions({})

# Default instance with default config
default_manager = SessionManager()

def get_session_manager(sessions_file: str = None) -> SessionManager:
    """Factory function to get session manager instance"""
    if sessions_file:
        return SessionManager(sessions_file)
    return default_manager