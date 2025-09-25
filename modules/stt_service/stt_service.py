"""
Speech-to-Text (STT) Service
Handles audio transcription using OpenAI Whisper
"""
import streamlit as st
from openai import OpenAI
from typing import Optional, Dict, Any, BinaryIO
import logging
import os
from pathlib import Path
import tempfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class STTService:
    def __init__(self, api_key: str = None):
        """Initialize STT service with configurable settings"""
        self.api_key = api_key or st.secrets.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided")
        self.client = OpenAI(api_key=self.api_key)

        # Model configurations - using latest models
        self.models = {
            "fast": "gpt-4o-mini-transcribe",  # Fastest, good for short clips
            "accurate": "gpt-4o-transcribe",     # Most accurate
            "legacy": "whisper-1"                # Fallback option
        }
        self.default_model = "accurate"
        self.supported_formats = ['mp3', 'mp4', 'mpeg', 'mpga', 'm4a', 'wav', 'webm', 'flac', 'ogg']

    def transcribe_audio(self,
                        audio_data: bytes = None,
                        audio_file_path: str = None,
                        filename: str = "audio.mp3",
                        language: str = None,
                        prompt: str = None,
                        response_format: str = "text",
                        temperature: float = 0.15,
                        model_type: str = "accurate",
                        stream: bool = False) -> Dict[str, Any]:
        """
        Transcribe audio to text using Whisper

        Args:
            audio_data: Audio data in bytes
            audio_file_path: Path to audio file
            filename: Name of the file (used for format detection)
            language: Language code (e.g., 'en' for English)
            prompt: Optional text to guide the model's style
            response_format: Output format (text, json, srt, verbose_json, vtt)
            temperature: Sampling temperature (0.15 default for consistency)
            model_type: Model to use ('fast', 'accurate', or 'legacy')
            stream: Enable streaming for supported models

        Returns:
            Dictionary with transcription results
        """
        try:
            # Prepare audio file
            if audio_file_path:
                with open(audio_file_path, "rb") as f:
                    audio_file = f
                    result = self._call_whisper_api(
                        audio_file,
                        language,
                        prompt,
                        response_format,
                        temperature
                    )
            elif audio_data:
                # Save to temporary file
                with tempfile.NamedTemporaryFile(suffix=f".{filename.split('.')[-1]}", delete=False) as tmp:
                    tmp.write(audio_data)
                    tmp_path = tmp.name

                try:
                    with open(tmp_path, "rb") as f:
                        result = self._call_whisper_api(
                            f,
                            language,
                            prompt,
                            response_format,
                            temperature,
                            model_type,
                            stream
                        )
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
            else:
                raise ValueError("Either audio_data or audio_file_path must be provided")

            return {
                "status": "success",
                "transcription": result.text if hasattr(result, 'text') else result,
                "format": response_format,
                "language": language or "auto-detected"
            }

        except Exception as e:
            logger.error(f"Transcription error: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "transcription": None
            }

    def _call_whisper_api(self, audio_file, language, prompt, response_format, temperature, model_type="accurate", stream=False):
        """Internal method to call Whisper API with latest models"""
        # Select model based on type
        model = self.models.get(model_type, self.models[self.default_model])

        params = {
            "model": model,
            "file": audio_file,
            "response_format": "json" if model.startswith("gpt-4o") else response_format,
            "temperature": temperature
        }

        if language:
            params["language"] = language
        if prompt:
            params["prompt"] = prompt

        # Add streaming support for newer models
        if stream and model.startswith("gpt-4o"):
            params["stream"] = True

        try:
            result = self.client.audio.transcriptions.create(**params)

            # Handle streaming response if available
            if stream and hasattr(result, '__iter__') and model.startswith("gpt-4o"):
                transcript = ""
                for event in result:
                    if hasattr(event, 'delta'):
                        transcript += event.delta
                    elif hasattr(event, 'text'):
                        transcript = event.text
                        break
                # Create a simple object with text attribute
                class TranscriptResult:
                    def __init__(self, text):
                        self.text = text
                return TranscriptResult(transcript)
            return result

        except Exception as e:
            # Fallback to legacy model on error
            if model != "whisper-1":
                logger.warning(f"Failed with {model}, falling back to whisper-1: {e}")
                audio_file.seek(0) if hasattr(audio_file, 'seek') else None
                params["model"] = "whisper-1"
                params["response_format"] = response_format
                params.pop("stream", None)
                return self.client.audio.transcriptions.create(**params)
            raise

    def translate_audio(self,
                       audio_data: bytes = None,
                       audio_file_path: str = None,
                       filename: str = "audio.mp3",
                       prompt: str = None,
                       response_format: str = "text",
                       temperature: float = 0) -> Dict[str, Any]:
        """
        Translate audio to English text

        Args:
            audio_data: Audio data in bytes
            audio_file_path: Path to audio file
            filename: Name of the file
            prompt: Optional text to guide the model's style
            response_format: Output format
            temperature: Sampling temperature (0.15 default for consistency)

        Returns:
            Dictionary with translation results
        """
        try:
            # Prepare audio file
            if audio_file_path:
                with open(audio_file_path, "rb") as f:
                    audio_file = f
                    result = self._call_translation_api(
                        audio_file,
                        prompt,
                        response_format,
                        temperature
                    )
            elif audio_data:
                # Save to temporary file
                with tempfile.NamedTemporaryFile(suffix=f".{filename.split('.')[-1]}", delete=False) as tmp:
                    tmp.write(audio_data)
                    tmp_path = tmp.name

                try:
                    with open(tmp_path, "rb") as f:
                        result = self._call_translation_api(
                            f,
                            prompt,
                            response_format,
                            temperature
                        )
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
            else:
                raise ValueError("Either audio_data or audio_file_path must be provided")

            return {
                "status": "success",
                "translation": result.text if hasattr(result, 'text') else result,
                "format": response_format
            }

        except Exception as e:
            logger.error(f"Translation error: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "translation": None
            }

    def _call_translation_api(self, audio_file, prompt, response_format, temperature):
        """Internal method to call translation API"""
        params = {
            "model": self.default_model,
            "file": audio_file,
            "response_format": response_format,
            "temperature": temperature
        }

        if prompt:
            params["prompt"] = prompt

        return self.client.audio.translations.create(**params)

    def is_format_supported(self, filename: str) -> bool:
        """Check if file format is supported"""
        extension = filename.split('.')[-1].lower()
        return extension in self.supported_formats

    def get_supported_formats(self) -> list:
        """Get list of supported audio formats"""
        return self.supported_formats

    def get_language_codes(self) -> Dict[str, str]:
        """Get common language codes for Whisper"""
        return {
            "English": "en",
            "Spanish": "es",
            "French": "fr",
            "German": "de",
            "Italian": "it",
            "Portuguese": "pt",
            "Russian": "ru",
            "Chinese": "zh",
            "Japanese": "ja",
            "Korean": "ko",
            "Arabic": "ar",
            "Hindi": "hi",
            "Auto-detect": None
        }

    def get_model_options(self) -> Dict[str, str]:
        """Get available model options with descriptions"""
        return {
            "fast": "gpt-4o-mini-transcribe - Fastest, best for short clips",
            "accurate": "gpt-4o-transcribe - Most accurate, supports streaming",
            "legacy": "whisper-1 - Stable fallback option"
        }

def get_stt_service(api_key: str = None) -> STTService:
    """Factory function to get STT service instance"""
    return STTService(api_key)