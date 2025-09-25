"""
Text-to-Speech (TTS) Service
Handles text to speech conversion using OpenAI
"""
import streamlit as st
from openai import OpenAI
from typing import Optional, Dict, Any
import logging
import os
import base64
import uuid
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TTSService:
    def __init__(self, api_key: str = None):
        """Initialize TTS service with configurable settings"""
        self.api_key = api_key or st.secrets.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided")
        self.client = OpenAI(api_key=self.api_key)

        # Model configurations - using latest models
        self.models = {
            "standard": "tts-1",
            "hd": "tts-1-hd",
            "advanced": "gpt-4o-mini-tts"  # Latest model with instructions support
        }
        self.default_model = "advanced"
        self.default_voice = "alloy"
        self.supported_formats = ['mp3', 'opus', 'aac', 'flac', 'wav', 'pcm']
        self.default_instructions = "Speak clearly with natural pacing and warm tone."

    def text_to_speech(self,
                      text: str,
                      voice: str = None,
                      model: str = None,
                      model_type: str = "advanced",
                      speed: float = 1.05,
                      response_format: str = "mp3",
                      instructions: str = None,
                      stream_format: str = "audio",
                      output_file: str = None) -> Dict[str, Any]:
        """
        Convert text to speech

        Args:
            text: Text to convert to speech
            voice: Voice to use (alloy, ash, ballad, coral, echo, fable, onyx, nova, sage, shimmer, verse)
            model: Specific model name (overrides model_type)
            model_type: Type of model ('standard', 'hd', or 'advanced')
            speed: Speed of speech (0.25 to 4.0, default 1.05)
            response_format: Output format (mp3, opus, aac, flac, wav, pcm)
            instructions: Voice instructions (only for gpt-4o-mini-tts)
            stream_format: Streaming format ('audio' or 'sse')
            output_file: Optional output file path

        Returns:
            Dictionary with audio data and metadata
        """
        try:
            if not text:
                raise ValueError("Text cannot be empty")

            # Validate parameters
            voice = voice or self.default_voice
            if not model:
                model = self.models.get(model_type, self.models["advanced"])
            speed = max(0.25, min(4.0, speed))

            # Build parameters
            params = {
                "model": model,
                "voice": voice,
                "input": text,
                "speed": speed,
                "response_format": response_format
            }

            # Add instructions for advanced model
            if model == "gpt-4o-mini-tts" and instructions:
                params["instructions"] = instructions or self.default_instructions
                # SSE streaming format for advanced model
                if stream_format == "sse" and model == "gpt-4o-mini-tts":
                    params["stream_format"] = "sse"

            try:
                # Generate speech
                response = self.client.audio.speech.create(**params)

                # Get the audio content
                audio_content = response.content
            except Exception as e:
                # Fallback to standard model if advanced fails
                if model == "gpt-4o-mini-tts":
                    logger.warning(f"Advanced model failed, falling back to HD: {e}")
                    params["model"] = "tts-1-hd"
                    params.pop("instructions", None)
                    params.pop("stream_format", None)
                    response = self.client.audio.speech.create(**params)
                    audio_content = response.content
                else:
                    raise

            # Save to file or return bytes
            if output_file:
                with open(output_file, 'wb') as f:
                    f.write(audio_content)
                audio_bytes = audio_content
            else:
                audio_bytes = audio_content

            return {
                "status": "success",
                "audio_data": audio_bytes,
                "format": response_format,
                "voice": voice,
                "model": model,
                "speed": speed,
                "text_length": len(text),
                "output_file": output_file
            }

        except Exception as e:
            logger.error(f"TTS error: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "audio_data": None
            }

    def text_to_speech_streaming(self, text: str, voice: str = None, model: str = None, model_type: str = "advanced", speed: float = 1.05, instructions: str = None):
        """
        Stream text to speech generation

        Args:
            text: Text to convert
            voice: Voice to use
            model: Model to use
            speed: Speed of speech

        Yields:
            Audio data chunks
        """
        try:
            voice = voice or self.default_voice
            if not model:
                model = self.models.get(model_type, self.models["advanced"])
            speed = max(0.25, min(4.0, speed))

            params = {
                "model": model,
                "voice": voice,
                "input": text,
                "speed": speed
            }

            # Add instructions for advanced model
            if model == "gpt-4o-mini-tts" and instructions:
                params["instructions"] = instructions or self.default_instructions

            try:
                with self.client.audio.speech.with_streaming_response.create(**params) as response:
                    for chunk in response.iter_bytes():
                        yield chunk
            except Exception as first_error:
                # Fallback to HD model if advanced fails
                if model == "gpt-4o-mini-tts":
                    logger.warning(f"Advanced streaming failed, falling back: {first_error}")
                    params["model"] = "tts-1-hd"
                    params.pop("instructions", None)
                    with self.client.audio.speech.with_streaming_response.create(**params) as response:
                        for chunk in response.iter_bytes():
                            yield chunk
                else:
                    raise

        except Exception as e:
            logger.error(f"TTS streaming error: {str(e)}")
            yield None

    def generate_autoplay_html(self, audio_data: bytes, format: str = "mp3", hidden: bool = True) -> str:
        """
        Generate HTML for autoplay audio

        Args:
            audio_data: Audio data in bytes
            format: Audio format
            hidden: Whether to hide audio controls

        Returns:
            HTML string for autoplay audio
        """
        try:
            # Convert to base64
            b64_audio = base64.b64encode(audio_data).decode("utf-8")

            # Generate HTML with preload for better performance
            style = "display:none;" if hidden else ""
            html = f"""
            <audio autoplay preload="auto" style="{style}">
                <source src="data:audio/{format};base64,{b64_audio}" type="audio/{format}">
            </audio>
            """
            return html

        except Exception as e:
            logger.error(f"Error generating autoplay HTML: {str(e)}")
            return ""

    def speak_tts_autoplay_chunk(self, text_chunk: str, voice: str = None, model_type: str = "advanced") -> str:
        """
        Convert text to speech with autoplay - matching frontend.py implementation

        Args:
            text_chunk: Text to convert
            voice: Voice to use (default: alloy)
            model_type: Model type to use

        Returns:
            HTML string for autoplay or empty string on error
        """
        text_chunk = text_chunk.strip()
        if not text_chunk or len(text_chunk) < 10:  # Skip very short chunks
            return ""

        voice = voice or self.default_voice

        # Generate TTS
        result = self.text_to_speech(
            text=text_chunk,
            voice=voice,
            model_type=model_type,
            speed=1.1,
            response_format="mp3"
        )

        if result["status"] == "success" and result["audio_data"]:
            return self.generate_autoplay_html(result["audio_data"])
        return ""

    def chunk_text(self, text: str, max_chunk_size: int = 1000) -> list:
        """
        Split long text into chunks for processing

        Args:
            text: Text to chunk
            max_chunk_size: Maximum characters per chunk

        Returns:
            List of text chunks
        """
        if len(text) <= max_chunk_size:
            return [text]

        chunks = []
        sentences = text.split('. ')
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 2 <= max_chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def get_voices(self) -> Dict[str, str]:
        """Get available voices and their descriptions"""
        return {
            "alloy": "Neutral and balanced",
            "ash": "Warm and confident (new)",
            "ballad": "Melodic and expressive (new)",
            "coral": "Clear and articulate (new)",
            "echo": "Warm and conversational",
            "fable": "Expressive and dynamic",
            "onyx": "Deep and authoritative",
            "nova": "Friendly and upbeat",
            "sage": "Wise and calming (new)",
            "shimmer": "Soft and gentle",
            "verse": "Versatile and engaging (new)"
        }

    def get_models(self) -> Dict[str, str]:
        """Get available TTS models"""
        return {
            "tts-1": "Standard quality, lower latency",
            "tts-1-hd": "High quality, higher latency",
            "gpt-4o-mini-tts": "Advanced model with voice instructions (latest)"
        }

    def get_model_types(self) -> Dict[str, str]:
        """Get model type descriptions"""
        return {
            "standard": "Fast, good for real-time applications",
            "hd": "Higher quality, suitable for production content",
            "advanced": "Latest model with voice customization via instructions"
        }

    def estimate_cost(self, text: str, model: str = "tts-1") -> float:
        """
        Estimate cost for TTS generation

        Args:
            text: Text to convert
            model: Model to use

        Returns:
            Estimated cost in USD
        """
        # Approximate costs (may vary)
        char_count = len(text)
        if model == "tts-1":
            cost_per_1k = 0.015  # $0.015 per 1K characters
        else:  # tts-1-hd
            cost_per_1k = 0.030  # $0.030 per 1K characters

        return (char_count / 1000) * cost_per_1k

def get_tts_service(api_key: str = None) -> TTSService:
    """Factory function to get TTS service instance"""
    return TTSService(api_key)