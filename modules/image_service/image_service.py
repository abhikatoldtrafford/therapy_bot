"""
Image Analysis Service
Handles image analysis using OpenAI Vision
"""
import streamlit as st
from openai import OpenAI
import base64
import logging
from typing import Dict, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageService:
    def __init__(self, api_key: str = None, model: str = "gpt-4o"):
        """Initialize image service"""
        self.api_key = api_key or st.secrets.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided")
        self.client = OpenAI(api_key=self.api_key)
        self.model = model

    def analyze_image(self, image_data: bytes, prompt: str = None,
                     detail: str = "high", max_tokens: int = 500) -> Dict:
        """Analyze an image with optional prompt"""
        try:
            # Encode image
            b64_image = base64.b64encode(image_data).decode('utf-8')

            # Default prompt
            if not prompt:
                prompt = "Analyze this image and provide a thorough summary including all elements."

            # Create message
            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64_image}",
                            "detail": detail
                        }
                    }
                ]
            }]

            # Get response
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens
            )

            return {
                "status": "success",
                "analysis": response.choices[0].message.content
            }

        except Exception as e:
            logger.error(f"Image analysis error: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }

def get_image_service(api_key: str = None) -> ImageService:
    """Factory function"""
    return ImageService(api_key)