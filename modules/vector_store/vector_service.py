"""
Vector Store / RAG Service
Handles vector store creation, file uploads, and retrieval
"""
import streamlit as st
from openai import OpenAI
from typing import List, Dict, Optional, BinaryIO
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStoreService:
    def __init__(self, api_key: str = None):
        """Initialize vector store service"""
        self.api_key = api_key or st.secrets.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided")
        self.client = OpenAI(api_key=self.api_key)

        # Default vector store ID with NARM Knowledge Base PDFs
        self.default_vector_store_id = "vs_68d4f901de948191bf47c56de33994e8"

    def create_vector_store(self, name: str = "Vector Store") -> str:
        """Create a new vector store"""
        try:
            vector_store = self.client.vector_stores.create(name=name)
            logger.info(f"Vector store created: {vector_store.id}")
            return vector_store.id
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            raise

    def upload_file_to_store(self,
                            vector_store_id: str,
                            file_path: str = None,
                            file_content: bytes = None,
                            file_name: str = None) -> str:
        """Upload a file to vector store"""
        try:
            # Upload file to OpenAI
            if file_path:
                with open(file_path, "rb") as f:
                    file_obj = self.client.files.create(file=f, purpose="assistants")
            elif file_content and file_name:
                # Create temporary file for upload
                temp_path = f"/tmp/{file_name}"
                with open(temp_path, "wb") as f:
                    f.write(file_content)
                try:
                    with open(temp_path, "rb") as f:
                        file_obj = self.client.files.create(file=f, purpose="assistants")
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
            else:
                raise ValueError("Either file_path or (file_content and file_name) must be provided")

            # Add file to vector store
            self.client.vector_stores.files.create(
                vector_store_id=vector_store_id,
                file_id=file_obj.id
            )

            logger.info(f"File {file_obj.id} added to vector store {vector_store_id}")
            return file_obj.id

        except Exception as e:
            logger.error(f"Error uploading file to vector store: {str(e)}")
            raise

    def list_vector_stores(self, limit: int = 20) -> List:
        """List all vector stores"""
        try:
            stores = self.client.vector_stores.list(limit=limit)
            return stores.data
        except Exception as e:
            logger.error(f"Error listing vector stores: {str(e)}")
            return []

    def get_vector_store(self, vector_store_id: str) -> Optional[Dict]:
        """Get vector store details"""
        try:
            return self.client.vector_stores.retrieve(vector_store_id=vector_store_id)
        except Exception as e:
            logger.error(f"Error retrieving vector store: {str(e)}")
            return None

    def list_files_in_store(self, vector_store_id: str) -> List:
        """List files in a vector store"""
        try:
            files = self.client.vector_stores.files.list(vector_store_id=vector_store_id)
            return files.data
        except Exception as e:
            logger.error(f"Error listing files in vector store: {str(e)}")
            return []

    def delete_vector_store(self, vector_store_id: str) -> bool:
        """Delete a vector store"""
        try:
            self.client.vector_stores.delete(vector_store_id=vector_store_id)
            logger.info(f"Vector store deleted: {vector_store_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting vector store: {str(e)}")
            return False

    def remove_file_from_store(self, vector_store_id: str, file_id: str) -> bool:
        """Remove a file from vector store"""
        try:
            self.client.vector_stores.files.delete(
                vector_store_id=vector_store_id,
                file_id=file_id
            )
            logger.info(f"File {file_id} removed from vector store {vector_store_id}")
            return True
        except Exception as e:
            logger.error(f"Error removing file from vector store: {str(e)}")
            return False

    def update_vector_store(self, vector_store_id: str, name: str = None, metadata: Dict = None) -> bool:
        """Update vector store properties"""
        try:
            update_params = {}
            if name:
                update_params["name"] = name
            if metadata:
                update_params["metadata"] = metadata

            if update_params:
                self.client.vector_stores.update(
                    vector_store_id=vector_store_id,
                    **update_params
                )
                logger.info(f"Vector store updated: {vector_store_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error updating vector store: {str(e)}")
            return False

    def attach_to_assistant(self, assistant_id: str, vector_store_id: str) -> bool:
        """Attach vector store to an assistant"""
        try:
            self.client.beta.assistants.update(
                assistant_id=assistant_id,
                tool_resources={
                    "file_search": {
                        "vector_store_ids": [vector_store_id]
                    }
                }
            )
            logger.info(f"Vector store {vector_store_id} attached to assistant {assistant_id}")
            return True
        except Exception as e:
            logger.error(f"Error attaching vector store to assistant: {str(e)}")
            return False

    def search_vector_store(self, vector_store_id: str, query: str, limit: int = 5) -> List[Dict]:
        """
        Search vector store (simplified - actual implementation would use embeddings)
        This is a placeholder for demonstration
        """
        try:
            # Note: This is a simplified version. Real implementation would:
            # 1. Generate embeddings for the query
            # 2. Search the vector store
            # 3. Return relevant documents
            logger.info(f"Searching vector store {vector_store_id} for: {query}")

            # For now, return files list as a simple demonstration
            files = self.list_files_in_store(vector_store_id)
            results = []
            for file in files[:limit]:
                results.append({
                    "file_id": file.id,
                    "score": 0.95,  # Placeholder score
                    "metadata": {}
                })
            return results

        except Exception as e:
            logger.error(f"Error searching vector store: {str(e)}")
            return []

def get_vector_service(api_key: str = None) -> VectorStoreService:
    """Factory function to get vector store service instance"""
    return VectorStoreService(api_key)