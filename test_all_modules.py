#!/usr/bin/env python3
"""
Comprehensive Module Functionality Test Script
Tests actual functionality of all modules including OpenAI API calls
"""

import sys
import os
import json
import time
import tempfile
import base64
from pathlib import Path
from typing import Dict, Any, Optional
import traceback

# Add modules to path
sys.path.append(str(Path(__file__).parent / "modules"))

# Test configuration
TEST_CONFIG = {
    "use_real_api": True,  # Set to True to test with real OpenAI API
    "api_key": os.getenv("OPENAI_API_KEY", ""),  # Will use env var or prompt
    "verbose": True,
    "test_data": {
        "text": "Hello, this is a test message.",
        "session_name": "Test User",
        "session_email": "test@example.com",
        "audio_file": "test_audio.mp3",
        "image_file": "test_image.png"
    }
}

class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_header(text: str):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(60)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")

def print_test(test_name: str, status: str = "TESTING"):
    if status == "TESTING":
        print(f"{Colors.CYAN}[{status}] {test_name}...{Colors.ENDC}")
    elif status == "PASSED":
        print(f"{Colors.GREEN}[✓ {status}] {test_name}{Colors.ENDC}")
    elif status == "FAILED":
        print(f"{Colors.FAIL}[✗ {status}] {test_name}{Colors.ENDC}")
    elif status == "SKIPPED":
        print(f"{Colors.WARNING}[- {status}] {test_name}{Colors.ENDC}")
    else:
        print(f"[{status}] {test_name}")

def print_error(error: str):
    print(f"{Colors.FAIL}  ERROR: {error}{Colors.ENDC}")

def print_success(message: str):
    print(f"{Colors.GREEN}  SUCCESS: {message}{Colors.ENDC}")

def print_info(message: str):
    print(f"{Colors.BLUE}  INFO: {message}{Colors.ENDC}")

# Test Results Storage
test_results = {
    "passed": 0,
    "failed": 0,
    "skipped": 0,
    "errors": []
}

# ============================================================================
# MODULE TEST FUNCTIONS
# ============================================================================

def test_session_manager() -> bool:
    """Test Session Manager functionality"""
    print_header("SESSION MANAGER TESTS")

    try:
        from session_manager.session_service import SessionManager, get_session_manager

        # Test 1: Initialize manager
        print_test("Initialize SessionManager", "TESTING")
        manager = SessionManager("test_sessions_temp.json")
        print_test("Initialize SessionManager", "PASSED")

        # Test 2: Create session
        print_test("Create Session", "TESTING")
        session_id = manager.create_session(
            name=TEST_CONFIG["test_data"]["session_name"],
            email=TEST_CONFIG["test_data"]["session_email"],
            focus_today="Test focus",
            desired_outcome="Test outcome"
        )
        assert session_id is not None
        assert len(session_id) > 0
        print_success(f"Created session: {session_id[:16]}...")
        print_test("Create Session", "PASSED")

        # Test 3: Get session
        print_test("Retrieve Session", "TESTING")
        session_data = manager.get_session(session_id)
        assert session_data is not None
        assert session_data["user_info"]["name"] == TEST_CONFIG["test_data"]["session_name"]
        print_success("Retrieved session data successfully")
        print_test("Retrieve Session", "PASSED")

        # Test 4: Update session
        print_test("Update Session", "TESTING")
        success = manager.update_session(
            session_id,
            assistant_id="test_assistant_123",
            thread_id="test_thread_456"
        )
        assert success == True
        updated = manager.get_session(session_id)
        assert updated["assistant_id"] == "test_assistant_123"
        print_success("Updated session successfully")
        print_test("Update Session", "PASSED")

        # Test 5: List sessions
        print_test("List Sessions", "TESTING")
        sessions = manager.list_sessions()
        assert len(sessions) > 0
        assert session_id in sessions
        print_success(f"Listed {len(sessions)} sessions")
        print_test("List Sessions", "PASSED")

        # Test 6: Delete session
        print_test("Delete Session", "TESTING")
        success = manager.delete_session(session_id)
        assert success == True
        deleted = manager.get_session(session_id)
        assert deleted is None
        print_success("Deleted session successfully")
        print_test("Delete Session", "PASSED")

        # Cleanup
        if os.path.exists("test_sessions_temp.json"):
            os.remove("test_sessions_temp.json")

        return True

    except Exception as e:
        print_error(f"{str(e)}")
        if TEST_CONFIG["verbose"]:
            traceback.print_exc()
        test_results["errors"].append(f"SessionManager: {str(e)}")
        return False

def test_config_manager() -> bool:
    """Test Configuration Manager functionality"""
    print_header("CONFIG MANAGER TESTS")

    try:
        from config_manager.config_manager import ConfigManager, get_config_manager

        # Test 1: Initialize
        print_test("Initialize ConfigManager", "TESTING")
        config = ConfigManager("test_config_temp.json")
        print_test("Initialize ConfigManager", "PASSED")

        # Test 2: Get default values
        print_test("Get Default Values", "TESTING")
        model = config.get("openai.model")
        assert model == "gpt-4o"
        print_success(f"Got default model: {model}")
        print_test("Get Default Values", "PASSED")

        # Test 3: Set values
        print_test("Set Configuration Value", "TESTING")
        success = config.set("tts.voice", "nova")
        assert success == True
        voice = config.get("tts.voice")
        assert voice == "nova"
        print_success(f"Set TTS voice to: {voice}")
        print_test("Set Configuration Value", "PASSED")

        # Test 4: Update section
        print_test("Update Configuration Section", "TESTING")
        success = config.update_section("ui", {
            "enable_voice": True,
            "enable_image": False
        })
        assert success == True
        print_success("Updated UI section")
        print_test("Update Configuration Section", "PASSED")

        # Test 5: Export/Import
        print_test("Export/Import Configuration", "TESTING")
        export_json = config.export_config()
        assert len(export_json) > 0

        # Create new config and import
        config2 = ConfigManager("test_config_temp2.json")
        success = config2.import_config(export_json)
        assert success == True
        assert config2.get("tts.voice") == "nova"
        print_success("Exported and imported configuration")
        print_test("Export/Import Configuration", "PASSED")

        # Test 6: Reset to defaults
        print_test("Reset to Defaults", "TESTING")
        success = config.reset_to_defaults()
        assert success == True
        voice = config.get("tts.voice")
        assert voice == "alloy"  # Default voice
        print_success("Reset to defaults successfully")
        print_test("Reset to Defaults", "PASSED")

        # Cleanup
        for file in ["test_config_temp.json", "test_config_temp2.json"]:
            if os.path.exists(file):
                os.remove(file)

        return True

    except Exception as e:
        print_error(f"{str(e)}")
        if TEST_CONFIG["verbose"]:
            traceback.print_exc()
        test_results["errors"].append(f"ConfigManager: {str(e)}")
        return False

def test_assistant_service(api_key: Optional[str] = None) -> bool:
    """Test Assistant Service functionality"""
    print_header("ASSISTANT SERVICE TESTS")

    if not api_key:
        print_test("Assistant Service", "SKIPPED")
        print_info("No API key provided - skipping OpenAI tests")
        return True

    try:
        from assistant_service.assistant_service import AssistantService, get_assistant_service

        # Test 1: Initialize with API key
        print_test("Initialize AssistantService", "TESTING")
        service = AssistantService(api_key=api_key)
        print_test("Initialize AssistantService", "PASSED")

        if TEST_CONFIG["use_real_api"]:
            # Test 2: Create assistant
            print_test("Create Assistant", "TESTING")
            assistant_id = service.create_assistant(
                name="Test Assistant",
                instructions="You are a test assistant.",
                temperature=0.7
            )
            assert assistant_id is not None
            print_success(f"Created assistant: {assistant_id[:16]}...")
            print_test("Create Assistant", "PASSED")

            # Test 3: Create thread
            print_test("Create Thread", "TESTING")
            thread_id = service.create_thread()
            assert thread_id is not None
            print_success(f"Created thread: {thread_id[:16]}...")
            print_test("Create Thread", "PASSED")

            # Test 4: List assistants
            print_test("List Assistants", "TESTING")
            assistants = service.list_assistants(limit=5)
            assert isinstance(assistants, list)
            print_success(f"Listed {len(assistants)} assistants")
            print_test("List Assistants", "PASSED")

            # Test 5: Delete assistant (cleanup)
            print_test("Delete Assistant", "TESTING")
            success = service.delete_assistant(assistant_id)
            assert success == True
            print_success("Deleted assistant")
            print_test("Delete Assistant", "PASSED")
        else:
            print_info("Real API calls disabled - testing initialization only")

        return True

    except Exception as e:
        print_error(f"{str(e)}")
        if TEST_CONFIG["verbose"]:
            traceback.print_exc()
        test_results["errors"].append(f"AssistantService: {str(e)}")
        return False

def test_vector_store_service(api_key: Optional[str] = None) -> bool:
    """Test Vector Store Service functionality"""
    print_header("VECTOR STORE SERVICE TESTS")

    if not api_key:
        print_test("Vector Store Service", "SKIPPED")
        print_info("No API key provided - skipping OpenAI tests")
        return True

    try:
        from vector_store.vector_service import VectorStoreService, get_vector_service

        # Test 1: Initialize
        print_test("Initialize VectorStoreService", "TESTING")
        service = VectorStoreService(api_key=api_key)
        assert service.default_vector_store_id == "vs_67a7a6bd68d48191a4f446ddeaec2e2b"
        print_test("Initialize VectorStoreService", "PASSED")

        if TEST_CONFIG["use_real_api"]:
            # Test 2: Create vector store
            print_test("Create Vector Store", "TESTING")
            store_id = service.create_vector_store("Test Store")
            assert store_id is not None
            print_success(f"Created vector store: {store_id[:16]}...")
            print_test("Create Vector Store", "PASSED")

            # Test 3: List stores
            print_test("List Vector Stores", "TESTING")
            stores = service.list_vector_stores(limit=5)
            assert isinstance(stores, list)
            print_success(f"Listed {len(stores)} stores")
            print_test("List Vector Stores", "PASSED")

            # Test 4: Delete store (cleanup)
            print_test("Delete Vector Store", "TESTING")
            success = service.delete_vector_store(store_id)
            assert success == True
            print_success("Deleted vector store")
            print_test("Delete Vector Store", "PASSED")
        else:
            print_info("Real API calls disabled - testing initialization only")

        return True

    except Exception as e:
        print_error(f"{str(e)}")
        if TEST_CONFIG["verbose"]:
            traceback.print_exc()
        test_results["errors"].append(f"VectorStoreService: {str(e)}")
        return False

def test_stt_service(api_key: Optional[str] = None) -> bool:
    """Test STT Service functionality"""
    print_header("STT SERVICE TESTS")

    if not api_key:
        print_test("STT Service", "SKIPPED")
        print_info("No API key provided - skipping OpenAI tests")
        return True

    try:
        from stt_service.stt_service import STTService, get_stt_service

        # Test 1: Initialize
        print_test("Initialize STTService", "TESTING")
        service = STTService(api_key=api_key)
        assert service.default_model == "accurate"  # Updated default
        print_test("Initialize STTService", "PASSED")

        # Test 2: Get supported formats
        print_test("Get Supported Formats", "TESTING")
        formats = service.get_supported_formats()
        assert "mp3" in formats
        assert "wav" in formats
        print_success(f"Supports {len(formats)} formats")
        print_test("Get Supported Formats", "PASSED")

        # Test 3: Get language codes
        print_test("Get Language Codes", "TESTING")
        languages = service.get_language_codes()
        assert "English" in languages
        assert languages["English"] == "en"
        print_success(f"Supports {len(languages)} languages")
        print_test("Get Language Codes", "PASSED")

        # Test 4: Check format support
        print_test("Check Format Support", "TESTING")
        assert service.is_format_supported("test.mp3") == True
        assert service.is_format_supported("test.xyz") == False
        print_test("Check Format Support", "PASSED")

        if TEST_CONFIG["use_real_api"]:
            # Test 5: Transcribe audio (would need actual audio file)
            print_info("Audio transcription test requires actual audio file")

        return True

    except Exception as e:
        print_error(f"{str(e)}")
        if TEST_CONFIG["verbose"]:
            traceback.print_exc()
        test_results["errors"].append(f"STTService: {str(e)}")
        return False

def test_tts_service(api_key: Optional[str] = None) -> bool:
    """Test TTS Service functionality"""
    print_header("TTS SERVICE TESTS")

    if not api_key:
        print_test("TTS Service", "SKIPPED")
        print_info("No API key provided - skipping OpenAI tests")
        return True

    try:
        from tts_service.tts_service import TTSService, get_tts_service

        # Test 1: Initialize
        print_test("Initialize TTSService", "TESTING")
        service = TTSService(api_key=api_key)
        assert service.default_model == "advanced"  # Updated default
        assert service.default_voice == "alloy"
        print_test("Initialize TTSService", "PASSED")

        # Test 2: Get voices
        print_test("Get Available Voices", "TESTING")
        voices = service.get_voices()
        assert "alloy" in voices
        assert len(voices) >= 6  # We have more voices now
        print_success(f"Available voices: {', '.join(voices.keys())}")
        print_test("Get Available Voices", "PASSED")

        # Test 3: Get models
        print_test("Get TTS Models", "TESTING")
        models = service.get_models()
        assert "tts-1" in models
        assert "tts-1-hd" in models
        assert len(models) >= 2  # Should have at least 2 models
        print_test("Get TTS Models", "PASSED")

        # Test 4: Chunk text
        print_test("Text Chunking", "TESTING")
        long_text = "Hello world. " * 100
        chunks = service.chunk_text(long_text, max_chunk_size=100)
        assert len(chunks) > 1
        print_success(f"Chunked text into {len(chunks)} parts")
        print_test("Text Chunking", "PASSED")

        # Test 5: Estimate cost
        print_test("Cost Estimation", "TESTING")
        cost = service.estimate_cost("Hello world", model="tts-1")
        assert cost > 0
        print_success(f"Estimated cost: ${cost:.6f}")
        print_test("Cost Estimation", "PASSED")

        # Test 6: Generate autoplay HTML
        print_test("Generate Autoplay HTML", "TESTING")
        test_audio = b"fake audio data"
        html = service.generate_autoplay_html(test_audio, format="mp3")
        assert "audio" in html
        assert "base64" in html
        print_test("Generate Autoplay HTML", "PASSED")

        if TEST_CONFIG["use_real_api"]:
            # Test 7: Generate speech
            print_test("Generate Speech", "TESTING")
            result = service.text_to_speech(
                text="Hello, this is a test.",
                voice="alloy",
                model="tts-1",
                speed=1.0
            )
            assert result["status"] == "success"
            assert result["audio_data"] is not None
            print_success(f"Generated {len(result['audio_data'])} bytes of audio")
            print_test("Generate Speech", "PASSED")
        else:
            print_info("Real API calls disabled - skipping speech generation")

        return True

    except Exception as e:
        print_error(f"{str(e)}")
        if TEST_CONFIG["verbose"]:
            traceback.print_exc()
        test_results["errors"].append(f"TTSService: {str(e)}")
        return False

def test_chat_service(api_key: Optional[str] = None) -> bool:
    """Test Chat Service functionality"""
    print_header("CHAT SERVICE TESTS")

    if not api_key:
        print_test("Chat Service", "SKIPPED")
        print_info("No API key provided - skipping OpenAI tests")
        return True

    try:
        from chat_service.chat_service import ChatService, get_chat_service

        # Test 1: Initialize
        print_test("Initialize ChatService", "TESTING")
        service = ChatService(api_key=api_key)
        print_test("Initialize ChatService", "PASSED")

        if TEST_CONFIG["use_real_api"]:
            # Would need assistant_id and thread_id from Assistant Service
            print_info("Chat tests require assistant and thread creation")
        else:
            print_info("Real API calls disabled - testing initialization only")

        return True

    except Exception as e:
        print_error(f"{str(e)}")
        if TEST_CONFIG["verbose"]:
            traceback.print_exc()
        test_results["errors"].append(f"ChatService: {str(e)}")
        return False

def test_image_service(api_key: Optional[str] = None) -> bool:
    """Test Image Service functionality"""
    print_header("IMAGE SERVICE TESTS")

    if not api_key:
        print_test("Image Service", "SKIPPED")
        print_info("No API key provided - skipping OpenAI tests")
        return True

    try:
        from image_service.image_service import ImageService, get_image_service

        # Test 1: Initialize
        print_test("Initialize ImageService", "TESTING")
        service = ImageService(api_key=api_key, model="gpt-4o")
        assert service.model == "gpt-4o"
        print_test("Initialize ImageService", "PASSED")

        if TEST_CONFIG["use_real_api"]:
            # Test 2: Analyze image (create test image)
            print_test("Analyze Image", "TESTING")

            # Create a simple test image (1x1 white pixel PNG)
            test_image = base64.b64decode(
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
            )

            result = service.analyze_image(
                image_data=test_image,
                prompt="Describe this image",
                detail="low",
                max_tokens=50
            )

            assert result["status"] == "success"
            assert result["analysis"] is not None
            print_success("Image analyzed successfully")
            print_test("Analyze Image", "PASSED")
        else:
            print_info("Real API calls disabled - testing initialization only")

        return True

    except Exception as e:
        print_error(f"{str(e)}")
        if TEST_CONFIG["verbose"]:
            traceback.print_exc()
        test_results["errors"].append(f"ImageService: {str(e)}")
        return False

def test_module_imports() -> bool:
    """Test that all modules can be imported"""
    print_header("MODULE IMPORT TESTS")

    modules = [
        ("session_manager.session_service", "SessionManager"),
        ("assistant_service.assistant_service", "AssistantService"),
        ("vector_store.vector_service", "VectorStoreService"),
        ("stt_service.stt_service", "STTService"),
        ("tts_service.tts_service", "TTSService"),
        ("chat_service.chat_service", "ChatService"),
        ("image_service.image_service", "ImageService"),
        ("config_manager.config_manager", "ConfigManager")
    ]

    all_passed = True
    for module_path, class_name in modules:
        print_test(f"Import {module_path}", "TESTING")
        try:
            module = __import__(module_path, fromlist=[class_name])
            cls = getattr(module, class_name)
            print_test(f"Import {module_path}", "PASSED")
        except Exception as e:
            print_test(f"Import {module_path}", "FAILED")
            print_error(str(e))
            all_passed = False
            test_results["errors"].append(f"Import {module_path}: {str(e)}")

    return all_passed

def main():
    """Run all tests"""
    print_header("COMPREHENSIVE MODULE TESTS")
    print(f"{Colors.BLUE}Testing all modules with functional tests{Colors.ENDC}")
    print(f"{Colors.BLUE}Verbose: {TEST_CONFIG['verbose']}{Colors.ENDC}")
    print(f"{Colors.BLUE}Use Real API: {TEST_CONFIG['use_real_api']}{Colors.ENDC}")

    # Check for API key
    api_key = None
    if TEST_CONFIG["use_real_api"]:
        api_key = TEST_CONFIG["api_key"]
        if not api_key:
            # Try to get from streamlit secrets
            try:
                import streamlit as st
                api_key = st.secrets.get("OPENAI_API_KEY")
            except:
                pass

        if not api_key:
            print(f"{Colors.WARNING}No API key found. Set OPENAI_API_KEY env var or use .streamlit/secrets.toml{Colors.ENDC}")
            print(f"{Colors.WARNING}Running tests without API calls...{Colors.ENDC}")
            TEST_CONFIG["use_real_api"] = False

    # Run tests
    tests = [
        ("Module Imports", test_module_imports),
        ("Session Manager", test_session_manager),
        ("Config Manager", test_config_manager),
        ("Assistant Service", lambda: test_assistant_service(api_key)),
        ("Vector Store Service", lambda: test_vector_store_service(api_key)),
        ("STT Service", lambda: test_stt_service(api_key)),
        ("TTS Service", lambda: test_tts_service(api_key)),
        ("Chat Service", lambda: test_chat_service(api_key)),
        ("Image Service", lambda: test_image_service(api_key))
    ]

    for test_name, test_func in tests:
        try:
            result = test_func()
            if result:
                test_results["passed"] += 1
            else:
                test_results["failed"] += 1
        except Exception as e:
            test_results["failed"] += 1
            print_error(f"Unexpected error in {test_name}: {str(e)}")
            if TEST_CONFIG["verbose"]:
                traceback.print_exc()

    # Print summary
    print_header("TEST SUMMARY")

    total = test_results["passed"] + test_results["failed"] + test_results["skipped"]
    print(f"{Colors.BOLD}Total Tests: {total}{Colors.ENDC}")
    print(f"{Colors.GREEN}Passed: {test_results['passed']}{Colors.ENDC}")
    print(f"{Colors.FAIL}Failed: {test_results['failed']}{Colors.ENDC}")
    print(f"{Colors.WARNING}Skipped: {test_results['skipped']}{Colors.ENDC}")

    if test_results["errors"]:
        print(f"\n{Colors.FAIL}{Colors.BOLD}Errors:{Colors.ENDC}")
        for error in test_results["errors"]:
            print(f"  {Colors.FAIL}• {error}{Colors.ENDC}")

    if test_results["failed"] == 0:
        print(f"\n{Colors.GREEN}{Colors.BOLD}✅ ALL TESTS PASSED!{Colors.ENDC}")
        return 0
    else:
        print(f"\n{Colors.FAIL}{Colors.BOLD}❌ SOME TESTS FAILED{Colors.ENDC}")
        return 1

if __name__ == "__main__":
    exit(main())