"""
Chat Service Test UI
Streamlit app for testing chat functionality with OpenAI Assistant API
"""
import sys
from pathlib import Path

# MUST add parent to path BEFORE any other imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

import streamlit as st
import time

# Import local module first (from same directory)
from chat_service import ChatService

# Now import from sibling directory
from assistant_service import AssistantService

st.set_page_config(page_title="Chat Service Test", page_icon="💬")

st.title("💬 Chat Service Module Test")
st.markdown("Test interface for chat functionality with streaming and attachments")

# Initialize services
if "chat_service" not in st.session_state:
    try:
        st.session_state.chat_service = ChatService()
        st.session_state.assistant_service = AssistantService()
        st.session_state.service_initialized = True
    except Exception as e:
        st.error(f"Failed to initialize service: {str(e)}")
        st.session_state.service_initialized = False

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")

    api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")

    st.divider()
    st.subheader("Chat Settings")
    streaming = st.checkbox("Enable Streaming", value=True)
    max_tokens = st.number_input("Max Tokens", value=500, min_value=50, max_value=4000)
    temperature = st.slider("Temperature", 0.0, 2.0, 0.7)

    if st.button("🔄 Reinitialize Service"):
        try:
            st.session_state.chat_service = ChatService(api_key)
            st.session_state.assistant_service = AssistantService(api_key)
            st.session_state.service_initialized = True
            st.success("Services reinitialized!")
        except Exception as e:
            st.error(f"Failed: {str(e)}")
            st.session_state.service_initialized = False

if not st.session_state.get('service_initialized', False):
    st.error("⚠️ Service not initialized. Please configure API key in sidebar.")
    st.stop()

chat_service = st.session_state.chat_service
assistant_service = st.session_state.assistant_service

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(["Quick Chat", "Thread Chat", "File Attachments", "Streaming Test"])

with tab1:
    st.header("Quick Chat Test")

    # Create or get assistant
    if "test_assistant_id" not in st.session_state:
        with st.spinner("Creating test assistant..."):
            assistant_id = assistant_service.create_assistant(
                name="Test Chat Assistant",
                instructions="You are a helpful assistant. Provide clear and concise responses.",
                temperature=temperature
            )
            st.session_state.test_assistant_id = assistant_id

    # Create or get thread
    if "test_thread_id" not in st.session_state:
        thread_id = assistant_service.create_thread()
        st.session_state.test_thread_id = thread_id
        st.session_state.messages = []

    # Display chat
    st.info(f"Assistant ID: {st.session_state.test_assistant_id[:16]}...")
    st.info(f"Thread ID: {st.session_state.test_thread_id[:16]}...")

    # Chat interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Type your message..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()

            if streaming:
                # Stream response
                full_response = ""
                for chunk in chat_service.stream_message(
                    st.session_state.test_thread_id,
                    st.session_state.test_assistant_id,
                    prompt
                ):
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
            else:
                # Non-streaming response
                message_placeholder.markdown("Thinking...")
                result = chat_service.send_message(
                    st.session_state.test_thread_id,
                    st.session_state.test_assistant_id,
                    prompt
                )
                if result["status"] == "success":
                    full_response = result["response"]
                    message_placeholder.markdown(full_response)
                else:
                    message_placeholder.error(f"Error: {result['message']}")
                    full_response = f"Error: {result['message']}"

            st.session_state.messages.append({"role": "assistant", "content": full_response})

with tab2:
    st.header("Thread Management")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Create New Thread")
        initial_message = st.text_area("Initial Message (optional)", height=100, key="thread_initial")

        if st.button("Create Thread", type="primary"):
            messages = []
            if initial_message:
                messages = [{"role": "user", "content": initial_message}]

            thread_id = assistant_service.create_thread(messages)
            st.success(f"Thread created: {thread_id}")
            st.session_state.last_created_thread = thread_id

    with col2:
        st.subheader("Thread Operations")
        thread_input = st.text_input("Thread ID",
                                    value=st.session_state.get('last_created_thread', ''),
                                    key="thread_ops_id")
        assistant_input = st.text_input("Assistant ID",
                                       value=st.session_state.get('test_assistant_id', ''),
                                       key="assistant_ops_id")

        message_text = st.text_area("Message", height=100, key="thread_message")

        if st.button("Send Message to Thread"):
            if thread_input and assistant_input and message_text:
                result = chat_service.send_message(
                    thread_input,
                    assistant_input,
                    message_text
                )
                if result["status"] == "success":
                    st.success("Message sent and response received!")
                    st.text_area("Response", value=result["response"], height=200)
                else:
                    st.error(f"Error: {result['message']}")

with tab3:
    st.header("File Attachments Test")

    st.info("Test sending messages with file attachments (requires file to be uploaded to OpenAI first)")

    col1, col2 = st.columns(2)

    with col1:
        file_id = st.text_input("File ID", placeholder="file-...", key="attachment_file_id")
        attachment_thread = st.text_input("Thread ID",
                                         value=st.session_state.get('test_thread_id', ''),
                                         key="attachment_thread")
        attachment_assistant = st.text_input("Assistant ID",
                                           value=st.session_state.get('test_assistant_id', ''),
                                           key="attachment_assistant")

    with col2:
        attachment_message = st.text_area("Message with attachment",
                                         value="Please analyze the attached file.",
                                         key="attachment_message")

        if st.button("Send with Attachment", type="primary"):
            if file_id and attachment_thread and attachment_assistant:
                attachments = [{"file_id": file_id, "tools": [{"type": "file_search"}]}]

                result = chat_service.send_message(
                    attachment_thread,
                    attachment_assistant,
                    attachment_message,
                    attachments
                )

                if result["status"] == "success":
                    st.success("Message with attachment sent!")
                    st.text_area("Response", value=result["response"], height=200)
                else:
                    st.error(f"Error: {result['message']}")

with tab4:
    st.header("Streaming Test")

    st.info("Test streaming responses with different message lengths")

    # Test prompts
    test_prompts = {
        "Short": "What is 2+2?",
        "Medium": "Explain photosynthesis in simple terms.",
        "Long": "Write a detailed explanation of how neural networks work, including backpropagation.",
        "Creative": "Write a short story about a robot learning to paint.",
        "Code": "Write a Python function to calculate fibonacci numbers with memoization."
    }

    selected_prompt = st.selectbox("Select test prompt", list(test_prompts.keys()))
    prompt_text = st.text_area("Prompt", value=test_prompts[selected_prompt], height=100)

    col1, col2 = st.columns(2)

    with col1:
        stream_thread = st.text_input("Thread ID",
                                     value=st.session_state.get('test_thread_id', ''),
                                     key="stream_thread")

    with col2:
        stream_assistant = st.text_input("Assistant ID",
                                        value=st.session_state.get('test_assistant_id', ''),
                                        key="stream_assistant")

    if st.button("🚀 Test Streaming", type="primary"):
        if stream_thread and stream_assistant:
            container = st.container()

            with container:
                st.markdown("**User:**")
                st.write(prompt_text)

                st.markdown("**Assistant:**")
                response_placeholder = st.empty()

                full_response = ""
                chunk_count = 0
                start_time = time.time()

                for chunk in chat_service.stream_message(
                    stream_thread,
                    stream_assistant,
                    prompt_text
                ):
                    full_response += chunk
                    chunk_count += 1
                    response_placeholder.markdown(full_response + "▌")

                response_placeholder.markdown(full_response)

                end_time = time.time()
                elapsed = end_time - start_time

                st.divider()
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Chunks", chunk_count)
                with col2:
                    st.metric("Characters", len(full_response))
                with col3:
                    st.metric("Time", f"{elapsed:.2f}s")

# Status bar
st.divider()
col1, col2, col3 = st.columns(3)
with col1:
    st.caption(f"🟢 Service: {'Connected' if st.session_state.get('service_initialized') else 'Not Connected'}")
with col2:
    st.caption(f"💬 Active Thread: {st.session_state.get('test_thread_id', 'None')[:16]}..." if st.session_state.get('test_thread_id') else "💬 No active thread")
with col3:
    st.caption(f"🤖 Assistant: {st.session_state.get('test_assistant_id', 'None')[:16]}..." if st.session_state.get('test_assistant_id') else "🤖 No assistant")

# Cleanup section
with st.expander("🧹 Cleanup"):
    st.warning("Delete test resources when done")

    if st.button("Delete Test Assistant", type="secondary"):
        if st.session_state.get('test_assistant_id'):
            if assistant_service.delete_assistant(st.session_state.test_assistant_id):
                st.success("Assistant deleted")
                del st.session_state.test_assistant_id
                st.rerun()

    if st.button("Clear Chat History", type="secondary"):
        st.session_state.messages = []
        st.success("Chat history cleared")
        st.rerun()