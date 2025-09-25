"""
Assistant Service Test UI
Streamlit app for testing OpenAI Assistant functionality
"""
import sys
from pathlib import Path

# MUST add parent to path BEFORE any other imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

import streamlit as st
import time
from assistant_service import AssistantService, DEFAULT_SYSTEM_PROMPT, DEFAULT_NARM_PROMPT

st.set_page_config(page_title="Assistant Service Test", page_icon="🤖")

st.title("🤖 Assistant Service Module Test")
st.markdown("Test interface for OpenAI Assistant functionality")

# Initialize assistant service
if "assistant_service" not in st.session_state:
    try:
        st.session_state.assistant_service = AssistantService()
        st.session_state.service_initialized = True
    except Exception as e:
        st.error(f"Failed to initialize service: {str(e)}")
        st.session_state.service_initialized = False

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")

    api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
    model = st.selectbox("Default Model", ["gpt-4o", "gpt-4", "gpt-3.5-turbo"])

    if st.button("🔄 Reinitialize Service"):
        try:
            st.session_state.assistant_service = AssistantService(api_key, model=model)
            st.session_state.service_initialized = True
            st.success("Service reinitialized!")
        except Exception as e:
            st.error(f"Failed: {str(e)}")
            st.session_state.service_initialized = False

if not st.session_state.get('service_initialized', False):
    st.error("⚠️ Service not initialized. Please configure API key in sidebar.")
    st.stop()

service = st.session_state.assistant_service

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(["Assistant Management", "Thread Management", "Running Assistants", "System Prompts"])

with tab1:
    st.header("Assistant Management")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Create New Assistant")

        with st.form("create_assistant_form"):
            name = st.text_input("Assistant Name", value="Test Assistant")
            instructions = st.text_area(
                "System Instructions",
                value="You are a helpful AI assistant. Be concise and informative.",
                height=150
            )
            model_choice = st.selectbox("Model", ["gpt-4.1", "gpt-4o", "gpt-4o-mini"])
            temperature = st.slider("Temperature", 0.0, 2.0, 0.7)

            tools = st.multiselect(
                "Tools",
                ["file_search", "code_interpreter"],
                default=["file_search"]
            )

            if st.form_submit_button("Create Assistant", type="primary"):
                tool_list = [{"type": tool} for tool in tools]
                assistant_id = service.create_assistant(
                    name=name,
                    instructions=instructions,
                    tools=tool_list,
                    model=model_choice,
                    temperature=temperature
                )
                st.success(f"✅ Created assistant: {assistant_id}")
                st.session_state.last_assistant_id = assistant_id

    with col2:
        st.subheader("Assistant Operations")

        assistant_id_input = st.text_input(
            "Assistant ID",
            value=st.session_state.get('last_assistant_id', ''),
            placeholder="asst_..."
        )

        col2_1, col2_2 = st.columns(2)
        with col2_1:
            if st.button("Get Assistant Info"):
                if assistant_id_input:
                    assistant = service.get_assistant(assistant_id_input)
                    if assistant:
                        # Convert assistant object to JSON-serializable dict
                        info = {
                            "id": assistant.id,
                            "name": assistant.name,
                            "model": assistant.model,
                            "instructions": assistant.instructions,
                            "tools": [tool.type for tool in assistant.tools] if assistant.tools else [],
                            "created_at": assistant.created_at,
                            "temperature": assistant.temperature
                        }
                        st.json(info)
                    else:
                        st.error("Assistant not found")

        with col2_2:
            if st.button("Delete Assistant", type="secondary"):
                if assistant_id_input:
                    if service.delete_assistant(assistant_id_input):
                        st.success("Assistant deleted")
                        if st.session_state.get('last_assistant_id') == assistant_id_input:
                            del st.session_state.last_assistant_id
                    else:
                        st.error("Failed to delete assistant")

with tab2:
    st.header("Thread Management")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Create Thread")

        initial_message = st.text_area(
            "Initial Message (optional)",
            placeholder="Start the conversation with an initial user message...",
            height=100
        )

        if st.button("Create Thread", type="primary"):
            messages = []
            if initial_message:
                messages = [{"role": "user", "content": initial_message}]

            thread_id = service.create_thread(messages)
            st.success(f"✅ Created thread: {thread_id}")
            st.session_state.last_thread_id = thread_id

    with col2:
        st.subheader("Thread Operations")

        thread_id_input = st.text_input(
            "Thread ID",
            value=st.session_state.get('last_thread_id', ''),
            placeholder="thread_..."
        )

        if st.button("Get Thread Messages"):
            if thread_id_input:
                messages = service.get_thread_messages(thread_id_input)
                if messages:
                    for msg in messages:
                        with st.container():
                            st.markdown(f"**{msg.role}**: {msg.content[0].text.value if msg.content else 'No content'}")
                            st.caption(f"ID: {msg.id}")
                else:
                    st.info("No messages in thread")

with tab3:
    st.header("Run Assistant on Thread")

    col1, col2 = st.columns(2)

    with col1:
        run_thread_id = st.text_input(
            "Thread ID for Run",
            value=st.session_state.get('last_thread_id', ''),
            placeholder="thread_...",
            key="run_thread"
        )
        run_assistant_id = st.text_input(
            "Assistant ID for Run",
            value=st.session_state.get('last_assistant_id', ''),
            placeholder="asst_...",
            key="run_assistant"
        )

        user_message = st.text_area(
            "User Message",
            placeholder="Type your message to the assistant...",
            height=100
        )

    with col2:
        st.subheader("Run Configuration")

        run_tools = st.multiselect(
            "Enable Tools for Run",
            ["file_search", "code_interpreter"],
            default=["file_search"],
            key="run_tools"
        )

        additional_instructions = st.text_area(
            "Additional Instructions (optional)",
            placeholder="Override or add to the assistant's instructions for this run...",
            height=100
        )

    if st.button("🚀 Run Assistant", type="primary"):
        if run_thread_id and run_assistant_id and user_message:
            # Add message to thread
            service.add_message_to_thread(run_thread_id, "user", user_message)

            # Create run
            tools_config = [{"type": tool} for tool in run_tools] if run_tools else None
            run = service.run_assistant(
                thread_id=run_thread_id,
                assistant_id=run_assistant_id,
                instructions=additional_instructions if additional_instructions else None,
                tools=tools_config
            )

            if run:
                st.info(f"Run created: {run.id}")

                # Poll for completion
                progress_bar = st.progress(0)
                status_text = st.empty()

                for i in range(30):  # Max 30 seconds
                    run_status = service.get_run_status(run_thread_id, run.id)
                    status_text.text(f"Status: {run_status.status}")

                    if run_status.status == 'completed':
                        progress_bar.progress(100)
                        st.success("Run completed!")

                        # Get and display messages
                        messages = service.get_thread_messages(run_thread_id)
                        if messages:
                            st.subheader("Response:")
                            # Display the most recent assistant message
                            for msg in messages:
                                if msg.role == "assistant":
                                    st.markdown(msg.content[0].text.value if msg.content else 'No content')
                                    break
                        break
                    elif run_status.status in ['failed', 'cancelled', 'expired']:
                        st.error(f"Run {run_status.status}")
                        break

                    progress_bar.progress(min((i + 1) * 3, 90))
                    time.sleep(1)

with tab4:
    st.header("System Prompt Templates")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Default System Prompt")
        st.text_area(
            "Template",
            value=DEFAULT_SYSTEM_PROMPT,
            height=300,
            disabled=True
        )

    with col2:
        st.subheader("NARM Therapy Prompt")
        st.text_area(
            "Template",
            value=DEFAULT_NARM_PROMPT,
            height=300,
            disabled=True
        )

# Status bar
st.divider()
col1, col2, col3 = st.columns(3)
with col1:
    st.caption(f"🟢 Service: {'Connected' if st.session_state.get('service_initialized') else 'Not Connected'}")
with col2:
    st.caption(f"🤖 Last Assistant: {st.session_state.get('last_assistant_id', 'None')[:16]}..." if st.session_state.get('last_assistant_id') else "🤖 No assistant")
with col3:
    st.caption(f"💬 Last Thread: {st.session_state.get('last_thread_id', 'None')[:16]}..." if st.session_state.get('last_thread_id') else "💬 No thread")

# Cleanup tools
with st.expander("🧹 Cleanup Tools"):
    st.warning("Delete test resources when done")

    if st.button("Delete All Test Assistants", type="secondary"):
        # This would need implementation in the service
        st.info("Feature to list and delete all assistants would go here")