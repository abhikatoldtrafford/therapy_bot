"""

# Add parent directory to path to import sibling modules
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Import modules
from session_manager import SessionManager

Session Manager Module Test UI
Streamlit app for testing the module functionality
"""
import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path to import sibling modules

# Import local module

"""
Session Manager Test UI
Streamlit app for testing session management functionality
"""
import streamlit as st
import json
from session_service import SessionManager

st.set_page_config(page_title="Session Manager Test", page_icon="📋")

st.title("📋 Session Manager Module Test")
st.markdown("Test interface for session management functionality")

# Initialize session manager
if "session_manager" not in st.session_state:
    st.session_state.session_manager = SessionManager("test_sessions.json")

manager = st.session_state.session_manager

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    sessions_file = st.text_input("Sessions File", value="test_sessions.json")
    if st.button("🔄 Reinitialize Manager"):
        st.session_state.session_manager = SessionManager(sessions_file)
        st.rerun()

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(["Create Session", "View Sessions", "Update Session", "Delete Session"])

with tab1:
    st.header("Create New Session")
    with st.form("create_session_form"):
        st.subheader("User Information")
        name = st.text_input("Name", placeholder="John Doe")
        email = st.text_input("Email", placeholder="john@example.com")
        focus_today = st.text_input("Focus Today", placeholder="What's the focus?")
        desired_outcome = st.text_input("Desired Outcome", placeholder="What outcome?")
        current_challenges = st.text_input("Current Challenges", placeholder="What challenges?")

        st.subheader("Additional Configuration")
        model_name = st.text_input("Model Name", value="gpt-4o")
        temperature = st.slider("Temperature", 0.0, 2.0, 0.7)
        max_tokens = st.number_input("Max Tokens", value=500)

        if st.form_submit_button("Create Session", type="primary"):
            session_id = manager.create_session(
                name=name,
                email=email,
                focus_today=focus_today,
                desired_outcome=desired_outcome,
                current_challenges=current_challenges,
                model_config={
                    "model": model_name,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
            )
            st.success(f"✅ Session created with ID: `{session_id}`")
            st.code(session_id, language="text")

with tab2:
    st.header("View All Sessions")

    sessions = manager.list_sessions()

    if not sessions:
        st.info("No sessions found. Create one in the first tab!")
    else:
        st.markdown(f"**Total Sessions:** {len(sessions)}")

        for session_id, session_data in sessions.items():
            with st.expander(f"Session: {session_id[:8]}..."):
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Session Info**")
                    st.text(f"Assistant ID: {session_data.get('assistant_id', 'Not set')}")
                    st.text(f"Thread ID: {session_data.get('thread_id', 'Not set')}")
                    st.text(f"Vector Store ID: {session_data.get('session_vs_id', 'Not set')}")
                    st.text(f"File IDs: {len(session_data.get('file_ids', []))}")

                with col2:
                    st.markdown("**User Info**")
                    user_info = session_data.get('user_info', {})
                    for key, value in user_info.items():
                        if key != 'model_config':
                            st.text(f"{key}: {value}")

                st.markdown("**Full JSON**")
                st.json(session_data)

with tab3:
    st.header("Update Session")

    sessions = manager.list_sessions()
    if not sessions:
        st.info("No sessions to update. Create one first!")
    else:
        session_ids = list(sessions.keys())
        selected_session = st.selectbox("Select Session",
                                       options=session_ids,
                                       format_func=lambda x: f"{x[:8]}... - {sessions[x].get('user_info', {}).get('name', 'Unknown')}")

        if selected_session:
            current_data = manager.get_session(selected_session)
            st.json(current_data)

            st.subheader("Update Fields")
            update_type = st.radio("What to update?",
                                  ["Assistant ID", "Thread ID", "Vector Store ID", "Add File ID", "User Info"])

            if update_type == "Assistant ID":
                new_assistant_id = st.text_input("New Assistant ID")
                if st.button("Update"):
                    if manager.update_session(selected_session, assistant_id=new_assistant_id):
                        st.success("✅ Updated successfully!")
                        st.rerun()

            elif update_type == "Thread ID":
                new_thread_id = st.text_input("New Thread ID")
                if st.button("Update"):
                    if manager.update_session(selected_session, thread_id=new_thread_id):
                        st.success("✅ Updated successfully!")
                        st.rerun()

            elif update_type == "Vector Store ID":
                new_vs_id = st.text_input("New Vector Store ID")
                if st.button("Update"):
                    if manager.update_session(selected_session, session_vs_id=new_vs_id):
                        st.success("✅ Updated successfully!")
                        st.rerun()

            elif update_type == "Add File ID":
                new_file_id = st.text_input("New File ID to Add")
                if st.button("Add"):
                    file_ids = current_data.get('file_ids', [])
                    file_ids.append(new_file_id)
                    if manager.update_session(selected_session, file_ids=file_ids):
                        st.success("✅ File ID added!")
                        st.rerun()

            elif update_type == "User Info":
                field_name = st.text_input("Field Name")
                field_value = st.text_input("Field Value")
                if st.button("Update"):
                    user_info = {field_name: field_value}
                    if manager.update_session(selected_session, user_info=user_info):
                        st.success("✅ User info updated!")
                        st.rerun()

with tab4:
    st.header("Delete Session")

    sessions = manager.list_sessions()
    if not sessions:
        st.info("No sessions to delete.")
    else:
        col1, col2 = st.columns([3, 1])

        with col1:
            session_to_delete = st.selectbox("Select Session to Delete",
                                            options=list(sessions.keys()),
                                            format_func=lambda x: f"{x[:8]}... - {sessions[x].get('user_info', {}).get('name', 'Unknown')}")

        with col2:
            st.write("")  # Spacer
            st.write("")  # Spacer
            if st.button("🗑️ Delete", type="secondary"):
                if manager.delete_session(session_to_delete):
                    st.success("✅ Session deleted!")
                    st.rerun()

        st.divider()

        st.warning("⚠️ Danger Zone")
        if st.button("🗑️ Clear All Sessions", type="secondary"):
            manager.clear_all_sessions()
            st.success("✅ All sessions cleared!")
            st.rerun()

# Display current state
st.divider()
st.subheader("📊 Current State Summary")
sessions = manager.list_sessions()
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Sessions", len(sessions))
with col2:
    st.metric("Active Sessions", sum(1 for s in sessions.values() if s.get('thread_id')))
with col3:
    st.metric("Sessions with Files", sum(1 for s in sessions.values() if s.get('file_ids')))