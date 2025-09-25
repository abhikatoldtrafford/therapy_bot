
"""
Vector Store / RAG Service Test UI
Streamlit app for testing vector store and retrieval functionality
"""
import streamlit as st
from vector_service import VectorStoreService
import pandas as pd

st.set_page_config(page_title="Vector Store Test", page_icon="📚")

st.title("📚 Vector Store / RAG Module Test")
st.markdown("Test interface for vector store and document retrieval")

# Initialize vector service
if "vector_service" not in st.session_state:
    try:
        st.session_state.vector_service = VectorStoreService()
        st.session_state.service_initialized = True
    except Exception as e:
        st.error(f"Failed to initialize service: {str(e)}")
        st.session_state.service_initialized = False

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")

    api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")

    st.divider()
    st.subheader("Default Store")
    st.code("vs_67a7a6bd68d48191a4f446ddeaec2e2b", language="text")
    st.caption("NARM knowledge base from original code")

    if st.button("🔄 Reinitialize Service"):
        try:
            st.session_state.vector_service = VectorStoreService(api_key)
            st.session_state.service_initialized = True
            st.success("Service reinitialized!")
        except Exception as e:
            st.error(f"Failed: {str(e)}")
            st.session_state.service_initialized = False

if not st.session_state.get('service_initialized', False):
    st.error("⚠️ Service not initialized. Please configure API key in sidebar.")
    st.stop()

service = st.session_state.vector_service

# Main tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Create Store", "Manage Stores", "Upload Files", "Search", "Assistant Integration"])

with tab1:
    st.header("Create New Vector Store")

    with st.form("create_store_form"):
        store_name = st.text_input("Store Name", value="My Knowledge Base")
        store_description = st.text_area("Description (optional)", placeholder="Describe the purpose of this store...")

        col1, col2 = st.columns(2)
        with col1:
            store_type = st.selectbox("Store Type", ["General", "NARM Therapy", "Technical Docs", "Custom"])
        with col2:
            auto_index = st.checkbox("Auto-index on upload", value=True)

        if st.form_submit_button("Create Vector Store", type="primary"):
            try:
                store_id = service.create_vector_store(name=store_name)
                st.success(f"✅ Vector store created successfully!")
                st.code(store_id, language="text")
                st.session_state.last_store_id = store_id
            except Exception as e:
                st.error(f"Failed to create vector store: {str(e)}")

with tab2:
    st.header("Manage Vector Stores")

    if st.button("🔄 Refresh List"):
        st.rerun()

    try:
        stores = service.list_vector_stores(limit=20)

        if not stores:
            st.info("No vector stores found. Create one in the first tab!")
        else:
            st.markdown(f"**Total Stores:** {len(stores)}")

            # Create a dataframe for better display
            store_data = []
            for store in stores:
                store_data.append({
                    "Name": store.name,
                    "ID": store.id[:16] + "...",
                    "Created": store.created_at,
                    "Files": store.file_counts.total if hasattr(store, 'file_counts') else 0
                })

            df = pd.DataFrame(store_data)
            st.dataframe(df, use_container_width=True)

            st.divider()

            # Detailed view
            selected_store = st.selectbox("Select store for details",
                                         options=[s.id for s in stores],
                                         format_func=lambda x: next(s.name for s in stores if s.id == x))

            if selected_store:
                store_details = service.get_vector_store(selected_store)
                if store_details:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Store Details**")
                        st.text(f"ID: {store_details.id}")
                        st.text(f"Name: {store_details.name}")
                        st.text(f"Created: {store_details.created_at}")
                    with col2:
                        st.markdown("**Actions**")
                        new_name = st.text_input("Rename store", value=store_details.name)
                        if st.button("Update Name"):
                            if service.update_vector_store(selected_store, name=new_name):
                                st.success("Updated!")
                                st.rerun()

                        if st.button("🗑️ Delete Store", type="secondary"):
                            if service.delete_vector_store(selected_store):
                                st.success("Deleted!")
                                st.rerun()

                    # List files in store
                    st.subheader("Files in Store")
                    files = service.list_files_in_store(selected_store)
                    if files:
                        st.markdown(f"**Total Files:** {len(files)}")
                        for file in files:
                            col3, col4 = st.columns([3, 1])
                            with col3:
                                st.text(f"📄 {file.id}")
                            with col4:
                                if st.button("Remove", key=f"rm_{file.id}"):
                                    if service.remove_file_from_store(selected_store, file.id):
                                        st.success("Removed!")
                                        st.rerun()
                    else:
                        st.info("No files in this store yet.")

    except Exception as e:
        st.error(f"Failed to manage stores: {str(e)}")

with tab3:
    st.header("Upload Files to Vector Store")

    store_id = st.text_input("Vector Store ID",
                            value=st.session_state.get('last_store_id', service.default_vector_store_id),
                            placeholder="vs_...")

    upload_method = st.radio("Upload Method", ["File Upload", "Text Input", "URL (placeholder)"])

    if upload_method == "File Upload":
        uploaded_files = st.file_uploader("Choose files",
                                         accept_multiple_files=True,
                                         type=['txt', 'pdf', 'md', 'json'])

        if uploaded_files and st.button("Upload Files", type="primary"):
            progress = st.progress(0)
            for i, file in enumerate(uploaded_files):
                try:
                    file_id = service.upload_file_to_store(
                        vector_store_id=store_id,
                        file_content=file.read(),
                        file_name=file.name
                    )
                    st.success(f"✅ Uploaded: {file.name}")
                    progress.progress((i + 1) / len(uploaded_files))
                except Exception as e:
                    st.error(f"Failed to upload {file.name}: {str(e)}")

    elif upload_method == "Text Input":
        text_title = st.text_input("Document Title", placeholder="My Document")
        text_content = st.text_area("Document Content", height=300,
                                   placeholder="Paste or type your content here...")

        if text_content and st.button("Upload Text", type="primary"):
            try:
                # Convert text to bytes
                content_bytes = text_content.encode('utf-8')
                file_name = f"{text_title or 'document'}.txt"

                file_id = service.upload_file_to_store(
                    vector_store_id=store_id,
                    file_content=content_bytes,
                    file_name=file_name
                )
                st.success(f"✅ Text uploaded as {file_name}")
                st.code(file_id, language="text")
            except Exception as e:
                st.error(f"Failed to upload text: {str(e)}")

    else:  # URL placeholder
        st.info("URL upload is a placeholder feature for future implementation")
        url = st.text_input("Enter URL", placeholder="https://example.com/document.pdf")
        if url and st.button("Fetch and Upload"):
            st.warning("This feature would fetch content from URL and upload to vector store")

with tab4:
    st.header("Search Vector Store")

    search_store_id = st.text_input("Vector Store ID to Search",
                                   value=st.session_state.get('last_store_id', service.default_vector_store_id),
                                   placeholder="vs_...")

    query = st.text_input("Search Query", placeholder="Enter your search query...")
    num_results = st.slider("Number of Results", 1, 20, 5)

    if query and st.button("🔍 Search", type="primary"):
        with st.spinner("Searching..."):
            try:
                results = service.search_vector_store(search_store_id, query, num_results)

                if results:
                    st.success(f"Found {len(results)} results")
                    for i, result in enumerate(results):
                        with st.expander(f"Result {i+1} - Score: {result['score']:.2f}"):
                            st.json(result)
                else:
                    st.info("No results found")

            except Exception as e:
                st.error(f"Search failed: {str(e)}")

    st.divider()
    st.subheader("Test Queries")
    st.markdown("Sample queries to test search functionality:")

    test_queries = [
        "NARM therapy principles",
        "emotional regulation",
        "attachment theory",
        "trauma healing",
        "somatic experiencing"
    ]

    for tq in test_queries:
        if st.button(f"Try: {tq}", key=f"tq_{tq}"):
            st.session_state.search_query = tq
            st.rerun()

with tab5:
    st.header("Assistant Integration")

    col1, col2 = st.columns(2)

    with col1:
        assistant_id = st.text_input("Assistant ID", placeholder="asst_...")
    with col2:
        attach_store_id = st.text_input("Vector Store ID",
                                       value=st.session_state.get('last_store_id', service.default_vector_store_id),
                                       placeholder="vs_...")

    if assistant_id and attach_store_id:
        st.markdown(f"""
        **Configuration:**
        - Assistant: `{assistant_id[:16]}...`
        - Vector Store: `{attach_store_id[:16]}...`
        """)

        if st.button("🔗 Attach Vector Store to Assistant", type="primary"):
            try:
                if service.attach_to_assistant(assistant_id, attach_store_id):
                    st.success("✅ Vector store attached to assistant!")
                    st.balloons()
                else:
                    st.error("Failed to attach vector store")
            except Exception as e:
                st.error(f"Error: {str(e)}")

    st.divider()
    st.subheader("Integration Code Example")

    code = '''
# Attach vector store to assistant
service.attach_to_assistant(
    assistant_id="asst_xxx",
    vector_store_id="vs_xxx"
)

# In assistant creation
assistant = client.beta.assistants.create(
    name="My Assistant",
    instructions="...",
    tools=[{"type": "file_search"}],
    tool_resources={
        "file_search": {
            "vector_store_ids": ["vs_xxx"]
        }
    }
)
'''
    st.code(code, language="python")

# Status bar
st.divider()
col1, col2, col3 = st.columns(3)
with col1:
    st.caption(f"🟢 Service: {'Connected' if st.session_state.get('service_initialized') else 'Not Connected'}")
with col2:
    st.caption(f"📚 Default Store: {service.default_vector_store_id[:16]}...")
with col3:
    stores = service.list_vector_stores(limit=100)
    st.caption(f"📊 Total Stores: {len(stores)}")