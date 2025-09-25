"""
Configuration Manager Test UI
Streamlit app for testing configuration management
"""
import streamlit as st
import json
from config_manager import ConfigManager, get_config_manager
import os

st.set_page_config(page_title="Config Manager Test", page_icon="⚙️")

st.title("⚙️ Configuration Manager Module Test")
st.markdown("Test interface for centralized configuration management")

# Initialize config manager
if "config_manager" not in st.session_state:
    st.session_state.config_manager = get_config_manager("test_config.json")

config = st.session_state.config_manager

# Sidebar
with st.sidebar:
    st.header("🗂️ Config Files")

    config_file = st.text_input("Config File", value="test_config.json")

    if st.button("Load Config File"):
        st.session_state.config_manager = ConfigManager(config_file)
        config = st.session_state.config_manager
        st.success(f"Loaded: {config_file}")
        st.rerun()

    st.divider()

    if st.button("🔄 Reload Current"):
        config.load_config()
        st.success("Configuration reloaded")
        st.rerun()

    if st.button("💾 Save Current"):
        if config.save_config():
            st.success("Configuration saved")

    if st.button("🔄 Reset to Defaults"):
        if config.reset_to_defaults():
            st.success("Reset to defaults")
            st.rerun()

# Main tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "View Config", "Edit Config", "Module Configs", "Import/Export", "Templates", "API Validation"
])

with tab1:
    st.header("Current Configuration")

    # Display full config
    config_view = config.config

    # Search/filter
    search = st.text_input("Search configuration keys", placeholder="e.g., openai, tts, assistant")

    if search:
        filtered = {}
        for key, value in config_view.items():
            if search.lower() in key.lower():
                filtered[key] = value
        config_view = filtered

    # Display as JSON
    st.json(config_view)

    # Display specific values
    st.divider()
    st.subheader("Quick Access")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**OpenAI Settings**")
        st.text(f"Model: {config.get('openai.model', 'Not set')}")
        st.text(f"Temperature: {config.get('openai.temperature', 'Not set')}")
        st.text(f"Max Tokens: {config.get('openai.max_tokens', 'Not set')}")

    with col2:
        st.markdown("**Assistant Settings**")
        st.text(f"Name: {config.get('assistant.name', 'Not set')}")
        st.text(f"Tools: {config.get('assistant.tools', [])}")
        vs_id = config.get('assistant.vector_store_id', 'Not set')
        if vs_id and len(vs_id) > 20:
            st.text(f"VS ID: {vs_id[:20]}...")
        else:
            st.text(f"VS ID: {vs_id}")

    with col3:
        st.markdown("**UI Settings**")
        st.text(f"Voice: {config.get('ui.enable_voice', False)}")
        st.text(f"Image: {config.get('ui.enable_image', False)}")
        st.text(f"Autoplay: {config.get('ui.autoplay_tts', False)}")

with tab2:
    st.header("Edit Configuration")

    edit_method = st.radio("Edit Method", ["By Section", "By Key", "Raw JSON"])

    if edit_method == "By Section":
        section = st.selectbox("Select Section", list(config.config.keys()))

        if section:
            st.subheader(f"Editing: {section}")

            # Get current section values
            section_data = config.config.get(section, {})

            # Create form for section
            updated_data = {}

            for key, value in section_data.items():
                if isinstance(value, bool):
                    updated_data[key] = st.checkbox(f"{key}", value=value, key=f"{section}_{key}")
                elif isinstance(value, (int, float)):
                    if isinstance(value, float):
                        updated_data[key] = st.number_input(f"{key}", value=value, key=f"{section}_{key}")
                    else:
                        updated_data[key] = st.number_input(f"{key}", value=value, step=1, key=f"{section}_{key}")
                elif isinstance(value, list):
                    # Handle list input as comma-separated
                    list_str = ", ".join(str(v) for v in value)
                    new_list_str = st.text_input(f"{key} (comma-separated)", value=list_str, key=f"{section}_{key}")
                    updated_data[key] = [v.strip() for v in new_list_str.split(",") if v.strip()]
                else:
                    updated_data[key] = st.text_input(f"{key}", value=str(value) if value else "", key=f"{section}_{key}")

            if st.button(f"Update {section}", type="primary"):
                if config.update_section(section, updated_data):
                    st.success(f"Updated {section} configuration")
                    st.rerun()

    elif edit_method == "By Key":
        key_path = st.text_input("Configuration Key (dot notation)",
                                placeholder="e.g., openai.model, tts.voice")

        if key_path:
            current_value = config.get(key_path)
            st.info(f"Current value: {current_value}")

            # Determine input type based on current value
            if isinstance(current_value, bool):
                new_value = st.checkbox("New Value", value=current_value)
            elif isinstance(current_value, (int, float)):
                new_value = st.number_input("New Value", value=current_value)
            elif isinstance(current_value, list):
                list_str = ", ".join(str(v) for v in current_value)
                new_list_str = st.text_input("New Value (comma-separated)", value=list_str)
                new_value = [v.strip() for v in new_list_str.split(",") if v.strip()]
            else:
                new_value = st.text_input("New Value", value=str(current_value) if current_value else "")

            if st.button("Update Value", type="primary"):
                if config.set(key_path, new_value):
                    st.success(f"Updated {key_path}")
                    st.rerun()

    else:  # Raw JSON
        st.warning("⚠️ Be careful when editing raw JSON. Invalid JSON will be rejected.")

        json_str = st.text_area("Configuration JSON",
                               value=json.dumps(config.config, indent=2),
                               height=500)

        if st.button("Apply JSON", type="primary"):
            try:
                new_config = json.loads(json_str)
                config.config = new_config
                if config.save_config():
                    st.success("Configuration updated from JSON")
                    st.rerun()
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON: {str(e)}")

with tab3:
    st.header("Module-Specific Configurations")

    modules = ["openai", "assistant", "tts", "stt", "chat", "ui", "session"]

    for module in modules:
        with st.expander(f"📦 {module.upper()} Configuration"):
            module_config = config.get_module_config(module)

            if module_config:
                st.json(module_config)

                # Quick edit for common settings
                if module == "openai":
                    new_api_key = st.text_input("API Key", type="password",
                                               value=module_config.get("api_key", ""),
                                               key=f"quick_{module}_api")
                    if st.button(f"Update API Key", key=f"update_{module}_api"):
                        config.set("openai.api_key", new_api_key)
                        st.success("API Key updated")

                elif module == "tts":
                    new_voice = st.selectbox("Voice",
                                           ["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
                                           index=["alloy", "echo", "fable", "onyx", "nova", "shimmer"].index(
                                               module_config.get("voice", "alloy")),
                                           key=f"quick_{module}_voice")
                    if st.button(f"Update Voice", key=f"update_{module}_voice"):
                        config.set("tts.voice", new_voice)
                        st.success("Voice updated")

            else:
                st.info(f"No configuration for {module}")

with tab4:
    st.header("Import/Export Configuration")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📥 Export")

        export_format = st.selectbox("Export Format", ["JSON", "Pretty JSON"])

        if export_format == "JSON":
            export_data = json.dumps(config.config)
        else:
            export_data = json.dumps(config.config, indent=2)

        st.download_button(
            label="📥 Download Configuration",
            data=export_data,
            file_name="chatbot_config.json",
            mime="application/json",
            use_container_width=True
        )

        # Display current config
        st.text_area("Current Configuration", value=export_data, height=300, disabled=True)

    with col2:
        st.subheader("📤 Import")

        uploaded_file = st.file_uploader("Choose config file", type="json")

        if uploaded_file:
            try:
                config_str = uploaded_file.read().decode('utf-8')
                imported_config = json.loads(config_str)

                st.json(imported_config)

                if st.button("Apply Imported Configuration", type="primary"):
                    if config.import_config(config_str):
                        st.success("Configuration imported successfully!")
                        st.rerun()
                    else:
                        st.error("Failed to import configuration")

            except Exception as e:
                st.error(f"Error reading file: {str(e)}")

        # Manual JSON import
        st.divider()
        json_input = st.text_area("Or paste JSON here", height=200, key="json_import")

        if json_input and st.button("Import from JSON", key="import_json"):
            try:
                if config.import_config(json_input):
                    st.success("Configuration imported!")
                    st.rerun()
            except Exception as e:
                st.error(f"Import failed: {str(e)}")

with tab5:
    st.header("Configuration Templates")

    templates = {
        "NARM Therapy Chatbot": {
            "openai": {
                "api_key": "",
                "model": "gpt-4o",
                "temperature": 0.7,
                "max_tokens": 500
            },
            "assistant": {
                "name": "NARM Therapy Assistant",
                "instructions": "You are a compassionate and intuitive NARM therapy assistant...",
                "tools": ["file_search", "code_interpreter"],
                "vector_store_id": "vs_67a7a6bd68d48191a4f446ddeaec2e2b"
            },
            "tts": {
                "model": "tts-1",
                "voice": "alloy",
                "speed": 1.0,
                "format": "mp3"
            },
            "ui": {
                "enable_voice": True,
                "enable_image": True,
                "autoplay_tts": True
            }
        },
        "Code Assistant": {
            "openai": {
                "api_key": "",
                "model": "gpt-4",
                "temperature": 0.2,
                "max_tokens": 1000
            },
            "assistant": {
                "name": "Code Helper",
                "instructions": "You are an expert programmer. Help with code questions and debugging.",
                "tools": ["code_interpreter"],
                "vector_store_id": None
            },
            "ui": {
                "enable_voice": False,
                "enable_image": True,
                "autoplay_tts": False
            }
        },
        "Customer Support Bot": {
            "openai": {
                "api_key": "",
                "model": "gpt-3.5-turbo",
                "temperature": 0.5,
                "max_tokens": 300
            },
            "assistant": {
                "name": "Support Agent",
                "instructions": "You are a helpful customer support agent. Be polite and professional.",
                "tools": ["file_search"],
                "vector_store_id": None
            },
            "ui": {
                "enable_voice": True,
                "enable_image": False,
                "autoplay_tts": False
            }
        }
    }

    selected_template = st.selectbox("Select Template", list(templates.keys()))

    if selected_template:
        st.json(templates[selected_template])

        col1, col2, col3 = st.columns([1, 1, 1])

        with col2:
            if st.button("Apply Template", type="primary", use_container_width=True):
                # Merge template with current config
                template_config = templates[selected_template]
                for section, values in template_config.items():
                    config.update_section(section, values)
                st.success(f"Applied {selected_template} template!")
                st.rerun()

with tab6:
    st.header("API Validation")

    st.markdown("Test your configuration settings")

    # API Key validation
    st.subheader("🔑 API Key Validation")

    api_key = config.get("openai.api_key", "")

    if api_key:
        if api_key.startswith("sk-"):
            st.success("✅ API key format looks valid")

            if st.button("Test API Connection"):
                try:
                    from openai import OpenAI
                    test_client = OpenAI(api_key=api_key)
                    # Try a simple API call
                    test_client.models.list()
                    st.success("✅ API connection successful!")
                except Exception as e:
                    st.error(f"❌ API connection failed: {str(e)}")
        else:
            st.error("❌ Invalid API key format")
    else:
        st.warning("⚠️ No API key configured")

    # Configuration validation
    st.divider()
    st.subheader("📋 Configuration Validation")

    validation_results = []

    # Check required fields
    required_fields = [
        ("openai.api_key", "API Key"),
        ("openai.model", "Model"),
        ("assistant.name", "Assistant Name"),
        ("assistant.instructions", "Assistant Instructions")
    ]

    for field, name in required_fields:
        value = config.get(field)
        if value:
            validation_results.append(f"✅ {name}: Configured")
        else:
            validation_results.append(f"❌ {name}: Missing")

    for result in validation_results:
        st.text(result)

    # Model compatibility check
    st.divider()
    st.subheader("🤖 Model Compatibility")

    model = config.get("openai.model", "")
    if model:
        if model in ["gpt-4o", "gpt-4", "gpt-3.5-turbo"]:
            st.success(f"✅ Model '{model}' is supported")
        else:
            st.warning(f"⚠️ Model '{model}' may not be fully supported")

# Status bar
st.divider()
col1, col2, col3 = st.columns(3)
with col1:
    st.caption(f"📁 Config File: {config.config_file}")
with col2:
    st.caption(f"✅ Valid API: {config.validate_api_key()}")
with col3:
    config_size = len(json.dumps(config.config))
    st.caption(f"📊 Config Size: {config_size} bytes")