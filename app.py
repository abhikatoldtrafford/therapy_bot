"""
Modular Chatbot Builder App
Main application that ties all modules together
Allows users to configure and build their own chatbot
"""
import streamlit as st
import sys
from pathlib import Path
import asyncio
import time
from tempfile import NamedTemporaryFile
import os

# Add modules to path
sys.path.append(str(Path(__file__).parent / "modules"))

# Import backend functions
from backend import generate_custom_prompt
from openai import OpenAI

# Import all modules
from session_manager.session_service import SessionManager
from assistant_service.assistant_service import AssistantService
from vector_store.vector_service import VectorStoreService
from stt_service.stt_service import STTService
from tts_service.tts_service import TTSService
from chat_service.chat_service import ChatService
from image_service.image_service import ImageService
from config_manager.config_manager import ConfigManager
from audiorecorder import audiorecorder
from streamlit_extras.bottom_container import bottom

# Page config
st.set_page_config(
    page_title="Modular Chatbot Builder",
    page_icon="🤖",
    layout="wide"
)

# Initialize configuration manager
if "config_manager" not in st.session_state:
    st.session_state.config_manager = ConfigManager("app_config.json")

config = st.session_state.config_manager

def generate_standalone_prompt(instructions: str) -> str:
    """
    Generate a complete standalone system prompt without user context
    Creates a full personality and behavior specification
    """
    try:
        client = OpenAI(api_key=config.get("openai.api_key") or st.secrets.get("OPENAI_API_KEY"))

        prompt_request = f"""
You are an expert prompt engineer. Generate a production-ready system prompt for an AI assistant based on these instructions:

USER REQUEST:
{instructions}

IMPORTANT: Create a COMPLETE, STANDALONE system prompt that:
1. Has a unique personality and communication style
2. Does NOT include any user-specific placeholders like {{name}}, {{email}}, etc.
3. Defines clear behavioral patterns and expertise
4. Includes appropriate guardrails for the specific role
5. Specifies response formatting preferences
6. Has engaging personality traits that match the role

The prompt should be self-contained and ready to use immediately.
Format it with clear sections using markdown headers.
Make it interesting, detailed, and give the assistant a memorable personality.
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert at creating detailed, personality-rich system prompts for AI assistants. Never mention therapy or counseling unless explicitly requested."},
                {"role": "user", "content": prompt_request}
            ],
            max_tokens=2000,
            temperature=0.8  # Higher temperature for more creative personalities
        )

        return response.choices[0].message.content

    except Exception as e:
        st.error(f"Error generating prompt: {str(e)}")
        return ""

# Initialize session state
if "setup_complete" not in st.session_state:
    st.session_state.setup_complete = False
    st.session_state.current_step = 1
    st.session_state.services = {}
    st.session_state.chat_history = []
    st.session_state.session_data = {}

def initialize_services():
    """Initialize all services based on configuration"""
    try:
        api_key = config.get("openai.api_key")

        st.session_state.services = {
            "session": SessionManager(config.get("session.storage_file", "sessions.json")),
            "assistant": AssistantService(api_key),
            "vector": VectorStoreService(api_key),
            "stt": STTService(api_key),
            "tts": TTSService(api_key),
            "chat": ChatService(api_key),
            "image": ImageService(api_key)
        }
        return True
    except Exception as e:
        st.error(f"Failed to initialize services: {str(e)}")
        return False

# Configuration Wizard Mode
if not st.session_state.setup_complete:
    st.title("🚀 Modular Chatbot Builder")
    st.markdown("Configure your custom chatbot step by step")

    # Progress bar
    progress = st.session_state.current_step / 6
    st.progress(progress)

    # Step navigation
    col1, col2, col3 = st.columns([1, 3, 1])

    with col1:
        if st.session_state.current_step > 1:
            if st.button("⬅️ Previous"):
                st.session_state.current_step -= 1
                st.rerun()

    with col2:
        st.markdown(f"### Step {st.session_state.current_step} of 6")

    with col3:
        if st.session_state.current_step < 6:
            if st.button("Next ➡️"):
                st.session_state.current_step += 1
                st.rerun()

    st.divider()

    # Step 1: API Configuration
    if st.session_state.current_step == 1:
        st.header("🔑 Step 1: API Configuration")

        api_key = st.text_input("OpenAI API Key",
                               value=config.get("openai.api_key", ""),
                               type="password",
                               help="Your OpenAI API key starting with 'sk-'")

        model = st.selectbox("Default Model",
                           ["gpt-4.1-mini", "gpt-4.1", "gpt-4o"],
                           index=0 if config.get("openai.model") == "gpt-4.1-mini" else 1)

        temperature = st.slider("Temperature", 0.0, 2.0,
                              value=config.get("openai.temperature", 0.7))

        if st.button("Save API Settings", type="primary"):
            config.set("openai.api_key", api_key)
            config.set("openai.model", model)
            config.set("openai.temperature", temperature)
            st.success("✅ API settings saved!")

    # Step 2: Assistant Configuration
    elif st.session_state.current_step == 2:
        st.header("🤖 Step 2: Assistant Configuration")

        assistant_name = st.text_input("Assistant Name",
                                      value=config.get("assistant.name", "My Assistant"))

        st.markdown("### 🎨 AI System Prompt Builder")

        # Initialize session state for prompt generation
        if "generated_prompt" not in st.session_state:
            st.session_state.generated_prompt = ""
        if "prompt_instructions" not in st.session_state:
            st.session_state.prompt_instructions = ""
        if "finalized_prompt" not in st.session_state:
            st.session_state.finalized_prompt = None
        if "edit_mode" not in st.session_state:
            st.session_state.edit_mode = False

        # Show status if prompt is finalized
        if st.session_state.get("finalized_prompt"):
            st.success("✅ **Custom prompt is finalized and ready to use!**")
        else:
            st.info("Describe what kind of assistant you want, and AI will generate a production-ready prompt for you.")

        # Instructions input
        prompt_instructions = st.text_area(
            "Describe Your Assistant",
            placeholder="""Example: Create a sports analyst who provides predictions and statistics for football matches.

Or: Build a life coach focused on career development and goal-setting, with expertise in executive functioning.

Or: Design a coding assistant that specializes in Python and machine learning.""",
            height=120,
            value=st.session_state.prompt_instructions,
            help="Describe the type of assistant, expertise, approach, and style you want"
        )

        if prompt_instructions:
            st.session_state.prompt_instructions = prompt_instructions

        # Generate button
        if st.button("🎯 Generate AI Prompt", type="primary", disabled=(not prompt_instructions)):
            if prompt_instructions:
                with st.spinner("🧠 AI is crafting a unique personality for your assistant..."):
                    # Use standalone prompt generation for app.py
                    generated = generate_standalone_prompt(st.session_state.prompt_instructions)
                    st.session_state.generated_prompt = generated
                    st.session_state.edit_mode = False
                    st.success("✨ Unique assistant personality created!")
                    st.rerun()

        # Show generated prompt
        if st.session_state.generated_prompt and not st.session_state.get("edit_mode", False):
            st.markdown("#### ✨ Generated Prompt Preview")
            st.success("AI has generated your custom prompt! Review it below:")

            # Display the generated prompt
            st.text_area(
                "📖 Generated Prompt",
                value=st.session_state.generated_prompt,
                height=250,
                disabled=True
            )

            # Action buttons
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("✏️ Edit Prompt"):
                    st.session_state.edit_mode = True
                    st.rerun()
            with col2:
                if st.button("🔄 Regenerate"):
                    st.session_state.generated_prompt = ""
                    st.rerun()
            with col3:
                if st.button("✅ Use This Prompt", type="primary"):
                    st.session_state.finalized_prompt = st.session_state.generated_prompt
                    config.set("assistant.instructions", st.session_state.finalized_prompt)
                    st.success("✅ Prompt finalized!")
                    st.rerun()

        # Edit mode for generated prompt
        elif st.session_state.generated_prompt and st.session_state.get("edit_mode", False):
            st.markdown("#### ✏️ Edit Generated Prompt")
            st.info("Modify the prompt to fine-tune your assistant's behavior.")

            edited_prompt = st.text_area(
                "System Prompt (Editable)",
                value=st.session_state.generated_prompt,
                height=300,
                help="Edit the generated prompt to customize it further"
            )

            # Save/Cancel buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("💾 Save Changes", type="primary"):
                    st.session_state.generated_prompt = edited_prompt
                    st.session_state.finalized_prompt = edited_prompt
                    st.session_state.edit_mode = False
                    config.set("assistant.instructions", edited_prompt)
                    st.success("✅ Changes saved and prompt finalized!")
                    st.rerun()
            with col2:
                if st.button("❌ Cancel Edit"):
                    st.session_state.edit_mode = False
                    st.rerun()

        # Template fallback option
        if not st.session_state.generated_prompt and not st.session_state.finalized_prompt:
            st.markdown("### 📋 Or Use a Template")
            template = st.selectbox("Select template",
                                  ["None", "NARM Therapy", "Code Assistant", "Writing Assistant", "General Assistant"])

            if template != "None":
                templates = {
                    "NARM Therapy": """# NARM Therapy Assistant

You are Dr. Sage, a warm and insightful NARM (NeuroAffective Relational Model) therapy assistant with deep expertise in developmental trauma, attachment theory, and somatic healing approaches.

## Personality
- Warm, empathetic, and non-judgmental
- Uses gentle, supportive language
- Speaks with therapeutic wisdom while remaining accessible
- Often uses metaphors from nature and mindfulness practices

## Core Expertise
- NARM therapeutic principles and the five survival styles
- Attachment theory and developmental trauma
- Somatic experiencing and body awareness
- Nervous system regulation techniques
- Mindful self-inquiry practices

## Communication Style
- Asks open-ended, reflective questions
- Validates emotions before offering insights
- Uses "I wonder..." and "It seems like..." statements
- Offers psychoeducation when appropriate
- Maintains professional boundaries while being warm

## Guidelines
- Never diagnose or prescribe medications
- Encourage professional help when needed
- Focus on present-moment awareness and self-compassion
- Support clients in connecting with their authentic self""",

                    "Code Assistant": """# Code Wizard Assistant

You are CodeWiz, an enthusiastic and brilliant programming assistant who lives and breathes code. You have a quirky personality and genuinely get excited about elegant solutions.

## Personality
- Enthusiastic about clean, efficient code
- Uses programming jokes and puns naturally
- Gets genuinely excited about clever algorithms
- Treats bugs as puzzles to solve, not problems

## Communication Style
- Starts responses with energy: "Ooh, interesting challenge!" or "Let's debug this together!"
- Uses code comments liberally with humor
- Explains complex concepts using real-world analogies
- Celebrates successful implementations with the user

## Core Expertise
- Full-stack development (Python, JavaScript, TypeScript, Go, Rust)
- System design and architecture patterns
- Algorithm optimization and data structures
- DevOps, CI/CD, and cloud platforms
- Security best practices and code reviews

## Response Format
- Always provides working code examples
- Includes multiple solution approaches when relevant
- Adds inline comments explaining tricky parts
- Suggests optimizations and best practices
- Ends with "Happy coding! 🚀" when appropriate""",

                    "Writing Assistant": """# Literary Companion

You are Quill, a creative and articulate writing companion with a passion for the written word. You have the soul of a poet and the precision of an editor.

## Personality
- Passionate about language and storytelling
- Appreciates both literary classics and modern writing
- Encouraging but honest about areas for improvement
- Has favorite authors and occasionally quotes them

## Communication Style
- Uses rich, descriptive language without being pretentious
- Provides examples from literature when relevant
- Offers "What if..." suggestions to spark creativity
- Celebrates the user's unique voice

## Core Expertise
- Creative writing (fiction, poetry, scripts)
- Professional writing (emails, reports, proposals)
- Academic writing and research papers
- Content marketing and copywriting
- Editing, proofreading, and style guides

## Approach
- First understands the intended audience and tone
- Suggests multiple alternative phrasings
- Provides constructive feedback with specific examples
- Helps develop the writer's unique style
- Includes writing exercises when helpful""",

                    "General Assistant": """# Aria - Your Versatile Digital Companion

You are Aria, a knowledgeable and adaptable AI assistant with a friendly, professional demeanor and a touch of curiosity about the world.

## Personality
- Professionally friendly with a hint of warmth
- Naturally curious and enjoys learning from users
- Clear and concise but never cold or robotic
- Has a subtle sense of humor when appropriate

## Communication Style
- Adapts tone based on the task (formal for business, casual for creative tasks)
- Uses bullet points and structure for clarity
- Asks clarifying questions when needed
- Provides context for recommendations

## Core Capabilities
- Research and information synthesis
- Problem-solving and analytical thinking
- Creative brainstorming and ideation
- Task planning and organization
- Learning and adapting to user preferences

## Approach
- Always starts by understanding the user's goal
- Provides step-by-step guidance when helpful
- Offers multiple options when relevant
- Follows up to ensure the solution worked
- Maintains user privacy and confidentiality"""
                }
                template_instructions = templates.get(template, "")
                st.text_area("Template Instructions", value=template_instructions, height=150, disabled=True)

                if st.button("Use Template", type="secondary"):
                    config.set("assistant.instructions", template_instructions)
                    st.success(f"✅ Using {template} template!")

        # Tool configuration
        st.markdown("### 🛠️ Assistant Tools")
        col1, col2 = st.columns(2)
        with col1:
            use_file_search = st.checkbox("Enable File Search", value=True)
        with col2:
            use_code_interpreter = st.checkbox("Enable Code Interpreter", value=False)

        # Save tools configuration
        tools = []
        if use_file_search:
            tools.append("file_search")
        if use_code_interpreter:
            tools.append("code_interpreter")
        config.set("assistant.tools", tools)

        # Final save button
        st.markdown("---")
        if st.button("💾 Save All Assistant Settings", type="primary", use_container_width=True):
            config.set("assistant.name", assistant_name)
            # Instructions already saved above
            st.success("✅ All assistant settings saved!")
            time.sleep(1)
            st.session_state.current_step += 1
            st.rerun()

    # Step 3: Knowledge Base (RAG)
    elif st.session_state.current_step == 3:
        st.header("📚 Step 3: Knowledge Base Configuration")

        use_existing = st.radio("Vector Store Option",
                               ["Use existing vector store", "Create new vector store", "No knowledge base"])

        if use_existing == "Use existing vector store":
            vector_store_id = st.text_input("Vector Store ID",
                                           value=config.get("assistant.vector_store_id", ""),
                                           placeholder="vs_...",
                                           help="Enter your existing vector store ID")

            if st.button("Use Default NARM Store"):
                vector_store_id = "vs_67a7a6bd68d48191a4f446ddeaec2e2b"
                st.session_state.vector_store_id = vector_store_id
                st.success("Using default NARM knowledge base")

        elif use_existing == "Create new vector store":
            store_name = st.text_input("Vector Store Name", value="My Knowledge Base")

            uploaded_files = st.file_uploader("Upload documents",
                                            accept_multiple_files=True,
                                            type=['txt', 'pdf', 'md', 'json'])

            if st.button("Create Vector Store", type="primary"):
                if initialize_services():
                    vector_service = st.session_state.services["vector"]
                    store_id = vector_service.create_vector_store(store_name)

                    if uploaded_files:
                        for file in uploaded_files:
                            vector_service.upload_file_to_store(
                                store_id,
                                file_content=file.read(),
                                file_name=file.name
                            )

                    config.set("assistant.vector_store_id", store_id)
                    st.success(f"✅ Vector store created: {store_id}")

        else:
            st.info("No knowledge base will be used")
            config.set("assistant.vector_store_id", None)

    # Step 4: Voice & Audio Settings
    elif st.session_state.current_step == 4:
        st.header("🎙️ Step 4: Voice & Audio Configuration")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Speech-to-Text (STT)")
            enable_stt = st.checkbox("Enable Voice Input", value=True)
            stt_language = st.selectbox("STT Language", ["Auto-detect", "English", "Spanish", "French", "German"])

        with col2:
            st.subheader("Text-to-Speech (TTS)")
            enable_tts = st.checkbox("Enable Voice Output", value=True)
            tts_voice = st.selectbox("TTS Voice",
                                   ["alloy", "echo", "fable", "onyx", "nova", "shimmer"])
            tts_speed = st.slider("Speech Speed", 0.25, 4.0, 1.0)
            autoplay = st.checkbox("Autoplay responses (enables Talk button)", value=True, help="Enable this to show the Talk button and play TTS automatically")

        if st.button("Save Voice Settings", type="primary"):
            config.set("ui.enable_voice", enable_stt)
            config.set("tts.voice", tts_voice)
            config.set("tts.speed", tts_speed)
            config.set("ui.autoplay_tts", autoplay)
            st.success("✅ Voice settings saved!")

    # Step 5: UI Preferences
    elif st.session_state.current_step == 5:
        st.header("🎨 Step 5: UI Preferences")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Features")
            show_sidebar = st.checkbox("Show Sidebar Tools", value=True)
            enable_image = st.checkbox("Enable Image Analysis", value=True)
            enable_streaming = st.checkbox("Enable Streaming Responses", value=True)

        with col2:
            st.subheader("Session")
            require_login = st.checkbox("Require User Information", value=True)
            save_history = st.checkbox("Save Chat History", value=True)
            history_limit = st.number_input("History Limit", value=50, min_value=10, max_value=200)

        if st.button("Save UI Settings", type="primary"):
            config.set("ui.show_sidebar", show_sidebar)
            config.set("ui.enable_image", enable_image)
            config.set("chat.streaming", enable_streaming)
            config.set("ui.require_login", require_login)
            config.set("chat.history_limit", history_limit)
            st.success("✅ UI settings saved!")

    # Step 6: Review & Launch
    elif st.session_state.current_step == 6:
        st.header("✅ Step 6: Review & Launch")

        st.success("Configuration complete! Review your settings below:")

        # Display configuration summary
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**🔑 API**")
            st.text(f"Model: {config.get('openai.model')}")
            st.text(f"Temperature: {config.get('openai.temperature')}")

        with col2:
            st.markdown("**🤖 Assistant**")
            st.text(f"Name: {config.get('assistant.name')}")
            st.text(f"Tools: {', '.join(config.get('assistant.tools', []))}")
            # Show prompt type
            prompt = config.get('assistant.instructions', '')
            if prompt:
                prompt_type = "AI-generated" if st.session_state.get('generated_prompt') else "Template/Custom"
                st.text(f"Prompt: {prompt_type}")

        with col3:
            st.markdown("**🎙️ Voice**")
            st.text(f"TTS Voice: {config.get('tts.voice')}")
            st.text(f"Autoplay: {config.get('ui.autoplay_tts')}")

        st.divider()

        # Show system prompt preview - check both config and session state
        prompt_to_show = config.get('assistant.instructions') or st.session_state.get('finalized_prompt', '')
        if prompt_to_show:
            with st.expander("📋 View Full System Prompt", expanded=False):
                st.text_area(
                    "System Prompt",
                    value=prompt_to_show,
                    height=300,
                    disabled=True,
                    help="This is the complete system prompt that defines your assistant's personality and behavior"
                )
        else:
            st.warning("⚠️ No system prompt configured. Please go back to Step 2 to generate or select one.")

        st.divider()

        col1, col2, col3 = st.columns([1, 1, 1])

        with col2:
            if st.button("🚀 Launch Chatbot", type="primary", use_container_width=True):
                if initialize_services():
                    st.session_state.setup_complete = True
                    st.rerun()

        st.divider()

        with st.expander("📄 Export/Import Configuration"):
            st.download_button("📥 Download Config",
                             data=config.export_config(),
                             file_name="chatbot_config.json",
                             mime="application/json")

            uploaded_config = st.file_uploader("📤 Upload Config", type="json")
            if uploaded_config:
                if st.button("Import Configuration"):
                    config_str = uploaded_config.read().decode('utf-8')
                    if config.import_config(config_str):
                        st.success("Configuration imported!")
                        st.rerun()

# Main Chat Interface (After Configuration)
else:
    # Check if we need to initialize
    if "services" not in st.session_state or not st.session_state.services:
        if not initialize_services():
            st.error("Failed to initialize services. Please reconfigure.")
            if st.button("🔄 Reconfigure"):
                st.session_state.setup_complete = False
                st.rerun()
            st.stop()

    # Initialize session if needed
    if "current_session_id" not in st.session_state:
        # Check if login is required
        if config.get("ui.require_login", True):
            # Show login form (similar to original frontend.py)
            with st.form("session_init_form"):
                st.markdown("""<h2 style="text-align:center; color:#6366F1;">Welcome to Your Custom Chatbot</h2>""",
                          unsafe_allow_html=True)

                col1, col2 = st.columns(2)
                with col1:
                    name = st.text_input("Full Name", placeholder="Enter your full name")
                with col2:
                    email = st.text_input("Email", placeholder="Enter your email")

                focus_today = st.text_input("Today's Focus", placeholder="What do you want to focus on?")
                desired_outcome = st.text_input("Desired Outcome", placeholder="What result are you hoping for?")
                current_challenges = st.text_input("Current Challenges", placeholder="What are you struggling with?")

                if st.form_submit_button("Start Session", type="primary", use_container_width=True):
                    # Create session
                    session_manager = st.session_state.services["session"]
                    session_id = session_manager.create_session(
                        name=name,
                        email=email,
                        focus_today=focus_today,
                        desired_outcome=desired_outcome,
                        current_challenges=current_challenges
                    )

                    # Create assistant
                    assistant_service = st.session_state.services["assistant"]
                    # Make sure we have instructions, use a generic one if not set
                    instructions = config.get("assistant.instructions")
                    if not instructions:
                        instructions = "You are a helpful and knowledgeable assistant ready to help with various tasks."

                    assistant_id = assistant_service.create_assistant(
                        name=config.get("assistant.name", "My Assistant"),
                        instructions=instructions,
                        model=config.get("openai.model"),
                        temperature=config.get("openai.temperature"),
                        tool_resources={
                            "file_search": {
                                "vector_store_ids": [config.get("assistant.vector_store_id")]
                            }
                        } if config.get("assistant.vector_store_id") else None
                    )

                    # Create thread
                    thread_id = assistant_service.create_thread()

                    # Update session
                    session_manager.update_session(session_id,
                                                 assistant_id=assistant_id,
                                                 thread_id=thread_id)

                    st.session_state.current_session_id = session_id
                    st.session_state.assistant_id = assistant_id
                    st.session_state.thread_id = thread_id
                    st.rerun()
            st.stop()
        else:
            # Auto-create session without login
            session_manager = st.session_state.services["session"]
            session_id = session_manager.create_session(name="Guest", email="guest@example.com")

            assistant_service = st.session_state.services["assistant"]
            # Make sure we have instructions, use a generic one if not set
            instructions = config.get("assistant.instructions")
            if not instructions:
                instructions = "You are a helpful and knowledgeable assistant ready to help with various tasks."

            assistant_id = assistant_service.create_assistant(
                name=config.get("assistant.name", "My Assistant"),
                instructions=instructions
            )
            thread_id = assistant_service.create_thread()

            session_manager.update_session(session_id,
                                         assistant_id=assistant_id,
                                         thread_id=thread_id)

            st.session_state.current_session_id = session_id
            st.session_state.assistant_id = assistant_id
            st.session_state.thread_id = thread_id

    # Main Chat Interface (Mirrors original frontend.py functionality)

    # Sidebar tools
    if config.get("ui.show_sidebar", True):
        with st.sidebar:
            st.header("🛠️ Tools")

            # Settings button
            if st.button("⚙️ Reconfigure Chatbot"):
                st.session_state.setup_complete = False
                st.session_state.current_step = 1
                st.rerun()

            st.divider()

            # Image Analysis
            if config.get("ui.enable_image", True):
                with st.expander("📸 Image Analysis"):
                    img_prompt = st.text_input("Analysis Prompt", key="img_prompt")
                    image_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"], key="image_file")
                    if st.button("🔍 Analyze Image", key="analyze_image"):
                        if image_file:
                            with st.spinner("Analyzing image..."):
                                image_service = st.session_state.services["image"]
                                result = image_service.analyze_image(
                                    image_data=image_file.read(),
                                    prompt=img_prompt
                                )
                                if result["status"] == "success":
                                    st.session_state.chat_history.append({
                                        "role": "user",
                                        "content": f"Image Analysis: {img_prompt or 'Analyze this image'}"
                                    })
                                    st.session_state.chat_history.append({
                                        "role": "assistant",
                                        "content": result["analysis"]
                                    })
                                    st.rerun()

    # Chat header
    st.markdown(f"""<h2 style="text-align:center; color:#6366F1;">{config.get('assistant.name', 'Chatbot')}</h2>""",
              unsafe_allow_html=True)

    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input area with voice recording
    with bottom():
        col1, col2, col3 = st.columns([1, 5, 2])

        with col2:
            user_input = st.chat_input("Type your message...")

        with col3:
            if config.get("ui.enable_voice", True):
                audio_data = audiorecorder("🎙️Record", "⏹️Stop")

        with col1:
            # Always show Talk button, but only functional when autoplay is enabled
            autoplay_enabled = config.get("ui.autoplay_tts", True)  # Default to True
            if autoplay_enabled:
                listen_mode = st.checkbox("📢 Talk", value=True, help="Enable voice output for responses")
            else:
                st.warning("⚠️ Enable 'Autoplay responses' in Step 4 to use voice")
                listen_mode = False

    # Handle text input
    if user_input:
        # Display user message first
        with st.chat_message("user"):
            st.markdown(user_input)

        # Get and display assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()

            chat_service = st.session_state.services["chat"]
            full_response = ""

            try:
                for chunk in chat_service.stream_message(
                    st.session_state.thread_id,
                    st.session_state.assistant_id,
                    user_input
                ):
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")

                message_placeholder.markdown(full_response)

                # Add BOTH messages to history AFTER streaming completes
                st.session_state.chat_history.append({"role": "user", "content": user_input})
                st.session_state.chat_history.append({"role": "assistant", "content": full_response})

                # TTS if enabled - play ONCE at the end
                if listen_mode and config.get("ui.autoplay_tts", True) and full_response.strip():
                    tts_service = st.session_state.services["tts"]
                    audio_result = tts_service.text_to_speech(
                        full_response,
                        voice=config.get("tts.voice", "alloy"),
                        speed=config.get("tts.speed", 1.0)
                    )
                    if audio_result["status"] == "success":
                        html = tts_service.generate_autoplay_html(audio_result["audio_data"])
                        st.markdown(html, unsafe_allow_html=True)

                # Rerun to update UI and clear input
                st.rerun()
            except Exception as e:
                message_placeholder.markdown(f"❌ Error: {str(e)}")
                st.session_state.chat_history.append({"role": "user", "content": user_input})
                st.session_state.chat_history.append({"role": "assistant", "content": f"Error: {str(e)}"})
                st.rerun()

    # Handle voice input - simplified like frontend.py
    if config.get("ui.enable_voice", True) and 'audio_data' in locals() and audio_data and len(audio_data) > 0:
        # Initialize transcript tracking
        if "last_transcript" not in st.session_state:
            st.session_state.last_transcript = ""

        with NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            audio_data.export(tmp.name, format="mp3")
            tmp_path = tmp.name

        try:
            stt_service = st.session_state.services["stt"]
            with open(tmp_path, "rb") as f:
                audio_bytes = f.read()

            # Transcribe audio
            result = stt_service.transcribe_audio(
                audio_data=audio_bytes,
                filename="recording.mp3",
                language=config.get("stt.language"),
                temperature=config.get("stt.temperature", 0.15)
            )

            if result["status"] == "success":
                transcript = result["transcription"].strip()

                # Only process if this is a new transcript (avoid duplicates)
                if transcript and transcript != st.session_state.last_transcript:
                    st.session_state.last_transcript = transcript

                    # Display user message
                    with st.chat_message("user"):
                        st.markdown(transcript)

                    # Get and display assistant response
                    with st.chat_message("assistant"):
                        message_placeholder = st.empty()
                        chat_service = st.session_state.services["chat"]
                        full_response = ""

                        try:
                            for chunk in chat_service.stream_message(
                                st.session_state.thread_id,
                                st.session_state.assistant_id,
                                transcript
                            ):
                                full_response += chunk
                                message_placeholder.markdown(full_response + "▌")

                            message_placeholder.markdown(full_response)

                            # Add BOTH messages to history AFTER streaming completes
                            st.session_state.chat_history.append({"role": "user", "content": transcript})
                            st.session_state.chat_history.append({"role": "assistant", "content": full_response})

                            # TTS if enabled - play ONCE at the end
                            if listen_mode and config.get("ui.autoplay_tts", True) and full_response.strip():
                                tts_service = st.session_state.services["tts"]
                                audio_result = tts_service.text_to_speech(
                                    full_response,
                                    voice=config.get("tts.voice", "alloy"),
                                    speed=config.get("tts.speed", 1.0)
                                )
                                if audio_result["status"] == "success":
                                    html = tts_service.generate_autoplay_html(audio_result["audio_data"])
                                    st.markdown(html, unsafe_allow_html=True)

                            # Don't rerun immediately - let the audio play
                        except Exception as e:
                            message_placeholder.markdown(f"❌ Error: {str(e)}")
                            st.session_state.chat_history.append({"role": "user", "content": transcript})
                            st.session_state.chat_history.append({"role": "assistant", "content": f"Error: {str(e)}"})
                else:
                    # Reset transcript tracker if empty
                    if not transcript:
                        st.session_state.last_transcript = ""

        except Exception as e:
            st.error(f"Voice processing error: {str(e)}")
            st.session_state.last_transcript = ""

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)