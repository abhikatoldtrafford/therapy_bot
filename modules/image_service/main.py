
import streamlit as st
import sys
from pathlib import Path

from image_service import ImageService
import base64
from PIL import Image
import io

st.set_page_config(page_title="Image Service Test", page_icon="📸")

st.title("📸 Image Analysis Module Test")
st.markdown("Test interface for image analysis using GPT-4 Vision")

# Initialize service
if "image_service" not in st.session_state:
    try:
        st.session_state.image_service = ImageService()
        st.session_state.service_initialized = True
    except Exception as e:
        st.error(f"Failed to initialize service: {str(e)}")
        st.session_state.service_initialized = False

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")

    api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
    model = st.selectbox("Model", ["gpt-4.1", "gpt-4o"], index=0)

    st.divider()
    st.subheader("Analysis Settings")
    detail_level = st.selectbox("Detail Level", ["high", "low", "auto"], index=0)
    max_tokens = st.number_input("Max Tokens", value=500, min_value=50, max_value=4000)

    if st.button("🔄 Reinitialize Service"):
        try:
            st.session_state.image_service = ImageService(api_key, model)
            st.session_state.service_initialized = True
            st.success("Service reinitialized!")
        except Exception as e:
            st.error(f"Failed: {str(e)}")
            st.session_state.service_initialized = False

if not st.session_state.get('service_initialized', False):
    st.error("⚠️ Service not initialized. Please configure API key in sidebar.")
    st.stop()

service = st.session_state.image_service

# Main tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Upload & Analyze", "Batch Analysis", "Comparison", "Templates", "History"])

with tab1:
    st.header("Single Image Analysis")

    # Image upload
    uploaded_file = st.file_uploader("Choose an image", type=['png', 'jpg', 'jpeg', 'gif', 'webp'])

    if uploaded_file:
        # Display image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Get file info
        file_size = len(uploaded_file.getvalue())
        st.caption(f"File: {uploaded_file.name} | Size: {file_size/1024:.1f} KB | Dimensions: {image.size}")

        # Analysis prompt
        st.subheader("Analysis Configuration")

        prompt_type = st.radio("Prompt Type", ["Default", "Custom", "Template"])

        if prompt_type == "Default":
            prompt = "Analyze this image and provide a thorough summary including all elements. If there's any text visible, include all the textual content."
            st.text_area("Prompt", value=prompt, disabled=True, height=100)
        elif prompt_type == "Custom":
            prompt = st.text_area("Custom Prompt",
                                placeholder="Enter your analysis prompt...",
                                height=150,
                                help="Describe what you want to analyze")
        else:  # Template
            templates = {
                "Medical": "Analyze this medical image and describe any visible anatomical structures, abnormalities, or relevant medical details.",
                "Document": "Extract and transcribe all text from this document image. Maintain formatting and structure.",
                "Art": "Describe this artwork including style, composition, colors, and artistic elements.",
                "Technical": "Analyze this technical diagram or schematic, explaining components and their relationships.",
                "Product": "Describe this product including features, condition, and any visible text or branding.",
                "NARM Therapy": "Analyze this image in the context of emotional expression, body language, and therapeutic relevance."
            }
            selected_template = st.selectbox("Select Template", list(templates.keys()))
            prompt = templates[selected_template]
            st.text_area("Template Prompt", value=prompt, disabled=True, height=100)

        # Analyze button
        if st.button("🔍 Analyze Image", type="primary"):
            with st.spinner("Analyzing image..."):
                # Convert to bytes
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='PNG')
                img_bytes = img_byte_arr.getvalue()

                # Analyze
                result = service.analyze_image(
                    image_data=img_bytes,
                    prompt=prompt,
                    detail=detail_level,
                    max_tokens=max_tokens
                )

                if result["status"] == "success":
                    st.success("Analysis Complete!")

                    # Display analysis
                    st.markdown("### Analysis Result")
                    st.write(result["analysis"])

                    # Save to history
                    if "analysis_history" not in st.session_state:
                        st.session_state.analysis_history = []

                    st.session_state.analysis_history.append({
                        "filename": uploaded_file.name,
                        "prompt": prompt,
                        "analysis": result["analysis"],
                        "timestamp": Path(__file__).stat().st_mtime
                    })

                    # Word count
                    word_count = len(result["analysis"].split())
                    st.caption(f"Analysis length: {word_count} words")

                else:
                    st.error(f"Analysis failed: {result['error']}")

with tab2:
    st.header("Batch Image Analysis")

    uploaded_files = st.file_uploader("Choose multiple images",
                                     type=['png', 'jpg', 'jpeg'],
                                     accept_multiple_files=True)

    if uploaded_files:
        st.info(f"Selected {len(uploaded_files)} images")

        # Display thumbnails
        cols = st.columns(min(len(uploaded_files), 4))
        for i, file in enumerate(uploaded_files):
            with cols[i % 4]:
                image = Image.open(file)
                st.image(image, caption=file.name[:20], use_column_width=True)

        batch_prompt = st.text_area("Batch Analysis Prompt",
                                   value="Describe this image concisely in 2-3 sentences.",
                                   height=100)

        if st.button("🔍 Analyze All", type="primary"):
            progress = st.progress(0)
            results = []

            for i, file in enumerate(uploaded_files):
                with st.spinner(f"Analyzing {file.name}..."):
                    # Convert to bytes
                    image = Image.open(file)
                    img_byte_arr = io.BytesIO()
                    image.save(img_byte_arr, format='PNG')
                    img_bytes = img_byte_arr.getvalue()

                    # Analyze
                    result = service.analyze_image(
                        image_data=img_bytes,
                        prompt=batch_prompt,
                        detail="low",  # Lower detail for batch
                        max_tokens=200
                    )

                    if result["status"] == "success":
                        results.append({
                            "filename": file.name,
                            "analysis": result["analysis"]
                        })

                    progress.progress((i + 1) / len(uploaded_files))

            # Display results
            st.success(f"Analyzed {len(results)} images")

            for res in results:
                with st.expander(f"📄 {res['filename']}"):
                    st.write(res["analysis"])

with tab3:
    st.header("Image Comparison")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Image 1")
        file1 = st.file_uploader("Choose first image", type=['png', 'jpg', 'jpeg'], key="comp1")
        if file1:
            st.image(file1, use_column_width=True)

    with col2:
        st.subheader("Image 2")
        file2 = st.file_uploader("Choose second image", type=['png', 'jpg', 'jpeg'], key="comp2")
        if file2:
            st.image(file2, use_column_width=True)

    if file1 and file2:
        comparison_prompt = st.text_area(
            "Comparison Prompt",
            value="Compare these two images and describe the similarities and differences.",
            height=100
        )

        if st.button("🔍 Compare Images", type="primary"):
            st.info("Note: Current implementation analyzes images separately. Multi-image analysis would require API updates.")

            col1, col2 = st.columns(2)

            # Analyze first image
            with col1:
                with st.spinner("Analyzing first image..."):
                    image1 = Image.open(file1)
                    img_byte_arr = io.BytesIO()
                    image1.save(img_byte_arr, format='PNG')
                    img_bytes = img_byte_arr.getvalue()

                    result1 = service.analyze_image(
                        image_data=img_bytes,
                        prompt="Describe this image for comparison.",
                        detail=detail_level,
                        max_tokens=300
                    )

                    if result1["status"] == "success":
                        st.markdown("**Image 1 Analysis:**")
                        st.write(result1["analysis"])

            # Analyze second image
            with col2:
                with st.spinner("Analyzing second image..."):
                    image2 = Image.open(file2)
                    img_byte_arr = io.BytesIO()
                    image2.save(img_byte_arr, format='PNG')
                    img_bytes = img_byte_arr.getvalue()

                    result2 = service.analyze_image(
                        image_data=img_bytes,
                        prompt="Describe this image for comparison.",
                        detail=detail_level,
                        max_tokens=300
                    )

                    if result2["status"] == "success":
                        st.markdown("**Image 2 Analysis:**")
                        st.write(result2["analysis"])

with tab4:
    st.header("Analysis Templates")

    st.markdown("Pre-configured prompts for different use cases")

    # Template categories
    template_categories = {
        "Medical & Healthcare": {
            "X-Ray Analysis": "Analyze this X-ray image, identifying visible anatomical structures and any potential abnormalities.",
            "Skin Condition": "Examine this skin image and describe visible characteristics, texture, color, and any notable features.",
            "Medical Report": "Extract all text and data from this medical report or prescription image."
        },
        "Documents & Text": {
            "OCR Extraction": "Extract all text from this image, maintaining original formatting and structure.",
            "Handwriting Recognition": "Transcribe the handwritten text in this image.",
            "Form Data": "Extract all form fields and their values from this document image."
        },
        "Technical & Engineering": {
            "Circuit Diagram": "Analyze this circuit diagram, identifying components and their connections.",
            "Architecture Blueprint": "Describe this architectural drawing including dimensions and structural elements.",
            "Flowchart": "Explain the process flow shown in this diagram."
        },
        "Art & Design": {
            "Artwork Analysis": "Analyze this artwork's composition, style, color palette, and artistic techniques.",
            "Logo Description": "Describe this logo design including colors, shapes, and text elements.",
            "UI/UX Screenshot": "Analyze this user interface design, describing layout, components, and user flow."
        },
        "Therapy & Psychology": {
            "Emotion Recognition": "Analyze facial expressions and body language visible in this image.",
            "Art Therapy": "Interpret this drawing or artwork from a therapeutic perspective.",
            "Environment Assessment": "Describe this space and its potential psychological impact."
        }
    }

    selected_category = st.selectbox("Select Category", list(template_categories.keys()))

    templates = template_categories[selected_category]

    for template_name, template_prompt in templates.items():
        with st.expander(f"📝 {template_name}"):
            st.text_area("Prompt", value=template_prompt, height=100, key=f"template_{template_name}")

            col1, col2 = st.columns([3, 1])
            with col2:
                if st.button("Use Template", key=f"use_{template_name}"):
                    st.session_state.selected_template_prompt = template_prompt
                    st.success("Template selected! Go to 'Upload & Analyze' tab to use it.")

with tab5:
    st.header("Analysis History")

    if "analysis_history" in st.session_state and st.session_state.analysis_history:
        st.info(f"Total analyses: {len(st.session_state.analysis_history)}")

        # Filter options
        search_term = st.text_input("Search in analyses", placeholder="Enter keyword...")

        # Display history
        for i, entry in enumerate(reversed(st.session_state.analysis_history)):
            if search_term and search_term.lower() not in entry["analysis"].lower():
                continue

            with st.expander(f"📸 {entry['filename']} - Analysis {len(st.session_state.analysis_history) - i}"):
                st.markdown("**Prompt:**")
                st.write(entry["prompt"])

                st.markdown("**Analysis:**")
                st.write(entry["analysis"])

                col1, col2 = st.columns([3, 1])
                with col2:
                    if st.button("📋 Copy", key=f"copy_{i}"):
                        st.code(entry["analysis"], language="text")

        # Clear history
        if st.button("🗑️ Clear History", type="secondary"):
            st.session_state.analysis_history = []
            st.success("History cleared")
            st.rerun()

    else:
        st.info("No analysis history yet. Analyze some images to see them here.")

# Example images section
st.divider()
st.subheader("🖼️ Example Use Cases")

example_cases = {
    "Text Extraction": "Upload a document, receipt, or screenshot to extract text",
    "Object Detection": "Upload a photo to identify and describe objects",
    "Medical Analysis": "Upload medical imagery for detailed description",
    "Art Interpretation": "Upload artwork for style and composition analysis",
    "Technical Diagrams": "Upload schematics or flowcharts for explanation"
}

cols = st.columns(len(example_cases))
for i, (case, description) in enumerate(example_cases.items()):
    with cols[i]:
        st.markdown(f"**{case}**")
        st.caption(description)

# Status bar
st.divider()
col1, col2, col3 = st.columns(3)
with col1:
    st.caption(f"🟢 Service: {'Connected' if st.session_state.get('service_initialized') else 'Not Connected'}")
with col2:
    st.caption(f"🎨 Model: {model}")
with col3:
    analyses_count = len(st.session_state.get('analysis_history', []))
    st.caption(f"📊 Total Analyses: {analyses_count}")