#!/usr/bin/env python3
"""
Create Default Vector Store with NARM Knowledge Base
Uploads NARM.pdf and Working-with-Developmental-Trauma.pdf to OpenAI Vector Store
"""

import os
import sys
from pathlib import Path
from openai import OpenAI
import time
import json

def create_narm_vector_store():
    """Create a vector store with NARM therapy knowledge base PDFs"""

    # Get API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # Try to get from streamlit secrets
        try:
            import streamlit as st
            api_key = st.secrets.get("OPENAI_API_KEY")
        except:
            pass

    if not api_key:
        print("❌ Error: OPENAI_API_KEY not found in environment or secrets")
        print("Please set: export OPENAI_API_KEY='your-key-here'")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    # PDF files to upload
    pdf_files = [
        "NARM.pdf",
        "Working-with-Developmental-Trauma.pdf"
    ]

    # Check if files exist
    for pdf in pdf_files:
        if not Path(pdf).exists():
            print(f"❌ Error: {pdf} not found in current directory")
            sys.exit(1)

    print("🚀 Creating NARM Knowledge Base Vector Store...")

    try:
        # Step 1: Create vector store
        print("📦 Creating vector store...")
        vector_store = client.vector_stores.create(
            name="NARM Therapy Knowledge Base - Default"
        )
        print(f"✅ Vector store created: {vector_store.id}")

        # Step 2: Upload files
        file_ids = []
        for pdf_path in pdf_files:
            print(f"📤 Uploading {pdf_path}...")
            with open(pdf_path, "rb") as f:
                file = client.files.create(
                    file=f,
                    purpose='assistants'
                )
                file_ids.append(file.id)
                print(f"✅ Uploaded {pdf_path}: {file.id}")

        # Step 3: Add files to vector store
        print("🔗 Adding files to vector store...")
        batch = client.vector_stores.file_batches.create(
            vector_store_id=vector_store.id,
            file_ids=file_ids
        )

        # Step 4: Wait for processing
        print("⏳ Processing files (this may take a minute)...")
        max_wait = 120  # 2 minutes max
        start_time = time.time()

        while time.time() - start_time < max_wait:
            batch_status = client.vector_stores.file_batches.retrieve(
                vector_store_id=vector_store.id,
                batch_id=batch.id
            )

            if batch_status.status == 'completed':
                print("✅ Files processed successfully!")
                break
            elif batch_status.status == 'failed':
                print(f"❌ Processing failed: {batch_status}")
                sys.exit(1)

            print(f"   Status: {batch_status.status} | Files: {batch_status.file_counts.completed}/{batch_status.file_counts.total}")
            time.sleep(5)

        # Step 5: Verify vector store
        print("\n📊 Vector Store Summary:")
        print(f"   ID: {vector_store.id}")
        print(f"   Name: {vector_store.name}")
        print(f"   Files: {len(file_ids)}")
        print(f"   Status: Active")

        # Step 6: Save configuration
        config = {
            "vector_store_id": vector_store.id,
            "vector_store_name": vector_store.name,
            "files": [
                {"name": pdf, "id": fid}
                for pdf, fid in zip(pdf_files, file_ids)
            ],
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "description": "Default NARM therapy knowledge base with comprehensive resources"
        }

        config_file = "default_vector_store_config.json"
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)
        print(f"\n💾 Configuration saved to {config_file}")

        # Print instructions
        print("\n" + "="*60)
        print("🎉 SUCCESS! Vector Store Created")
        print("="*60)
        print(f"\n📌 Vector Store ID: {vector_store.id}")
        print("\n📝 To use this vector store, update backend.py:")
        print(f'   DEFAULT_VECTOR_STORE_ID = "{vector_store.id}"')
        print("\n✨ Features included:")
        print("   • NARM therapy principles and framework")
        print("   • Developmental trauma treatment approaches")
        print("   • Therapeutic techniques and interventions")
        print("   • Case studies and clinical examples")
        print("\n💡 Users can still upload custom documents to override this default")

        return vector_store.id

    except Exception as e:
        print(f"❌ Error creating vector store: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    vector_store_id = create_narm_vector_store()
    print(f"\n✅ Default Vector Store Ready: {vector_store_id}")