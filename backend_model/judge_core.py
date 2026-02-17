# judge_core.py
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def run_prediction_with_uploaded_pdfs(
    pdf_paths: list[str],
    prompt_text: str,
    model: str = "gpt-5-nano-2025-08-07",
) -> str:
    """
    Upload PDFs, call the model with file attachments, return raw text.
    """
    uploaded_files = []
    file_handles = []

    try:
        # Upload PDFs
        for path in pdf_paths:
            fh = open(path, "rb")
            file_handles.append(fh)

            uploaded = client.files.create(
                file=fh,
                purpose="user_data",  # recommended for file inputs
            )
            uploaded_files.append(uploaded)

        # Build message with PDFs attached
        message_content = [{"type": "text", "text": prompt_text}]
        for uf in uploaded_files:
            message_content.append({"type": "file", "file": {"file_id": uf.id}})

        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": message_content}],
        )
        return resp.choices[0].message.content

    finally:
        # Close local file handles
        for fh in file_handles:
            try:
                fh.close()
            except:
                pass

        # Delete uploaded OpenAI files
        for uf in uploaded_files:
            try:
                client.files.delete(uf.id)
            except:
                pass
