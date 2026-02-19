# judge_core.py
import os
from typing import List

from openai import OpenAI

# The SDK reads OPENAI_API_KEY from the environment automatically.
client = OpenAI()


def run_prediction_with_uploaded_pdfs(
    pdf_paths: List[str],
    prompt_text: str,
    model: str | None = None,
) -> str:
    """
    Upload PDFs, call the Responses API with file inputs, return output text.
    """
    model = model or os.getenv("OPENAI_MODEL", "gpt-5")

    uploaded_file_ids: List[str] = []
    file_handles = []

    try:
        # Upload PDFs (recommended purpose for model inputs is "user_data")
        for path in pdf_paths:
            fh = open(path, "rb")
            file_handles.append(fh)

            uploaded = client.files.create(
                file=fh,
                purpose="user_data",
            )
            uploaded_file_ids.append(uploaded.id)

        # Build Responses API input with multiple files + your prompt text.
        # (For PDF inputs, Responses API is the recommended path.) :contentReference[oaicite:1]{index=1}
        content = []
        for fid in uploaded_file_ids:
            content.append({"type": "input_file", "file_id": fid})
        content.append({"type": "input_text", "text": prompt_text})

        response = client.responses.create(
            model=model,
            input=[
                {
                    "role": "user",
                    "content": content,
                }
            ],
        )

        # The docs show using response.output_text for Responses API output. :contentReference[oaicite:2]{index=2}
        return response.output_text or ""

    finally:
        # Close local file handles
        for fh in file_handles:
            try:
                fh.close()
            except Exception:
                pass

        # Best-effort cleanup: delete uploaded OpenAI files
        for fid in uploaded_file_ids:
            try:
                client.files.delete(fid)
            except Exception:
                pass
