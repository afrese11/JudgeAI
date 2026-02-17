import os
import time
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from docx import Document
from docx.shared import Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Define paths
SCRIPT_DIR = Path(__file__).resolve().parent
TEST_DATA_DIR = SCRIPT_DIR / "test_data_full"
OUTPUT_DIR = SCRIPT_DIR / "case_outputs"

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(exist_ok=True)

# Prompt template
PROMPT_TEMPLATE = """
You are serving as an appellate judge reviewing an appellate case which includes {num_docs} document(s).

CRITICAL INSTRUCTIONS:
- You must PREDICT the outcome based ONLY on the legal arguments, facts, and law presented in these documents.
- DO NOT use any external knowledge about this case, parties, or docket number.
- DO NOT simply extract or report any actual court decision that appears in the documents.
- Treat this as if YOU are the appellate court making the decision for the first time.
- Base your prediction solely on: the strength of legal arguments presented, applicable law and precedent discussed in the briefs, the quality of each party's legal reasoning, and the facts of the case.

The documents may include:
- Appellant's brief (arguing the lower court was wrong)
- Appellee's brief (defending the lower court decision)
- Reply briefs
- Addendums with relevant statutes, exhibits, or the lower court's decision

Your task is to produce TWO separate PREDICTIVE outputs:

**OUTPUT 1: CASE SUMMARY DOCUMENT (PREDICTIVE)**
Predict and provide:
1. A brief 3-5 sentence summary of the key legal issues on appeal.
2. Statement of the lower court's decision (what ruling is being appealed from).
3. Your PREDICTED recommendation for the length of oral argument, based on the legal complexity of the issues on appeal.
4. An explanation of the case complexity and your reasoning for the oral argument time recommendation.

**OUTPUT 2: CASE DECISION DOCUMENT (PREDICTIVE)**
Predict and provide:
1. A written judicial opinion that decides all of the issues raised on the appeal, based on your analysis of the arguments presented.
2. Your PREDICTED determination of whether the case should be AFFIRMED, REVERSED, VACATED, or another disposition.
3. Legal reasoning supporting your predicted decision, citing the arguments and law from the briefs.

Base your predictions on:
- Strength and persuasiveness of each party's legal arguments
- How well applicable law and precedent support each side
- Quality of legal reasoning and authority cited in the briefs
- The facts as presented and their legal significance

DO NOT base your prediction on:
- Any actual appellate decision if mentioned in the documents
- External knowledge of how this case was actually decided
- Recognition of the parties, docket numbers, or jurisdiction

Please structure your response exactly as follows:

===CASE SUMMARY===
[Your predicted case summary here]

===CASE DECISION===
[Your predicted case decision here]
"""

# Redaction prompt template
REDACTION_PROMPT = """
Please read this legal document and return the EXACT same text, but with the following redactions:

1. Replace all docket numbers, case numbers, and case citations that identify THIS specific case (e.g., "No. 22-3593", "Case No. 23-1414") with "[DOCKET NUMBER REDACTED]"
2. Replace all party names (plaintiff, defendant, appellant, appellee names - both individuals and organizations) with generic terms:
   - Use "Appellant" for the appealing party
   - Use "Appellee" for the responding party
   - Use "Petitioner" or "Respondent" where applicable
3. Replace all judge names with "[JUDGE NAME REDACTED]"
4. Replace all attorney names and law firm names with "[ATTORNEY NAME REDACTED]" and "[LAW FIRM REDACTED]"
5. Replace specific court names (e.g., "Fifth Circuit Court of Appeals") with generic terms like "the Court of Appeals" or "the District Court"

IMPORTANT: 
- Only redact identifying information about THIS case and its parties
- DO NOT redact case law citations, precedent names, or statutory references (e.g., keep "Brown v. Board of Education", "42 U.S.C. § 1983")
- Keep all legal arguments, statutes, facts, reasoning, and structure EXACTLY as written
- Preserve all legal content and formatting

Return the full redacted text with no preamble or explanation.
"""

def create_word_document(title, content, output_path):
    """Create a Word document with the given title and content."""
    doc = Document()
    
    # Add title
    title_paragraph = doc.add_heading(title, level=1)
    title_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    # Add content
    paragraphs = content.strip().split('\n')
    for para_text in paragraphs:
        if para_text.strip():
            p = doc.add_paragraph(para_text.strip())
            p.style.font.size = Pt(11)
    
    # Save document
    doc.save(output_path)
    print(f"  Created: {output_path.name}")

def parse_gpt_response(response_text):
    """Parse GPT response into case summary and case decision."""
    # Split by the markers
    parts = response_text.split('===CASE SUMMARY===')
    
    if len(parts) < 2:
        return None, None
    
    remainder = parts[1]
    decision_parts = remainder.split('===CASE DECISION===')
    
    if len(decision_parts) < 2:
        return None, None
    
    case_summary = decision_parts[0].strip()
    case_decision = decision_parts[1].strip()
    
    return case_summary, case_decision

def redact_pdf_content(pdf_path, max_retries=3):
    """Use OpenAI to redact identifying information from a PDF."""
    
    print(f"    - Redacting {pdf_path.name}")
    
    for attempt in range(max_retries):
        try:
            # Upload the PDF
            with open(pdf_path, "rb") as f:
                uploaded_file = client.files.create(
                    file=f,
                    purpose="assistants"
                )
            
            # Add a small delay to avoid rate limiting
            time.sleep(1)
            
            response = client.chat.completions.create(
                model="gpt-5-nano-2025-08-07",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": REDACTION_PROMPT},
                            {
                                "type": "file",
                                "file": {
                                    "file_id": uploaded_file.id
                                }
                            }
                        ]
                    }
                ],
                timeout=120.0  # Increase timeout to 2 minutes
            )
            
            redacted_text = response.choices[0].message.content
            
            # Clean up the uploaded file
            client.files.delete(uploaded_file.id)
            
            return redacted_text
            
        except Exception as e:
            print(f"    Attempt {attempt + 1}/{max_retries} failed: {str(e)}")
            
            # Try to clean up the file if it was uploaded
            try:
                if 'uploaded_file' in locals():
                    client.files.delete(uploaded_file.id)
            except:
                pass
            
            # If this was the last attempt, return None
            if attempt == max_retries - 1:
                print(f"    Error redacting {pdf_path.name} after {max_retries} attempts")
                return None
            
            # Wait before retrying (exponential backoff)
            wait_time = 2 ** attempt  # 1s, 2s, 4s
            print(f"    Waiting {wait_time} seconds before retry...")
            time.sleep(wait_time)
    
    return None

def process_case_directory(case_dir, redact=False):
    """Process all PDFs in a single case directory.
    
    Args:
        case_dir: Path to the case directory
        redact: If True, redact identifying information before prediction
    """
    case_id = case_dir.name
    pdf_files = sorted(case_dir.glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDFs found in {case_id}")
        return None
    
    # Check if this case has already been processed
    case_output_dir = OUTPUT_DIR / case_id
    summary_path = case_output_dir / f"{case_id}_summary.docx"
    decision_path = case_output_dir / f"{case_id}_decision.docx"
    
    if case_output_dir.exists() and summary_path.exists() and decision_path.exists():
        print(f"\nSkipping case: {case_id} (already processed)")
        return None
    
    print(f"\nProcessing case: {case_id}")
    print(f"Found {len(pdf_files)} PDF(s)")
    print(f"Redaction: {'ENABLED' if redact else 'DISABLED'}")
    
    if redact:
        # REDACTION PATH: Use GPT to redact, then send as text
        print(f"  Redacting documents...")
        redacted_texts = []
        
        # Create output directory for this case
        case_output_dir = OUTPUT_DIR / case_id
        case_output_dir.mkdir(exist_ok=True)
        
        for i, pdf_path in enumerate(pdf_files):
            # Add delay between files to avoid rate limiting
            if i > 0:
                time.sleep(2)
            
            redacted_text = redact_pdf_content(pdf_path)
            if redacted_text:
                redacted_texts.append({
                    "filename": pdf_path.name,
                    "content": redacted_text
                })
                
                # Save each redacted document as a separate text file
                txt_filename = pdf_path.stem + "_redacted.txt"
                txt_filepath = case_output_dir / txt_filename
                
                with open(txt_filepath, 'w', encoding='utf-8') as f:
                    f.write(f"REDACTED DOCUMENT: {pdf_path.name}\n")
                    f.write(f"{'='*80}\n\n")
                    f.write(redacted_text)
                
                print(f"    Saved: {txt_filename}")
            else:
                print(f"    WARNING: Skipping {pdf_path.name} - redaction failed")
        
        if not redacted_texts:
            print(f"  Failed to redact documents for {case_id}")
            return None
        
        # Combine redacted texts for the prediction prompt
        combined_content = ""
        for doc in redacted_texts:
            combined_content += f"\n\n{'='*60}\n"
            combined_content += f"DOCUMENT: {doc['filename']}\n"
            combined_content += f"{'='*60}\n\n"
            combined_content += doc['content']
        
        # Create the prediction prompt
        prompt = PROMPT_TEMPLATE.format(num_docs=len(redacted_texts))
        full_prompt = prompt + "\n\nDOCUMENTS:\n" + combined_content
        
        try:
            # Make prediction with redacted content (text only, no file uploads)
            print(f"  Making prediction with redacted content...")
            response = client.chat.completions.create(
                model="gpt-5-nano-2025-08-07",
                messages=[
                    {
                        "role": "user",
                        "content": full_prompt
                    }
                ],
                timeout=180.0  # 3 minute timeout for large combined text
            )
            
            result = response.choices[0].message.content
            num_docs = len(redacted_texts)
            doc_names = [doc['filename'] for doc in redacted_texts]
            
        except Exception as e:
            print(f"Error processing {case_id}: {str(e)}")
            return None
    
    else:
        # NON-REDACTION PATH: Upload PDFs directly
        uploaded_files = []
        for pdf_path in pdf_files:
            print(f"  - Uploading {pdf_path.name}")
            with open(pdf_path, "rb") as f:
                uploaded_file = client.files.create(
                    file=f,
                    purpose="assistants"
                )
                uploaded_files.append(uploaded_file)
        
        # Create the prompt
        prompt = PROMPT_TEMPLATE.format(num_docs=len(pdf_files))
        
        try:
            # Build the message content with file references
            message_content = [
                {"type": "text", "text": prompt}
            ]
            
            # Add each PDF file to the message
            for uploaded_file in uploaded_files:
                message_content.append({
                    "type": "file",
                    "file": {
                        "file_id": uploaded_file.id
                    }
                })
            
            # Make API call to OpenAI with PDF files
            print(f"  Making prediction with original documents...")
            response = client.chat.completions.create(
                model="gpt-5-nano-2025-08-07",
                messages=[
                    {
                        "role": "user",
                        "content": message_content
                    }
                ],
                timeout=180.0  # 3 minute timeout
            )
            
            result = response.choices[0].message.content
            num_docs = len(pdf_files)
            doc_names = [f.name for f in pdf_files]
            
            # Clean up uploaded files
            for uploaded_file in uploaded_files:
                client.files.delete(uploaded_file.id)
            
        except Exception as e:
            # Clean up uploaded files in case of error
            for uploaded_file in uploaded_files:
                try:
                    client.files.delete(uploaded_file.id)
                except:
                    pass
            print(f"Error processing {case_id}: {str(e)}")
            return None
    
    # Parse the response (same for both paths)
    case_summary, case_decision = parse_gpt_response(result)
    
    if case_summary and case_decision:
        # Create output directory for this case (may already exist if redacted)
        case_output_dir = OUTPUT_DIR / case_id
        case_output_dir.mkdir(exist_ok=True)
        
        # Create Word documents
        summary_path = case_output_dir / f"{case_id}_summary.docx"
        decision_path = case_output_dir / f"{case_id}_decision.docx"
        
        create_word_document(
            f"Case Summary: {case_id}",
            case_summary,
            summary_path
        )
        
        create_word_document(
            f"Case Decision: {case_id}",
            case_decision,
            decision_path
        )
    else:
        print(f"  Warning: Could not parse response into two sections for {case_id}")
    
    return {
        "case_id": case_id,
        "summary": case_summary,
        "decision": case_decision,
        "num_documents": num_docs,
        "documents": doc_names,
        "redacted": redact
    }

def main(redact=False):
    """Main function to process all case directories.
    
    Args:
        redact: If True, redact identifying information before prediction
    """
    
    # Get all case directories
    case_dirs = sorted([d for d in TEST_DATA_DIR.iterdir() if d.is_dir()])
    
    print(f"Found {len(case_dirs)} case directories to process")
    print(f"Output will be saved to: {OUTPUT_DIR}")
    print(f"Redaction mode: {'ENABLED' if redact else 'DISABLED'}")
    
    results = []
    
    # Process each case directory
    for case_dir in case_dirs:
        result = process_case_directory(case_dir, redact=redact)
        if result:
            results.append(result)
    
    print(f"\nProcessing complete! Processed {len(results)} cases.")
    print(f"All outputs saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    import sys
    
    # Check for command line arguments
    redact = False
    if len(sys.argv) > 1 and sys.argv[1] == '--redact':
        redact = True
    
    # Set redact=True to enable redaction, redact=False to disable
    main(redact=redact)