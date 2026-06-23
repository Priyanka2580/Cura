"""
Gemini Summarizer Service
Calls the Gemini API with separate prompts for prescriptions and medical reports.
Phase 1 prompts use OCR text only; phase 2 prompts also ground the model with
structured entities already pulled out by the BioBERT NER model.
"""
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types

from config import GEMINI_API_KEY_ENV

load_dotenv()

GEMINI_MODEL_NAME = "gemini-2.5-flash"

# Without this, a stalled network call hangs indefinitely instead of failing
# and letting the caller move on to the next image.
REQUEST_TIMEOUT_MS = 60_000

PRESCRIPTION_PROMPT = """You are a medical documentation assistant generating a structured reference summary of a prescription/outpatient record for evaluation purposes.

Below is the raw text extracted from a prescription image via OCR. It may contain minor OCR errors (misspelled drug names, missing punctuation).

---
{raw_text}
---

Write a complete, detailed summary, as a single dense paragraph (no headers, no bullet points, no markdown), covering everything below that is present in the text:
- Document Type (e.g. Prescription, Outpatient Record, Discharge Summary)
- Hospital/clinic name and address/contact details
- Doctor name, qualifications, and registration number
- Patient name, age, gender, and date
- Complaints, symptoms, or diagnosis mentioned
- Every medicine, listed individually as "1) ...", "2) ...", with: exact name and strength, form (tablet/capsule/syrup/etc), dosage (e.g. 1-0-1), frequency, duration, and instructions (e.g. after food, before bedtime)
- Any advice, diet instructions, or follow-up instructions

Rules:
- Include everything present in the text above; do not omit details
- Do NOT add information that is not present in the text
- Do NOT give medical advice or interpretation beyond what is written
- Write in clear, factual English using field labels followed by a colon (e.g. "Patient Name: ...", "Medications: 1) ... 2) ..."), matching the style of a clinical record summary, not a patient-friendly explanation

If a field is missing or the text is too garbled to make sense of, state that plainly instead of guessing."""

REPORT_PROMPT = """You are a medical documentation assistant generating a structured reference summary of a lab/diagnostic report for evaluation purposes.

Below is the raw text extracted from a medical report image via OCR. It may contain minor OCR errors.

---
{raw_text}
---

Write a complete, detailed summary, as a single dense paragraph (no headers, no bullet points, no markdown), covering everything below that is present in the text:
- Document Type (e.g. Laboratory Investigation Report, Echocardiography Report, etc.)
- Hospital/lab name and address/contact details
- Doctor/consultant name
- Patient name, age, gender, and date
- Report type (e.g. CBC, Lipid Profile, Biochemistry)
- Every test, with: test name, value with unit, reference range, and status (Normal/High/Low) if stated
- Any interpretation or remarks given in the text

Rules:
- Include everything present in the text above; do not omit details
- Do NOT add information that is not present in the text
- Do NOT give medical advice or interpretation beyond what is written
- Write in clear, factual English using field labels followed by a colon (e.g. "Patient Name: ...", "Hemoglobin: 10.5 g/dl (Low, Reference Range: 11 - 13.5)"), matching the style of a clinical lab record summary, not a patient-friendly explanation

If a field is missing or the text is too garbled to make sense of, state that plainly instead of guessing."""

PROMPTS_BY_DOC_TYPE = {
    "prescription": PRESCRIPTION_PROMPT,
    "report": REPORT_PROMPT,
}

PRESCRIPTION_PROMPT_WITH_NER = """You are a medical documentation assistant generating a structured reference summary of a prescription/outpatient record for evaluation purposes.

Raw text extracted from the prescription image via OCR (may contain minor OCR errors):
---
{raw_text}
---

Structured entities already pulled out by a medical NER model — trust these over the raw text above for drug names, dosages, strengths, frequency, route and duration:
{entities_block}

Write a complete, detailed summary, as a single dense paragraph (no headers, no bullet points, no markdown), covering everything below that is present:
- Document Type (e.g. Prescription, Outpatient Record, Discharge Summary)
- Hospital/clinic name and address/contact details
- Doctor name, qualifications, and registration number
- Patient name, age, gender, and date
- Complaints, symptoms, or diagnosis mentioned
- Every medicine, listed individually as "1) ...", "2) ...", with: exact name and strength, form (tablet/capsule/syrup/etc), dosage (e.g. 1-0-1), frequency, route, duration, and instructions (e.g. after food, before bedtime)
- Any advice, diet instructions, or follow-up instructions

Rules:
- Include everything present in the source above; do not omit details
- Do NOT add information that is not present in the source
- Do NOT give medical advice or interpretation beyond what is written
- Write in clear, factual English using field labels followed by a colon (e.g. "Patient Name: ...", "Medications: 1) ... 2) ..."), matching the style of a clinical record summary, not a patient-friendly explanation

If a field is missing or unclear, state that plainly instead of guessing."""

REPORT_PROMPT_WITH_NER = """You are a medical documentation assistant generating a structured reference summary of a lab/diagnostic report for evaluation purposes.

Raw text extracted from the report image via OCR (may contain minor OCR errors):
---
{raw_text}
---

Structured entities already pulled out by a medical NER model — trust these over the raw text above for test names, values, units and reference ranges:
{entities_block}

Write a complete, detailed summary, as a single dense paragraph (no headers, no bullet points, no markdown), covering everything below that is present:
- Document Type (e.g. Laboratory Investigation Report, Echocardiography Report, etc.)
- Hospital/lab name and address/contact details
- Doctor/consultant name
- Patient name, age, gender, and date
- Report type (e.g. CBC, Lipid Profile, Biochemistry)
- Every test, with: test name, value with unit, reference range, and status (Normal/High/Low) if stated or determinable
- Any interpretation or remarks given in the source

Rules:
- Include everything present in the source above; do not omit details
- Do NOT add information that is not present in the source
- Do NOT give medical advice or interpretation beyond what is written
- Write in clear, factual English using field labels followed by a colon (e.g. "Patient Name: ...", "Hemoglobin: 10.5 g/dl (Low, Reference Range: 11 - 13.5)"), matching the style of a clinical lab record summary, not a patient-friendly explanation

If a field is missing or unclear, state that plainly instead of guessing."""

PROMPTS_WITH_NER_BY_DOC_TYPE = {
    "prescription": PRESCRIPTION_PROMPT_WITH_NER,
    "report": REPORT_PROMPT_WITH_NER,
}


def _format_entities(entities_by_type):
    """Render a {entity_type: [values]} dict as a plain-text block for prompts."""
    if not entities_by_type:
        return "(no entities extracted)"
    return "\n".join(
        f"{entity_type}: {', '.join(values)}"
        for entity_type, values in entities_by_type.items()
    )


class GeminiSummarizer:
    """Generates a structured clinical summary from OCR text using the Gemini API."""

    def __init__(self, api_key=None, model_name=GEMINI_MODEL_NAME):
        api_key = api_key or os.environ.get(GEMINI_API_KEY_ENV)
        if not api_key:
            raise ValueError(f"{GEMINI_API_KEY_ENV} not set. Add it to your .env file.")
        self.client = genai.Client(
            api_key=api_key,
            http_options=genai_types.HttpOptions(timeout=REQUEST_TIMEOUT_MS),
        )
        self.model_name = model_name

    def _generate(self, doc_type, prompt):
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )
            return {"summary": response.text, "doc_type": doc_type, "error": None}
        except Exception as e:
            return {"summary": None, "doc_type": doc_type, "error": str(e)}

    def summarize(self, doc_type, raw_text):
        """Generate a structured clinical-record-style summary from OCR text alone (phase 1).

        Args:
            doc_type: "prescription" or "report" — callers should filter out
                      "non_medical" before reaching here, there is no prompt for it
            raw_text: OCR-extracted text of the document

        Returns a dict with: summary, doc_type, error (error is None on success)
        """
        prompt_template = PROMPTS_BY_DOC_TYPE.get(doc_type)
        if prompt_template is None:
            return {
                "summary": None,
                "doc_type": doc_type,
                "error": f"No summarizer prompt for doc_type '{doc_type}'",
            }
        return self._generate(doc_type, prompt_template.format(raw_text=raw_text))

    def summarize_with_entities(self, doc_type, raw_text, entities_by_type):
        """Generate a structured clinical-record-style summary from OCR text + NER entities (phase 2).

        Args:
            doc_type:         "prescription" or "report"
            raw_text:         OCR-extracted text of the document
            entities_by_type: {entity_type: [matched text, ...]} from
                               NERExtractor.get_entities_by_type()

        Returns a dict with: summary, doc_type, error (error is None on success)
        """
        prompt_template = PROMPTS_WITH_NER_BY_DOC_TYPE.get(doc_type)
        if prompt_template is None:
            return {
                "summary": None,
                "doc_type": doc_type,
                "error": f"No NER-aware summarizer prompt for doc_type '{doc_type}'",
            }
        prompt = prompt_template.format(
            raw_text=raw_text,
            entities_block=_format_entities(entities_by_type),
        )
        return self._generate(doc_type, prompt)
