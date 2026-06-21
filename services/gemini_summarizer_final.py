"""
Gemini Summarizer Service (Production)
Calls the Gemini API with a single NER-grounded prompt per doc type.
Production pipeline is classifier -> OCR -> NER -> AbnormalityDetector (reports
only) -> Gemini, so structured entities (and, for reports, abnormality
results) are always available by the time this runs. Abnormality detection
itself happens upstream, in the pipeline -- this module only formats the
results it's given into the report prompt.
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

# Placeholder prompt -- will be revised later.
PRESCRIPTION_PROMPT = """You are a medical assistant explaining a prescription to a patient in plain, simple language.

Raw text extracted from the prescription image via OCR (may contain minor OCR errors):
---
{raw_text}
---

Structured entities already pulled out by a medical NER model -- trust these over the raw text above for drug names, dosages, strengths, frequency, route and duration:
{entities_block}

Write a clear, patient-friendly summary that:
1. Mentions the doctor's name and the patient's name/details and date if available
2. Lists each medicine with its dosage, strength, frequency, route and duration if available
3. Explains in simple terms what each medicine is generally used for
4. Notes any special instructions
5. Uses short sentences and avoids medical jargon

If information is missing or unclear, say so plainly instead of guessing."""

# Placeholder prompt -- will be revised later.
REPORT_PROMPT = """You are a medical assistant explaining a lab/diagnostic report to a patient in plain, simple language.

Raw text extracted from the report image via OCR (may contain minor OCR errors):
---
{raw_text}
---

Structured entities already pulled out by a medical NER model -- trust these over the raw text above for test names, values, units and reference ranges:
{entities_block}

Abnormality check already computed by a rule-based detector that compared each value against its reference range -- trust this over your own judgement for which values are abnormal and how severe:
{abnormality_block}

Write a clear, patient-friendly summary that:
1. Mentions the patient's name/details if available
2. Lists each test with its value, unit, and reference range if available
3. Clearly states which values are LOW or HIGH and their severity, in plain terms
4. Gives an overall plain-language takeaway of what the report shows
5. If any value is SEVERE, suggests what type of specialist the patient may want to consult
6. Avoids medical jargon

If information is missing or unclear, say so plainly instead of guessing."""

PROMPTS_BY_DOC_TYPE = {
    "prescription": PRESCRIPTION_PROMPT,
    "report": REPORT_PROMPT,
}


def _format_entities(entities_by_type):
    """Render a {entity_type: [values]} dict as a plain-text block for prompts."""
    if not entities_by_type:
        return "(no entities extracted)"
    return "\n".join(
        f"{entity_type}: {', '.join(values)}"
        for entity_type, values in entities_by_type.items()
    )


def _format_abnormalities(abnormality_results, abnormality_summary):
    """Render AbnormalityDetector detect()/get_summary() output as a plain-text block for prompts."""
    if not abnormality_results:
        return "(no test values available to evaluate)"

    lines = []
    for r in abnormality_results:
        if r["status"] in ("incomplete_data", "could_not_parse"):
            lines.append(f"{r['test_name']}: {r['status']}")
            continue
        unit = f" {r['unit']}" if r["unit"] else ""
        severity = f" ({r['severity']})" if r["severity"] else ""
        lines.append(
            f"{r['test_name']}: {r['value']}{unit} vs ref range {r['ref_range']} -> {r['status']}{severity}"
        )

    if not abnormality_summary:
        return "\n".join(lines)

    summary_line = (
        f"Summary: {abnormality_summary['abnormal_count']} abnormal "
        f"({abnormality_summary['severe_count']} severe) out of {abnormality_summary['total_tests']} tests evaluated"
    )
    return "\n".join(lines) + "\n" + summary_line


class GeminiSummarizer:
    """Generates a patient-friendly summary from OCR text + NER entities using the Gemini API."""

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

    def summarize(self, doc_type, raw_text, entities_by_type,
                  abnormality_results=None, abnormality_summary=None):
        """Generate a plain-language summary grounded with NER entities (production path).

        Args:
            doc_type:            "prescription" or "report" -- callers should filter out
                                  "non_medical" before reaching here, there is no prompt for it
            raw_text:             OCR-extracted text of the document
            entities_by_type:     {entity_type: [matched text, ...]} from
                                   NERExtractor.get_entities_by_type()
            abnormality_results:  AbnormalityDetector.detect() output -- pass for doc_type
                                  "report" so the prompt is grounded with computed
                                  LOW/HIGH/NORMAL statuses; ignored for "prescription"
            abnormality_summary:  AbnormalityDetector.get_summary() output -- pass alongside
                                  abnormality_results for doc_type "report"

        Returns a dict with: summary, doc_type, error (error is None on success)
        """
        prompt_template = PROMPTS_BY_DOC_TYPE.get(doc_type)
        if prompt_template is None:
            return {
                "summary": None,
                "doc_type": doc_type,
                "error": f"No summarizer prompt for doc_type '{doc_type}'",
            }

        if doc_type == "report":
            prompt = prompt_template.format(
                raw_text=raw_text,
                entities_block=_format_entities(entities_by_type),
                abnormality_block=_format_abnormalities(abnormality_results, abnormality_summary),
            )
        else:
            prompt = prompt_template.format(
                raw_text=raw_text,
                entities_block=_format_entities(entities_by_type),
            )

        return self._generate(doc_type, prompt)
