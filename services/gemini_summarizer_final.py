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
import time

from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from config import GEMINI_API_KEY_ENV

load_dotenv()

GEMINI_MODEL_NAME = "gemini-2.5-flash"

# Without this, a stalled network call hangs indefinitely instead of failing
# and letting the caller move on to the next image.
REQUEST_TIMEOUT_MS = 60_000

# Gemini returns these as transient APIError.code values (429 = rate limited,
# 503 = "model currently experiencing high demand") -- both are worth a short
# retry instead of immediately failing the whole pipeline run for the image.
RETRYABLE_STATUS_CODES = {429, 503}
MAX_RETRIES = 2
RETRY_BACKOFF_SECONDS = 2

PRESCRIPTION_PROMPT = """You are a medical assistant explaining a prescription to a patient in simple, friendly language that anyone can understand.

Raw text extracted from the prescription image via OCR (may contain minor OCR errors) -- this is the COMPLETE prescription and may contain MORE medicines or details than what appears below in the structured data:
---
{raw_text}
---

Structured entities already pulled out by a medical NER model -- use these for drug names, dosages, strengths, frequency, route and duration when available, but the raw text above may contain additional medicines or details not captured here:
{entities_block}

IMPORTANT: The structured entities above may be INCOMPLETE -- they only cover what the NER model successfully extracted. The raw text is the full prescription. You MUST scan the raw text yourself and include EVERY medicine mentioned in the prescription, not just the ones present in the structured entities.

Generate a response in valid JSON format only, with this exact structure:

{{
  "patient_summary": "2-3 sentence plain-language overview: doctor name, patient name/details, date, and the diagnosis/condition being treated if available -- name the condition explicitly here",
  "diagnosis_explained": "1-2 sentences explaining what the diagnosed condition is, in simple terms a non-medical person would understand, or null if no diagnosis is mentioned",
  "medicines": [
    {{
      "name": "medicine name",
      "dosage": "e.g. 650mg, or 'Not specified' if not mentioned",
      "frequency": "e.g. twice daily, or 'Not specified' if not mentioned",
      "duration": "e.g. 5 days, or 'Not specified' if not mentioned",
      "usage": "what this medicine is generally used for, in simple terms",
      "common_side_effects": "2-3 most common side effects, plain language",
      "alternatives": "1-2 commonly known alternative medicines if widely known, else 'Consult your doctor'",
      "source": "structured_data or raw_text_only"
    }}
  ],
  "recommended_tests": [
    {{
      "test_name": "name of any lab test, scan, or follow-up investigation the doctor has advised in this prescription",
      "purpose": "1 simple sentence on why this test was likely advised, based on the diagnosis/medicines",
      "timing": "when to get it done if mentioned, e.g. 'after 2 weeks' or 'Not specified'"
    }}
  ],
  "do_list": ["actionable care point NOT already stated in the medicines table -- e.g. specific foods to eat, timing habits, hydration, rest, hygiene related to this condition/medicine class"],
  "dont_list": ["actionable restriction NOT already stated in the medicines table -- e.g. foods/drinks/activities to avoid, interactions to watch for, habits that worsen this condition"],
  "lifestyle_tips": ["broader lifestyle guidance specific to the diagnosed condition or medicine class -- e.g. exercise, sleep, diet pattern -- NOT a repeat of dosage/timing instructions"],
  "missing_info_note": "state plainly if any critical info like dosage or duration was missing or unclear, else null"
}}

RULES FOR "medicines" -- THIS IS CRITICAL:
- Include EVERY single medicine mentioned anywhere in the raw text, even if it's not in the structured entities
- For medicines covered by the structured entities: use their exact dosage/frequency/duration values, set "source": "structured_data"
- For medicines found ONLY in raw text (not in structured entities): read the dosage/frequency/duration directly from the raw text yourself, set "source": "raw_text_only". If any of these fields genuinely cannot be found in the raw text either, write "Not specified" for that field
- Do not skip, drop, or summarize away any medicine found in the raw text -- the patient needs to see all of them
- If the same medicine appears in both structured entities and raw text, only include it once, using the structured entities version

RULES FOR "recommended_tests":
- Scan the raw text and entities for ANY mention of a lab test, blood test, scan, X-ray, or follow-up investigation the doctor advised (e.g. "CBC after 2 weeks", "Repeat blood sugar test", "Get an X-ray done")
- If no such advice is found anywhere in the document, return an empty list []
- Do not confuse a test name with a medicine name -- only include genuine diagnostic tests/investigations here

CRITICAL RULES ON REPETITION:
- Do NOT repeat dosage, frequency, timing, or duration instructions in "do_list", "dont_list", or "lifestyle_tips" -- that information belongs ONLY in the "medicines" table
- "do_list", "dont_list", and "lifestyle_tips" must contain NEW information not found anywhere else in the JSON -- think of them as general health/diet/activity guidance related to the condition or medicine class, not medicine-taking instructions
- Example of what NOT to do: "Don't forget to give the second dose" (this is a timing instruction, belongs in medicines table, not here)
- Example of what TO do: "Avoid giving iron supplements with tea or milk, as it reduces absorption" (this is new actionable knowledge)

OTHER RULES:
- Keep "patient_summary" short -- 2-3 sentences only
- "do_list" and "dont_list" should have 2-4 points each, specific to the diagnosed condition and medicine class -- not generic filler
- "lifestyle_tips" should have 1-3 points only
- Do not invent specific dosage/frequency/duration numbers not present in the raw text -- only fill in fields you can actually read in the raw text
- Do not include any text outside the JSON object -- no markdown, no explanation, just valid JSON"""

REPORT_PROMPT = """You are a medical assistant explaining a lab/diagnostic report to a patient in simple, friendly language that anyone can understand.

Raw text extracted from the report image via OCR (may contain minor OCR errors) -- this is the COMPLETE report and may contain MORE tests than what appears below in the structured data:
---
{raw_text}
---

Structured entities already pulled out by a medical NER model -- use these for test names, values, units and reference ranges when available, but the raw text above may contain additional tests not captured here:
{entities_block}

Abnormality check already computed by a rule-based detector for the tests it had complete data for -- trust this completely for the status/severity of those specific tests:
{abnormality_block}

IMPORTANT: The structured data and abnormality check above may be INCOMPLETE -- they only cover tests where extraction succeeded. The raw text is the full report. You MUST scan the raw text yourself and include EVERY test mentioned in the report, not just the ones with structured abnormality data.

Generate a response in valid JSON format only, with this exact structure:

{{
  "patient_summary": "2-3 sentence plain-language overview: patient name/details and date if available, and an overall takeaway of what this report shows",
  "likely_condition": "1-2 sentences naming what condition or issue the abnormal results may indicate (e.g. 'low hemoglobin suggests possible anemia'), in plain language, or 'No significant issues detected' if all values are normal",
  "test_results": [
    {{
      "test_name": "test name",
      "value": "value with unit",
      "reference_range": "range as given in report, or 'Not provided in report' if missing",
      "status": "NORMAL or LOW or HIGH or UNKNOWN",
      "severity": "MILD, MODERATE, SEVERE, or null if NORMAL or UNKNOWN",
      "meaning": "1 simple sentence on what this specific result means for the patient",
      "source": "structured_data or raw_text_only"
    }}
  ],
  "mentioned_medicines": [
    {{
      "name": "medicine name found anywhere in the report text, e.g. in a doctor's note, advice section, or already-prescribed medication list",
      "usage": "what this medicine is generally used for, in simple terms",
      "common_side_effects": "2-3 most common side effects, plain language",
      "alternatives": "1-2 commonly known alternative medicines if widely known, else 'Consult your doctor'"
    }}
  ],
  "overall_condition": "1-2 sentence plain-language statement of the patient's general health picture based on these results",
  "specialist_suggestion": "type of doctor/specialist to consult if any result is SEVERE or multiple are abnormal, else 'No specialist visit indicated based on these results, routine follow-up as advised by your doctor'",
  "do_list": ["actionable care point related to managing/improving the abnormal result(s) -- e.g. specific foods to eat, habits that help -- NOT a repeat of the test_results table"],
  "dont_list": ["actionable restriction related to the abnormal result(s) -- e.g. foods/habits to avoid that worsen this condition -- NOT a repeat of the test_results table"],
  "lifestyle_tips": ["broader lifestyle guidance specific to the likely condition -- e.g. diet pattern, exercise, hydration, sleep -- NOT a repeat of test values or status already shown"],
  "missing_info_note": "state plainly if any test was missing data or could not be parsed, else null"
}}

RULES FOR "test_results" -- THIS IS CRITICAL:
- Include EVERY single test mentioned anywhere in the raw text, even if it's not in the structured data or abnormality check
- For tests covered by the abnormality detector: use its exact status and severity, set "source": "structured_data"
- For tests found ONLY in raw text (not in structured/abnormality data): determine status yourself using standard medical reference ranges for that test if you know them, or if no reference range is given in the report and you don't know a standard range, set status to "UNKNOWN" and reference_range to "Not provided in report". Set "source": "raw_text_only" for these
- Do not skip, drop, or summarize away any test value found in the raw text -- the patient needs to see all of them
- If the same test appears in both structured data and raw text, only include it once, using the structured data version

RULES FOR "mentioned_medicines":
- Scan the raw text and entities for ANY medicine name mentioned anywhere in the report -- this could be in a doctor's remarks, advice section, current medication list, or follow-up notes
- If no medicine is mentioned anywhere in the document, return an empty list []
- Do not confuse a test name or instrument name with a medicine name

CRITICAL RULES ON REPETITION:
- Do NOT repeat test values, statuses, or severities in "do_list", "dont_list", or "lifestyle_tips" -- that information belongs ONLY in the "test_results" table
- Do NOT repeat medicine dosage/usage details in "do_list", "dont_list", or "lifestyle_tips" -- that belongs ONLY in "mentioned_medicines"
- "do_list", "dont_list", and "lifestyle_tips" must contain NEW information -- think of them as condition-specific diet/habit/exercise guidance, not a restatement of which values are high/low or which medicines are listed

OTHER RULES:
- Keep "patient_summary" short -- 2-3 sentences only
- "do_list" and "dont_list" should have 2-4 points each, tied to the likely condition -- not generic filler
- "lifestyle_tips" should have 1-3 points only
- If all results are NORMAL, keep do/dont/lifestyle lists short and general wellness-focused, and set "likely_condition" to "No significant issues detected"
- Do not invent specific numeric values not present in the raw text -- only assess status for values you can actually read in the raw text
- Do not include any text outside the JSON object -- no markdown, no explanation, just valid JSON"""

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
        last_error = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    # Forces the API itself to return a single JSON object -- no
                    # markdown fences or stray preamble/postamble text -- instead
                    # of just asking nicely for JSON in the prompt.
                    config=genai_types.GenerateContentConfig(
                        response_mime_type="application/json",
                    ),
                )
                return {"summary": response.text, "doc_type": doc_type, "error": None}
            except Exception as e:
                last_error = e
                is_retryable = (
                    isinstance(e, genai_errors.APIError)
                    and e.code in RETRYABLE_STATUS_CODES
                )
                if is_retryable and attempt < MAX_RETRIES:
                    time.sleep(RETRY_BACKOFF_SECONDS * (attempt + 1))
                    continue
                break
        return {"summary": None, "doc_type": doc_type, "error": str(last_error)}

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
