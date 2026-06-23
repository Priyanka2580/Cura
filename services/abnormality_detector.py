"""
Abnormality Detector Service
Flags abnormal medical test values using rule-based reference ranges.
"""
import re

MILD_THRESHOLD_PERCENT = 20
MODERATE_THRESHOLD_PERCENT = 50

# Matches a two-sided range like "13.0 - 17.0" or "13.0-17.0"
RANGE_PATTERN = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)\s*$")
# Matches a one-sided range like "< 110" or "> 40"
BOUND_PATTERN = re.compile(r"^\s*([<>])\s*=?\s*(-?\d+(?:\.\d+)?)\s*$")
# BERT-style tokenizers split decimals into separate tokens (e.g. "16.00"
# becomes "16 . 00"), so collapse whitespace around a decimal point back out.
NUMERIC_SPACING_PATTERN = re.compile(r"(?<=\d)\s*\.\s*(?=\d)")


class AbnormalityDetector:
    """Compares lab test values against NER-extracted reference ranges and
    flags abnormalities with a severity rating.
    """

    def detect(self, entities):
        """Compare each test's value against its reference range.

        Args:
            entities: dict grouped by entity type, with aligned lists under
                      "TEST_NAME", "TEST_VALUE", "TEST_UNIT" and "REF_RANGE"
                      (same index across all four lists belongs to one test)

        Returns:
            List of dicts, one per test, each containing test_name, value,
            unit, ref_range, status, severity and deviation_percent. Tests
            that can't be evaluated get status "incomplete_data" or
            "could_not_parse" instead, with the rest of the fields set to None.
        """
        test_names = entities.get("TEST_NAME", [])
        test_values = entities.get("TEST_VALUE", [])
        test_units = entities.get("TEST_UNIT", [])
        ref_ranges = entities.get("REF_RANGE", [])

        results = []
        for i, test_name in enumerate(test_names):
            raw_value = test_values[i] if i < len(test_values) else None
            unit = test_units[i] if i < len(test_units) else None
            ref_range = ref_ranges[i] if i < len(ref_ranges) else None

            if not raw_value or not ref_range:
                results.append({
                    "test_name": test_name,
                    "value": None,
                    "unit": unit,
                    "ref_range": ref_range,
                    "status": "incomplete_data",
                    "severity": None,
                    "deviation_percent": None,
                })
                continue

            value = self._parse_value(raw_value)
            if value is None:
                results.append({
                    "test_name": test_name,
                    "value": None,
                    "unit": unit,
                    "ref_range": ref_range,
                    "status": "could_not_parse",
                    "severity": None,
                    "deviation_percent": None,
                })
                continue

            ref_low, ref_high = self._parse_ref_range(ref_range)
            if ref_low is None and ref_high is None:
                results.append({
                    "test_name": test_name,
                    "value": value,
                    "unit": unit,
                    "ref_range": ref_range,
                    "status": "could_not_parse",
                    "severity": None,
                    "deviation_percent": None,
                })
                continue

            status, deviation_percent = self._evaluate_status(value, ref_low, ref_high)
            severity = self._severity_for_deviation(deviation_percent) if status != "NORMAL" else None

            results.append({
                "test_name": test_name,
                "value": value,
                "unit": unit,
                "ref_range": ref_range,
                "status": status,
                "severity": severity,
                "deviation_percent": deviation_percent,
            })

        return results

    def get_summary(self, results):
        """Summarize the output of detect() into aggregate counts.

        Args:
            results: list of dicts as returned by detect()

        Returns:
            Dict with total_tests, normal_count, abnormal_count, severe_count
            and abnormal_tests (names of tests flagged LOW or HIGH).
        """
        abnormal_tests = [r["test_name"] for r in results if r["status"] in ("LOW", "HIGH")]
        normal_count = sum(1 for r in results if r["status"] == "NORMAL")
        severe_count = sum(1 for r in results if r["severity"] == "SEVERE")

        return {
            "total_tests": len(results),
            "normal_count": normal_count,
            "abnormal_count": len(abnormal_tests),
            "severe_count": severe_count,
            "abnormal_tests": abnormal_tests,
        }

    def _parse_value(self, raw_value):
        """Parse a TEST_VALUE string into a float, stripping any units. Returns None on failure."""
        raw_value = self._normalize_numeric_spacing(raw_value)
        match = re.search(r"-?\d+(?:\.\d+)?", raw_value)
        if not match:
            return None
        try:
            return float(match.group())
        except ValueError:
            return None

    def _parse_ref_range(self, ref_range):
        """Parse a REF_RANGE string into (ref_low, ref_high) floats.

        Supports two-sided ranges ("13.0 - 17.0"), and one-sided bounds
        ("< 110" or "> 40"), where the missing side is treated as unbounded.
        Returns (None, None) if the string can't be parsed.
        """
        ref_range = self._normalize_numeric_spacing(ref_range).strip()

        range_match = RANGE_PATTERN.match(ref_range)
        if range_match:
            return float(range_match.group(1)), float(range_match.group(2))

        bound_match = BOUND_PATTERN.match(ref_range)
        if bound_match:
            operator, bound_value = bound_match.group(1), float(bound_match.group(2))
            if operator == "<":
                return None, bound_value
            return bound_value, None

        return None, None

    def _normalize_numeric_spacing(self, text):
        """Collapse tokenizer-introduced whitespace around decimal points (e.g. '16 . 00' -> '16.00')."""
        return NUMERIC_SPACING_PATTERN.sub(".", str(text))

    def _evaluate_status(self, value, ref_low, ref_high):
        """Compare value against (ref_low, ref_high) and return (status, deviation_percent)."""
        if ref_low is not None and value < ref_low:
            return "LOW", abs(value - ref_low) / ref_low * 100
        if ref_high is not None and value > ref_high:
            return "HIGH", abs(value - ref_high) / ref_high * 100
        return "NORMAL", None

    def _severity_for_deviation(self, deviation_percent):
        """Map a deviation percentage to a MILD/MODERATE/SEVERE severity rating."""
        if deviation_percent < MILD_THRESHOLD_PERCENT:
            return "MILD"
        if deviation_percent <= MODERATE_THRESHOLD_PERCENT:
            return "MODERATE"
        return "SEVERE"
