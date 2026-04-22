import hashlib
import json
import random
import re
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd

PIPELINE_VERSION = "2026-04-22-streamlit-cache-v2"
MODEL_DEFAULT = "gemini-2.5-pro"

RECOMMEND_COL = "Would you recommend studying with University of Aberdeen Online to other people?"
CONSENT_COL = "Are you happy for us to use your feedback in University of Aberdeen marketing materials, including our website and on social media?"

TEXT_COLS = [
    ("typical_day", "Typical day like for you as an online student?"),
    ("learning_helped", "Did your online learning help you with any of the following?"),
    ("why_recommend", "Thanks for recommending us! Can you tell us why?"),
    ("negative_feedback", "We’re sorry to hear that. Let us know if there's any feedback you'd like us to pass to a relevant member of staff"),
    ("anything_else", "Is there anything else you’d like to share?"),
    ("why_aberdeen", "What attracted you to the University of Aberdeen?"),
]

MATCH_COLUMNS = ["row_key", "content_hash", "pipeline_version"]
INTERNAL_COLUMNS = MATCH_COLUMNS + ["comment_text", "comment_word_count", "marketing_consent_yes"]

RESULT_COLUMNS = [
    "overall_sentiment_score_0_100",
    "overall_sentiment_score_raw_model_0_100",
    "overall_sentiment_label",
    "overall_feedback_summary_1_sentence",
    "themes_tags_csv",
    "area_sentiment.university",
    "area_sentiment.tutors",
    "area_sentiment.vle_myaberdeen",
    "area_sentiment.learning_course_content",
    "area_sentiment.job_prospects",
    "confidence_0_1",
    "source_fields_present",
    "evidence_word_count",
    "sentiment_low_evidence_flag",
    "severe_distress_flag",
    "manual_review_required",
    "manual_review_reason",
    "analysis_notes",
    "quote_short_marketing",
    "quote_long_case_study",
    "quote_source_field",
    "quote_length_fit",
    "quote_sentiment_fit",
    "quote_uniqueness_score_1_5",
    "quote_selection_notes",
    "analysis_status",
    "analysis_mode",
    "analysis_error_type",
    "analysis_error_message",
    "analysis_raw_text",
    "quote_status",
    "quote_mode",
    "quote_error_type",
    "quote_error_message",
    "quote_raw_text",
]

DERIVED_COLUMNS = [
    "themes_tags",
    "regular_theme_tags",
    "short_quote_word_count",
    "long_quote_word_count",
    "quote_length_points",
    "quote_sentiment_points",
    "quote_uniqueness_points",
    "quote_quality_score_0_100",
    "quote_marketing_ready",
    "quote_case_study_ready",
    "quote_internal_highlight_ready",
]

GENERATED_FIELD_METADATA = [
    {
        "field": "comment_text",
        "category": "Input prep",
        "appears_in": "Scored upload CSV and master cache CSV",
        "description": "Normalized combined comment built from the available testimonial text fields and used as the main analysis evidence.",
    },
    {
        "field": "comment_word_count",
        "category": "Input prep",
        "appears_in": "Scored upload CSV and master cache CSV",
        "description": "Word count of `comment_text`.",
    },
    {
        "field": "marketing_consent_yes",
        "category": "Input prep",
        "appears_in": "Scored upload CSV and master cache CSV",
        "description": "Boolean consent flag derived from the marketing-consent survey question.",
    },
    {
        "field": "row_key",
        "category": "Cache matching",
        "appears_in": "Master cache CSV only",
        "description": "Stable row identifier used to match uploads against cached results; uses `Response ID` when available, otherwise a derived key.",
    },
    {
        "field": "content_hash",
        "category": "Cache matching",
        "appears_in": "Master cache CSV only",
        "description": "Hash of the row's content used to detect whether a previously seen testimonial has changed.",
    },
    {
        "field": "pipeline_version",
        "category": "Cache matching",
        "appears_in": "Master cache CSV only",
        "description": "Version of the analysis pipeline that produced the cached result; helps invalidate stale rows after logic changes.",
    },
    {
        "field": "overall_sentiment_score_0_100",
        "category": "Sentiment analysis",
        "appears_in": "Scored upload CSV and master cache CSV",
        "description": "Final calibrated sentiment score on a 0-100 scale.",
    },
    {
        "field": "overall_sentiment_score_raw_model_0_100",
        "category": "Sentiment analysis",
        "appears_in": "Scored upload CSV and master cache CSV",
        "description": "Raw sentiment score returned by the model before post-processing and evidence-based caps.",
    },
    {
        "field": "overall_sentiment_label",
        "category": "Sentiment analysis",
        "appears_in": "Scored upload CSV and master cache CSV",
        "description": "Overall sentiment bucket: `very_positive`, `positive`, `mixed`, `neutral`, `negative`, or `very_negative`.",
    },
    {
        "field": "overall_feedback_summary_1_sentence",
        "category": "Sentiment analysis",
        "appears_in": "Scored upload CSV and master cache CSV",
        "description": "One-sentence summary of the testimonial.",
    },
    {
        "field": "themes_tags_csv",
        "category": "Sentiment analysis",
        "appears_in": "Scored upload CSV and master cache CSV",
        "description": "Comma-separated list of normalized theme tags extracted from the testimonial.",
    },
    {
        "field": "area_sentiment.university",
        "category": "Sentiment analysis",
        "appears_in": "Scored upload CSV and master cache CSV",
        "description": "Sentiment about the university overall, including reputation, administration, or institution-level experience.",
    },
    {
        "field": "area_sentiment.tutors",
        "category": "Sentiment analysis",
        "appears_in": "Scored upload CSV and master cache CSV",
        "description": "Sentiment about tutors, lecturers, or staff support.",
    },
    {
        "field": "area_sentiment.vle_myaberdeen",
        "category": "Sentiment analysis",
        "appears_in": "Scored upload CSV and master cache CSV",
        "description": "Sentiment about the learning platform, VLE, or MyAberdeen experience.",
    },
    {
        "field": "area_sentiment.learning_course_content",
        "category": "Sentiment analysis",
        "appears_in": "Scored upload CSV and master cache CSV",
        "description": "Sentiment about course content, teaching, structure, and learning materials.",
    },
    {
        "field": "area_sentiment.job_prospects",
        "category": "Sentiment analysis",
        "appears_in": "Scored upload CSV and master cache CSV",
        "description": "Sentiment about employability, career impact, or job prospects.",
    },
    {
        "field": "confidence_0_1",
        "category": "Sentiment analysis",
        "appears_in": "Scored upload CSV and master cache CSV",
        "description": "Confidence score between 0 and 1, reduced when the evidence is thin or ambiguous.",
    },
    {
        "field": "source_fields_present",
        "category": "Sentiment analysis",
        "appears_in": "Scored upload CSV and master cache CSV",
        "description": "List of source text sections that contained usable content for analysis.",
    },
    {
        "field": "evidence_word_count",
        "category": "Sentiment analysis",
        "appears_in": "Scored upload CSV and master cache CSV",
        "description": "Word count across the usable evidence considered during analysis.",
    },
    {
        "field": "sentiment_low_evidence_flag",
        "category": "Sentiment analysis",
        "appears_in": "Scored upload CSV and master cache CSV",
        "description": "True when the sentiment result is based on limited evidence and should be interpreted cautiously.",
    },
    {
        "field": "severe_distress_flag",
        "category": "Safeguarding",
        "appears_in": "Scored upload CSV and master cache CSV",
        "description": "True when severe distress or extreme negative cues were detected in the testimonial.",
    },
    {
        "field": "manual_review_required",
        "category": "Safeguarding",
        "appears_in": "Scored upload CSV and master cache CSV",
        "description": "True when the row should be reviewed manually before operational or marketing use.",
    },
    {
        "field": "manual_review_reason",
        "category": "Safeguarding",
        "appears_in": "Scored upload CSV and master cache CSV",
        "description": "Reason the row was flagged for manual review.",
    },
    {
        "field": "analysis_notes",
        "category": "Sentiment analysis",
        "appears_in": "Scored upload CSV and master cache CSV",
        "description": "Short diagnostic note from the sentiment analysis stage.",
    },
    {
        "field": "quote_short_marketing",
        "category": "Quote extraction",
        "appears_in": "Scored upload CSV and master cache CSV",
        "description": "Short verbatim quote intended for marketing-style use when a suitable quote exists.",
    },
    {
        "field": "quote_long_case_study",
        "category": "Quote extraction",
        "appears_in": "Scored upload CSV and master cache CSV",
        "description": "Longer verbatim quote intended for case-study or fuller editorial use.",
    },
    {
        "field": "quote_source_field",
        "category": "Quote extraction",
        "appears_in": "Scored upload CSV and master cache CSV",
        "description": "Source text section from which the retained quote was taken, such as `why_recommend` or `why_aberdeen`.",
    },
    {
        "field": "quote_length_fit",
        "category": "Quote extraction",
        "appears_in": "Scored upload CSV and master cache CSV",
        "description": "Length suitability of the retained quote(s): `short_ready`, `long_ready`, `both`, or `none`.",
    },
    {
        "field": "quote_sentiment_fit",
        "category": "Quote extraction",
        "appears_in": "Scored upload CSV and master cache CSV",
        "description": "How suitable the quote sentiment is for reuse: `strong_positive`, `positive`, `neutral`, `mixed`, `negative`, or `not_suitable`.",
    },
    {
        "field": "quote_uniqueness_score_1_5",
        "category": "Quote extraction",
        "appears_in": "Scored upload CSV and master cache CSV",
        "description": "Distinctiveness score for the quote from 1 to 5, where higher means more specific and less generic.",
    },
    {
        "field": "quote_selection_notes",
        "category": "Quote extraction",
        "appears_in": "Scored upload CSV and master cache CSV",
        "description": "Short note explaining the selected quote or why no quote was kept.",
    },
    {
        "field": "analysis_status",
        "category": "Operational metadata",
        "appears_in": "Scored upload CSV and master cache CSV",
        "description": "Outcome of the sentiment analysis stage, typically `ok` or `fallback`.",
    },
    {
        "field": "analysis_mode",
        "category": "Operational metadata",
        "appears_in": "Scored upload CSV and master cache CSV",
        "description": "Execution mode used for sentiment analysis, such as `structured` or `prompt_only_json`.",
    },
    {
        "field": "analysis_error_type",
        "category": "Operational metadata",
        "appears_in": "Scored upload CSV and master cache CSV",
        "description": "Normalized error type when sentiment analysis does not complete cleanly.",
    },
    {
        "field": "analysis_error_message",
        "category": "Operational metadata",
        "appears_in": "Scored upload CSV and master cache CSV",
        "description": "Raw error message captured from the sentiment analysis stage.",
    },
    {
        "field": "analysis_raw_text",
        "category": "Operational metadata",
        "appears_in": "Scored upload CSV and master cache CSV",
        "description": "Raw model response text from the sentiment analysis call, kept for debugging.",
    },
    {
        "field": "quote_status",
        "category": "Operational metadata",
        "appears_in": "Scored upload CSV and master cache CSV",
        "description": "Outcome of quote extraction: `ok_short_only`, `ok_long_only`, `ok_short_and_long`, or `skipped`.",
    },
    {
        "field": "quote_mode",
        "category": "Operational metadata",
        "appears_in": "Scored upload CSV and master cache CSV",
        "description": "Execution mode used for quote extraction, including deterministic fallback when applicable.",
    },
    {
        "field": "quote_error_type",
        "category": "Operational metadata",
        "appears_in": "Scored upload CSV and master cache CSV",
        "description": "Normalized quote skip or failure reason, such as `no_consent`, `too_short`, or `verbatim_quote_unusable`.",
    },
    {
        "field": "quote_error_message",
        "category": "Operational metadata",
        "appears_in": "Scored upload CSV and master cache CSV",
        "description": "Raw error message captured from the quote extraction stage.",
    },
    {
        "field": "quote_raw_text",
        "category": "Operational metadata",
        "appears_in": "Scored upload CSV and master cache CSV",
        "description": "Raw model response text from the quote extraction call, kept for debugging.",
    },
    {
        "field": "themes_tags",
        "category": "Derived scoring",
        "appears_in": "Scored upload CSV and master cache CSV",
        "description": "List version of `themes_tags_csv` after normalization and de-duplication.",
    },
    {
        "field": "regular_theme_tags",
        "category": "Derived scoring",
        "appears_in": "Scored upload CSV and master cache CSV",
        "description": "Subset of theme tags that recur across the dataset often enough to count as regular themes.",
    },
    {
        "field": "short_quote_word_count",
        "category": "Derived scoring",
        "appears_in": "Scored upload CSV and master cache CSV",
        "description": "Word count of `quote_short_marketing`.",
    },
    {
        "field": "long_quote_word_count",
        "category": "Derived scoring",
        "appears_in": "Scored upload CSV and master cache CSV",
        "description": "Word count of `quote_long_case_study`.",
    },
    {
        "field": "quote_length_points",
        "category": "Derived scoring",
        "appears_in": "Scored upload CSV and master cache CSV",
        "description": "Length-based contribution to the overall quote quality score.",
    },
    {
        "field": "quote_sentiment_points",
        "category": "Derived scoring",
        "appears_in": "Scored upload CSV and master cache CSV",
        "description": "Sentiment-fit contribution to the overall quote quality score.",
    },
    {
        "field": "quote_uniqueness_points",
        "category": "Derived scoring",
        "appears_in": "Scored upload CSV and master cache CSV",
        "description": "Uniqueness-based contribution to the overall quote quality score.",
    },
    {
        "field": "quote_quality_score_0_100",
        "category": "Derived scoring",
        "appears_in": "Scored upload CSV and master cache CSV",
        "description": "Combined quote quality score derived from length, sentiment fit, and uniqueness.",
    },
    {
        "field": "quote_marketing_ready",
        "category": "Derived scoring",
        "appears_in": "Scored upload CSV and master cache CSV",
        "description": "True when a short quote meets the marketing-use thresholds and consent is present.",
    },
    {
        "field": "quote_case_study_ready",
        "category": "Derived scoring",
        "appears_in": "Scored upload CSV and master cache CSV",
        "description": "True when a longer quote meets the case-study-use thresholds and consent is present.",
    },
    {
        "field": "quote_internal_highlight_ready",
        "category": "Derived scoring",
        "appears_in": "Scored upload CSV and master cache CSV",
        "description": "True when the quote is usable for internal highlighting even if it is not marketing-ready.",
    },
]

OVERALL_LABELS = [
    "very_positive",
    "positive",
    "mixed",
    "neutral",
    "negative",
    "very_negative",
]

AREA_LABELS = [
    "positive",
    "mixed",
    "neutral",
    "negative",
    "not_mentioned",
]

QUOTE_LENGTH_LABELS = ["short_ready", "long_ready", "both", "none"]
QUOTE_SENTIMENT_LABELS = ["strong_positive", "positive", "neutral", "mixed", "negative", "not_suitable"]

QUOTE_SOURCE_PRIORITY = {
    "why_recommend": 0,
    "why_aberdeen": 1,
    "anything_else": 2,
    "learning_helped": 3,
    "typical_day": 4,
    "negative_feedback": 9,
}

GENERIC_QUOTE_VALUES = {
    "",
    "yes",
    "no",
    "ok",
    "okay",
    "na",
    "n/a",
    "ntr",
    "other",
    "reputation",
    "referral",
    "course variety",
    "online learning",
    "only available course",
}

QUOTE_VALUE_HINTS = [
    "career",
    "job",
    "support",
    "community",
    "reputation",
    "flexible",
    "flexibility",
    "affordable",
    "accessible",
    "inclusive",
    "research",
    "industry",
    "energy",
    "clinical",
    "nutrition",
    "website",
    "platform",
    "teaching",
    "tutor",
    "lecturer",
    "employability",
    "qualification",
    "pathway",
]

ANALYSIS_SCHEMA = {
    "type": "object",
    "propertyOrdering": [
        "score",
        "label",
        "summary",
        "themes_csv",
        "uni",
        "tut",
        "vle",
        "content",
        "jobs",
        "confidence",
        "notes",
    ],
    "properties": {
        "score": {"type": "number", "minimum": 0, "maximum": 100},
        "label": {"type": "string", "enum": OVERALL_LABELS},
        "summary": {"type": "string"},
        "themes_csv": {"type": "string"},
        "uni": {"type": "string", "enum": AREA_LABELS},
        "tut": {"type": "string", "enum": AREA_LABELS},
        "vle": {"type": "string", "enum": AREA_LABELS},
        "content": {"type": "string", "enum": AREA_LABELS},
        "jobs": {"type": "string", "enum": AREA_LABELS},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "notes": {"type": "string"},
    },
    "required": ["score", "label", "summary", "themes_csv", "uni", "tut", "vle", "content", "jobs", "confidence", "notes"],
    "additionalProperties": False,
}

QUOTE_SCHEMA = {
    "type": "object",
    "propertyOrdering": [
        "short_quote",
        "long_quote",
        "length_fit",
        "sentiment_fit",
        "unique_score",
        "notes",
    ],
    "properties": {
        "short_quote": {"type": "string"},
        "long_quote": {"type": "string"},
        "length_fit": {"type": "string", "enum": QUOTE_LENGTH_LABELS},
        "sentiment_fit": {"type": "string", "enum": QUOTE_SENTIMENT_LABELS},
        "unique_score": {"type": "integer", "minimum": 1, "maximum": 5},
        "notes": {"type": "string"},
    },
    "required": ["short_quote", "long_quote", "length_fit", "sentiment_fit", "unique_score", "notes"],
    "additionalProperties": False,
}

ANALYSIS_SYSTEM = """Analyse University of Aberdeen Online testimonial comments.
Return JSON only.

Rules:
- Use COMMENT_TEXT as the main evidence and RECOMMEND only as minor context.
- Use the full 0-100 sentiment scale carefully:
  95-100 = exceptional praise with no meaningful criticism
  80-94 = clearly positive with only minor caveats
  60-79 = mostly positive but with meaningful caveats
  45-59 = mixed or close to neutral
  25-44 = clearly negative but not severe
  10-24 = very negative
  0-9 = only for extreme negative wording or severe harm
- If label=mixed, keep the score between 40 and 68.
- If label=very_positive and the student gives clear criticism, keep the score below 95.
- If label=very_negative, only go below 10 for truly extreme wording.
- Keep summary to one sentence and at most 24 words.
- themes_csv must be 2 to 6 concise lower-case tags separated by commas.
- Use not_mentioned when an area is not clearly discussed.
- notes must be short and factual.
- Do not invent details.
"""

QUOTE_SYSTEM = """Select quotes from University of Aberdeen Online testimonial comments.
Return JSON only.

Rules:
- Copy quotes from COMMENT_TEXT only. Do not paraphrase or invent.
- Prefer exact wording that is specific, distinctive, and usable in marketing.
- short_quote should usually be 8 to 25 words.
- long_quote should usually be 1 to 3 sentences and at most 75 words.
- If no good quote exists, return empty strings, length_fit=none, sentiment_fit=not_suitable, and explain briefly in notes.
"""

POSITIVE_CUES = [
    "excellent",
    "great",
    "fantastic",
    "amazing",
    "supportive",
    "helpful",
    "flexible",
    "engaging",
    "informative",
    "interesting",
    "easy to navigate",
    "user-friendly",
    "well-structured",
    "high-quality",
    "valuable",
    "recommend",
    "career progression",
    "good reputation",
    "prestigious",
]

NEGATIVE_CUES = [
    "poor",
    "bad",
    "disappoint",
    "frustrat",
    "difficult",
    "hard to navigate",
    "lack",
    "no support",
    "slow",
    "broken",
    "outdated",
    "inconsistent",
    "limited support",
    "poor value",
    "waste of money",
    "confusing",
    "unresponsive",
    "temperamental",
    "mediocre",
    "below the academic standard",
]

SEVERE_NEGATIVE_CUES = [
    "depression",
    "lost 5k",
    "wasted money",
    "want to shoot myself",
    "diabolical",
    "mind numbing",
    "haggard",
    "loss of will to go on",
]

CONTRAST_CUES = [
    "but",
    "however",
    "although",
    "despite",
    "though",
    "while",
    "except",
]


def clean(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = re.sub(r"\s+", " ", str(value)).strip()
    return "" if text.lower() in {"nan", "none", "null"} else text


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", clean(text)).strip()


def normalize_for_hash(text: str) -> str:
    return normalize_whitespace(text).lower()


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z']+", clean(text)))


def yes_like(value: Any) -> bool:
    return clean(value).lower() in {"yes", "y", "true", "1"}


def normalize_row_id(value: Any) -> str:
    text = clean(value)
    if re.fullmatch(r"-?\d+\.0+", text):
        return text.split(".", 1)[0]
    return text


def parse_json_safely(text: str) -> Optional[Dict[str, Any]]:
    text = clean(text)
    if not text:
        return None
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def backoff_sleep(attempt: int) -> None:
    time.sleep((2**attempt) * 0.6 + random.random() * 0.3)


def make_config(system_instruction: str, schema: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    config: Dict[str, Any] = {
        "system_instruction": system_instruction,
        "temperature": 0,
        "response_mime_type": "application/json",
    }
    if schema is not None:
        config["response_json_schema"] = schema
    return config


def get_client(api_key: str):
    from google import genai

    return genai.Client(api_key=clean(api_key))


def available_text_cols(df: pd.DataFrame) -> List[Tuple[str, str]]:
    return [(short_label, column_name) for short_label, column_name in TEXT_COLS if column_name in df.columns]


def build_comment_text(row: pd.Series, text_cols: Sequence[Tuple[str, str]]) -> str:
    parts = []
    for short_label, column_name in text_cols:
        value = clean(row.get(column_name, ""))
        if value:
            parts.append(f"{short_label}: {value}")
    return "\n".join(parts)


def parse_comment_sections(comment_text: str) -> List[Tuple[str, str]]:
    sections: List[Tuple[str, str]] = []
    if pd.isna(comment_text):
        return sections

    for raw_line in str(comment_text).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if ":" in line:
            label, text = line.split(":", 1)
            label = clean(label)
            text = normalize_whitespace(text)
        else:
            label = "comment"
            text = normalize_whitespace(line)
        if text:
            sections.append((label, text))
    return sections


def split_sentences(text: str) -> List[str]:
    raw = normalize_whitespace(text)
    if not raw:
        return []
    sentences = [part.strip() for part in re.findall(r"[^.!?]+(?:[.!?]+|$)", raw) if clean(part)]
    return sentences or [raw]


def iter_verbatim_quote_texts(text: str) -> List[str]:
    raw = normalize_whitespace(text)
    if not raw:
        return []
    candidates: List[str] = []
    for candidate in [raw] + split_sentences(raw):
        normalized = normalize_whitespace(candidate)
        if normalized and normalized not in candidates:
            candidates.append(normalized)
    return candidates


def is_generic_quote_text(text: str) -> bool:
    lowered = normalize_for_hash(text)
    if lowered in GENERIC_QUOTE_VALUES:
        return True
    if word_count(lowered) < 4:
        return True
    return False


def quote_candidate_score(text: str, label: str, target: str) -> float:
    lowered = normalize_for_hash(text)
    words = word_count(text)
    positive_hits = count_cues(lowered, POSITIVE_CUES)
    value_hits = sum(1 for hint in QUOTE_VALUE_HINTS if hint in lowered)
    long_words = len({word.lower() for word in re.findall(r"[A-Za-z']+", text) if len(word) >= 7})
    priority_bonus = max(0, 24 - QUOTE_SOURCE_PRIORITY.get(label, 9) * 4)

    score = float(priority_bonus + positive_hits * 7 + value_hits * 3 + min(long_words, 6))
    if target == "short":
        if 8 <= words <= 25:
            score += 12
        elif 6 <= words <= 30:
            score += 6
        else:
            score -= abs(words - 16)
    else:
        if 18 <= words <= 75:
            score += 10
        elif 12 <= words <= 75:
            score += 4
        else:
            score -= 10

    if text.endswith("?"):
        score -= 6
    return score


def build_quote_candidate_profile(comment_text: str) -> Dict[str, Any]:
    sections = parse_comment_sections(comment_text)
    short_candidates_by_text: Dict[str, Dict[str, Any]] = {}
    long_candidates_by_text: Dict[str, Dict[str, Any]] = {}
    max_section_words = 0

    for label, text in sections:
        section_words = word_count(text)
        max_section_words = max(max_section_words, section_words)
        if label == "negative_feedback":
            continue

        for candidate_text in iter_verbatim_quote_texts(text):
            candidate_words = word_count(candidate_text)
            if is_generic_quote_text(candidate_text) or candidate_words < 6:
                continue

            if candidate_words <= 30:
                score = quote_candidate_score(candidate_text, label, "short")
                current = short_candidates_by_text.get(candidate_text)
                if current is None or score > current["score"]:
                    short_candidates_by_text[candidate_text] = {
                        "text": candidate_text,
                        "label": label,
                        "word_count": candidate_words,
                        "score": score,
                    }

            if 12 <= candidate_words <= 75:
                score = quote_candidate_score(candidate_text, label, "long")
                current = long_candidates_by_text.get(candidate_text)
                if current is None or score > current["score"]:
                    long_candidates_by_text[candidate_text] = {
                        "text": candidate_text,
                        "label": label,
                        "word_count": candidate_words,
                        "score": score,
                    }

    short_candidates = sorted(short_candidates_by_text.values(), key=lambda item: (-item["score"], item["word_count"]))
    long_candidates = sorted(long_candidates_by_text.values(), key=lambda item: (-item["score"], item["word_count"]))
    return {
        "sections": sections,
        "source_fields_present": [label for label, _ in sections],
        "short_candidates": short_candidates,
        "long_candidates": long_candidates,
        "has_promising_candidate": bool(short_candidates or long_candidates),
        "max_section_words": max_section_words,
    }


def best_quote_candidate(profile: Dict[str, Any], target: str) -> Optional[Dict[str, Any]]:
    candidates = profile.get(f"{target}_candidates", [])
    return candidates[0] if candidates else None


def prefix_quote(text: str, max_words: int = 24) -> str:
    matches = list(re.finditer(r"[A-Za-z']+", text))
    if len(matches) < 6:
        return ""
    end = matches[min(len(matches), max_words) - 1].end()
    return text[:end].strip()


def derive_short_quote_from_text(text: str) -> str:
    normalized = clean(text)
    if not normalized:
        return ""

    for sentence in split_sentences(normalized):
        words = word_count(sentence)
        if 6 <= words <= 30 and not is_generic_quote_text(sentence):
            return sentence

    words = word_count(normalized)
    if 6 <= words <= 30 and not is_generic_quote_text(normalized):
        return normalized

    prefix = prefix_quote(normalized, 24)
    if 6 <= word_count(prefix) <= 30 and not is_generic_quote_text(prefix):
        return prefix
    return ""


def compute_quote_length_fit(short_quote: str, long_quote: str) -> str:
    has_short = bool(clean(short_quote))
    has_long = bool(clean(long_quote))
    if has_short and has_long:
        return "both"
    if has_short:
        return "short_ready"
    if has_long:
        return "long_ready"
    return "none"


def infer_quote_sentiment_fit(label: str, score: float) -> str:
    if score >= 90 or label == "very_positive":
        return "strong_positive"
    if score >= 70 or label == "positive":
        return "positive"
    if label == "mixed":
        return "mixed"
    if label == "neutral":
        return "neutral"
    return "negative"


def estimate_quote_uniqueness(text: str, label: str) -> int:
    lowered = normalize_for_hash(text)
    long_words = {word.lower() for word in re.findall(r"[A-Za-z']+", clean(text)) if len(word) >= 7}
    score = 1
    if len(long_words) >= 2:
        score += 1
    if len(long_words) >= 5:
        score += 1
    if any(hint in lowered for hint in QUOTE_VALUE_HINTS):
        score += 1
    if label in {"why_aberdeen", "anything_else"}:
        score += 1
    return min(5, score)


def find_quote_source_fields(quotes: Sequence[str], sections: Sequence[Tuple[str, str]]) -> List[str]:
    labels: List[str] = []
    for quote in quotes:
        if not clean(quote):
            continue
        for label, text in sections:
            if is_verbatim_quote(quote, text) and label not in labels:
                labels.append(label)
    return labels


def join_note_parts(*parts: Any) -> str:
    merged: List[str] = []
    for part in parts:
        if isinstance(part, (list, tuple)):
            values = part
        else:
            values = [part]
        for value in values:
            normalized = clean(value)
            if normalized and normalized not in merged:
                merged.append(normalized)
    return " ".join(merged)


def quote_status_from_payload(payload: Dict[str, Any]) -> str:
    has_short = bool(clean(payload.get("short_quote", "")))
    has_long = bool(clean(payload.get("long_quote", "")))
    if has_short and has_long:
        return "ok_short_and_long"
    if has_short:
        return "ok_short_only"
    if has_long:
        return "ok_long_only"
    return "skipped"


def quote_status_is_ok(value: Any) -> bool:
    return clean(value).startswith("ok")


def quote_skip_reason(
    analysis: Dict[str, Any],
    marketing_consent_yes: bool,
    comment_text: str,
    profile: Dict[str, Any],
) -> str:
    if not marketing_consent_yes:
        return "no_consent"
    if analysis["label"] not in {"very_positive", "positive", "mixed"} or analysis["score"] < 60:
        return "low_sentiment"
    if profile["has_promising_candidate"]:
        return ""
    if profile["max_section_words"] < 6 or word_count(comment_text) < 12:
        return "too_short"
    return "fragmented_text"


def evidence_confidence_cap(comment_text: str) -> float:
    evidence_words = word_count(comment_text)
    if evidence_words < 15:
        return 0.72
    if evidence_words < 25:
        return 0.82
    if evidence_words < 40:
        return 0.9
    return 1.0


def build_manual_review_flags(comment_text: str, label: str, score: float) -> Dict[str, Any]:
    sections = parse_comment_sections(comment_text)
    evidence_words = word_count(comment_text)
    low_evidence = evidence_words < 25
    severe_distress = count_cues(comment_text, SEVERE_NEGATIVE_CUES) > 0
    reasons: List[str] = []

    if severe_distress:
        reasons.append("severe_distress")
    if low_evidence and ((label == "very_positive" and score >= 88) or (label == "very_negative" and score <= 12)):
        reasons.append("low_evidence_extreme_sentiment")

    return {
        "source_fields_present": ", ".join(label_name for label_name, _ in sections),
        "evidence_word_count": evidence_words,
        "sentiment_low_evidence_flag": low_evidence,
        "severe_distress_flag": severe_distress,
        "manual_review_required": bool(reasons),
        "manual_review_reason": ", ".join(reasons),
    }


def build_content_hash(row: pd.Series) -> str:
    payload = "\n||\n".join(
        [
            normalize_for_hash(row.get("comment_text", "")),
            normalize_for_hash(row.get(RECOMMEND_COL, "")),
            normalize_for_hash(row.get(CONSENT_COL, "")),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_row_key(row: pd.Series) -> str:
    response_id = normalize_row_id(row.get("Response ID", ""))
    if response_id:
        return f"response_id::{response_id}"
    return f"content_hash::{build_content_hash(row)}"


def prepare_input_df(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    text_cols = available_text_cols(prepared)
    prepared["comment_text"] = prepared.apply(lambda row: build_comment_text(row, text_cols), axis=1)
    prepared["comment_word_count"] = prepared["comment_text"].fillna("").astype(str).str.findall(r"[A-Za-z']+").str.len()
    prepared["marketing_consent_yes"] = prepared[CONSENT_COL].apply(yes_like) if CONSENT_COL in prepared.columns else True
    prepared["row_key"] = prepared.apply(build_row_key, axis=1)
    prepared["content_hash"] = prepared.apply(build_content_hash, axis=1)
    prepared["pipeline_version"] = PIPELINE_VERSION
    return prepared


def normalize_existing_cache_df(df: pd.DataFrame) -> pd.DataFrame:
    cache_df = df.copy()

    if "comment_text" not in cache_df.columns:
        text_cols = available_text_cols(cache_df)
        if text_cols:
            cache_df["comment_text"] = cache_df.apply(lambda row: build_comment_text(row, text_cols), axis=1)
        else:
            cache_df["comment_text"] = ""
    if "comment_word_count" not in cache_df.columns:
        cache_df["comment_word_count"] = cache_df["comment_text"].fillna("").astype(str).str.findall(r"[A-Za-z']+").str.len()
    if "marketing_consent_yes" not in cache_df.columns:
        cache_df["marketing_consent_yes"] = cache_df[CONSENT_COL].apply(yes_like) if CONSENT_COL in cache_df.columns else True
    if "row_key" not in cache_df.columns:
        cache_df["row_key"] = cache_df.apply(build_row_key, axis=1)
    if "content_hash" not in cache_df.columns:
        cache_df["content_hash"] = cache_df.apply(build_content_hash, axis=1)
    if "pipeline_version" not in cache_df.columns:
        cache_df["pipeline_version"] = ""

    for column in RESULT_COLUMNS:
        if column not in cache_df.columns:
            cache_df[column] = ""

    return cache_df


def coerce_enum(value: Any, allowed: Sequence[str], default: str) -> str:
    value = clean(value)
    return value if value in allowed else default


def normalize_themes_csv(value: Any) -> List[str]:
    raw = clean(value).lower()
    if not raw:
        return []
    tags = []
    for part in raw.split(","):
        tag = re.sub(r"[^a-z0-9 /&+-]", "", part).strip()
        tag = re.sub(r"\s+", " ", tag)
        if tag and tag not in tags:
            tags.append(tag)
    return tags[:6]


def generated_field_dictionary_df(include_cache_only: bool = True) -> pd.DataFrame:
    dictionary = pd.DataFrame(GENERATED_FIELD_METADATA)
    if not include_cache_only:
        dictionary = dictionary[dictionary["appears_in"] != "Master cache CSV only"].copy()
    return dictionary[["field", "category", "appears_in", "description"]]


def count_cues(text: str, cues: Sequence[str]) -> int:
    lowered = clean(text).lower()
    return sum(1 for cue in cues if cue in lowered)


def clamp_score(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def recalibrate_sentiment_score(
    raw_score: float,
    label: str,
    comment_text: str,
    summary: str,
    notes: str,
) -> float:
    evidence_text = " ".join([clean(comment_text), clean(summary), clean(notes)]).lower()
    evidence_words = word_count(comment_text)
    positive_hits = count_cues(evidence_text, POSITIVE_CUES)
    negative_hits = count_cues(evidence_text, NEGATIVE_CUES)
    severe_negative_hits = count_cues(evidence_text, SEVERE_NEGATIVE_CUES)
    contrast_hits = count_cues(evidence_text, CONTRAST_CUES)

    base_scores = {
        "very_positive": 90.0,
        "positive": 76.0,
        "mixed": 55.0,
        "neutral": 50.0,
        "negative": 32.0,
        "very_negative": 20.0,
    }
    band_limits = {
        "very_positive": (88.0, 98.0),
        "positive": (70.0, 89.0),
        "mixed": (40.0, 68.0),
        "neutral": (48.0, 58.0),
        "negative": (25.0, 44.0),
        "very_negative": (12.0, 24.0),
    }

    score = base_scores.get(label, 50.0)

    if label == "very_positive":
        score += min(positive_hits * 1.8, 8.0)
        score -= min(negative_hits * 4.0 + contrast_hits * 2.0, 10.0)
    elif label == "positive":
        score += min(positive_hits * 1.5, 7.0)
        score -= min(negative_hits * 3.0 + contrast_hits * 2.0, 9.0)
    elif label == "mixed":
        score += clamp_score((positive_hits - negative_hits) * 3.0, -10.0, 10.0)
        score -= min(contrast_hits * 1.5, 4.0)
    elif label == "neutral":
        score += clamp_score((positive_hits - negative_hits) * 1.5, -4.0, 4.0)
    elif label == "negative":
        score -= min(negative_hits * 2.0 + contrast_hits * 1.0, 7.0)
        score += min(positive_hits * 2.5, 8.0)
    elif label == "very_negative":
        score -= min(negative_hits * 1.0, 3.0)
        score += min(positive_hits * 3.0, 8.0)
        if severe_negative_hits:
            score = min(score, 8.0)
            score -= min(severe_negative_hits * 2.0, 5.0)

    score += (raw_score - 50.0) * 0.05

    if severe_negative_hits == 0:
        if evidence_words < 15:
            score = 50.0 + (score - 50.0) * 0.72
        elif evidence_words < 25:
            score = 50.0 + (score - 50.0) * 0.85

    lower, upper = band_limits.get(label, (0.0, 100.0))
    if label == "very_positive" and (negative_hits > 0 or contrast_hits > 0):
        upper = min(upper, 93.0)
    if label == "positive" and (negative_hits > 0 or contrast_hits > 1):
        upper = min(upper, 83.0)
    if label == "very_positive":
        if evidence_words < 15:
            upper = min(upper, 84.0)
        elif evidence_words < 25:
            upper = min(upper, 88.0)
        elif evidence_words < 40 and positive_hits < 2:
            upper = min(upper, 91.0)
    if label == "positive" and evidence_words < 15:
        upper = min(upper, 78.0)
    if label == "very_negative" and severe_negative_hits:
        lower = 1.0
        upper = min(upper, 12.0)

    return round(clamp_score(score, lower, upper), 1)


def analysis_fallback(note: str) -> Dict[str, Any]:
    return {
        "score": 50.0,
        "raw_score_model": 50.0,
        "label": "neutral",
        "summary": "",
        "themes_csv": "",
        "uni": "not_mentioned",
        "tut": "not_mentioned",
        "vle": "not_mentioned",
        "content": "not_mentioned",
        "jobs": "not_mentioned",
        "confidence": 0.0,
        "source_fields_present": "",
        "evidence_word_count": 0,
        "sentiment_low_evidence_flag": False,
        "severe_distress_flag": False,
        "manual_review_required": False,
        "manual_review_reason": "",
        "notes": note,
    }


def quote_fallback(note: str) -> Dict[str, Any]:
    return {
        "short_quote": "",
        "long_quote": "",
        "source_field": "",
        "length_fit": "none",
        "sentiment_fit": "not_suitable",
        "unique_score": 1,
        "notes": note,
    }


def normalize_analysis_payload(payload: Dict[str, Any], comment_text: str) -> Dict[str, Any]:
    result = analysis_fallback("normalized")
    raw_score = pd.to_numeric(payload.get("score", result["score"]), errors="coerce")
    raw_score = float(raw_score) if pd.notna(raw_score) else 50.0
    raw_score = max(0.0, min(100.0, raw_score))
    result["label"] = coerce_enum(payload.get("label", result["label"]), OVERALL_LABELS, "neutral")
    result["summary"] = clean(payload.get("summary", ""))[:240]
    result["themes_csv"] = ", ".join(normalize_themes_csv(payload.get("themes_csv", "")))
    result["uni"] = coerce_enum(payload.get("uni", "not_mentioned"), AREA_LABELS, "not_mentioned")
    result["tut"] = coerce_enum(payload.get("tut", "not_mentioned"), AREA_LABELS, "not_mentioned")
    result["vle"] = coerce_enum(payload.get("vle", "not_mentioned"), AREA_LABELS, "not_mentioned")
    result["content"] = coerce_enum(payload.get("content", "not_mentioned"), AREA_LABELS, "not_mentioned")
    result["jobs"] = coerce_enum(payload.get("jobs", "not_mentioned"), AREA_LABELS, "not_mentioned")
    result["confidence"] = pd.to_numeric(payload.get("confidence", result["confidence"]), errors="coerce")
    result["confidence"] = float(result["confidence"]) if pd.notna(result["confidence"]) else 0.0
    result["confidence"] = max(0.0, min(1.0, result["confidence"]))
    result["notes"] = clean(payload.get("notes", ""))
    result["raw_score_model"] = raw_score
    result["score"] = recalibrate_sentiment_score(
        raw_score=raw_score,
        label=result["label"],
        comment_text=comment_text,
        summary=result["summary"],
        notes=result["notes"],
    )
    result["confidence"] = min(result["confidence"], evidence_confidence_cap(comment_text))
    result.update(build_manual_review_flags(comment_text, result["label"], result["score"]))
    return result


def is_verbatim_quote(quote: str, comment_text: str) -> bool:
    normalized_quote = normalize_whitespace(quote).lower()
    normalized_comment = normalize_whitespace(comment_text).lower()
    return bool(normalized_quote) and normalized_quote in normalized_comment


def normalize_quote_payload(payload: Dict[str, Any], comment_text: str) -> Dict[str, Any]:
    result = quote_fallback("normalized")
    short_quote = clean(payload.get("short_quote", ""))
    long_quote = clean(payload.get("long_quote", ""))
    attempted_quote = bool(short_quote or long_quote)

    if short_quote and not is_verbatim_quote(short_quote, comment_text):
        short_quote = ""
    if long_quote and not is_verbatim_quote(long_quote, comment_text):
        long_quote = ""

    result["short_quote"] = short_quote
    result["long_quote"] = long_quote
    result["source_field"] = ""
    result["length_fit"] = coerce_enum(payload.get("length_fit", "none"), QUOTE_LENGTH_LABELS, "none")
    result["sentiment_fit"] = coerce_enum(payload.get("sentiment_fit", "not_suitable"), QUOTE_SENTIMENT_LABELS, "not_suitable")
    unique_score = pd.to_numeric(payload.get("unique_score", 1), errors="coerce")
    unique_score = int(round(float(unique_score))) if pd.notna(unique_score) else 1
    result["unique_score"] = max(1, min(5, unique_score))
    result["notes"] = clean(payload.get("notes", ""))

    if not result["short_quote"] and not result["long_quote"]:
        result["length_fit"] = "none"
        result["sentiment_fit"] = "not_suitable"
        result["unique_score"] = 1
        if not result["notes"]:
            result["notes"] = "no verbatim quote selected"
        elif attempted_quote:
            result["notes"] = f"{result['notes']} Verbatim quote could not be preserved.".strip()

    return result


def apply_verbatim_quote_fallback(
    quote_payload: Dict[str, Any],
    comment_text: str,
    analysis: Dict[str, Any],
    profile: Dict[str, Any],
) -> Tuple[Dict[str, Any], bool]:
    result = quote_fallback("normalized")
    result.update(quote_payload)
    result["notes"] = clean(result["notes"]).replace("Verbatim quote could not be preserved.", "").strip()

    fallback_notes: List[str] = []
    used_fallback = False
    source_labels: List[str] = []

    if not clean(result["long_quote"]):
        long_candidate = best_quote_candidate(profile, "long")
        if long_candidate is not None:
            result["long_quote"] = long_candidate["text"]
            source_labels.append(long_candidate["label"])
            fallback_notes.append(f"Fallback long quote extracted from {long_candidate['label']}.")
            used_fallback = True

    if not clean(result["short_quote"]):
        derived_short = derive_short_quote_from_text(result["long_quote"])
        if derived_short:
            result["short_quote"] = derived_short
            fallback_notes.append("Short quote derived from verbatim long quote.")
            used_fallback = True
        else:
            short_candidate = best_quote_candidate(profile, "short")
            if short_candidate is not None:
                result["short_quote"] = short_candidate["text"]
                source_labels.append(short_candidate["label"])
                fallback_notes.append(f"Fallback short quote extracted from {short_candidate['label']}.")
                used_fallback = True

    if not clean(result["long_quote"]) and clean(result["short_quote"]) and word_count(result["short_quote"]) >= 12:
        result["long_quote"] = result["short_quote"]
        fallback_notes.append("Short quote reused as long quote.")
        used_fallback = True

    detected_labels = find_quote_source_fields([result["short_quote"], result["long_quote"]], profile["sections"])
    for label in source_labels + detected_labels:
        if label and label not in source_labels:
            source_labels.append(label)
    result["source_field"] = ", ".join(source_labels)
    result["length_fit"] = compute_quote_length_fit(result["short_quote"], result["long_quote"])

    if clean(result["short_quote"]) or clean(result["long_quote"]):
        if result["sentiment_fit"] == "not_suitable":
            result["sentiment_fit"] = infer_quote_sentiment_fit(analysis["label"], analysis["score"])
        source_label = source_labels[0] if source_labels else ""
        uniqueness_text = clean(result["long_quote"]) or clean(result["short_quote"])
        result["unique_score"] = max(result["unique_score"], estimate_quote_uniqueness(uniqueness_text, source_label))
        result["notes"] = join_note_parts(result["notes"], fallback_notes)
    else:
        result["length_fit"] = "none"
        result["sentiment_fit"] = "not_suitable"
        result["unique_score"] = 1

    return result, used_fallback


def call_model(client: Any, model: str, prompt: str, system_instruction: str, schema: Dict[str, Any], retries: int = 4) -> Tuple[Optional[Dict[str, Any]], Dict[str, str]]:
    last_type = ""
    last_message = ""
    last_raw = ""
    mode = "structured"

    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=make_config(system_instruction, schema if mode == "structured" else None),
            )
            last_raw = clean(getattr(response, "text", ""))
            parsed = parse_json_safely(last_raw)
            if parsed is None:
                raise ValueError("Could not parse JSON response")
            return parsed, {
                "status": "ok",
                "mode": mode,
                "error_type": "",
                "error_message": "",
                "raw_text": last_raw,
            }
        except Exception as exc:
            last_type = type(exc).__name__
            last_message = clean(str(exc))
            if mode == "structured" and last_type == "ClientError":
                mode = "prompt_only_json"
                backoff_sleep(attempt)
                continue
            backoff_sleep(attempt)

    return None, {
        "status": "fallback",
        "mode": mode,
        "error_type": last_type,
        "error_message": last_message,
        "raw_text": last_raw,
    }


def analyse_comment(client: Any, model: str, comment_text: str, recommend: str) -> Tuple[Dict[str, Any], Dict[str, str]]:
    if not clean(comment_text):
        return analysis_fallback("no_comment_text"), {
            "status": "fallback",
            "mode": "skipped",
            "error_type": "",
            "error_message": "",
            "raw_text": "",
        }

    prompt = f"COMMENT_TEXT:\n{comment_text}\n\nRECOMMEND:\n{recommend}"
    parsed, meta = call_model(client, model, prompt, ANALYSIS_SYSTEM, ANALYSIS_SCHEMA)
    if parsed is None:
        note = meta["error_type"] or "analysis_failed"
        return analysis_fallback(note), meta
    return normalize_analysis_payload(parsed, comment_text), meta


def should_extract_quote(
    analysis: Dict[str, Any],
    marketing_consent_yes: bool,
    comment_text: str,
    profile: Dict[str, Any],
) -> Tuple[bool, str]:
    reason = quote_skip_reason(analysis, marketing_consent_yes, comment_text, profile)
    return reason == "", reason


def extract_quote(
    client: Any,
    model: str,
    comment_text: str,
    analysis: Dict[str, Any],
    profile: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    if not clean(comment_text):
        return quote_fallback("no_comment_text"), {
            "status": "fallback",
            "mode": "skipped",
            "error_type": "",
            "error_message": "",
            "raw_text": "",
        }

    prompt = (
        f"COMMENT_TEXT:\n{comment_text}\n\n"
        f"ANALYSIS:\n"
        f"- label: {analysis['label']}\n"
        f"- score: {analysis['score']}\n"
        f"- summary: {analysis['summary']}\n"
        f"- themes_csv: {analysis['themes_csv']}\n"
    )
    parsed, meta = call_model(client, model, prompt, QUOTE_SYSTEM, QUOTE_SCHEMA)
    normalized_quote = normalize_quote_payload(parsed or {}, comment_text) if parsed is not None else quote_fallback(meta["error_type"] or "quote_failed")
    normalized_quote, used_fallback = apply_verbatim_quote_fallback(normalized_quote, comment_text, analysis, profile)
    final_status = quote_status_from_payload(normalized_quote)

    if final_status == "skipped":
        meta["status"] = "skipped"
        if not clean(meta.get("error_type", "")):
            meta["error_type"] = "verbatim_quote_unusable" if profile["has_promising_candidate"] else "fragmented_text"
        return normalized_quote, meta

    meta["status"] = final_status
    if used_fallback:
        if clean(meta.get("mode", "")) in {"structured", "prompt_only_json"}:
            meta["mode"] = f"{meta['mode']}+deterministic_fallback"
        else:
            meta["mode"] = "deterministic_fallback"
    meta["error_type"] = ""
    meta["error_message"] = ""
    return normalized_quote, meta


def build_result_payload(analysis: Dict[str, Any], quote: Dict[str, Any], analysis_meta: Dict[str, str], quote_meta: Dict[str, str]) -> Dict[str, Any]:
    return {
        "overall_sentiment_score_0_100": analysis["score"],
        "overall_sentiment_score_raw_model_0_100": analysis["raw_score_model"],
        "overall_sentiment_label": analysis["label"],
        "overall_feedback_summary_1_sentence": analysis["summary"],
        "themes_tags_csv": analysis["themes_csv"],
        "area_sentiment.university": analysis["uni"],
        "area_sentiment.tutors": analysis["tut"],
        "area_sentiment.vle_myaberdeen": analysis["vle"],
        "area_sentiment.learning_course_content": analysis["content"],
        "area_sentiment.job_prospects": analysis["jobs"],
        "confidence_0_1": analysis["confidence"],
        "source_fields_present": analysis["source_fields_present"],
        "evidence_word_count": analysis["evidence_word_count"],
        "sentiment_low_evidence_flag": analysis["sentiment_low_evidence_flag"],
        "severe_distress_flag": analysis["severe_distress_flag"],
        "manual_review_required": analysis["manual_review_required"],
        "manual_review_reason": analysis["manual_review_reason"],
        "analysis_notes": analysis["notes"],
        "quote_short_marketing": quote["short_quote"],
        "quote_long_case_study": quote["long_quote"],
        "quote_source_field": quote["source_field"],
        "quote_length_fit": quote["length_fit"],
        "quote_sentiment_fit": quote["sentiment_fit"],
        "quote_uniqueness_score_1_5": quote["unique_score"],
        "quote_selection_notes": quote["notes"],
        "analysis_status": analysis_meta["status"],
        "analysis_mode": analysis_meta["mode"],
        "analysis_error_type": analysis_meta["error_type"],
        "analysis_error_message": analysis_meta["error_message"],
        "analysis_raw_text": analysis_meta["raw_text"],
        "quote_status": quote_meta["status"],
        "quote_mode": quote_meta["mode"],
        "quote_error_type": quote_meta["error_type"],
        "quote_error_message": quote_meta["error_message"],
        "quote_raw_text": quote_meta["raw_text"],
    }


def analyse_rows(
    df: pd.DataFrame,
    api_key: str,
    model: str = MODEL_DEFAULT,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> pd.DataFrame:
    client = get_client(api_key)
    records: List[Dict[str, Any]] = []
    total = len(df)

    for position, (_, row) in enumerate(df.iterrows(), start=1):
        comment_text = clean(row.get("comment_text", ""))
        recommend = clean(row.get(RECOMMEND_COL, ""))
        analysis, analysis_meta = analyse_comment(client, model, comment_text, recommend)
        profile = build_quote_candidate_profile(comment_text)

        should_extract, skip_reason = should_extract_quote(analysis, bool(row.get("marketing_consent_yes", True)), comment_text, profile)
        if should_extract:
            quote, quote_meta = extract_quote(client, model, comment_text, analysis, profile)
        else:
            quote = quote_fallback(skip_reason)
            quote_meta = {
                "status": "skipped",
                "mode": "skipped",
                "error_type": skip_reason,
                "error_message": "",
                "raw_text": "",
            }

        record = {
            "row_key": row["row_key"],
            "content_hash": row["content_hash"],
            "pipeline_version": PIPELINE_VERSION,
        }
        record.update(build_result_payload(analysis, quote, analysis_meta, quote_meta))
        records.append(record)

        if progress_callback is not None:
            progress_callback(position, total, normalize_row_id(row.get("Response ID", "")) or row["row_key"])

    return pd.DataFrame(records)


def compute_quote_scores(df_scored: pd.DataFrame) -> pd.DataFrame:
    df_scored = df_scored.copy()
    for column in [
        "quote_short_marketing",
        "quote_long_case_study",
        "quote_length_fit",
        "quote_sentiment_fit",
        "quote_uniqueness_score_1_5",
        "quote_status",
    ]:
        if column not in df_scored.columns:
            df_scored[column] = ""

    if "marketing_consent_yes" not in df_scored.columns:
        df_scored["marketing_consent_yes"] = False
    consent_series = df_scored["marketing_consent_yes"].apply(yes_like)

    df_scored["quote_uniqueness_score_1_5"] = pd.to_numeric(df_scored["quote_uniqueness_score_1_5"], errors="coerce").fillna(1).clip(1, 5).round().astype(int)
    df_scored["short_quote_word_count"] = df_scored["quote_short_marketing"].fillna("").astype(str).str.findall(r"[A-Za-z']+").str.len()
    df_scored["long_quote_word_count"] = df_scored["quote_long_case_study"].fillna("").astype(str).str.findall(r"[A-Za-z']+").str.len()

    quote_present = (
        df_scored["quote_short_marketing"].fillna("").astype(str).str.strip().ne("")
        | df_scored["quote_long_case_study"].fillna("").astype(str).str.strip().ne("")
    )
    quote_status_series = df_scored["quote_status"].fillna("").astype(str)
    valid_quote = quote_present & quote_status_series.apply(quote_status_is_ok)
    short_ok = quote_status_series.isin(["ok_short_only", "ok_short_and_long"])
    long_ok = quote_status_series.isin(["ok_long_only", "ok_short_and_long"])

    length_points = {"both": 25, "short_ready": 18, "long_ready": 18, "none": 0}
    sentiment_points = {
        "strong_positive": 45,
        "positive": 35,
        "neutral": 20,
        "mixed": 10,
        "negative": 0,
        "not_suitable": 0,
    }

    df_scored["quote_length_points"] = df_scored["quote_length_fit"].map(length_points).fillna(0)
    df_scored["quote_sentiment_points"] = df_scored["quote_sentiment_fit"].map(sentiment_points).fillna(0)
    df_scored["quote_uniqueness_points"] = ((df_scored["quote_uniqueness_score_1_5"] - 1) / 4 * 30).round(2)
    df_scored.loc[~valid_quote, "quote_uniqueness_score_1_5"] = 1
    df_scored.loc[~valid_quote, "quote_length_points"] = 0
    df_scored.loc[~valid_quote, "quote_sentiment_points"] = 0
    df_scored.loc[~valid_quote, "quote_uniqueness_points"] = 0
    df_scored["quote_quality_score_0_100"] = (
        df_scored["quote_length_points"]
        + df_scored["quote_sentiment_points"]
        + df_scored["quote_uniqueness_points"]
    ).round(2)
    df_scored["quote_marketing_ready"] = (
        short_ok
        & consent_series
        & df_scored["quote_sentiment_fit"].isin(["strong_positive", "positive"])
        & (df_scored["quote_uniqueness_score_1_5"] >= 3)
        & (df_scored["short_quote_word_count"].between(6, 30))
    )
    df_scored["quote_case_study_ready"] = (
        long_ok
        & consent_series
        & df_scored["quote_sentiment_fit"].isin(["strong_positive", "positive", "neutral"])
        & (df_scored["quote_uniqueness_score_1_5"] >= 3)
        & (df_scored["long_quote_word_count"].between(18, 75))
    )
    df_scored["quote_internal_highlight_ready"] = (
        valid_quote
        & df_scored["quote_sentiment_fit"].isin(["strong_positive", "positive", "neutral", "mixed"])
    )
    return df_scored


def finalize_scored_df(df_scored: pd.DataFrame) -> pd.DataFrame:
    finalized = df_scored.copy()
    if "themes_tags_csv" not in finalized.columns:
        finalized["themes_tags_csv"] = ""
    finalized["themes_tags"] = finalized["themes_tags_csv"].fillna("").apply(normalize_themes_csv)
    finalized["regular_theme_tags"] = [[] for _ in range(len(finalized))]

    tag_series = finalized["themes_tags"].explode().dropna().astype(str).str.strip().str.lower()
    tag_series = tag_series[tag_series != ""]
    if not tag_series.empty:
        tag_counts = tag_series.value_counts()
        regular_tags = set(tag_counts[tag_counts >= 3].head(25).index.tolist())
        finalized["regular_theme_tags"] = finalized["themes_tags"].apply(
            lambda tags: [tag for tag in tags if tag in regular_tags] if isinstance(tags, list) else []
        )

    return compute_quote_scores(finalized)


def reusable_cache_subset(cache_df: pd.DataFrame) -> pd.DataFrame:
    columns = [column for column in MATCH_COLUMNS + RESULT_COLUMNS if column in cache_df.columns]
    subset = cache_df[columns].copy()
    return subset.drop_duplicates(subset=MATCH_COLUMNS, keep="last")


def merge_results_into_upload(prepared_upload_df: pd.DataFrame, result_df: pd.DataFrame) -> pd.DataFrame:
    merged = prepared_upload_df.merge(result_df, on=MATCH_COLUMNS, how="left")
    return finalize_scored_df(merged)


def upsert_master_cache(existing_cache_df: pd.DataFrame, current_output_df: pd.DataFrame) -> pd.DataFrame:
    cache_df = normalize_existing_cache_df(existing_cache_df)
    replacement = current_output_df.copy()
    replacement = normalize_existing_cache_df(replacement)

    preserved = cache_df[~cache_df["row_key"].isin(replacement["row_key"])].copy()
    updated = pd.concat([preserved, replacement], ignore_index=True)
    updated = updated.drop_duplicates(subset=["row_key"], keep="last").reset_index(drop=True)
    return finalize_scored_df(updated)


def strip_internal_columns(df: pd.DataFrame) -> pd.DataFrame:
    visible = df.copy()
    drop_cols = [column for column in ["row_key", "content_hash", "pipeline_version"] if column in visible.columns]
    if drop_cols:
        visible = visible.drop(columns=drop_cols)
    return visible
