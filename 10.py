"""Crawl ICLR papers, reviews/comments, and scores from OpenReview.

Usage:
    python 10.py --year 2026 --output iclr_2026.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import requests

LOGGER = logging.getLogger(__name__)
OPENREVIEW_API_BASE = "https://api2.openreview.net"
OPENREVIEW_WEB_BASE = "https://openreview.net"
API_MAX_RETRIES = 5
API_BASE_BACKOFF_SECONDS = 1.5


@dataclass(frozen=True)
class CrawlConfig:
    """Runtime configuration for crawling one conference year.

    Attributes:
        year: Conference year, e.g., 2026.
        output: Output JSON file path.
        page_size: Number of items requested per API page.
        max_papers: Optional cap for quick sampling runs.
        download_pdf: Whether to download paper PDF files.
        pdf_dir: Local directory for downloaded PDFs.
        sleep_seconds: Sleep interval between requests/downloads.
        retry: Number of retries for PDF download.
        mapping_output: Optional path to save paper-text/pdf mapping JSONL.
        per_status: Optional sample count per status bucket.
        per_tab: Optional sample count per OpenReview tab bucket.
        ratings_only_output: Optional JSONL path for flattened rating records.
    """

    year: int
    output: Path
    page_size: int = 1000
    max_papers: int | None = None
    download_pdf: bool = False
    pdf_dir: Path = Path("pdfs")
    sleep_seconds: float = 0.0
    retry: int = 3
    mapping_output: Path | None = None
    per_status: int | None = None
    per_tab: int | None = None
    ratings_only_output: Path | None = None


def create_session() -> requests.Session:
    """Create a resilient HTTP session for OpenReview API calls.

    Returns:
        A configured `requests.Session` object.
    """
    session: requests.Session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (compatible; ICLR-Crawler/1.0)",
            "Accept": "application/json",
        }
    )
    return session


def get_json(session: requests.Session, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
    """Send a GET request and parse JSON response.

    Args:
        session: HTTP session.
        endpoint: API endpoint path starting with `/`.
        params: Query parameters.

    Returns:
        Parsed JSON response.

    Raises:
        requests.HTTPError: If server returns non-2xx status code.
        ValueError: If response cannot be decoded as JSON.
    """
    url: str = f"{OPENREVIEW_API_BASE}{endpoint}"
    for attempt in range(1, API_MAX_RETRIES + 1):
        try:
            response = session.get(url, params=params, timeout=(30, 180))
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            if attempt >= API_MAX_RETRIES:
                raise
            sleep_seconds = API_BASE_BACKOFF_SECONDS * attempt
            LOGGER.warning(
                "API request failed attempt %d/%d for %s: %s. Retrying in %.1fs",
                attempt,
                API_MAX_RETRIES,
                endpoint,
                exc,
                sleep_seconds,
            )
            time.sleep(sleep_seconds)
    return {}


def get_group_content_value(session: requests.Session, group_id: str, key: str) -> str | None:
    """Get one value from OpenReview group content.

    Args:
        session: HTTP session.
        group_id: Group id.
        key: Content key.

    Returns:
        Group content value if found, else None.
    """
    payload = get_json(session, "/groups", params={"id": group_id})
    groups = payload.get("groups", [])
    if not groups:
        return None
    content = groups[0].get("content", {})
    raw_value = content.get(key)
    if isinstance(raw_value, dict) and "value" in raw_value:
        value = raw_value["value"]
    else:
        value = raw_value
    return str(value) if value is not None else None


def paged_notes(
    session: requests.Session,
    invitation: str,
    *,
    details: str | None = None,
    page_size: int = 1000,
    max_items: int | None = None,
    extra_params: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Fetch all notes for a given invitation with pagination.

    Args:
        session: HTTP session.
        invitation: OpenReview invitation id.
        details: Optional details expansion, e.g. "replies".
        page_size: Number of notes per page.
        max_items: Optional hard cap on returned note count.
        extra_params: Optional additional query parameters.

    Returns:
        List of note objects.
    """
    offset: int = 0
    all_notes: list[dict[str, Any]] = []
    while True:
        current_limit = page_size
        if max_items is not None:
            remaining = max_items - len(all_notes)
            if remaining <= 0:
                break
            current_limit = min(page_size, remaining)
        params: dict[str, Any] = {
            "invitation": invitation,
            "limit": current_limit,
            "offset": offset,
            "sort": "number:asc",
        }
        if extra_params:
            params.update(extra_params)
        if details:
            params["details"] = details
        payload = get_json(session, "/notes", params=params)
        notes: list[dict[str, Any]] = payload.get("notes", [])
        if not notes:
            break
        all_notes.extend(notes)
        offset += len(notes)
        LOGGER.info("Fetched %d notes from %s", len(all_notes), invitation)
    return all_notes


def fetch_note_with_replies(session: requests.Session, note_id: str) -> dict[str, Any] | None:
    """Fetch one note with details=replies.

    Args:
        session: HTTP session.
        note_id: Note identifier.

    Returns:
        Note with replies if found, else None.
    """
    payload = get_json(session, "/notes", params={"id": note_id, "details": "replies"})
    notes = payload.get("notes", [])
    if isinstance(notes, list) and notes:
        return notes[0]
    return None


def extract_score(content: dict[str, Any]) -> str | None:
    """Extract score-like value from a review content payload.

    Args:
        content: Review note content dictionary.

    Returns:
        Score text if found, else None.
    """
    candidate_keys: tuple[str, ...] = (
        "rating",
        "recommendation",
        "overall_assessment",
        "overall_rating",
        "score",
    )
    for key in candidate_keys:
        value = content.get(key)
        if isinstance(value, dict):
            maybe_value: Any = value.get("value")
            if maybe_value is not None:
                return str(maybe_value)
        elif value is not None:
            return str(value)
    return None


def extract_content_field(content: dict[str, Any], key: str) -> Any:
    """Extract one field value from OpenReview content with value-wrapper support.

    Args:
        content: Raw note content dictionary.
        key: Content key to extract.

    Returns:
        Normalized field value if present, else None.
    """
    value = content.get(key)
    if isinstance(value, dict):
        return value.get("value")
    return value


def normalize_numeric_score(value: Any) -> int | None:
    """Try converting score value to integer.

    Args:
        value: Raw score-like value.

    Returns:
        Integer score if conversion succeeds, else None.
    """
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if text.isdigit():
            return int(text)
    return None


def describe_review_metric(metric: str, value: Any) -> str | None:
    """Convert numeric review metrics to human-readable text.

    Args:
        metric: Metric name, e.g., rating/confidence/soundness.
        value: Metric value.

    Returns:
        Readable description string, or None if unavailable.
    """
    score = normalize_numeric_score(value)
    if score is None:
        return None

    generic_quality_map: dict[int, str] = {
        1: "poor",
        2: "fair",
        3: "good",
        4: "very good",
    }
    rating_map: dict[int, str] = {
        1: "far below the acceptance threshold",
        2: "well below the acceptance threshold",
        3: "below the acceptance threshold",
        4: "slightly below the acceptance threshold",
        5: "borderline acceptance",
        6: "marginally above the acceptance threshold. But would not mind if paper is rejected",
        7: "above the acceptance threshold, and I vote for accepting",
        8: "strong accept",
        9: "top 15% of accepted papers, clear accept",
        10: "top 5% of accepted papers, champion this paper",
    }
    confidence_map: dict[int, str] = {
        1: "You are not confident in your assessment and your evaluation is an educated guess.",
        2: "You are somewhat confident in your assessment, but there are significant gaps in your understanding.",
        3: "You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.",
        4: "You are confident in your assessment, and you understand the paper well. You are familiar with related work and checked most details.",
        5: "You are absolutely certain in your assessment. You carefully checked all details and are very familiar with related work.",
    }

    if metric in {"soundness", "presentation", "contribution"}:
        label = generic_quality_map.get(score)
        return f"{score}: {label}" if label else str(score)
    if metric == "rating":
        label = rating_map.get(score)
        return f"{score}: {label}" if label else str(score)
    if metric == "confidence":
        label = confidence_map.get(score)
        return f"{score}: {label}" if label else str(score)
    return str(score)


def normalize_content_value(value: Any) -> Any:
    """Normalize OpenReview content field value for serialization.

    Args:
        value: Raw value from OpenReview note content.

    Returns:
        Normalized primitive/structured value.
    """
    if isinstance(value, dict) and "value" in value:
        return value["value"]
    return value


def normalize_content(content: dict[str, Any]) -> dict[str, Any]:
    """Normalize an OpenReview note content dictionary.

    Args:
        content: Raw content dictionary.

    Returns:
        Normalized content dictionary.
    """
    return {key: normalize_content_value(value) for key, value in content.items()}


def enrich_metric_descriptions(content: dict[str, Any]) -> dict[str, Any]:
    """Enrich normalized content with readable metric descriptions.

    Args:
        content: Normalized note content.

    Returns:
        Content with score/text dual fields for key review metrics.
    """
    enriched = dict(content)
    for metric in ("soundness", "presentation", "contribution", "rating", "confidence"):
        metric_value = enriched.get(metric)
        metric_description = describe_review_metric(metric, metric_value)
        if metric_description is None:
            continue
        enriched[f"{metric}_score"] = metric_value
        enriched[metric] = metric_description
    return enriched


def extract_pdf_url(submission_content: dict[str, Any]) -> str | None:
    """Extract full PDF URL from submission content.

    Args:
        submission_content: Normalized submission content dictionary.

    Returns:
        Absolute PDF URL if available, else None.
    """
    pdf_value = submission_content.get("pdf")
    if not isinstance(pdf_value, str) or not pdf_value.strip():
        # Fallback: some notes embed PDF URL inside markdown text fields.
        text_fields: tuple[str, ...] = ("article", "comment", "summary", "abstract")
        for field_name in text_fields:
            field_value = submission_content.get(field_name)
            if not isinstance(field_value, str):
                continue
            pdf_url = extract_pdf_url_from_text(field_value)
            if pdf_url is not None:
                return pdf_url
        return None
    if pdf_value.startswith("http://") or pdf_value.startswith("https://"):
        return pdf_value
    return urljoin(f"{OPENREVIEW_WEB_BASE}/", pdf_value.lstrip("/"))


def extract_pdf_url_from_text(text: str) -> str | None:
    """Extract a PDF URL from free-form markdown/plain text.

    Args:
        text: Text potentially containing URLs.

    Returns:
        First matched PDF URL, else None.
    """
    url_pattern = re.compile(r"https?://[^\s\)\]]+")
    for match in url_pattern.findall(text):
        candidate = match.strip(".,;:!\"'")
        lower_candidate = candidate.lower()
        if lower_candidate.endswith(".pdf"):
            return candidate
        if "arxiv.org/abs/" in lower_candidate:
            # Normalize arXiv abstract URLs to direct PDF URLs.
            return re.sub(r"arxiv\.org/abs/([^?#]+)", r"arxiv.org/pdf/\1.pdf", candidate)
        if "arxiv.org/pdf/" in lower_candidate:
            return candidate if lower_candidate.endswith(".pdf") else f"{candidate}.pdf"
    return None


def download_pdf_file(
    session: requests.Session,
    pdf_url: str,
    output_path: Path,
    retry: int,
    sleep_seconds: float,
) -> str:
    """Download one PDF with retry and basic status reporting.

    Args:
        session: HTTP session.
        pdf_url: Absolute PDF URL.
        output_path: Local path to save PDF.
        retry: Maximum retry attempts.
        sleep_seconds: Sleep seconds between attempts.

    Returns:
        Download status string.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    last_exception: Exception | None = None
    chunk_size: int = 1024 * 1024

    for attempt in range(1, retry + 1):
        temp_path = output_path.with_suffix(output_path.suffix + ".part")
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        try:
            with session.get(
                pdf_url,
                stream=True,
                timeout=(30, 600),
            ) as response:
                if response.status_code in {401, 403}:
                    return f"forbidden_{response.status_code}"
                if response.status_code == 404:
                    return "not_found"
                if response.status_code != 200:
                    LOGGER.warning(
                        "Download failed (%s), attempt %d/%d: %s",
                        response.status_code,
                        attempt,
                        retry,
                        pdf_url,
                    )
                    continue

                expected_len: int | None = None
                content_length = response.headers.get("Content-Length")
                if content_length is not None and str(content_length).isdigit():
                    expected_len = int(content_length)

                bytes_written = 0
                with temp_path.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            handle.write(chunk)
                            bytes_written += len(chunk)

                if expected_len is not None and bytes_written != expected_len:
                    raise ValueError(
                        f"Incomplete download: got {bytes_written} bytes, expected {expected_len}"
                    )

            temp_path.replace(output_path)
            return "ok"
        except (requests.RequestException, ValueError, OSError) as exc:
            last_exception = exc
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)
            LOGGER.warning("Download exception attempt %d/%d: %s", attempt, retry, exc)

        if attempt < retry and sleep_seconds > 0:
            time.sleep(sleep_seconds)

    if last_exception is not None:
        return f"error_{type(last_exception).__name__}"
    return "failed"


def split_replies_by_type(replies: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split forum replies into review-like and comment-like notes.

    Args:
        replies: Replies from submission details.

    Returns:
        A tuple of (reviews, comments).
    """
    reviews: list[dict[str, Any]] = []
    comments: list[dict[str, Any]] = []

    for note in replies:
        invitation = str(note.get("invitation", "")).lower()
        if "review" in invitation:
            reviews.append(note)
            continue
        if "comment" in invitation:
            comments.append(note)
            continue

    return reviews, comments


def classify_reply_type(note: dict[str, Any], normalized_content: dict[str, Any]) -> str:
    """Classify one reply note into a coarse discussion type.

    Args:
        note: Raw reply note.
        normalized_content: Normalized note content.

    Returns:
        Reply type label for downstream filtering.
    """
    invitation = str(note.get("invitation", "")).lower()
    title_value = str(normalized_content.get("title", "")).lower()

    if "official_review" in invitation or "review" in invitation:
        return "review"
    if "decision" in invitation:
        return "decision"
    if "meta_review" in invitation or "metareview" in invitation:
        return "meta_review"
    if "response" in invitation or title_value.startswith("response by authors"):
        return "author_response"
    if "comment" in invitation:
        return "comment"
    return "other"


def normalize_reply_note(note: dict[str, Any]) -> dict[str, Any]:
    """Normalize one reply note into a compact serializable record.

    Args:
        note: Raw OpenReview reply note.

    Returns:
        Normalized reply record.
    """
    content_raw = note.get("content", {})
    content = enrich_metric_descriptions(normalize_content(content_raw)) if isinstance(content_raw, dict) else {}
    raw_content = content
    return {
        "id": note.get("id"),
        "forum": note.get("forum"),
        "replyto": note.get("replyto"),
        "invitation": note.get("invitation"),
        "type": classify_reply_type(note, content),
        "score": extract_score(raw_content),
        "title": content.get("title"),
        "summary": content.get("summary"),
        "comment": content.get("comment"),
        "reviewer_scores": content.get("reviewer_scores"),
        "reviewer_concerns": content.get("reviewer_concerns"),
        "soundness_score": extract_content_field(raw_content, "soundness"),
        "soundness": describe_review_metric("soundness", extract_content_field(raw_content, "soundness")),
        "presentation_score": extract_content_field(raw_content, "presentation"),
        "presentation": describe_review_metric("presentation", extract_content_field(raw_content, "presentation")),
        "contribution_score": extract_content_field(raw_content, "contribution"),
        "contribution": describe_review_metric("contribution", extract_content_field(raw_content, "contribution")),
        "rating_score": extract_content_field(raw_content, "rating"),
        "rating": describe_review_metric("rating", extract_content_field(raw_content, "rating")),
        "confidence_score": extract_content_field(raw_content, "confidence"),
        "confidence": describe_review_metric("confidence", extract_content_field(raw_content, "confidence")),
        "raw_content": raw_content,
        "content": content,
        "tcdate": note.get("tcdate"),
        "cdate": note.get("cdate"),
    }


def build_discussion_views(replies: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Build multiple filtered views for all forum replies.

    Args:
        replies: Raw replies list.

    Returns:
        Dictionary containing full and filtered reply collections.
    """
    all_notes: list[dict[str, Any]] = [normalize_reply_note(note) for note in replies]
    return {
        "all_replies": all_notes,
        "author_responses": [note for note in all_notes if note["type"] == "author_response"],
        "meta_reviews": [note for note in all_notes if note["type"] in {"meta_review", "decision"}],
        "review_notes": [note for note in all_notes if note["type"] == "review"],
    }


def extract_editorial_signals(replies: list[dict[str, Any]]) -> dict[str, Any]:
    """Extract meta-level fields like reviewer scores/concerns from replies.

    Args:
        replies: Replies from submission details.

    Returns:
        Dictionary containing extracted fields and source note metadata.
    """
    extracted: dict[str, Any] = {
        "summary": None,
        "reviewer_scores": None,
        "reviewer_concerns": None,
        "source_note_id": None,
        "source_invitation": None,
    }

    # Prefer decision/meta-review style notes first.
    priority_markers: tuple[str, ...] = (
        "decision",
        "meta_review",
        "metareview",
        "area_chair",
        "senior_area_chair",
    )
    sorted_replies = sorted(
        replies,
        key=lambda note: (
            0
            if any(marker in str(note.get("invitation", "")).lower() for marker in priority_markers)
            else 1
        ),
    )

    for note in sorted_replies:
        content_raw = note.get("content", {})
        if not isinstance(content_raw, dict):
            continue
        content = normalize_content(content_raw)

        reviewer_scores = content.get("reviewer_scores")
        reviewer_concerns = content.get("reviewer_concerns")
        summary = content.get("summary")

        if reviewer_scores is None and reviewer_concerns is None and summary is None:
            continue

        extracted["summary"] = summary
        extracted["reviewer_scores"] = reviewer_scores
        extracted["reviewer_concerns"] = reviewer_concerns
        extracted["source_note_id"] = note.get("id")
        extracted["source_invitation"] = note.get("invitation")
        break

    return extracted


def extract_decision_text_from_replies(replies: list[dict[str, Any]]) -> str | None:
    """Extract decision text from reply notes if available.

    Args:
        replies: Replies under one submission forum.

    Returns:
        Decision text if found, else None.
    """
    decision_markers: tuple[str, ...] = ("decision", "meta_review", "metareview")
    for note in replies:
        invitation = str(note.get("invitation", "")).lower()
        if not any(marker in invitation for marker in decision_markers):
            continue
        content = note.get("content", {})
        if not isinstance(content, dict):
            continue
        for key in ("decision", "final_decision", "recommendation", "verdict"):
            value = extract_content_field(content, key)
            if value is not None:
                return str(value)
    return None


def classify_submission_status(decision_text: str | None) -> str:
    """Classify submission status from decision text.

    Args:
        decision_text: Decision text extracted from notes.

    Returns:
        One of oral/poster/reject/withdrawn/other/unknown.
    """
    if decision_text is None:
        return "unknown"
    text = decision_text.lower()
    if "withdraw" in text:
        return "withdrawn"
    if "oral" in text:
        return "oral"
    if "poster" in text or "spotlight" in text:
        return "poster"
    if "reject" in text:
        return "reject"
    return "other"


def classify_status_from_venue(venue_text: str | None) -> str:
    """Classify submission status from venue text.

    Args:
        venue_text: Venue string from submission content.

    Returns:
        One of oral/poster/reject/withdrawn/other/unknown.
    """
    if venue_text is None:
        return "unknown"
    text = venue_text.lower()
    if "withdraw" in text:
        return "withdrawn"
    if "oral" in text:
        return "oral"
    if "poster" in text or "spotlight" in text:
        return "poster"
    if "reject" in text:
        return "reject"
    return "other"


def get_submission_status(submission: dict[str, Any]) -> tuple[str, str | None]:
    """Resolve submission status with decision and venue fallback.

    Args:
        submission: One submission note.

    Returns:
        A tuple of (status, decision_text).
    """
    details = submission.get("details", {})
    replies = details.get("replies", [])
    if not isinstance(replies, list):
        replies = []
    decision_text = extract_decision_text_from_replies(replies)
    status_from_decision = classify_submission_status(decision_text)
    if status_from_decision not in {"unknown", "other"}:
        return status_from_decision, decision_text

    content = submission.get("content", {})
    venue_text: str | None = None
    if isinstance(content, dict):
        venue_value = extract_content_field(content, "venue")
        if venue_value is not None:
            venue_text = str(venue_value)
    status_from_venue = classify_status_from_venue(venue_text)
    return status_from_venue, decision_text


def select_submissions_by_status(
    submissions: list[dict[str, Any]],
    per_status: int,
) -> list[dict[str, Any]]:
    """Select balanced subset by status buckets.

    Args:
        submissions: All crawled submissions.
        per_status: Number of papers to keep per status bucket.

    Returns:
        Selected submissions.
    """
    target_statuses: tuple[str, ...] = ("oral", "poster", "reject", "withdrawn")
    counters: dict[str, int] = {status: 0 for status in target_statuses}
    selected: list[dict[str, Any]] = []

    for submission in submissions:
        status, _ = get_submission_status(submission)
        if status not in counters:
            continue
        if counters[status] >= per_status:
            continue
        counters[status] += 1
        selected.append(submission)
        if all(count >= per_status for count in counters.values()):
            break

    LOGGER.info("Selected by status counters: %s", counters)
    return selected


def fetch_submissions_by_status_tabs(
    session: requests.Session,
    venue_id: str,
    per_status: int,
    page_size: int,
) -> list[dict[str, Any]]:
    """Fetch submissions using status-style venue filters like OpenReview tabs.

    Args:
        session: HTTP session.
        venue_id: Venue id, e.g. ICLR.cc/2026/Conference.
        per_status: Number of papers per status.
        page_size: API page size.

    Returns:
        Aggregated selected submissions.
    """
    submission_invitation = f"{venue_id}/-/Submission"
    # These labels use OpenReview venue values used by ICLR tabs.
    status_to_venue_labels: dict[str, tuple[str, ...]] = {
        "oral": (f"ICLR {venue_id.split('/')[1]} Oral",),
        "poster": (f"ICLR {venue_id.split('/')[1]} Poster", f"ICLR {venue_id.split('/')[1]} Spotlight"),
        "reject": (
            f"Submitted to ICLR {venue_id.split('/')[1]}",
        ),
        "withdrawn": (
            f"ICLR {venue_id.split('/')[1]} Conference Withdrawn Submission",
        ),
    }
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    counters: dict[str, int] = {status: 0 for status in status_to_venue_labels}

    for status, venue_labels in status_to_venue_labels.items():
        for venue_label in venue_labels:
            remaining = per_status - counters[status]
            if remaining <= 0:
                break
            notes = paged_notes(
                session=session,
                invitation=submission_invitation,
                details="replies",
                page_size=page_size,
                max_items=remaining,
                extra_params={"content.venue": venue_label},
            )
            for note in notes:
                note_id = str(note.get("id", ""))
                if note_id in selected_ids:
                    continue
                selected_ids.add(note_id)
                selected.append(note)
                counters[status] += 1
                if counters[status] >= per_status:
                    break

    LOGGER.info("Selected by tab-like venue filters: %s", counters)
    return selected


def fetch_submissions_by_openreview_tabs(
    session: requests.Session,
    venue_id: str,
    per_tab: int,
    page_size: int,
) -> dict[str, list[dict[str, Any]]]:
    """Fetch submissions by OpenReview UI tab semantics.

    Args:
        session: HTTP session.
        venue_id: Venue id, e.g. ICLR.cc/2026/Conference.
        per_tab: Number of papers to fetch per tab.
        page_size: API page size.

    Returns:
        Mapping from tab name to list of submissions.
    """
    year_token = venue_id.split("/")[1]
    submission_invitation = f"{venue_id}/-/Submission"
    rejected_venue_id = get_group_content_value(session, venue_id, "rejected_venue_id")
    withdrawn_venue_id = get_group_content_value(session, venue_id, "withdrawn_venue_id")
    desk_rejected_venue_id = get_group_content_value(session, venue_id, "desk_rejected_venue_id")
    if rejected_venue_id is None:
        rejected_venue_id = f"{venue_id}/Rejected_Submission"
    if withdrawn_venue_id is None:
        withdrawn_venue_id = f"{venue_id}/Withdrawn_Submission"
    if desk_rejected_venue_id is None:
        desk_rejected_venue_id = f"{venue_id}/Desk_Rejected_Submission"

    # Strict tab-driven retrieval: do not use local fallback classification.
    # Values are aligned with ICLR 2026 venue tags shown by OpenReview API.
    tab_to_filters: dict[str, list[dict[str, Any]]] = {
        "accept_oral": [{"content.venue": f"ICLR {year_token} Oral"}],
        "accept_poster": [
            {"content.venue": f"ICLR {year_token} Poster"},
            {"content.venue": f"ICLR {year_token} Spotlight"},
        ],
        "reject": [{"content.venueid": rejected_venue_id}],
        "withdrawn_submissions": [{"content.venueid": withdrawn_venue_id}],
        "desk_rejected_submissions": [{"content.venueid": desk_rejected_venue_id}],
    }
    tab_samples: dict[str, list[dict[str, Any]]] = {}
    is_unbounded = per_tab <= 0
    for tab_name, filters in tab_to_filters.items():
        selected: list[dict[str, Any]] = []
        selected_ids: set[str] = set()
        for filter_params in filters:
            remaining = None if is_unbounded else per_tab - len(selected)
            if (not is_unbounded) and remaining <= 0:
                break
            notes = paged_notes(
                session=session,
                invitation=submission_invitation,
                details=None,
                page_size=page_size,
                max_items=remaining,
                extra_params=filter_params,
            )
            for note in notes:
                note_id = str(note.get("id", ""))
                if note_id in selected_ids:
                    continue
                selected_ids.add(note_id)
                selected.append(note)
                if (not is_unbounded) and len(selected) >= per_tab:
                    break
        tab_samples[tab_name] = selected

    # "Recent Activity" tab is time-ordered latest submissions/updates.
    recent_activity = paged_notes(
        session=session,
        invitation=submission_invitation,
        details=None,
        page_size=page_size,
        max_items=None if is_unbounded else per_tab,
        extra_params={"sort": "tmdate:desc"},
    )
    tab_samples["recent_activity"] = recent_activity

    LOGGER.info(
        "Selected by OpenReview tabs: %s",
        {tab: len(items) for tab, items in tab_samples.items()},
    )
    return tab_samples


def crawl_iclr(config: CrawlConfig) -> list[dict[str, Any]]:
    """Crawl ICLR submissions with review/comment data.

    Args:
        config: Crawl configuration.

    Returns:
        List of paper bundles.
    """
    venue_id: str = f"ICLR.cc/{config.year}/Conference"
    submission_invitation: str = f"{venue_id}/-/Submission"

    session = create_session()
    submissions = paged_notes(
        session,
        submission_invitation,
        details="replies",
        page_size=config.page_size,
        max_items=config.max_papers,
    )
    LOGGER.info("Found %d submissions for %s", len(submissions), venue_id)

    if config.per_status is not None:
        tab_selected_submissions = fetch_submissions_by_status_tabs(
            session=session,
            venue_id=venue_id,
            per_status=config.per_status,
            page_size=config.page_size,
        )
        if tab_selected_submissions:
            submissions = tab_selected_submissions
            LOGGER.info("Using %d submissions from tab-style venue filters", len(submissions))
        else:
            submissions = select_submissions_by_status(submissions, config.per_status)
            LOGGER.info("Using %d submissions after fallback per-status sampling", len(submissions))

    papers: list[dict[str, Any]] = []

    for submission in submissions:
        forum_id = str(submission.get("forum", ""))
        normalized_submission_content = normalize_content(submission.get("content", {}))
        pdf_url = extract_pdf_url(normalized_submission_content)
        pdf_local_path: str | None = None
        pdf_download_status: str | None = None
        if config.download_pdf and pdf_url is not None:
            paper_id = str(submission.get("id", forum_id or "paper_unknown"))
            target_path = config.pdf_dir / f"{paper_id}.pdf"
            pdf_download_status = download_pdf_file(
                session=session,
                pdf_url=pdf_url,
                output_path=target_path,
                retry=config.retry,
                sleep_seconds=config.sleep_seconds,
            )
            pdf_local_path = target_path.as_posix()
            if config.sleep_seconds > 0:
                time.sleep(config.sleep_seconds)

        details = submission.get("details", {})
        replies = details.get("replies", [])
        if not isinstance(replies, list):
            replies = []
        reviews, comments = split_replies_by_type(replies)
        editorial_signals = extract_editorial_signals(replies)
        discussion_views = build_discussion_views(replies)
        status, decision_text = get_submission_status(submission)
        papers.append(
            {
                "paper_id": submission.get("id"),
                "forum_id": forum_id,
                "number": submission.get("number"),
                "title": normalized_submission_content.get("title"),
                "abstract": normalized_submission_content.get("abstract"),
                "keywords": normalized_submission_content.get("keywords"),
                "decision": decision_text,
                "status": status,
                "pdf_url": pdf_url,
                "pdf_local_path": pdf_local_path,
                "pdf_download_status": pdf_download_status,
                "submission_content": normalized_submission_content,
                "editorial_signals": editorial_signals,
                "all_replies": discussion_views["all_replies"],
                "author_responses": discussion_views["author_responses"],
                "meta_reviews": discussion_views["meta_reviews"],
                "review_notes": discussion_views["review_notes"],
                "reviews": [
                    {
                        "id": note.get("id"),
                        "forum": note.get("forum"),
                        "replyto": note.get("replyto"),
                        "invitation": note.get("invitation"),
                        "score": extract_score(note.get("content", {})),
                        "soundness_score": extract_content_field(note.get("content", {}), "soundness"),
                        "soundness": describe_review_metric(
                            "soundness", extract_content_field(note.get("content", {}), "soundness")
                        ),
                        "presentation_score": extract_content_field(note.get("content", {}), "presentation"),
                        "presentation": describe_review_metric(
                            "presentation", extract_content_field(note.get("content", {}), "presentation")
                        ),
                        "contribution_score": extract_content_field(note.get("content", {}), "contribution"),
                        "contribution": describe_review_metric(
                            "contribution", extract_content_field(note.get("content", {}), "contribution")
                        ),
                        "rating_score": extract_content_field(note.get("content", {}), "rating"),
                        "rating": describe_review_metric("rating", extract_content_field(note.get("content", {}), "rating")),
                        "confidence_score": extract_content_field(note.get("content", {}), "confidence"),
                        "confidence": describe_review_metric(
                            "confidence", extract_content_field(note.get("content", {}), "confidence")
                        ),
                        "raw_content": enrich_metric_descriptions(
                            normalize_content(note.get("content", {}))
                        ),
                        "content": enrich_metric_descriptions(
                            normalize_content(note.get("content", {}))
                        ),
                        "tcdate": note.get("tcdate"),
                    }
                    for note in reviews
                ],
                "comments": [
                    {
                        "id": note.get("id"),
                        "forum": note.get("forum"),
                        "replyto": note.get("replyto"),
                        "invitation": note.get("invitation"),
                        "content": enrich_metric_descriptions(
                            normalize_content(note.get("content", {}))
                        ),
                        "tcdate": note.get("tcdate"),
                    }
                    for note in comments
                ],
            }
        )
    return papers


def crawl_iclr_by_tabs(config: CrawlConfig) -> dict[str, list[dict[str, Any]]]:
    """Crawl ICLR submissions grouped by OpenReview tab categories.

    Args:
        config: Crawl configuration.

    Returns:
        Mapping of tab name to crawled paper bundles.
    """
    if config.per_tab is None:
        return {}

    venue_id: str = f"ICLR.cc/{config.year}/Conference"
    session = create_session()
    tab_submissions = fetch_submissions_by_openreview_tabs(
        session=session,
        venue_id=venue_id,
        per_tab=config.per_tab,
        page_size=config.page_size,
    )

    tab_papers: dict[str, list[dict[str, Any]]] = {}
    for tab_name, submissions in tab_submissions.items():
        papers: list[dict[str, Any]] = []
        for submission in submissions:
            papers.append(build_tab_paper_record(session, submission, tab_name, config))
        tab_papers[tab_name] = papers
    return tab_papers


def build_tab_paper_record(
    session: requests.Session,
    submission: dict[str, Any],
    tab_name: str,
    config: CrawlConfig,
) -> dict[str, Any]:
    """Build one paper record for tab-based crawling.

    Args:
        session: HTTP session.
        submission: Submission note.
        tab_name: Tab bucket name.
        config: Crawl config.

    Returns:
        One paper record.
    """
    note_id = str(submission.get("id", ""))
    if "details" not in submission and note_id:
        hydrated = fetch_note_with_replies(session, note_id)
        if hydrated is not None:
            submission = hydrated
    forum_id = str(submission.get("forum", ""))
    normalized_submission_content = normalize_content(submission.get("content", {}))
    pdf_url = extract_pdf_url(normalized_submission_content)
    pdf_local_path: str | None = None
    pdf_download_status: str | None = None
    if config.download_pdf and pdf_url is not None:
        paper_id = str(submission.get("id", forum_id or "paper_unknown"))
        target_path = config.pdf_dir / f"{paper_id}.pdf"
        pdf_download_status = download_pdf_file(
            session=session,
            pdf_url=pdf_url,
            output_path=target_path,
            retry=config.retry,
            sleep_seconds=config.sleep_seconds,
        )
        pdf_local_path = target_path.as_posix()
        if config.sleep_seconds > 0:
            time.sleep(config.sleep_seconds)

    details = submission.get("details", {})
    replies = details.get("replies", [])
    if not isinstance(replies, list):
        replies = []
    reviews, comments = split_replies_by_type(replies)
    editorial_signals = extract_editorial_signals(replies)
    discussion_views = build_discussion_views(replies)
    status, decision_text = get_submission_status(submission)
    return {
        "tab": tab_name,
        "paper_id": submission.get("id"),
        "forum_id": forum_id,
        "number": submission.get("number"),
        "title": normalized_submission_content.get("title"),
        "abstract": normalized_submission_content.get("abstract"),
        "keywords": normalized_submission_content.get("keywords"),
        "decision": decision_text,
        "status": status,
        "pdf_url": pdf_url,
        "pdf_local_path": pdf_local_path,
        "pdf_download_status": pdf_download_status,
        "submission_content": normalized_submission_content,
        "editorial_signals": editorial_signals,
        "all_replies": discussion_views["all_replies"],
        "author_responses": discussion_views["author_responses"],
        "meta_reviews": discussion_views["meta_reviews"],
        "review_notes": discussion_views["review_notes"],
        "reviews": [
            {
                "id": note.get("id"),
                "forum": note.get("forum"),
                "replyto": note.get("replyto"),
                "invitation": note.get("invitation"),
                "score": extract_score(note.get("content", {})),
                "soundness_score": extract_content_field(note.get("content", {}), "soundness"),
                "soundness": describe_review_metric(
                    "soundness", extract_content_field(note.get("content", {}), "soundness")
                ),
                "presentation_score": extract_content_field(note.get("content", {}), "presentation"),
                "presentation": describe_review_metric(
                    "presentation", extract_content_field(note.get("content", {}), "presentation")
                ),
                "contribution_score": extract_content_field(note.get("content", {}), "contribution"),
                "contribution": describe_review_metric(
                    "contribution", extract_content_field(note.get("content", {}), "contribution")
                ),
                "rating_score": extract_content_field(note.get("content", {}), "rating"),
                "rating": describe_review_metric(
                    "rating", extract_content_field(note.get("content", {}), "rating")
                ),
                "confidence_score": extract_content_field(note.get("content", {}), "confidence"),
                "confidence": describe_review_metric(
                    "confidence", extract_content_field(note.get("content", {}), "confidence")
                ),
                "raw_content": enrich_metric_descriptions(
                    normalize_content(note.get("content", {}))
                ),
                "content": enrich_metric_descriptions(
                    normalize_content(note.get("content", {}))
                ),
                "tcdate": note.get("tcdate"),
            }
            for note in reviews
        ],
        "comments": [
            {
                "id": note.get("id"),
                "forum": note.get("forum"),
                "replyto": note.get("replyto"),
                "invitation": note.get("invitation"),
                "content": enrich_metric_descriptions(
                    normalize_content(note.get("content", {}))
                ),
                "tcdate": note.get("tcdate"),
            }
            for note in comments
        ],
    }


def crawl_iclr_by_tabs_stream(
    config: CrawlConfig,
    output_path: Path,
    mapping_output: Path | None = None,
    ratings_only_output: Path | None = None,
) -> dict[str, int]:
    """Stream tab-based papers directly into JSONL outputs.

    Args:
        config: Crawl configuration.
        output_path: JSONL output path for paper records.
        mapping_output: Optional JSONL output path for mapping records.
        ratings_only_output: Optional JSONL path for flattened rating records.

    Returns:
        Tab counts.
    """
    venue_id: str = f"ICLR.cc/{config.year}/Conference"
    session = create_session()
    tab_submissions = fetch_submissions_by_openreview_tabs(
        session=session,
        venue_id=venue_id,
        per_tab=config.per_tab if config.per_tab is not None else 0,
        page_size=config.page_size,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if mapping_output is not None:
        mapping_output.parent.mkdir(parents=True, exist_ok=True)
    if ratings_only_output is not None:
        ratings_only_output.parent.mkdir(parents=True, exist_ok=True)

    tab_counts: dict[str, int] = {}
    with output_path.open("w", encoding="utf-8") as out_f:
        map_f = mapping_output.open("w", encoding="utf-8") if mapping_output is not None else None
        try:
            for tab_name, submissions in tab_submissions.items():
                tab_counts[tab_name] = 0
                for submission in submissions:
                    paper = build_tab_paper_record(session, submission, tab_name, config)
                    out_f.write(json.dumps(paper, ensure_ascii=False) + "\n")
                    if map_f is not None:
                        map_f.write(json.dumps(build_pdf_text_mapping_record(paper), ensure_ascii=False) + "\n")
                    if ratings_only_output is not None:
                        append_jsonl_records(ratings_only_output, build_rating_records(paper))
                    tab_counts[tab_name] += 1
                LOGGER.info("Streamed %d papers for tab %s", tab_counts[tab_name], tab_name)
        finally:
            if map_f is not None:
                map_f.close()
    return tab_counts


def build_pdf_text_mapping_record(paper: dict[str, Any]) -> dict[str, Any]:
    """Build one compact mapping record from a crawled paper entry.

    Args:
        paper: One paper object from crawl output.

    Returns:
        Mapping record linking identifiers, PDF, and text fields.
    """
    review_notes = paper.get("review_notes", [])
    author_responses = paper.get("author_responses", [])
    meta_reviews = paper.get("meta_reviews", [])
    all_replies = paper.get("all_replies", [])
    return {
        "paper_id": paper.get("paper_id"),
        "forum_id": paper.get("forum_id"),
        "number": paper.get("number"),
        "title": paper.get("title"),
        "abstract": paper.get("abstract"),
        "pdf_url": paper.get("pdf_url"),
        "pdf_local_path": paper.get("pdf_local_path"),
        "pdf_download_status": paper.get("pdf_download_status"),
        "openreview_forum_url": (
            f"{OPENREVIEW_WEB_BASE}/forum?id={paper.get('forum_id')}" if paper.get("forum_id") else None
        ),
        "text_blocks": {
            "submission_content": paper.get("submission_content"),
            "editorial_signals": paper.get("editorial_signals"),
            "reviews": review_notes,
            "author_responses": author_responses,
            "meta_reviews": meta_reviews,
            "all_replies": all_replies,
        },
        "counts": {
            "review_count": len(review_notes),
            "author_response_count": len(author_responses),
            "meta_review_count": len(meta_reviews),
            "reply_count": len(all_replies),
        },
    }


def build_rating_records(paper: dict[str, Any]) -> list[dict[str, Any]]:
    """Build flattened rating records from one paper.

    Args:
        paper: Paper record.

    Returns:
        List of rating-centric records (one per review with rating).
    """
    records: list[dict[str, Any]] = []
    for review in paper.get("reviews", []):
        rating = review.get("rating")
        rating_score = review.get("rating_score")
        if rating is None and rating_score is None:
            continue
        records.append(
            {
                "tab": paper.get("tab"),
                "paper_id": paper.get("paper_id"),
                "forum_id": paper.get("forum_id"),
                "title": paper.get("title"),
                "review_id": review.get("id"),
                "invitation": review.get("invitation"),
                "rating": rating,
                "rating_score": rating_score,
                "confidence": review.get("confidence"),
                "confidence_score": review.get("confidence_score"),
                "soundness": review.get("soundness"),
                "soundness_score": review.get("soundness_score"),
                "presentation": review.get("presentation"),
                "presentation_score": review.get("presentation_score"),
                "contribution": review.get("contribution"),
                "contribution_score": review.get("contribution_score"),
                "score": review.get("score"),
                "tcdate": review.get("tcdate"),
            }
        )
    return records


def write_mapping_jsonl(mapping_output: Path, papers: list[dict[str, Any]]) -> None:
    """Write one-json-per-line mapping file for downstream processing.

    Args:
        mapping_output: Output JSONL path.
        papers: Crawled papers.
    """
    mapping_output.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(build_pdf_text_mapping_record(paper), ensure_ascii=False) for paper in papers]
    mapping_output.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_papers_jsonl(output_path: Path, papers: list[dict[str, Any]]) -> None:
    """Write paper records as JSONL.

    Args:
        output_path: Target JSONL file path.
        papers: Paper records to write.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(paper, ensure_ascii=False) for paper in papers]
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def append_jsonl_records(output_path: Path, records: list[dict[str, Any]]) -> None:
    """Append records to a JSONL file.

    Args:
        output_path: Target JSONL path.
        records: Records to append.
    """
    if not records:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> CrawlConfig:
    """Parse CLI arguments.

    Returns:
        Parsed and validated crawl config.
    """
    parser = argparse.ArgumentParser(description="Crawl ICLR papers and OpenReview discussions.")
    parser.add_argument("--year", type=int, default=2026, help="Conference year (default: 2026).")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("iclr_2026_with_reviews.json"),
        help="Output JSON path.",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=100,
        help="OpenReview API page size (default: 100).",
    )
    parser.add_argument(
        "--max-papers",
        type=int,
        default=None,
        help="Only crawl first N papers (for quick test).",
    )
    parser.add_argument(
        "--download-pdf",
        action="store_true",
        help="Download PDF for each crawled paper if pdf URL is available.",
    )
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=Path("pdfs"),
        help="Directory to store downloaded PDFs (default: pdfs).",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Sleep interval between network calls/downloads.",
    )
    parser.add_argument(
        "--retry",
        type=int,
        default=3,
        help="Retry attempts for PDF download (default: 3).",
    )
    parser.add_argument(
        "--mapping-output",
        type=Path,
        default=None,
        help="Optional JSONL path to save PDF-text mapping records.",
    )
    parser.add_argument(
        "--per-status",
        type=int,
        default=None,
        help="Sample N papers per status: oral/poster/reject/withdrawn.",
    )
    parser.add_argument(
        "--per-tab",
        type=int,
        default=None,
        help=(
            "Sample N papers per OpenReview tab: accept_oral, accept_poster, reject, "
            "withdrawn_submissions, desk_rejected_submissions, recent_activity. "
            "Use 0 for all papers in each tab."
        ),
    )
    parser.add_argument(
        "--ratings-only-output",
        type=Path,
        default=None,
        help="Optional JSONL path to output flattened rating records only.",
    )
    args = parser.parse_args()
    return CrawlConfig(
        year=args.year,
        output=args.output,
        page_size=args.page_size,
        max_papers=args.max_papers,
        download_pdf=args.download_pdf,
        pdf_dir=args.pdf_dir,
        sleep_seconds=args.sleep_seconds,
        retry=args.retry,
        mapping_output=args.mapping_output,
        per_status=args.per_status,
        per_tab=args.per_tab,
        ratings_only_output=args.ratings_only_output,
    )


def main() -> None:
    """Run the crawler and persist results to disk."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    config = parse_args()
    if config.per_tab is not None:
        if config.output.suffix.lower() == ".jsonl":
            tab_counts = crawl_iclr_by_tabs_stream(
                config=config,
                output_path=config.output,
                mapping_output=config.mapping_output,
                ratings_only_output=config.ratings_only_output,
            )
            total_count = sum(tab_counts.values())
            LOGGER.info(
                "Streamed %d tab-grouped papers as JSONL to %s",
                total_count,
                config.output.as_posix(),
            )
            LOGGER.info("Tab counts: %s", tab_counts)
        else:
            tab_papers = crawl_iclr_by_tabs(config)
            total_count = sum(len(items) for items in tab_papers.values())
            flat_papers = [paper for papers in tab_papers.values() for paper in papers]
            config.output.write_text(
                json.dumps(
                    {
                        "conference": f"ICLR {config.year}",
                        "paper_count": total_count,
                        "tab_counts": {tab: len(items) for tab, items in tab_papers.items()},
                        "tabs": tab_papers,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            LOGGER.info("Saved %d tab-grouped papers to %s", total_count, config.output.as_posix())
            if config.mapping_output is not None:
                write_mapping_jsonl(config.mapping_output, flat_papers)
                LOGGER.info("Saved mapping JSONL to %s", config.mapping_output.as_posix())
            if config.ratings_only_output is not None:
                rating_records = []
                for paper in flat_papers:
                    rating_records.extend(build_rating_records(paper))
                write_papers_jsonl(config.ratings_only_output, rating_records)
                LOGGER.info("Saved %d rating records to %s", len(rating_records), config.ratings_only_output.as_posix())
        return

    papers = crawl_iclr(config)
    config.output.write_text(
        json.dumps(
            {
                "conference": f"ICLR {config.year}",
                "paper_count": len(papers),
                "papers": papers,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    if config.mapping_output is not None:
        write_mapping_jsonl(config.mapping_output, papers)
        LOGGER.info("Saved mapping JSONL to %s", config.mapping_output.as_posix())
    if config.ratings_only_output is not None:
        rating_records = []
        for paper in papers:
            rating_records.extend(build_rating_records(paper))
        write_papers_jsonl(config.ratings_only_output, rating_records)
        LOGGER.info("Saved %d rating records to %s", len(rating_records), config.ratings_only_output.as_posix())
    LOGGER.info("Saved %d papers to %s", len(papers), config.output.as_posix())


if __name__ == "__main__":
    main()