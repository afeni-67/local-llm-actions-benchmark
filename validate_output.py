#!/usr/bin/env python3
"""Validate model JSON against the expected schema."""
from __future__ import annotations

import json
import re
from typing import Any, Optional, Tuple

ALLOWED_CATEGORIES = {
    "billing",
    "technical",
    "shipping",
    "account",
    "product",
    "other",
}
ALLOWED_SENTIMENT = {"negative", "neutral", "positive"}


def extract_json_object(text: str) -> Optional[dict]:
    """Best-effort extract of a single JSON object from model output."""
    if not text:
        return None
    text = text.strip()
    # strip common markdown fences
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    # direct parse
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    # find first {...}
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        return None
    return None


def validate_record(obj: Any) -> Tuple[bool, str]:
    if not isinstance(obj, dict):
        return False, "not a JSON object"
    cat = obj.get("category")
    sent = obj.get("sentiment")
    conf = obj.get("confidence")
    if cat not in ALLOWED_CATEGORIES:
        return False, f"invalid category: {cat!r}"
    if sent not in ALLOWED_SENTIMENT:
        return False, f"invalid sentiment: {sent!r}"
    if not isinstance(conf, (int, float)) or not (0.0 <= float(conf) <= 1.0):
        return False, f"invalid confidence: {conf!r}"
    return True, "ok"
