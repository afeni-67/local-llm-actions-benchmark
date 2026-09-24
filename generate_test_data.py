#!/usr/bin/env python3
"""Generate synthetic support-ticket records for classification benchmark."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

CATEGORIES = [
    "billing",
    "technical",
    "shipping",
    "account",
    "product",
    "other",
]

SENTIMENTS = ["negative", "neutral", "positive"]

TEMPLATES = {
    "billing": [
        "I was charged twice for invoice {n}. Please refund the extra charge.",
        "My card ending {n} failed; update payment method needed.",
        "Subscription auto-renewed without notice. Order ref SYN-{n}.",
    ],
    "technical": [
        "App crashes when opening settings on device id {n}.",
        "API returns 500 for endpoint /v1/items/{n}.",
        "Cannot reset password; link expires immediately for user{n}.",
    ],
    "shipping": [
        "Package SYN-{n} is 5 days late. Tracking shows no movement.",
        "Wrong item delivered for order SYN-{n}.",
        "Need to change shipping address for order SYN-{n}.",
    ],
    "account": [
        "Cannot log in after password change for account A{n}.",
        "Please delete my account and all data for user{n}@example.test.",
        "Two-factor codes not arriving for account A{n}.",
    ],
    "product": [
        "Product P{n} arrived damaged. Requesting replacement.",
        "Feature request: dark mode for dashboard on plan {n}.",
        "Documentation for module M{n} is incomplete.",
    ],
    "other": [
        "General inquiry about partnership option {n}.",
        "Where is your privacy policy for region {n}?",
        "Feedback only: site is slow on page {n}.",
    ],
}


def make_record(i: int, rng: random.Random) -> dict:
    cat = CATEGORIES[i % len(CATEGORIES)]
    text = rng.choice(TEMPLATES[cat]).format(n=1000 + i)
    # mild noise so the model must actually classify
    if rng.random() < 0.15:
        text = text + " " + rng.choice(["Thanks.", "Urgent.", "Please advise."])
    return {
        "id": f"syn-{i:04d}",
        "text": text,
        "gold_category": cat,  # for offline analysis only; model must not see this in prompt
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("data/synthetic_tickets.jsonl"))
    args = ap.parse_args()
    rng = random.Random(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for i in range(args.count):
            f.write(json.dumps(make_record(i, rng), ensure_ascii=False) + "\n")
    print(f"Wrote {args.count} records to {args.out}")


if __name__ == "__main__":
    main()
