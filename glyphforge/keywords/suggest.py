from __future__ import annotations

from typing import List


def suggest_default_keywords(style: str = "athlete") -> List[str]:
    base = [
        "Leader",
        "Discipline",
        "Legacy",
        "Grit",
        "Champion",
        "Focus",
        "Determination",
        "Resilience",
        "Vision",
        "Greatness",
    ]
    if style == "tribute":
        base.extend(["Honor", "Respect", "Excellence", "Inspiration"])
    return base
