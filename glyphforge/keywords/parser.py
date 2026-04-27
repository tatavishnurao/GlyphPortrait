from __future__ import annotations

import re
from collections import Counter
from typing import List, Sequence, Tuple


def parse_words(text: str) -> List[str]:
    if not text:
        return []
    normalized = text.replace("\n", ",")
    tokens = [t.strip() for t in normalized.split(",")]
    words = [re.sub(r"\s+", " ", t) for t in tokens if t]
    return [w for w in words if w]


def parse_weighted_words(text: str, max_words: int = 600) -> Sequence[Tuple[str, float]]:
    words = parse_words(text)[:max_words]
    if not words:
        return [("Legend", 1.0), ("Focus", 0.92), ("Discipline", 0.86), ("Resilience", 0.8)]

    counts = Counter(w.lower() for w in words)
    out: List[Tuple[str, float]] = []
    n = max(1, len(words))
    for idx, word in enumerate(words):
        base = 1.0 - (idx / (n * 1.4))
        freq_boost = min(0.25, 0.05 * (counts[word.lower()] - 1))
        weight = max(0.18, min(1.0, base + freq_boost))
        out.append((word, weight))
    return out
