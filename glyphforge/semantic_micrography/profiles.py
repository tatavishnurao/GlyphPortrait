from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from glyphforge.keywords.parser import parse_words


@dataclass(frozen=True)
class WordProfile:
    subject_name: str = "subject"
    hero_words: list[str] = field(default_factory=list)
    region_words: dict[str, list[str]] = field(default_factory=dict)
    texture_words: list[str] = field(default_factory=list)
    anchor_words: list[str] = field(default_factory=list)
    colors: dict[str, Any] = field(default_factory=dict)
    theme: dict[str, Any] = field(default_factory=dict)

    def words_for_region(self, region: str) -> list[str]:
        words: list[str] = []
        words.extend(self.region_words.get(region, []))
        words.extend(self.texture_words)
        words.extend(self.hero_words)
        return _dedupe(words) or [self.subject_name, "Legacy", "Focus"]


def _dedupe(words: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for word in words:
        cleaned = " ".join(str(word).strip().split())
        if not cleaned:
            continue
        key = cleaned.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(cleaned)
    return out


def profile_from_prompt(prompt: str, subject: str | None = None) -> WordProfile:
    words = _dedupe(parse_words(prompt))
    subject_name = subject or (words[0] if words else "subject")
    hero_words = words[:6] if words else [subject_name]
    texture_words = words[6:] if len(words) > 6 else words
    return WordProfile(
        subject_name=subject_name,
        hero_words=hero_words,
        texture_words=texture_words or hero_words,
        anchor_words=hero_words[:5],
    )


def profile_from_words_file(path: Path, subject: str | None = None) -> WordProfile:
    return profile_from_prompt(path.read_text(encoding="utf-8"), subject=subject)


def profile_from_json(path: Path) -> WordProfile:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return profile_from_mapping(payload)


def profile_from_mapping(payload: dict[str, Any]) -> WordProfile:
    subject = str(payload.get("subject") or payload.get("subject_name") or "subject")
    hero_words = _dedupe([str(item) for item in payload.get("hero_words", [])])
    texture_words = _dedupe([str(item) for item in payload.get("texture_words", [])])
    anchor_words = _dedupe([str(item) for item in payload.get("anchor_words", [])])
    region_words = {
        str(region): _dedupe([str(item) for item in words])
        for region, words in dict(payload.get("region_words", {})).items()
    }
    if not hero_words:
        hero_words = [subject]
    if not texture_words:
        texture_words = hero_words
    if not anchor_words:
        anchor_words = hero_words[:5]
    return WordProfile(
        subject_name=subject,
        hero_words=hero_words,
        region_words=region_words,
        texture_words=texture_words,
        anchor_words=anchor_words,
        colors=dict(payload.get("colors", {})),
        theme=dict(payload.get("theme", {})),
    )


def load_profile(
    prompt: str | None = None,
    words_path: Path | None = None,
    profile_path: Path | None = None,
    subject: str | None = None,
) -> WordProfile:
    provided = [prompt is not None, words_path is not None, profile_path is not None]
    if sum(provided) != 1:
        raise ValueError("Provide exactly one of --prompt, --words, or --profile")
    if profile_path is not None:
        return profile_from_json(profile_path)
    if words_path is not None:
        return profile_from_words_file(words_path, subject=subject)
    return profile_from_prompt(prompt or "", subject=subject)
