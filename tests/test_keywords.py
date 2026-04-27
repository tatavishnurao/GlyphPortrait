from glyphforge.keywords.parser import parse_weighted_words, parse_words


def test_parse_words_handles_commas_and_lines():
    text = "MVP, Leader\nChampion,  Focus  "
    words = parse_words(text)
    assert words == ["MVP", "Leader", "Champion", "Focus"]


def test_parse_weighted_words_defaults_when_empty():
    weighted = parse_weighted_words("")
    assert len(weighted) >= 1
    assert all(len(item) == 2 for item in weighted)
