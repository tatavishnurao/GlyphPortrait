from glyphforge.keywords.parser import parse_weighted_words, parse_words


def test_parse_words_handles_commas_and_lines():
    text = "MVP, Leader\nChampion,  Focus  "
    words = parse_words(text)
    assert words == ["MVP", "Leader", "Champion", "Focus"]


def test_parse_weighted_words_defaults_when_empty():
    weighted = parse_weighted_words("")
    assert len(weighted) >= 1
    assert all(len(item) == 2 for item in weighted)


def test_parse_words_handles_extra_delimiters_and_spaces():
    text = " , , MVP  ;  The  GOAT \n\n ; Clutch Performer , "
    words = parse_words(text)
    assert words == ["MVP", "The GOAT", "Clutch Performer"]


def test_parse_weighted_words_respects_max_words_limit():
    weighted = parse_weighted_words("a,b,c,d,e", max_words=3)
    assert [word for word, _ in weighted] == ["a", "b", "c"]
