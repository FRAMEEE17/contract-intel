"""Unit tests for app.adapters.documents.text_parser.TextDocParser."""
from __future__ import annotations

from app.adapters.documents.text_parser import TextDocParser
from app.domain.ports import ParsedDocument


def test_parse_returns_parsed_document_with_expected_fields():
    parser = TextDocParser()
    content = "Hello world".encode("utf-8")

    result = parser.parse(content=content, filename="hello.txt")

    assert isinstance(result, ParsedDocument)
    assert result.document_id == "hello.txt"
    assert result.parser_version == "text/v1"
    assert result.text == "Hello world"
    assert result.pages == ["Hello world"]


def test_parse_decodes_utf8_with_replace_on_invalid_bytes():
    parser = TextDocParser()
    # 0xff is not valid utf-8 on its own -> should be replaced, not raise.
    content = b"valid text \xff more text"

    result = parser.parse(content=content, filename="bad.txt")

    assert isinstance(result.text, str)
    assert "�" in result.text
    assert "valid text" in result.text
    assert "more text" in result.text


def test_parse_splits_pages_on_form_feed():
    parser = TextDocParser()
    content = "page one\fpage two\fpage three".encode("utf-8")

    result = parser.parse(content=content, filename="multi.txt")

    assert result.pages == ["page one", "page two", "page three"]
    assert result.text == "page one\fpage two\fpage three"


def test_parse_no_form_feed_yields_single_page():
    parser = TextDocParser()
    content = "just one page of text".encode("utf-8")

    result = parser.parse(content=content, filename="single.txt")

    assert result.pages == ["just one page of text"]


def test_parse_document_id_matches_given_filename():
    parser = TextDocParser()
    content = b"content"

    result = parser.parse(content=content, filename="some/path/name.txt")

    assert result.document_id == "some/path/name.txt"
