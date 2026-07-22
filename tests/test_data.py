from src.data.preprocessing import TextPreprocessor


class TestTextPreprocessor:
    def _make_preprocessor(self):
        config = {
            "data": {
                "preprocessing": {
                    "max_length": 128,
                    "tokenizer": "microsoft/deberta-v3-base",
                    "lowercase": False,
                    "strip_whitespace": True,
                    "deduplication": True,
                    "min_token_length": 3,
                    "max_token_length": 120,
                }
            }
        }
        return TextPreprocessor(config)

    def test_clean_text_basic(self):
        preprocessor = self._make_preprocessor()
        result = preprocessor._clean_text("  Hello World  ")
        assert result == "Hello World"

    def test_clean_text_html_tags(self):
        preprocessor = self._make_preprocessor()
        result = preprocessor._clean_text("Hello <b>World</b>")
        assert "<b>" not in result
        assert "World" in result

    def test_clean_text_urls(self):
        preprocessor = self._make_preprocessor()
        result = preprocessor._clean_text("Visit https://example.com for more")
        assert "https://" not in result
        assert "[URL]" in result

    def test_clean_text_whitespace(self):
        preprocessor = self._make_preprocessor()
        result = preprocessor._clean_text("Hello    World   Test")
        assert "    " not in result
