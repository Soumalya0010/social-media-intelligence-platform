"""
Unit and integration tests for Step 2: Text Cleaning, Noise Removal, Emoji Handling, Language Detection.
"""

import unittest
from preprocessing.text_cleaner import TextCleaner


class TestStep2TextCleaning(unittest.TestCase):
    """Verifies noise removal, emoji translation/demojizing, and language detection."""

    def setUp(self):
        self.cleaner = TextCleaner()

    def test_noise_removal(self):
        raw = "Check this out https://example.com/test &amp; join @dev_user! <b>Bold text</b>"
        cleaned = self.cleaner.remove_noise(raw)
        self.assertNotIn("https://", cleaned)
        self.assertNotIn("&amp;", cleaned)
        self.assertNotIn("<b>", cleaned)
        self.assertNotIn("@dev_user", cleaned)
        self.assertIn("&", cleaned)
        self.assertIn("Bold text", cleaned)

    def test_emoji_handling(self):
        raw = "AI is fascinating! 😊🔥"
        demojized, emojis = self.cleaner.handle_emojis(raw)
        self.assertIn("😊", emojis)
        self.assertIn("🔥", emojis)
        self.assertIn(":smiling_face", demojized)
        self.assertIn(":fire:", demojized)

    def test_language_detection(self):
        en_text = "This is a comprehensive natural language processing pipeline for social intelligence."
        lang_en, conf_en = self.cleaner.detect_language(en_text)
        self.assertEqual(lang_en, "en")

        es_text = "Los modelos de lenguaje y la inteligencia artificial son fascinantes y revolucionarios."
        lang_es, conf_es = self.cleaner.detect_language(es_text)
        self.assertEqual(lang_es, "es")

    def test_full_clean_record(self):
        raw = "Breaking news at https://news.org! 🚨 Massive cyber breach exposes data 😱. Follow @reporter!"
        res = self.cleaner.clean_record(raw)
        self.assertIn("clean_text", res)
        self.assertNotIn("https://", res["clean_text"])
        self.assertGreaterEqual(res["emoji_count"], 2)
        self.assertIn("detected_language", res)


if __name__ == "__main__":
    unittest.main()
