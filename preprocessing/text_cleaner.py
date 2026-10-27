"""
Text Preprocessing and Cleaning Module.
Step 2: Clean text: remove noise, handle emojis, detect language.
"""

import html
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import emoji
from langdetect import DetectorFactory, detect, detect_langs

from config import PROCESSED_DATA_DIR, REDDIT_RAW_DIR, YOUTUBE_RAW_DIR

# Enforce deterministic language detection
DetectorFactory.seed = 42

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class TextCleaner:
    """
    Cleans social media text by:
    1. Removing noise (URLs, mentions, markdown, HTML entities, excess symbols).
    2. Handling emojis (demojizing, extracting emoji counts/sentiments).
    3. Detecting language (using langdetect with probabilistic fallback).
    """

    # URL regex
    URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
    # HTML tag regex
    HTML_TAG_PATTERN = re.compile(r"<.*?>")
    # Mention regex (Reddit u/, r/, Twitter/YouTube @mentions)
    MENTION_PATTERN = re.compile(r"[@/][uUrR]?[\w\-_]+")
    # Markdown symbols (e.g. **bold**, *italic*, headers, blockquotes)
    MARKDOWN_PATTERN = re.compile(r"[\*\_~`#>\[\]\(\)]+")
    # Excessive whitespace and newline repetitions
    WHITESPACE_PATTERN = re.compile(r"\s+")
    # Repetitive punctuation like !!!! or ???
    PUNCT_REPEAT_PATTERN = re.compile(r"([!?,.]){2,}")

    def __init__(self, demojize: bool = True, default_language: str = "en"):
        self.demojize = demojize
        self.default_language = default_language

    def remove_noise(self, text: str) -> str:
        """Removes URLs, HTML tags, mentions, excess symbols, and decodes HTML entities."""
        if not text:
            return ""

        # Decode HTML entities (e.g., &amp; -> &, &lt; -> <)
        cleaned = html.unescape(text)

        # Remove URLs
        cleaned = self.URL_PATTERN.sub(" ", cleaned)

        # Remove HTML tags
        cleaned = self.HTML_TAG_PATTERN.sub(" ", cleaned)

        # Remove mentions (@username, u/username, r/subreddit)
        cleaned = self.MENTION_PATTERN.sub(" ", cleaned)

        # Simplify markdown symbols
        cleaned = self.MARKDOWN_PATTERN.sub(" ", cleaned)

        # Normalize repeated punctuation (e.g., '!!!!' -> '!')
        cleaned = self.PUNCT_REPEAT_PATTERN.sub(r"\1", cleaned)

        # Collapse excess whitespace
        cleaned = self.WHITESPACE_PATTERN.sub(" ", cleaned).strip()

        return cleaned

    def handle_emojis(self, text: str) -> Tuple[str, List[str]]:
        """
        Detects emojis in text and demojizes them into textual descriptive tokens.
        Example: 'Great news! 😊🔥' -> ('Great news! :smiling_face_with_smiling_eyes: :fire:', ['😊', '🔥'])
        """
        if not text:
            return "", []

        # Extract list of distinct emojis found
        emojis_found = [c for c in text if emoji.is_emoji(c)]

        if self.demojize:
            # Convert emojis to text representation with clear spaces
            demojized = emoji.demojize(text, delimiters=(" :", ": "))
            # Collapse whitespace around demojized tokens
            demojized = self.WHITESPACE_PATTERN.sub(" ", demojized).strip()
            return demojized, emojis_found

        return text, emojis_found

    def detect_language(self, text: str) -> Tuple[str, float]:
        """
        Detects the primary language of the text and returns (lang_code, confidence).
        Falls back gracefully for very short or non-alphabetic strings.
        """
        if not text or len(text.strip()) < 3:
            return self.default_language, 0.0

        # Strip demojized tokens and digits for cleaner language detection
        stripped_for_lang = re.sub(r":\w+:", " ", text)
        stripped_for_lang = re.sub(r"[0-9\W]+", " ", stripped_for_lang).strip()

        if len(stripped_for_lang) < 3:
            return self.default_language, 0.0

        try:
            predictions = detect_langs(stripped_for_lang)
            if predictions:
                best = predictions[0]
                return best.lang, round(best.prob, 3)
            return detect(stripped_for_lang), 1.0
        except Exception:
            return self.default_language, 0.0

    def clean_record(self, raw_text: str) -> Dict[str, Any]:
        """
        Executes full Step 2 cleaning pipeline on a raw text string:
        Noise Removal -> Emoji Handling -> Language Detection.
        """
        # Step 2a: Handle emojis first so their textual meaning is preserved
        text_with_emojis_handled, emojis_found = self.handle_emojis(raw_text or "")

        # Step 2b: Remove noise
        clean_text = self.remove_noise(text_with_emojis_handled)

        # Step 2c: Detect language
        detected_lang, lang_confidence = self.detect_language(clean_text)

        return {
            "clean_text": clean_text,
            "emojis_detected": emojis_found,
            "emoji_count": len(emojis_found),
            "detected_language": detected_lang,
            "language_confidence": lang_confidence,
        }


def extract_posts_from_raw_json(cleaner: Optional[TextCleaner] = None) -> List[Dict[str, Any]]:
    """
    Parses all raw JSON files from Reddit and YouTube directories, cleans their text,
    and returns a standardized list of structured post and comment records.
    """
    cleaner = cleaner or TextCleaner()
    all_records: List[Dict[str, Any]] = []

    # 1. Process Raw Reddit Files
    reddit_files = list(Path(REDDIT_RAW_DIR).glob("*.json"))
    for r_file in reddit_files:
        try:
            with open(r_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            children = data.get("data", {}).get("children", [])
            for child in children:
                pdata = child.get("data", {})
                post_id = pdata.get("id", "")
                title = pdata.get("title", "")
                selftext = pdata.get("selftext", "")
                raw_text = f"{title}. {selftext}".strip()

                cleaned_meta = cleaner.clean_record(raw_text)

                record = {
                    "id": f"reddit_{post_id}",
                    "source": "reddit",
                    "type": "post",
                    "parent_id": None,
                    "author": pdata.get("author", "[deleted]"),
                    "created_utc": pdata.get("created_utc", datetime.now(timezone.utc).timestamp()),
                    "iso_timestamp": datetime.fromtimestamp(
                        pdata.get("created_utc", datetime.now(timezone.utc).timestamp()), timezone.utc
                    ).isoformat(),
                    "raw_text": raw_text,
                    "clean_text": cleaned_meta["clean_text"],
                    "emojis_detected": cleaned_meta["emojis_detected"],
                    "emoji_count": cleaned_meta["emoji_count"],
                    "detected_language": cleaned_meta["detected_language"],
                    "language_confidence": cleaned_meta["language_confidence"],
                    "subreddit": pdata.get("subreddit", "unknown"),
                    "score": pdata.get("score", 0),
                    "num_comments": pdata.get("num_comments", 0),
                    "url": pdata.get("url", ""),
                }
                all_records.append(record)

                # Process nested comments if available
                raw_comments = pdata.get("raw_comments", [])
                for comment_wrap in raw_comments:
                    if isinstance(comment_wrap, dict):
                        c_children = comment_wrap.get("data", {}).get("children", [])
                        for c in c_children:
                            cdata = c.get("data", {})
                            c_body = cdata.get("body", "")
                            if c_body:
                                c_clean = cleaner.clean_record(c_body)
                                c_time = cdata.get("created_utc", record["created_utc"] + 60)
                                all_records.append({
                                    "id": f"reddit_comment_{cdata.get('id', '')}",
                                    "source": "reddit",
                                    "type": "comment",
                                    "parent_id": record["id"],
                                    "author": cdata.get("author", "[deleted]"),
                                    "created_utc": c_time,
                                    "iso_timestamp": datetime.fromtimestamp(c_time, timezone.utc).isoformat(),
                                    "raw_text": c_body,
                                    "clean_text": c_clean["clean_text"],
                                    "emojis_detected": c_clean["emojis_detected"],
                                    "emoji_count": c_clean["emoji_count"],
                                    "detected_language": c_clean["detected_language"],
                                    "language_confidence": c_clean["language_confidence"],
                                    "subreddit": record["subreddit"],
                                    "score": cdata.get("score", 0),
                                    "num_comments": 0,
                                    "url": record["url"],
                                })
        except Exception as e:
            logger.error(f"Error parsing Reddit raw file {r_file}: {e}")

    # 2. Process Raw YouTube Files
    youtube_files = list(Path(YOUTUBE_RAW_DIR).glob("*.json"))
    for yt_file in youtube_files:
        try:
            with open(yt_file, "r", encoding="utf-8") as f:
                yt_data = json.load(f)

            items = yt_data.get("items", [])
            for item in items:
                vid_id = item.get("id", "")
                snippet = item.get("snippet", {})
                stats = item.get("statistics", {})

                title = snippet.get("title", "")
                desc = snippet.get("description", "")
                raw_text = f"{title}. {desc}".strip()
                cleaned_meta = cleaner.clean_record(raw_text)

                pub_at = snippet.get("publishedAt", datetime.now(timezone.utc).isoformat())
                try:
                    dt = datetime.fromisoformat(pub_at.replace("Z", "+00:00"))
                    utc_ts = dt.timestamp()
                except Exception:
                    utc_ts = datetime.now(timezone.utc).timestamp()
                    dt = datetime.fromtimestamp(utc_ts, timezone.utc)

                video_record = {
                    "id": f"youtube_{vid_id}",
                    "source": "youtube",
                    "type": "video",
                    "parent_id": None,
                    "author": snippet.get("channelTitle", "Unknown Channel"),
                    "created_utc": utc_ts,
                    "iso_timestamp": dt.isoformat(),
                    "raw_text": raw_text,
                    "clean_text": cleaned_meta["clean_text"],
                    "emojis_detected": cleaned_meta["emojis_detected"],
                    "emoji_count": cleaned_meta["emoji_count"],
                    "detected_language": cleaned_meta["detected_language"],
                    "language_confidence": cleaned_meta["language_confidence"],
                    "channel_title": snippet.get("channelTitle", ""),
                    "score": int(stats.get("likeCount", 0)),
                    "views": int(stats.get("viewCount", 0)),
                    "num_comments": int(stats.get("commentCount", 0)),
                    "url": f"https://www.youtube.com/watch?v={vid_id}",
                }
                all_records.append(video_record)

                # Process top comments for this video
                comments_threads = item.get("commentThreads", {}).get("items", [])
                for thread in comments_threads:
                    top_c = thread.get("snippet", {}).get("topLevelComment", {}).get("snippet", {})
                    c_text = top_c.get("textOriginal", "")
                    if c_text:
                        c_clean = cleaner.clean_record(c_text)
                        c_pub = top_c.get("publishedAt", video_record["iso_timestamp"])
                        try:
                            c_dt = datetime.fromisoformat(c_pub.replace("Z", "+00:00"))
                            c_ts = c_dt.timestamp()
                        except Exception:
                            c_ts = video_record["created_utc"] + 120
                            c_dt = datetime.fromtimestamp(c_ts, timezone.utc)

                        all_records.append({
                            "id": f"youtube_comment_{thread.get('id', '')}",
                            "source": "youtube",
                            "type": "comment",
                            "parent_id": video_record["id"],
                            "author": top_c.get("authorDisplayName", "Anonymous"),
                            "created_utc": c_ts,
                            "iso_timestamp": c_dt.isoformat(),
                            "raw_text": c_text,
                            "clean_text": c_clean["clean_text"],
                            "emojis_detected": c_clean["emojis_detected"],
                            "emoji_count": c_clean["emoji_count"],
                            "detected_language": c_clean["detected_language"],
                            "language_confidence": c_clean["language_confidence"],
                            "channel_title": video_record["channel_title"],
                            "score": int(top_c.get("likeCount", 0)),
                            "num_comments": 0,
                            "url": video_record["url"],
                        })

        except Exception as e:
            logger.error(f"Error parsing YouTube raw file {yt_file}: {e}")

    return all_records


def process_raw_dataset(output_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """
    Executes full Step 2: reads raw JSONs, applies TextCleaner, and saves
    the clean dataset to `data/processed/clean_posts.json`.
    """
    target = Path(output_path or (PROCESSED_DATA_DIR / "clean_posts.json"))
    target.parent.mkdir(parents=True, exist_ok=True)

    cleaner = TextCleaner()
    logger.info("Step 2: Cleaning raw social media posts (noise removal, emoji handling, language detection)...")
    cleaned_records = extract_posts_from_raw_json(cleaner)

    # Sort deterministically by timestamp
    cleaned_records.sort(key=lambda x: x.get("created_utc", 0))

    with open(target, "w", encoding="utf-8") as f:
        json.dump(cleaned_records, f, ensure_ascii=False, indent=2)

    logger.info(f"Step 2 completed: Successfully cleaned and saved {len(cleaned_records)} records to {target}")
    return cleaned_records


if __name__ == "__main__":
    records = process_raw_dataset()
    print(f"Total processed and cleaned records: {len(records)}")
