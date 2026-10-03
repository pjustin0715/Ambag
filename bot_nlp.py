import html
import re
from collections import Counter

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from bot_teams import read

BOT_PREFIX = "🔔"

# NLTK setup


def _ensure_nltk():
    for pkg, path in [
        ("punkt_tab", "tokenizers/punkt_tab"),
        ("punkt", "tokenizers/punkt"),
        ("stopwords", "corpora/stopwords"),
    ]:
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass


_ensure_nltk()

# Words with task meanig, not be treated as stop words
KEEP_WORDS = {"do", "will", "after", "before",
              "until", "on", "by", "not", "no"}

try:
    STOP_WORDS = set(stopwords.words("english")) - KEEP_WORDS
except LookupError:
    STOP_WORDS = {"the", "is", "a", "an", "and",
                  "or", "in", "for", "to", "of", "it"}

# Signals that a message might create/assign/complete a task
TASK_SIGNALS = {
    "do", "will", "assign", "assigned", "done", "finish", "finished",
    "complete", "completed", "after", "before", "until", "deadline", "due",
    "start", "submit", "handle", "take", "taking", "create", "make", "need",
    "needs", "todo", "task",
}
DATE_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}|"
    r"\b(mon|tues|wednes|thurs|fri|satur|sun)day\b|"
    r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b|"
    r"\b(today|tomorrow|tonight)\b",
    re.I,
)


# Cleaning
def clean_teams_text(raw_text):
    """Teams HTML -> plain text. Mentions keep the person's name."""
    if not raw_text:
        return ""

    text = re.sub(r"<br\s*/?>|</p>|</div>|</li>", " ", raw_text, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)   # strips <p>, <at ...>, <a ...>, etc.
    text = html.unescape(text)            # unescape AFTER stripping tags
    text = text.replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).strip()


def is_bot_message(text):
    return text.startswith(BOT_PREFIX)


# Tokenizing
def tokenize_text(text):
    text = text.lower()
    try:
        words = word_tokenize(text)
    except LookupError:                    # punkt missing and download failed
        words = re.findall(r"[\w-]+", text)

    # keep anything with a letter OR digit ("sprint", "3", "2026-09-25")
    return [w for w in words if re.search(r"\w", w) and w not in STOP_WORDS]


def get_keywords(text, limit=5):
    counts = Counter(tokenize_text(text))
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [word for word, _ in ranked[:limit]]


# Pre-filter before calling Gemini
def looks_like_task(text):
    """Cheap, lenient check. False means 'definitely just chatter'."""
    words = set(re.findall(r"[a-z']+", text.lower()))
    return bool(words & TASK_SIGNALS) or bool(DATE_RE.search(text))


# Reading from Teams
def get_chat_text(limit=10):
    """Cleaned human messages, oldest -> newest."""
    raw_messages = read(limit)
    cleaned = [clean_teams_text(m) for m in reversed(raw_messages)]
    return [m for m in cleaned if m and not is_bot_message(m)]


if __name__ == "__main__":
    for msg in get_chat_text(5):
        print("Text:", msg)
        print("Tokens:", tokenize_text(msg))
        print("Keywords:", get_keywords(msg, 5))
        print("Task-like:", looks_like_task(msg))
        print("---")
