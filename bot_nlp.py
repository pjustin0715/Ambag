import html
import re
import difflib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from bot_teams import read

# Download quietly if missing
for pkg in ["wordnet", "stopwords", "omw-1.4"]:
    try:
        nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass

_lemmatizer = WordNetLemmatizer()
try:
    _STOP_WORDS = set(stopwords.words("english"))
except Exception:
    _STOP_WORDS = {"the", "a", "an", "to", "for", "of", "in", "on", "at", "by", "with"}

BOT_PREFIX = "🔔"

TASK_SIGNALS = {
    "do", "will", "assign", "assigned", "done", "finish", "finished",
    "complete", "completed", "after", "before", "until", "deadline", "due",
    "start", "submit", "handle", "take", "taking", "create", "make", "need",
    "needs", "todo", "task", "can", "review", "work", "working", "i'll",
}

DATE_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}|"
    r"\b(mon|tues|wednes|thurs|fri|satur|sun)day\b|"
    r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b|"
    r"\b(today|tomorrow|tonight)\b",
    re.I,
)

def clean_teams_text(raw_text: str) -> str:
    """Strip HTML tags/entities and normalize whitespace."""
    if not raw_text:
        return ""
    text = re.sub(r"<br\s*/?>|</p>|</div>|</li>", " ", raw_text, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text).replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).strip()

def looks_like_task(text: str) -> bool:
    """Pre-filter: True if message contains task keywords or dates."""
    words = set(re.findall(r"[a-z']+", text.lower()))
    return bool(words & TASK_SIGNALS) or bool(DATE_RE.search(text))

def get_chat_text(limit: int = 10) -> list[tuple[str, str, str]]:
    """Cleaned human messages (oldest -> newest): [(msg_id, text, sender), ...]."""
    messages = []
    for m in reversed(read(limit)):
        text = clean_teams_text(m.get("content", ""))
        sender = m.get("imdisplayname") or m.get("fromGivenNameInToken") or ""
        if text and not text.startswith(BOT_PREFIX):
            messages.append((m.get("id", ""), text, sender))
    return messages

def normalize_tokens(text: str) -> set[str]:
    """Lowercase, remove stopwords, and lemmatize to base forms."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    lemmas = set()
    for w in words:
        if w not in _STOP_WORDS:
            # Lemmatize both verb and noun inflections (e.g., 'reviewing' -> 'review')
            lemma = _lemmatizer.lemmatize(w, "v")
            lemma = _lemmatizer.lemmatize(lemma, "n")
            lemmas.add(lemma)
    return lemmas

def task_similarity(a: str, b: str) -> float:
    """Jaccard token set similarity + containment overlap + character fallback."""
    t_a, t_b = normalize_tokens(a), normalize_tokens(b)
    if not t_a or not t_b:
        return 0.0
    inter = len(t_a & t_b)
    jaccard = inter / len(t_a | t_b)
    # Reward complete containment (e.g., 'eat' inside 'steven is done eating')
    containment = (inter / min(len(t_a), len(t_b))) * 0.85 if inter == min(len(t_a), len(t_b)) else 0.0
    seq = difflib.SequenceMatcher(None, " ".join(sorted(t_a)), " ".join(sorted(t_b))).ratio()
    return max(jaccard, seq, containment)

def match_task_candidate(target_name: str, candidates: list[dict], threshold: float = 0.6) -> dict | None:
    """Finds the best matching Notion page from candidates."""
    best_page, best_score = None, 0.0
    for page in candidates:
        title_objs = page.get("properties", {}).get("Task Name", {}).get("title", [])
        if not title_objs:
            continue
        title = title_objs[0].get("text", {}).get("content", "")
        score = task_similarity(target_name, title)
        if score > best_score:
            best_score, best_page = score, page
    return best_page if best_score >= threshold else None
