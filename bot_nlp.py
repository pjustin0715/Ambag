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
_TAGALOG_STOP_WORDS = {
    "ang", "ng", "nang", "sa", "si", "ni", "kay", "sina", "nina", "kina",
    "mga", "yung", "ung", "yun", "yan", "ito", "iyon", "na", "pa", "po",
    "ko", "mo", "ka", "ako", "ikaw", "siya", "kami", "tayo", "kayo", "sila",
    "namin", "natin", "nila", "akin", "sakin", "iyo", "kanya", "lang", "lng",
    "din", "rin", "naman", "nmn", "kasi", "kse", "para", "ba", "eh", "nga",
    "daw", "raw", "muna", "ulit", "pala", "talaga",
}
try:
    _STOP_WORDS = set(stopwords.words("english")) | _TAGALOG_STOP_WORDS
except Exception:
    _STOP_WORDS = {"the", "a", "an", "to", "for", "of", "in", "on", "at", "by", "with"} | _TAGALOG_STOP_WORDS

BOT_PREFIX = "🔔"

TASK_SIGNALS = {
    "do", "will", "assign", "assigned", "done", "finish", "finished",
    "complete", "completed", "after", "before", "until", "deadline", "due",
    "start", "submit", "handle", "take", "taking", "create", "make", "need",
    "needs", "todo", "task", "can", "review", "work", "working", "i'll", "then", "once",
    "have", "has", "i've", "we've", "already", "just", "went", "gone", "did", "sent",
    "submitted", "pushed", "merged", "reviewed", "drafted", "wrote", "written",
    # Tagalog / Taglish task, completion, and sequencing signals
    "bahala", "gawa", "gawin", "gagawa", "gagawin", "nagawa", "ginawa",
    "kailangan", "kelangan", "kailangang", "paki", "pakigawa", "simulan", "umpisahan",
    "kunin", "ipasa", "pasa", "pwede", "pede", "tapos", "tpos", "natapos",
    "napasa", "nasubmit", "oks", "goods", "pagkatapos", "sunod", "kasunod",
    "bago", "hanggang", "kapag", "pag", "tapusin",
}

TAGLISH_RE = re.compile(
    r"\b("
    r"ako\s+na|si\s+\w+\s+na|kay\s+\w+|"
    r"(ko|mo|namin|natin|nila)\s+na|"
    r"(mag|nag|pag|ni|na|in|ipa)-[a-z]+|"
    r"(mag|nag|nai|naka|nakapag)[a-z]{3,}|"
    r"na[a-z]{3,}\s+(na|ko|ni|yung|ung|ang)"
    r")\b",
    re.I,
)

DATE_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}|"
    r"\b(mon|tues?|wed(nes)?|thu(rs)?|fri|sat(ur)?|sun)(day)?\b|"
    r"\b(lunes|martes|miy?erkules|hu?webes|biy?ernes|sabado|linggo)\b|"
    r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b|"
    r"\b(today|tomorrow|tonight|this week|next week|weekend|in \d+ (day|week|month)s?)\b|"
    r"\b(ngayon|ngayong|bukas|bkas|mamaya|makalawa|susunod|kinabukasan)\b",
    re.I,
)

def clean_teams_text(raw_text: str) -> str:
    """Strip HTML tags/entities and preserve line breaks."""
    if not raw_text:
        return ""
    text = re.sub(r"<br\s*/?>|</p>|</div>|</li>", "\n", raw_text, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text).replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n+", "\n", text).strip()

def looks_like_task(text: str) -> bool:
    """Pre-filter: True if message contains English/Tagalog task cues, dates, or lists."""
    words = set(re.findall(r"[a-z']+", text.lower()))
    return (
        bool(words & TASK_SIGNALS)
        or bool(TAGLISH_RE.search(text))
        or bool(DATE_RE.search(text))
        or ("\n" in text and " - " in text)
        or bool(re.search(r"^\s*\d+[.)]\s+\w+", text, re.M))
    )

def get_chat_text(limit: int = 10, chat_id: str | None = None) -> list[tuple[str, str, str, list[str]]]:
    """Cleaned human messages (oldest -> newest): [(msg_id, text, sender, img_urls), ...]."""
    messages = []
    for m in reversed(read(limit, chat_id=chat_id)):
        raw = m.get("content", "")
        imgs = [u for u in re.findall(r'<img[^>]+src="([^"]+)"', raw, flags=re.I) if "emoticon" not in u.lower() and "emoji" not in u.lower()]
        text = clean_teams_text(raw)
        sender = m.get("imdisplayname") or m.get("fromGivenNameInToken") or ""
        if (text or imgs) and not text.startswith(BOT_PREFIX) and "just a reminder, your task" not in text and "You can start with your task" not in text and "Your task tracker & flowchart is ready:" not in text and "that image was a bit hard to read!" not in text and "couldn't quite make out the writing" not in text:
            messages.append((m.get("id", ""), text, sender, imgs))
    return messages

def normalize_tokens(text: str) -> set[str]:
    """Lowercase, strip Taglish hyphenated verb prefixes, remove EN/TL stopwords, and lemmatize."""
    # Strip hyphenated Taglish prefixes (e.g., 'nag-review' -> 'review', 'mag-push' -> 'push')
    cleaned = re.sub(r"\b(?:mag|nag|pag|ni|na|in|ipa)-([a-z0-9]+)\b", r"\1", text.lower())
    words = re.findall(r"[a-z0-9]+", cleaned)
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
    return max(jaccard, containment, seq if seq >= 0.85 else 0.0)

def match_task_candidate(target_name: str, candidates: list[dict], threshold: float = 0.75) -> dict | None:
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


if __name__ == "__main__":
    # Self-check for Tagalog / Taglish pre-filter & fuzzy matching
    for msg in (
        "ako na bahala sa intro",
        "si jonas na sa slides bukas",
        "tpos ko na ung chapter 1",
        "nag-push na ko ng code",
        "sa makalawa yung draft",
        "kay steven yung wiring",
        "naprint ko na ang paper",
        "nasend na yung slides",
    ):
        assert looks_like_task(msg), f"Should pass looks_like_task: {msg}"
    for chatter in ("haha oo nga eh", "kain muna tayo", "sige g", "gutom na ako"):
        assert not looks_like_task(chatter), f"Should skip chatter: {chatter}"
    assert task_similarity("ni-review na pr", "review pr") == 1.0
    assert task_similarity("yung slides ko", "slides") == 1.0
    assert task_similarity("gawin yung paper", "gawin yung draft") < 0.75
    print("bot_nlp Taglish self-check passed!")
