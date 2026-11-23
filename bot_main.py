import re
import sys
import time
import json
import base64
import threading
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from datetime import date, datetime, timedelta
from pydantic import BaseModel
from google import genai
from google.genai import types
from notion_client import Client
import bot_teams

from collections import deque
from bot_nlp import get_chat_text, looks_like_task, match_task_candidate

import os
from urllib.parse import unquote
from dotenv import load_dotenv, set_key

load_dotenv()

# ================= CONFIG =================
GEMINI_API_KEY = os.getenv("gemini")
GROQ_API_KEY = os.getenv("groq") or os.getenv("GROQ_API_KEY")
OPENROUTER_API_KEY = os.getenv("openrouter") or os.getenv("OPENROUTER_API_KEY")
NOTION_TOKEN = os.getenv("notion")
NOTION_DB_ID = os.getenv("notion_db_id", "").strip()
NOTION_ROOT_PAGE_ID = os.getenv("notion_root_page_id", "").strip()
OCR_CONFIDENCE_THRESHOLD = float(os.getenv("ocr_threshold", "0.90"))  # Raise to 0.95 for stricter confirmation, or 1.01 to always confirm images
CHATS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chats.json")


_http = requests.Session()
_http.mount("https://", HTTPAdapter(max_retries=Retry(total=3, connect=3, read=2, backoff_factor=0.3, allowed_methods=frozenset({"GET", "POST"}))))
llm = genai.Client(api_key=GEMINI_API_KEY, http_options=types.HttpOptions(timeout=20000)) if GEMINI_API_KEY else None
notion = Client(auth=NOTION_TOKEN)
_DB_INFO = notion.databases.retrieve(NOTION_DB_ID) if (NOTION_DB_ID and NOTION_DB_ID != "..." and "--setup" not in sys.argv) else {}
DATA_SOURCE_ID = _DB_INFO["data_sources"][0]["id"] if _DB_INFO else ""
_flowchart_block_id = None
_pulse_block_id = None
_sync_lock = threading.Lock()
_chats_lock = threading.Lock()
recent_by_chat: dict[str, deque] = {}  # chat_id -> deque[(task_name, assignee, sender)]
open_tasks_by_chat: dict[str, list[tuple[str, str]]] = {}  # chat_id -> [(title, assignee)]
notion_state_by_chat: dict[str, dict[str, tuple]] = {}  # chat_id -> {page_id: snapshot_tuple}


def _get_recent(chat_id: str) -> deque:
    return recent_by_chat.setdefault(chat_id, deque(maxlen=20))


def _first_name(raw: str | None, default: str = "Someone") -> str:
    parts = (raw or "").strip().split()
    return parts[0].capitalize() if parts else default


def _valid_iso_date(val: str | None) -> bool:
    if not val or len(val) != 10:
        return False
    try:
        date.fromisoformat(val)
        return True
    except ValueError:
        return False


def _record_msg_tasks(cfg: dict, msg_ids: list[str], actions: list[dict]) -> bool:
    """Map Teams message ID(s) -> created/assigned task names so replies to photos resolve accurately."""
    names = [a["task"].strip() for a in actions if a.get("action") in ("create", "assign") and (a.get("task") or "").strip()]
    if not names:
        return False
    mt = cfg.setdefault("msg_tasks", {})
    for mid in msg_ids:
        if mid:
            mt[mid] = names
    while len(mt) > 30:
        mt.pop(next(iter(mt)))
    return True


def _record_member(cfg: dict, sender: str, mri: str) -> bool:
    """Cache sender's Teams MRI by first name for real push @mentions."""
    first = _first_name(sender, "")
    if not first or not mri or not mri.startswith("8:"):
        return False
    members = cfg.setdefault("members", {})
    if members.get(first.lower(), {}).get("mri") != mri:
        members[first.lower()] = {"mri": mri, "name": first}
        return True
    return False


def _mention(cfg: dict | None, raw_name: str) -> tuple[str, list[tuple[str, str]] | None]:
    """Return ('<at id="0">Name</at>', [(mri, Name)]) if MRI known, else ('@Name', None)."""
    first = _first_name(raw_name, "Someone")
    info = ((cfg or {}).get("members") or {}).get(first.lower())
    if info and info.get("mri"):
        return f'<at id="0">{first}</at>', [(info["mri"], first)]
    return f"@{first}", None


def _creates_cycle(task_id: str, prereq_id: str, all_tasks: list[dict]) -> bool:
    """Return True if linking prereq_id -> task_id would create a circular DAG dependency."""
    if task_id == prereq_id:
        return True
    parents = {
        t["id"]: [r["id"] for r in t["properties"].get("Prerequisite", {}).get("relation", [])]
        for t in all_tasks
    }
    stack, visited = [prereq_id], set()
    while stack:
        cur = stack.pop()
        if cur == task_id:
            return True
        if cur not in visited:
            visited.add(cur)
            stack.extend(parents.get(cur, []))
    return False


def load_chats() -> dict:
    chats = json.load(open(CHATS_FILE)) if os.path.exists(CHATS_FILE) else {}
    # Keep existing single-chat .env setup working automatically
    if bot_teams.CHAT_ID and NOTION_DB_ID and DATA_SOURCE_ID and bot_teams.CHAT_ID not in chats:
        chats[bot_teams.CHAT_ID] = {
            "db_id": NOTION_DB_ID,
            "data_source_id": DATA_SOURCE_ID,
            "flowchart_block_id": _flowchart_block_id,
            "name": "Default Chat",
        }
    return chats


def save_chats(chats: dict):
    with _chats_lock:
        tmp = CHATS_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(chats, f, indent=2)
        os.replace(tmp, CHATS_FILE)


def provision_chat(chat_id: str, topic: str) -> dict:
    """Create a dedicated Notion page containing an inline Task List DB + Mermaid flowchart, then send its link."""
    page = notion.pages.create(
        parent={"page_id": NOTION_ROOT_PAGE_ID},
        properties={"title": [{"text": {"content": topic}}]},
    )
    db = notion.databases.create(
        parent={"type": "page_id", "page_id": page["id"]},
        is_inline=True,
        title=[{"type": "text", "text": {"content": "Task List"}}],
        initial_data_source={
            "properties": {
                "Task Name": {"title": {}},
                "Assignee": {"rich_text": {}},
                "Deadline": {"date": {}},
                "Completed": {"checkbox": {}},
            }
        },
    )
    ds_id = db["data_sources"][0]["id"]
    notion.data_sources.update(
        data_source_id=ds_id,
        properties={
            "Prerequisite": {
                "relation": {"data_source_id": ds_id, "type": "single_property", "single_property": {}}
            }
        },
    )
    safe_topic = topic.replace('"', "'").replace("\n", " ").strip() or "Project"
    created = notion.blocks.children.append(
        block_id=page["id"],
        children=[
            {
                "object": "block",
                "type": "callout",
                "callout": {
                    "rich_text": [{"type": "text", "text": {"content": "Progress: 0/0 tasks done"}}],
                    "color": "gray_background",
                },
            },
            {
                "object": "block",
                "type": "code",
                "code": {
                    "language": "mermaid",
                    "rich_text": [{"type": "text", "text": {"content": f'---\ntitle: "{safe_topic}\'s Tasks"\n---\nflowchart LR\n    empty["No tasks yet"]'}}],
                },
            },
        ],
    )
    cfg = {
        "name": topic,
        "page_id": page["id"],
        "page_url": page["url"],
        "db_id": db["id"],
        "data_source_id": ds_id,
        "pulse_block_id": created["results"][0]["id"],
        "flowchart_block_id": created["results"][1]["id"],
    }
    welcome = (
        f'Hey everyone! I\'m Ambag, and I\'ll help keep track of your tasks in this chat. '
        f'Everything we assign here will automatically show up in your <a href="{page["url"]}">Notion tracker</a>.<br/><br/>'
        "<b>Quick privacy note:</b> Task-related messages and shared planning photos are sent to external AI APIs "
        "(Groq / OpenRouter / Gemini) only to extract tasks, assignees, deadlines, and dependencies. "
        "Casual chatter stays out of Notion, and you can remove me from the chat anytime to stop monitoring."
    )
    bot_teams.send(welcome, chat_id=chat_id)
    print(f"[Provisioned] '{topic}' -> {page['url']}")
    return cfg


class TaskAction(BaseModel):
    action: str  # create | assign | complete | uncomplete | delete | query_all | query_person | query_link | ignore
    task: str | None
    assignee: str | None
    deadline: str | None
    prerequisite_task: str | None


class TaskList(BaseModel):
    actions: list[TaskAction]
    ocr_confidence: float = 1.0

# ================= LLM BRAIN =================


def _call_openai_compat(prompt: str, url: str, api_key: str, model: str, images: list[tuple[bytes, str]] | None = None) -> tuple[list[dict], float]:
    full_prompt = prompt + '\nReturn ONLY valid JSON: {"ocr_confidence": float, "actions": [{"action": "create|assign|complete|uncomplete|delete|query_all|query_person|query_link|ignore", "task": str|null, "assignee": str|null, "deadline": str|null, "prerequisite_task": str|null}]}'
    if images:
        content = [{"type": "text", "text": full_prompt}] + [
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{base64.b64encode(b).decode()}"}}
            for b, mime in images
        ]
    else:
        content = full_prompt
    body = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": 1024,
    }
    if "openrouter.ai" in url and model == "openrouter/free":
        body["reasoning"] = {"enabled": False}
    if model == "openai/gpt-oss-20b":
        body["reasoning_effort"] = "low"
    if model == "openai/gpt-4o-mini":
        body["response_format"] = {"type": "json_object"}
    r = _http.post(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
        json=body,
        timeout=15 if images else 10,
    )
    if not r.ok:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
    raw = (r.json()["choices"][0]["message"].get("content") or "").strip()
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.S).strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start == -1 or end <= start:
            raise
        data = json.loads(raw[start:end + 1])
    actions = [a for a in data.get("actions", []) if a.get("action") != "ignore"]
    conf = float(data.get("ocr_confidence", 1.0)) if images else 1.0
    return actions, conf


def analyze_message(
    text: str,
    sender: str = "",
    images: list[tuple[bytes, str]] | None = None,
    chat_id: str | None = None,
    pending_draft: list[dict] | None = None,
) -> tuple[list[dict], float]:
    """Returns (actions, ocr_confidence)."""
    today = date.today()
    cal = ", ".join((today + timedelta(days=i)).strftime("%a %Y-%m-%d") for i in range(14))
    cid = chat_id or bot_teams.CHAT_ID
    recent = _get_recent(cid)
    recent_str = ", ".join(
        f"'{t}' (assignee: {a or 'none'}, by: {s or 'unknown'})" for t, a, s in list(recent)[-5:]
    )
    open_str = ", ".join(
        f"'{t}' (assignee: {a or 'unassigned'})" for t, a in open_tasks_by_chat.get(cid, [])[:25]
    ) or "none"
    last_sender_task = next(
        (t for t, _, s in reversed(recent) if s == sender),
        recent[-1][0] if recent else None,
    )
    draft_line = (
        f"Pending OCR draft awaiting confirmation (if the message corrects/patches any field, return the FULL corrected actions list): {json.dumps(pending_draft)}\n"
        if pending_draft
        else ""
    )
    prompt = (
        f"Today is {today.isoformat()} ({today.strftime('%A')}). "
        f"Message sender: {sender or 'Unknown'}.\n"
        f"Upcoming 14 days: {cal}.\n"
        f"Recent tasks (oldest -> newest): {recent_str or 'none'}.\n"
        f"Active open tasks in Notion: {open_str}.\n"
        f"{draft_line}"
        "Extract ALL task actions from the message and/or any attached whiteboard/notes image. Understand natural English, Tagalog, and colloquial Taglish (split across lines OR connectors like 'and', 'then', 'also', 'at', 'tapos', ',', ';'):\n"
        "- ocr_confidence: float 0.0-1.0 for attached images (1.0 if no image). Be strict: 0.95-1.0 ONLY for crisp typed/printed text; 0.75-0.89 for clear handwriting; 0.50-0.74 for messy/blurry/skewed handwriting; <0.50 if mostly illegible. Lower the score if ANY word, name, or date had to be guessed.\n"
        "- action:\n"
        "  * 'assign': someone takes, is assigned, reassigns, or updates a task (e.g. '[Name] will [task]', 'I will/can [task]', 'ako na sa [task]', 'ako na bahala sa [task]', 'si [Name] na sa [task]', 'kay [Name] yung [task]', '[task] - [Name] - [date]'). Also use 'assign' when a reply reassigns or updates a quoted task in [Replying to: ...] (e.g. 'ako na pala dito', 'I'll take this', 'si [Name] na lang dito', 'sa Friday na lang pala'), using the task from [Replying to: ...].\n"
        f"  * 'complete': task is finished/done or stated as already performed, including unhyphenated Taglish perfect tense (e.g. 'I have gone to class', 'tapos na', 'tapos ko na', 'naprint ko na', 'nasend na', 'nagawa ko na', 'napasa ko na', 'oks na', 'goods na', 'na-push ko na'). Strip prefixes like 'na-'/'nag-' so task name matches the base task (e.g. 'naprint ko na ang paper' -> task='print paper'). If the message just says 'done'/'finished'/'tapos na'/'oks na' without naming a task, prefer any task listed in [Replying to: ...] first; otherwise infer the open task assigned to '{sender}' from Active open tasks in Notion (or the only open task if unassigned). Set assignee=null for 'complete'. Note: 'g' / 'g na' means 'let's go', NOT complete.\n"
        "  * 'uncomplete': reopen a mistakenly completed task (e.g. 'reopen [task]', 'undo [task]', 'di pa pala tapos yung [task]', 'hindi pa tapos [task]').\n"
        "  * 'delete': cancel or remove a task (e.g. 'cancel [task]', 'delete [task]', 'remove [task]', 'wag na yung [task]', 'tanggalin yung [task]', 'burahin yung [task]').\n"
        "  * 'create': new task is created without assignment (e.g. 'kailangan gawin yung [task]').\n"
        "  * 'query_all': user asks for a summary of all remaining/pending tasks (e.g. 'what's left?', 'ano pa kulang?', 'what are we still missing?', 'status?'). Set task=null, assignee=null.\n"
        "  * 'query_person': user asks what tasks a specific person or themselves has left (e.g. 'what tasks do I have left?', 'ano pa gagawin ko?', 'ano pa gagawin ni steven?', 'may pending pa ba si jonas?'). Set task=null and assignee to that person's name (or sender if first-person).\n"
        "  * 'query_link': user asks for the Notion tracker / board / flowchart link (e.g. 'link?', 'asan yung notion?', 'pahingi ng tracker link'). Set task=null, assignee=null.\n"
        "  * 'ignore': casual chatter or no task action.\n"
        f"- assignee: Person assigned. If first-person ('I', 'me', 'myself', 'ako', 'ko', 'akin', 'sakin', 'ako na'), use sender '{sender}'. Only set if explicitly stated, else null.\n"
        "- task: Clean, concise task name (strip filler words like 'yung', 'ang', 'na', 'ko'). If referring to an existing task in Recent tasks or Active open tasks (even in another language, e.g. Tagalog vs English, or via a quoted [Replying to: ...] block), reuse that exact existing task name.\n"
        "- deadline: ISO 8601 (YYYY-MM-DD) or null. Resolve relative phrases using the calendar above: "
        "'today'/'ngayon'/'mamaya' = today; 'tomorrow'/'bukas'/'bkas' = +1 day; 'sa makalawa' = +2 days; "
        "'this <day>'/'on <day>'/'<day>'/'sa <day>'/'ngayong <day>' (lunes..linggo) = today if today IS <day>, otherwise the nearest upcoming <day>; "
        "'next <day>'/'sa susunod na <day>' = +7 days if today IS <day>, otherwise <day> in the following calendar week (or +7 days from this <day>); "
        "each extra 'next' (e.g. 'next next <day>') adds +7 days; "
        "'this week'/'ngayong linggo' = upcoming Friday (or Sunday if past Fri); 'next week'/'sa susunod na linggo' = Friday of next week; "
        "'in N days/weeks' = exact offset from today.\n"
        "- prerequisite_task: Clean underlying task name only (e.g. if 'after Steven is done eating' or 'pagkatapos kumain ni Steven', prerequisite_task is 'eat'/'kumain'). "
        f"Sequential/conditional connectors ('then', 'once', 'before', 'after', 'afterwards', 'pagkatapos', 'kapag tapos na', 'pag tapos na', 'bago', 'sunod', 'kasunod', or 'tapos' used as 'then') link tasks: in 'A then B' or 'A tapos B', B's prerequisite_task is A; if a message starts with or ends with 'after'/'then'/'tapos'/'pagkatapos'/'sunod' with no task named, ALWAYS set prerequisite_task to '{last_sender_task or 'null'}' (or the previous task in this message). "
        "If it says 'after [Person]' / 'pagkatapos ni [Person]', use [Person]'s most recent task from Recent tasks above or earlier in this message. "
        "If tasks are numbered in order (e.g. '1. A', '2. B', '3. C' in text or image), strip the numbers from task names and chain each item's prerequisite_task to the previous numbered task (unless another prerequisite is stated). "
        "Do NOT include person names or status phrases like 'is done'/'tapos na'/'after', and NEVER set prerequisite_task to the task itself.\n"
        f"Examples:\n"
        f"  1) 'I will go to school tomorrow and steven will bring the laptop after' -> (1) task='go to school', assignee='{sender or 'sender'}', deadline=tomorrow, prerequisite_task=null; (2) task='bring the laptop', assignee='steven', deadline=null, prerequisite_task='go to school'.\n"
        f"  2) 'ako na bahala sa slides bukas tapos si jonas mag-review sa makalawa' -> (1) task='slides', assignee='{sender or 'sender'}', deadline=+1d, prerequisite_task=null; (2) task='review slides', assignee='jonas', deadline=+2d, prerequisite_task='slides'.\n"
        f"Message:\n{text or '(see attached image)'}"
    )
    last_err = None
    if GROQ_API_KEY:
        groq_models = ("qwen/qwen3.8-27b",) if images else ("qwen/qwen3.8-27b", "openai/gpt-oss-20b")
        for g_model in groq_models:
            try:
                return _call_openai_compat(prompt, "https://api.groq.com/openai/v1/chat/completions", GROQ_API_KEY, g_model, images)
            except Exception as e:
                last_err = e
                print(f"[LLM Fallback] Groq ({g_model}) failed ({e}), trying next...")

    if OPENROUTER_API_KEY:
        try:
            return _call_openai_compat(prompt, "https://openrouter.ai/api/v1/chat/completions", OPENROUTER_API_KEY, "openrouter/free", images)
        except Exception as e:
            last_err = e
            print(f"[LLM Fallback] openrouter/free failed ({e}), trying Gemini...")

    if llm:
        gemini_contents = [prompt, *[types.Part.from_bytes(data=b, mime_type=mime) for b, mime in images]] if images else prompt
        for model, level in (
            ("gemini-3.5-flash-lite", types.ThinkingLevel.MINIMAL),
            ("gemini-3.8-flash", types.ThinkingLevel.LOW),
            ("gemini-4-hydrogen", types.ThinkingLevel.LOW),
        ):
            try:
                cfg = types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=TaskList,
                    temperature=0,
                    thinking_config=types.ThinkingConfig(thinking_level=level),
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                )
                res = llm.models.generate_content(model=model, contents=gemini_contents, config=cfg)
                data = json.loads(res.text or "")
                actions = [a for a in data.get("actions", []) if a.get("action") != "ignore"]
                conf = float(data.get("ocr_confidence", 1.0)) if images else 1.0
                return actions, conf
            except Exception as e:
                last_err = e
                print(f"[LLM Fallback] {model} failed ({e}), trying next...")

    if OPENROUTER_API_KEY:
        try:
            return _call_openai_compat(prompt, "https://openrouter.ai/api/v1/chat/completions", OPENROUTER_API_KEY, "openai/gpt-4o-mini", images)
        except Exception as e:
            last_err = e
            print(f"[LLM Fallback] openai/gpt-4o-mini failed ({e})")
    raise last_err or RuntimeError("No LLM API keys configured in .env")

# ================= NOTION DB =================


def _query_all(ds_id: str, filter: dict | None = None) -> list[dict]:
    """Query all pages from a Notion data source, following pagination cursors."""
    results, cursor = [], None
    while True:
        kwargs = {"data_source_id": ds_id}
        if filter is not None:
            kwargs["filter"] = filter
        if cursor:
            kwargs["start_cursor"] = cursor
        res = notion.data_sources.query(**kwargs)
        results.extend(res.get("results", []))
        if not res.get("has_more"):
            break
        cursor = res.get("next_cursor")
    return results


def _safe_title(page: dict) -> str:
    tl = page.get("properties", {}).get("Task Name", {}).get("title", [])
    return tl[0]["text"]["content"].strip() if tl else "Untitled"


def _mermaid_escape(text: str) -> str:
    return re.sub(r"\s+", " ", text).replace('"', "'").replace("<", "&lt;").replace(">", "&gt;").strip()


def find_task(name: str, ds_id: str | None = None, prefer_completed: bool = False):
    if not name:
        return None
    ds = ds_id or DATA_SOURCE_ID
    # 1. Exact match (preferring the expected completion state if duplicates exist)
    exact = _query_all(ds, filter={"property": "Task Name", "title": {"equals": name}})
    if exact:
        for p in exact:
            if p["properties"].get("Completed", {}).get("checkbox", False) == prefer_completed:
                return p
        return exact[0]

    # 2. NLP fuzzy match across preferred completion state first
    preferred = _query_all(ds, filter={"property": "Completed", "checkbox": {"equals": prefer_completed}})
    if match := match_task_candidate(name, preferred):
        return match

    # 3. Fallback: check all tasks
    return match_task_candidate(name, _query_all(ds))


def _task_snapshot(tasks: list[dict]) -> dict[str, tuple]:
    """Build {page_id: (completed, title, assignee, deadline, prereq_ids)} for manual-edit detection."""
    snap = {}
    for t in tasks:
        props = t["properties"]
        title = _safe_title(t)
        if title.lower() == "task flowchart":
            continue
        done = props.get("Completed", {}).get("checkbox", False)
        ar = props.get("Assignee", {}).get("rich_text", [])
        assignee = ar[0]["text"]["content"].strip() if ar else ""
        dl = (props.get("Deadline", {}).get("date") or {}).get("start", "")
        prereqs = tuple(sorted(r["id"] for r in props.get("Prerequisite", {}).get("relation", [])))
        snap[t["id"]] = (done, title, assignee, dl, prereqs)
    return snap


def sync_flowchart(cfg: dict | None = None, tasks: list[dict] | None = None, chat_id: str | None = None):
    """Renders all Notion tasks + prerequisites into a native Notion Mermaid block + Project Pulse callout."""
    global _flowchart_block_id, _pulse_block_id
    with _sync_lock:
        try:
            ds = (cfg or {}).get("data_source_id") or DATA_SOURCE_ID
            cid = chat_id or (None if cfg else bot_teams.CHAT_ID)
            block_id = (cfg or {}).get("flowchart_block_id") or (None if cfg else _flowchart_block_id)
            pulse_id = (cfg or {}).get("pulse_block_id") or (None if cfg else _pulse_block_id)
            raw_tasks = tasks if tasks is not None else _query_all(ds)

            real_tasks = []
            for t in raw_tasks:
                title = _safe_title(t)
                if title.lower() != "task flowchart":
                    real_tasks.append((t, title))

            valid_ids = {t["id"] for t, _ in real_tasks}
            done_ids = {t["id"] for t, _ in real_tasks if t["properties"].get("Completed", {}).get("checkbox", False)}
            if cid:
                snap = _task_snapshot(raw_tasks)
                notion_state_by_chat[cid] = snap
                open_tasks_by_chat[cid] = [(v[1], _first_name(v[2], "")) for v in snap.values() if not v[0]]

            today = date.today()
            chat_name = _mermaid_escape((cfg or {}).get("name") or "Project")
            header = f"{chat_name}'s Tasks" if not chat_name.lower().endswith("tasks") else chat_name
            lines = [
                "---",
                f'title: "{header}"',
                "---",
                "flowchart LR",
                "    classDef done fill:#dcfce7,stroke:#15803d,stroke-width:1.5px,color:#111827,rx:8,ry:8;",
                "    classDef overdue fill:#fee2e2,stroke:#dc2626,stroke-width:1.5px,color:#111827,rx:8,ry:8;",
                "    classDef ready fill:#eff6ff,stroke:#2563eb,stroke-width:1.5px,color:#111827,rx:8,ry:8;",
                "    classDef blocked fill:#f3f4f6,stroke:#9ca3af,stroke-width:1.5px,color:#4b5563,rx:8,ry:8;",
            ]
            if not real_tasks:
                lines.append('    empty["No tasks yet"]')

            counts = {"done": 0, "ready": 0, "blocked": 0, "overdue": 0}
            next_due = None  # (date, title, assignee)
            for t, title in real_tasks:
                nid = "n" + t["id"].replace("-", "")
                props = t["properties"]
                assignee_list = props.get("Assignee", {}).get("rich_text", [])
                raw_assignee = assignee_list[0]["text"]["content"].strip() if assignee_list else ""
                assignee = _mermaid_escape(_first_name(raw_assignee, "Unassigned"))

                deadline = (props.get("Deadline", {}).get("date") or {}).get("start", "")
                dl_date, dl_str = None, ""
                if deadline:
                    try:
                        dl_date = date.fromisoformat(deadline[:10])
                        dl_str = dl_date.strftime("%m/%d (%a)")
                    except ValueError:
                        dl_str = _mermaid_escape(deadline)

                prereq_ids = [
                    pre["id"] for pre in props.get("Prerequisite", {}).get("relation", [])
                    if pre["id"] in valid_ids and pre["id"] != t["id"]
                ]
                if t["id"] in done_ids:
                    cls = "done"
                elif dl_date and dl_date < today:
                    cls = "overdue"
                elif any(pid not in done_ids for pid in prereq_ids):
                    cls = "blocked"
                else:
                    cls = "ready"
                counts[cls] += 1
                if cls != "done" and dl_date and (next_due is None or dl_date < next_due[0]):
                    next_due = (dl_date, title, assignee)

                safe_title = _mermaid_escape(title)
                footer = f"{assignee} - {dl_str}" if dl_str else assignee
                footer = footer.replace(" ", "\u00a0").replace("-", "\u2011")
                card = f"<b>&nbsp;&nbsp;{safe_title}&nbsp;&nbsp;</b><br/><hr/>{footer}"
                lines.append(f'    {nid}["{card}"]:::{cls}')
                for pid_raw in prereq_ids:
                    pid = "n" + pid_raw.replace("-", "")
                    lines.append(f"    {pid} --> {nid}")

            total = len(real_tasks)
            pct = int(round(counts["done"] * 100 / total)) if total else 0
            pulse = (
                f"Progress: {counts['done']}/{total} done ({pct}%) · "
                f"Ready: {counts['ready']} · Blocked: {counts['blocked']} · Overdue: {counts['overdue']}"
            )
            if next_due:
                pulse += f" · Next due: {next_due[1]} ({next_due[2]}, {next_due[0].strftime('%m/%d')})"

            code_str = "\n".join(lines)
            body = {
                "language": "mermaid",
                "rich_text": [{"type": "text", "text": {"content": code_str[i:i + 2000]}} for i in range(0, len(code_str), 2000)],
            }

            page_id = (cfg or {}).get("page_id")
            if not block_id or not pulse_id:
                if not page_id and (cfg or {}).get("db_id"):
                    page_id = notion.databases.retrieve(cfg["db_id"]).get("parent", {}).get("page_id")
                    if page_id and cfg is not None:
                        cfg["page_id"] = page_id
                if not page_id:
                    page_id = os.getenv("notion_flowchart_id", "").strip()
                if not page_id:
                    res = notion.search(query="Task Flowchart").get("results", [])
                    pages = [r for r in res if r.get("object") == "page"]
                    page_id = pages[0]["id"] if pages else _DB_INFO.get("parent", {}).get("page_id")
                if page_id:
                    for b in notion.blocks.children.list(block_id=page_id).get("results", []):
                        if not block_id and b.get("type") == "code" and b.get("code", {}).get("language") == "mermaid":
                            block_id = b["id"]
                        elif not pulse_id and b.get("type") == "callout":
                            pulse_id = b["id"]
                    if not pulse_id:
                        created_p = notion.blocks.children.append(
                            block_id=page_id,
                            children=[{
                                "object": "block",
                                "type": "callout",
                                "callout": {"rich_text": [{"type": "text", "text": {"content": pulse}}], "color": "gray_background"},
                            }],
                        )
                        pulse_id = created_p["results"][0]["id"]
                    if cfg is not None:
                        cfg["pulse_block_id"] = pulse_id
                    else:
                        _pulse_block_id = pulse_id
                    if not block_id:
                        created = notion.blocks.children.append(
                            block_id=page_id,
                            children=[{"object": "block", "type": "code", "code": body}],
                        )
                        block_id = created["results"][0]["id"]
                        if cfg is not None:
                            cfg["flowchart_block_id"] = block_id
                        else:
                            _flowchart_block_id = block_id
                        print("[Done] Created Mermaid flowchart in Notion")
                        return
                    if cfg is not None:
                        cfg["flowchart_block_id"] = block_id
                    else:
                        _flowchart_block_id = block_id

            if not block_id:
                print("[Skip] Flowchart: create a Notion page 'Task Flowchart' (shared with integration) or set notion_flowchart_id in .env")
                return

            if pulse_id:
                notion.blocks.update(block_id=pulse_id, callout={"rich_text": [{"type": "text", "text": {"content": pulse}}]})
            notion.blocks.update(block_id=block_id, code=body)
            print("[Done] Synced Notion flowchart")
        except Exception as e:
            print(f"[Flowchart Error] {e}")


def _notify_dependents(page: dict, prev_sender: str, chat_id: str | None, cfg: dict | None, ds_id: str):
    """Notify assignees of open tasks whose prerequisite just completed."""
    prev_rt = page["properties"].get("Assignee", {}).get("rich_text", [])
    prev_raw = (prev_rt[0]["text"]["content"].strip() if prev_rt else "") or prev_sender
    prev_assignee = _first_name(prev_raw, "Someone")
    prev_title = _safe_title(page)

    deps = _query_all(
        ds_id,
        filter={
            "and": [
                {"property": "Prerequisite", "relation": {"contains": page["id"]}},
                {"property": "Completed", "checkbox": {"equals": False}},
            ]
        },
    )
    state = notion_state_by_chat.get(chat_id or bot_teams.CHAT_ID, {})
    for dep in deps:
        pre_rels = dep["properties"].get("Prerequisite", {}).get("relation", [])
        if len(pre_rels) > 1 and any(not state.get(r["id"], (True,))[0] for r in pre_rels if r["id"] != page["id"]):
            continue
        dep_rt = dep["properties"].get("Assignee", {}).get("rich_text", [])
        dep_raw = dep_rt[0]["text"]["content"].strip() if dep_rt else "Someone"
        dep_name = _safe_title(dep)
        mention, mentions = _mention(cfg, dep_raw)
        bot_teams.send(
            f'Hey {mention}! {prev_assignee} just finished "{prev_title}", so you\'re all clear to start on "{dep_name}"!',
            chat_id=chat_id,
            mentions=mentions,
        )
        print(f"[Notification] Reminded {dep_raw} for {dep_name}")


def handle_action(data: dict, sender: str = "", sync: bool = True, chat_id: str | None = None, cfg: dict | None = None) -> bool:
    action = data.get("action")
    cid = chat_id or bot_teams.CHAT_ID
    if action in ("query_all", "query_person", "query_link"):
        handle_query(action, data.get("assignee") or "", sender, cid, cfg or {})
        return False
    task_name = (data.get("task") or "").strip()
    if not task_name:
        return False
    db_id = (cfg or {}).get("db_id") or NOTION_DB_ID
    ds_id = (cfg or {}).get("data_source_id") or DATA_SOURCE_ID

    if action in ("create", "assign"):
        page = find_task(task_name, ds_id, prefer_completed=False)
        props = {}
        if data.get("assignee") and str(data["assignee"]).strip():
            props["Assignee"] = {"rich_text": [{"text": {"content": str(data["assignee"]).strip()}}]}
        if _valid_iso_date(data.get("deadline")):
            props["Deadline"] = {"date": {"start": data["deadline"]}}
        if data.get("prerequisite_task") and data["prerequisite_task"].strip().lower() != task_name.lower():
            prereq = find_task(data["prerequisite_task"].strip(), ds_id, prefer_completed=False)
            if prereq:
                if page and _creates_cycle(page["id"], prereq["id"], _query_all(ds_id)):
                    print(f"[Skip] Blocked circular dependency: {data['prerequisite_task']} -> {task_name}")
                else:
                    props["Prerequisite"] = {"relation": [{"id": prereq["id"]}]}

        if page:
            if not props:
                print(f"[Skip] Task already exists: {task_name}")
                return False
            notion.pages.update(page["id"], properties=props)
            print(f"[Done] Updated task {task_name}: {', '.join(props.keys())}")
        else:
            props["Task Name"] = {"title": [{"text": {"content": task_name}}]}
            notion.pages.create(parent={"database_id": db_id}, properties=props)
            assigned_str = f" assigned to {data['assignee']}" if data.get("assignee") else ""
            print(f"[Done] Created task: {task_name}{assigned_str}")
        _get_recent(cid).append((task_name, data.get("assignee"), sender))
        if sync:
            sync_flowchart(cfg, chat_id=chat_id)
        return True

    elif action == "complete":
        page = find_task(task_name, ds_id, prefer_completed=False)
        if page:
            notion.pages.update(page["id"], properties={"Completed": {"checkbox": True}})
            with _sync_lock:
                if cid in notion_state_by_chat and page["id"] in notion_state_by_chat[cid]:
                    notion_state_by_chat[cid][page["id"]] = (True, *notion_state_by_chat[cid][page["id"]][1:])
            print(f"[Done] Completed task: {task_name}")
            if sync:
                sync_flowchart(cfg, chat_id=chat_id)
            _notify_dependents(page, sender, chat_id, cfg, ds_id)
            return True
        print(f"[Skip] Task not found to complete: {task_name}")

    elif action == "uncomplete":
        page = find_task(task_name, ds_id, prefer_completed=True)
        if page:
            notion.pages.update(page["id"], properties={"Completed": {"checkbox": False}})
            print(f"[Done] Reopened task: {task_name}")
            if sync:
                sync_flowchart(cfg, chat_id=chat_id)
            return True
        print(f"[Skip] Task not found to reopen: {task_name}")

    elif action == "delete":
        page = find_task(task_name, ds_id, prefer_completed=False)
        if page:
            notion.pages.update(page["id"], archived=True)
            print(f"[Done] Deleted task: {task_name}")
            if sync:
                sync_flowchart(cfg, chat_id=chat_id)
            return True
        print(f"[Skip] Task not found to delete: {task_name}")
    return False


def _human_due(dl_date: date | None, today: date) -> tuple[str, str]:
    """Return (due_phrase like ', due tomorrow', short_label like 'tomorrow' or 'Nov 26')."""
    if not dl_date:
        return "", ""
    delta = (dl_date - today).days
    if delta == 0:
        return ", due today", "today"
    if delta == 1:
        return ", due tomorrow", "tomorrow"
    if delta == -1:
        return ", due yesterday", "yesterday"
    label = dl_date.strftime("%b %d").replace(" 0", " ")
    return f", due {label}", label


def handle_query(kind: str, target_name: str, sender: str, cid: str, cfg: dict):
    """Answer status/link queries conversationally from Notion (all tasks, self, another member, or tracker link)."""
    url = cfg.get("page_url", "")
    if not url and cfg.get("page_id"):
        try:
            url = notion.pages.retrieve(cfg["page_id"]).get("url", "")
            if url:
                cfg["page_url"] = url
        except Exception:
            pass
    if kind == "query_link":
        msg = f'Here\'s the <a href="{url}">Notion tracker</a> for this chat!' if url else "I don't have a Notion tracker link saved for this chat yet!"
        bot_teams.send(msg, chat_id=cid)
        return

    ds = cfg.get("data_source_id") or DATA_SOURCE_ID
    tasks = _query_all(ds)
    by_id = {t["id"]: t for t in tasks}
    done_ids = {t["id"] for t in tasks if t["properties"].get("Completed", {}).get("checkbox", False)}
    today = date.today()
    first = _first_name(sender, "there")
    sender_first = _first_name(sender, "").lower()
    target_first = _first_name(target_name, "").lower() or sender_first
    is_self = (kind == "query_person") and (not target_first or target_first == sender_first)
    who_label = first if is_self else (target_first.capitalize() if target_first else "Someone")

    items = []
    for t in tasks:
        if t["id"] in done_ids:
            continue
        props = t["properties"]
        title = _safe_title(t)
        if title.lower() == "task flowchart":
            continue
        ar = props.get("Assignee", {}).get("rich_text", [])
        raw_a = ar[0]["text"]["content"].strip() if ar else ""
        assignee = _first_name(raw_a, "")
        if kind == "query_person" and (not target_first or target_first not in raw_a.lower()):
            continue

        dl = (props.get("Deadline", {}).get("date") or {}).get("start", "")
        dl_date = None
        if dl:
            try:
                dl_date = date.fromisoformat(dl[:10])
            except ValueError:
                pass
        due_phrase, dl_label = _human_due(dl_date, today)
        if dl and not dl_date:
            due_phrase, dl_label = f", due {dl}", dl

        blockers = []
        for r in props.get("Prerequisite", {}).get("relation", []):
            pid = r["id"]
            if pid in done_ids or pid not in by_id or pid == t["id"]:
                continue
            p_props = by_id[pid]["properties"]
            p_title = _safe_title(by_id[pid])
            p_ar = p_props.get("Assignee", {}).get("rich_text", [])
            p_raw = p_ar[0]["text"]["content"].strip() if p_ar else ""
            p_who = _first_name(p_raw, "")
            blockers.append(f'{p_who} to finish "{p_title}"' if p_who else f'"{p_title}"')
        blocker_str = " and ".join(blockers)
        is_overdue = bool(dl_date and dl_date < today)
        items.append((title, assignee, due_phrase, dl_label, is_overdue, blocker_str))

    if not items:
        if kind == "query_person":
            msg = (
                f"You're all caught up, {first}, nothing on your plate right now!"
                if is_self
                else f"{who_label} is all caught up, no open tasks right now!"
            )
        else:
            msg = "Everyone's all caught up, no pending tasks left!"
        bot_teams.send(msg, chat_id=cid)
        return

    if len(items) == 1:
        title, assignee, due_phrase, dl_label, is_overdue, blocker_str = items[0]
        if kind == "query_person" and is_self:
            if blocker_str and is_overdue:
                msg = f'Hey {first}, heads up, "{title}" was due {dl_label}, though it\'s still waiting on {blocker_str}. You\'ll want to wrap that up soon!'
            elif blocker_str:
                msg = f'Hey {first}, you have "{title}"{due_phrase}, though it\'s still waiting on {blocker_str}.'
            elif is_overdue:
                msg = f'Hey {first}, heads up, "{title}" was due {dl_label}, so you\'ll want to wrap that up soon!'
            else:
                msg = f'Hey {first}, you just have "{title}" left{due_phrase}.'
        elif kind == "query_person":
            if blocker_str and is_overdue:
                msg = f'Heads up, {who_label} still has "{title}" (was due {dl_label}), though it\'s still waiting on {blocker_str}.'
            elif blocker_str:
                msg = f'{who_label} has "{title}"{due_phrase}, though it\'s still waiting on {blocker_str}.'
            elif is_overdue:
                msg = f'Heads up, {who_label} still has "{title}" (was due {dl_label}), so they\'ll want to wrap that up soon!'
            else:
                msg = f'{who_label} just has "{title}" left{due_phrase}.'
        else:
            who = f" ({assignee})" if assignee else " (unassigned)"
            if blocker_str:
                tail = f", still waiting on {blocker_str}."
            elif is_overdue:
                tail = f", was due {dl_label} (overdue)."
            else:
                tail = "."
            msg = f'We just have one task left: "{title}"{who}{"" if is_overdue else due_phrase}{tail}'
        bot_teams.send(msg, chat_id=cid)
        return

    lines = []
    for title, assignee, due_phrase, dl_label, is_overdue, blocker_str in items:
        who = "" if kind == "query_person" else (f" ({assignee})" if assignee else " (unassigned)")
        if is_overdue:
            wait = f", still waiting on {blocker_str}" if blocker_str else ""
            status = f", was due {dl_label} (overdue{wait})"
            due_part = ""
        elif blocker_str:
            status = f", still waiting on {blocker_str}"
            due_part = due_phrase
        else:
            status = ""
            due_part = due_phrase
        lines.append(f"• <b>{title}</b>{who}{due_part}{status}")

    if kind == "query_person":
        intro = (
            f"Hey {first}, you've got {len(items)} tasks left on your plate:"
            if is_self
            else f"Here's what {who_label} still has left:"
        )
    else:
        intro = f"Here's where we're at, {len(items)} tasks left:"
    bot_teams.send(f"{intro}<br/>" + "<br/>".join(lines), chat_id=cid)


def poll_notion_changes(cid: str, cfg: dict):
    """Detect manual edits/completions made directly in Notion and sync flowchart + notifications."""
    ds = cfg.get("data_source_id")
    if not ds:
        return
    try:
        tasks = _query_all(ds)
        with _sync_lock:
            prev = notion_state_by_chat.get(cid)
        if prev is None:
            sync_flowchart(cfg, tasks=tasks, chat_id=cid)
            return
        curr = _task_snapshot(tasks)
        newly_done = [
            t for t in tasks
            if t["id"] in prev and not prev[t["id"]][0] and curr.get(t["id"], (False,))[0]
        ]
        if curr != prev:
            print(f"[Notion Sync] Manual change detected in '{cfg.get('name', cid[:12])}'")
            sync_flowchart(cfg, tasks=tasks, chat_id=cid)
            for t in newly_done:
                _notify_dependents(t, "", cid, cfg, ds)
    except Exception as e:
        print(f"[Notion Poll Error] {e}")


def check_deadlines(chats: dict | None = None) -> bool:
    """Send @mention reminders for open tasks across all active chats (once per task per day)."""
    today = date.today()
    today_iso = today.isoformat()
    all_chats = chats if chats is not None else load_chats()
    changed = False
    all_ok = True
    for cid, cfg in list(all_chats.items()):
        if cfg.get("ignored") or not cfg.get("data_source_id"):
            continue
        try:
            reminded = cfg.setdefault("reminded", {})
            # Prune stale dates from earlier days
            stale = [k for k, v in reminded.items() if v != today_iso]
            for k in stale:
                reminded.pop(k, None)
                changed = True

            open_tasks = _query_all(
                cfg["data_source_id"],
                filter={"property": "Completed", "checkbox": {"equals": False}},
            )

            for t in open_tasks:
                if reminded.get(t["id"]) == today_iso:
                    continue
                props = t["properties"]
                deadline = (props.get("Deadline", {}).get("date") or {}).get("start", "")
                if not deadline:
                    continue
                try:
                    days_left = (date.fromisoformat(deadline[:10]) - today).days
                except ValueError:
                    continue

                title = _safe_title(t)
                if title.lower() == "task flowchart":
                    continue
                assignee_list = props.get("Assignee", {}).get("rich_text", [])
                assignee = (assignee_list[0]["text"]["content"].strip() if assignee_list else "") or "Someone"
                mention, mentions = _mention(cfg, assignee)

                if days_left == 7:
                    msg = f'Hey {mention}, quick heads up, "{title}" is due in a week.'
                elif days_left == 1:
                    msg = f'Hey {mention}, just a reminder that "{title}" is due tomorrow!'
                elif days_left == 0:
                    msg = f'Hey {mention}, friendly nudge, "{title}" is due today!'
                elif days_left < 0:
                    msg = f'Hey {mention}, heads up, "{title}" is past due!'
                else:
                    continue

                if bot_teams.send(msg, chat_id=cid, mentions=mentions):
                    reminded[t["id"]] = today_iso
                    changed = True
                    print(f"[Reminder] Sent for '{title}' ({days_left}d) to {assignee}")
                else:
                    all_ok = False
        except Exception as e:
            all_ok = False
            print(f"[Reminder Error] {cfg.get('name', cid[:12])}: {e}")
    if changed:
        save_chats(all_chats)
    return all_ok


def run_setup():
    """Interactive .env setup: parses 'Copy as cURL' from Teams DevTools + prompts for missing keys."""
    open(".env", "a").close()
    print("Paste Teams 'Copy as cURL' (messages and/or imgo), then press Enter on a blank line (or just Enter to skip):")
    lines = []
    while line := input():
        lines.append(line)
    blob = "\n".join(lines)

    updates = {}
    if m := re.search(r"conversations/([^/?'\"\s]+)", blob):
        updates["teams_chat_id"] = unquote(m.group(1))
    if m := re.search(r"authorization:\s*(skype_token\s+[^\r\n'\"]+)", blob, re.I):
        updates["teams_media_token"] = m.group(1).strip()
    elif m := re.search(r"(skypetoken_asm=[^\r\n;'\"\s]+)", blob, re.I):
        updates["teams_media_token"] = m.group(1).strip()
    elif m := re.search(r"authorization:\s*(?:bearer\s+)?([^\r\n'\"]+)", blob, re.I):
        updates["teams_token"] = m.group(1).strip()
    elif m := re.search(r"authentication:\s*(skypetoken=[^\r\n'\"]+)", blob, re.I):
        updates["teams_token"] = m.group(1).strip()

    for key in ("notion", "notion_root_page_id", "notion_db_id", "groq", "openrouter", "gemini"):
        if not os.getenv(key):
            if val := input(f"Enter {key} (Enter to skip): ").strip():
                updates[key] = val

    for k, v in updates.items():
        set_key(".env", k, v, quote_mode="never")
        print(f"[Saved] {k}")
    print("Setup complete." if updates else "Nothing changed.")


# ================= LOOP =================
if __name__ == "__main__":
    if "--setup" in sys.argv:
        run_setup()
        sys.exit(0)
    if "--remind" in sys.argv:
        check_deadlines()
        sys.exit(0)

    print("Bot running... Polling for messages.")
    chats = load_chats()
    seen_ids: dict[str, None] = {}  # insertion-ordered for true FIFO eviction
    pending_confirm: dict[str, dict] = {}  # chat_id -> pending low-confidence OCR actions

    def _mark_seen(mid: str):
        if mid:
            seen_ids[mid] = None
            while len(seen_ids) > 500:
                seen_ids.pop(next(iter(seen_ids)))

    # On startup, ignore pre-existing group chats not in chats.json so we don't spam old chats
    if NOTION_ROOT_PAGE_ID:
        try:
            changed = False
            for cid, topic in bot_teams.list_group_chats():
                if cid not in chats:
                    chats[cid] = {"ignored": True, "name": topic}
                    changed = True
            if changed:
                save_chats(chats)
        except Exception as e:
            print(f"[Discovery Warning] {e}")

    # Sync flowcharts & mark existing message history as seen for active chats
    for cid, cfg in chats.items():
        if cfg.get("ignored"):
            continue
        sync_flowchart(cfg, chat_id=cid)
        try:
            for msg_id, _, sender, sender_mri, _ in get_chat_text(10, chat_id=cid):
                _mark_seen(msg_id)
                _record_member(cfg, sender, sender_mri)
        except Exception as e:
            print(f"[Init Error] {cid}: {e}")
    if chats:
        save_chats(chats)

    poll_count = 0
    last_remind_date = None
    while True:
        try:
            # Check for newly joined or renamed group chats every ~10s
            if poll_count % 2 == 0:
                try:
                    for cid, topic in bot_teams.list_group_chats():
                        if cid not in chats and NOTION_ROOT_PAGE_ID:
                            chats[cid] = provision_chat(cid, topic)
                            for msg_id, _, sender, sender_mri, _ in get_chat_text(10, chat_id=cid):
                                _mark_seen(msg_id)
                                _record_member(chats[cid], sender, sender_mri)
                            save_chats(chats)
                        elif cid in chats and not chats[cid].get("ignored") and topic != chats[cid].get("name"):
                            chats[cid]["name"] = topic
                            pid = chats[cid].get("page_id")
                            if not pid and chats[cid].get("db_id"):
                                pid = notion.databases.retrieve(chats[cid]["db_id"]).get("parent", {}).get("page_id")
                                if pid:
                                    chats[cid]["page_id"] = pid
                            if pid:
                                notion.pages.update(pid, properties={"title": [{"text": {"content": topic}}]})
                                print(f"[Renamed] Notion page -> '{topic}'")
                            save_chats(chats)
                            threading.Thread(target=sync_flowchart, args=(chats[cid], None, cid), daemon=True).start()
                except Exception as e:
                    print(f"[Discovery Error] {e}")

            # Two-way Notion sync every ~10s
            if poll_count > 0 and poll_count % 2 == 0:
                for cid, cfg in list(chats.items()):
                    if not cfg.get("ignored"):
                        poll_notion_changes(cid, cfg)

            # Built-in daily 9:00 AM deadline reminders (deduped per task/day in chats.json)
            now = datetime.now()
            if now.hour >= 9 and last_remind_date != now.date():
                if check_deadlines(chats):
                    last_remind_date = now.date()

            poll_count += 1

            for cid, cfg in list(chats.items()):
                if cfg.get("ignored"):
                    continue
                try:
                    # Check if a pending low-confidence OCR confirmation received 👍 or 👎/❌
                    if cid in pending_confirm:
                        choice = bot_teams.check_confirmation_reaction(pending_confirm[cid]["client_msg_id"], chat_id=cid)
                        if choice == "yes":
                            pend = pending_confirm.pop(cid)
                            print(f"\n[Confirmed 👍] Committing {len(pend['actions'])} pending task(s) to Notion...")
                            if any([handle_action(a, pend["sender"], sync=False, chat_id=cid, cfg=cfg) for a in pend["actions"]]):
                                if _record_msg_tasks(cfg, [pend["orig_msg_id"]], pend["actions"]):
                                    save_chats(chats)
                                threading.Thread(target=sync_flowchart, args=(cfg, None, cid), daemon=True).start()
                                bot_teams.react(pend["orig_msg_id"], chat_id=cid)
                        elif choice == "no":
                            pending_confirm.pop(cid)
                            print("\n[Cancelled 👎] Discarded pending OCR draft")
                            bot_teams.send("No worries, I scrapped that! Just drop the task here whenever you're ready.", chat_id=cid)

                    for msg_id, text, sender, sender_mri, img_urls in get_chat_text(10, chat_id=cid, msg_tasks=cfg.get("msg_tasks")):      # oldest -> newest
                        if msg_id in seen_ids:
                            continue
                        if _record_member(cfg, sender, sender_mri):
                            save_chats(chats)

                        print(f"\n[{cfg.get('name', cid[:12])} | {sender or 'unknown'}] {text or '(image)'}" + (f" [{len(img_urls)} img]" if img_urls else ""))
                        if not img_urls and cid not in pending_confirm and not looks_like_task(text, [t for t, _ in open_tasks_by_chat.get(cid, [])]):
                            _mark_seen(msg_id)
                            print("[Skip] No task signal")
                            continue

                        images = [d for u in img_urls if (d := bot_teams.download_image(u))]
                        if img_urls and not images and not text:
                            _mark_seen(msg_id)
                            print("[Skip] Image download failed and no text")
                            continue
                        draft = pending_confirm[cid]["actions"] if (cid in pending_confirm and not images) else None
                        actions, confidence = analyze_message(text, sender, images, chat_id=cid, pending_draft=draft)
                        # Preserve unchanged fields (e.g. deadline) from pending OCR draft when user only patches one field
                        if draft and actions:
                            for i, a in enumerate(actions):
                                if a.get("action") not in ("create", "assign"):
                                    continue
                                base = next(
                                    (d for d in draft if (d.get("task") or "").lower() == (a.get("task") or "").lower()),
                                    draft[i] if len(actions) == len(draft) else None,
                                )
                                if base:
                                    for field in ("assignee", "deadline", "prerequisite_task"):
                                        if not a.get(field) and base.get(field):
                                            a[field] = base[field]
                        # Deterministic chain when message starts with 'Tapos...'/'Then...'/'After...'/'Sunod...'
                        if (
                            actions
                            and actions[0].get("action") in ("create", "assign")
                            and not actions[0].get("prerequisite_task")
                            and re.match(r"^\s*(tapos|then|after|afterwards|pagkatapos|sunod|kasunod|next)\b", text, re.I)
                            and not re.match(r"^\s*tapos\s+(na|ko\s+na)\b", text, re.I)
                        ):
                            recent = _get_recent(cid)
                            prev = next((t for t, _, s in reversed(recent) if s == sender), recent[-1][0] if recent else None)
                            if prev and prev.lower() != (actions[0].get("task") or "").lower():
                                actions[0]["prerequisite_task"] = prev
                        if len(re.findall(r"^\s*\d+[.)]\s+", text, re.M)) >= 2 and len(actions) >= 2:
                            for i in range(1, len(actions)):
                                if (
                                    actions[i].get("action") in ("create", "assign")
                                    and actions[i - 1].get("action") in ("create", "assign")
                                    and actions[i - 1].get("task")
                                    and not actions[i].get("prerequisite_task")
                                ):
                                    actions[i]["prerequisite_task"] = actions[i - 1].get("task")
                        _mark_seen(msg_id)
                        needs_confirm = bool(images and confidence < OCR_CONFIDENCE_THRESHOLD)
                        conf_tag = f" [OCR conf: {confidence:.2f}/{OCR_CONFIDENCE_THRESHOLD:.2f}]" if images else ""
                        print(f"[Parsed actions] {actions}{conf_tag}")

                        if needs_confirm:
                            ocr_tasks = [a for a in actions if (a.get("task") or "").strip()]
                            first_name = _first_name(sender, "there")
                            bot_teams.react(msg_id, key="2753_questionmark", chat_id=cid)
                            if ocr_tasks:
                                summary = "<br/>".join(
                                    f'• <b>{a["task"].strip()}</b>'
                                    + (f' for {_first_name(a.get("assignee"), "")}' if _first_name(a.get("assignee"), "") else "")
                                    + (f', due {a["deadline"]}' if a.get("deadline") else "")
                                    + (f' (after "{a["prerequisite_task"]}")' if a.get("prerequisite_task") else "")
                                    for a in ocr_tasks
                                )
                                ask = (
                                    f"Hey {first_name}, that photo was a little tricky to read, here's what I got:<br/>"
                                    f"{summary}<br/>"
                                    "React with 👍 if that looks good, 👎 to scrap it, or just reply with any fixes!"
                                )
                                if confirm_cid := bot_teams.send(ask, chat_id=cid):
                                    pending_confirm[cid] = {"client_msg_id": confirm_cid, "orig_msg_id": msg_id, "actions": ocr_tasks, "sender": sender}
                            else:
                                bot_teams.send(f"Hey {first_name}, I couldn't quite make out the writing in that photo, mind typing it out or sending a clearer shot?", chat_id=cid)
                            continue

                        resolved_pend = None
                        if any(a.get("action") not in ("query_all", "query_person", "query_link") for a in actions):
                            resolved_pend = pending_confirm.pop(cid, None)  # User replied with corrected tasks
                        if any([handle_action(a, sender, sync=False, chat_id=cid, cfg=cfg) for a in actions]):
                            linked_ids = [msg_id] + ([resolved_pend["orig_msg_id"]] if resolved_pend else [])
                            if _record_msg_tasks(cfg, linked_ids, actions):
                                save_chats(chats)
                            threading.Thread(target=sync_flowchart, args=(cfg, None, cid), daemon=True).start()
                            bot_teams.react(msg_id, chat_id=cid)
                except Exception as e:
                    print(f"[Chat Error] {cfg.get('name', cid[:12])}: {e}")
        except Exception as e:
            print(f"Error: {e}")

        time.sleep(5)
