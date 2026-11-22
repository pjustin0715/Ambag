import re
import sys
import time
import json
import base64
import threading
import requests
from datetime import date, timedelta
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
llm = genai.Client(api_key=GEMINI_API_KEY, http_options=types.HttpOptions(timeout=20000))
notion = Client(auth=NOTION_TOKEN)
_DB_INFO = notion.databases.retrieve(NOTION_DB_ID) if (NOTION_DB_ID and NOTION_DB_ID != "..." and "--setup" not in sys.argv) else {}
DATA_SOURCE_ID = _DB_INFO["data_sources"][0]["id"] if _DB_INFO else ""
_flowchart_block_id = None
recent_by_chat: dict[str, deque] = {}  # chat_id -> deque[(task_name, assignee, sender)]


def _get_recent(chat_id: str) -> deque:
    return recent_by_chat.setdefault(chat_id, deque(maxlen=20))


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
    with open(CHATS_FILE, "w") as f:
        json.dump(chats, f, indent=2)


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
    created = notion.blocks.children.append(
        block_id=page["id"],
        children=[{
            "object": "block",
            "type": "code",
            "code": {
                "language": "mermaid",
                "rich_text": [{"type": "text", "text": {"content": 'flowchart LR\n    empty["No tasks yet"]'}}],
            },
        }],
    )
    cfg = {
        "name": topic,
        "page_url": page["url"],
        "db_id": db["id"],
        "data_source_id": ds_id,
        "flowchart_block_id": created["results"][0]["id"],
    }
    bot_teams.send(f'Hi! Your task tracker & flowchart is ready: {page["url"]}', chat_id=chat_id)
    print(f"[Provisioned] '{topic}' -> {page['url']}")
    return cfg


class TaskAction(BaseModel):
    action: str  # create | assign | complete | ignore
    task: str | None
    assignee: str | None
    deadline: str | None
    prerequisite_task: str | None


class TaskList(BaseModel):
    actions: list[TaskAction]
    ocr_confidence: float = 1.0

# ================= LLM BRAIN =================


def _call_openai_compat(prompt: str, url: str, api_key: str, model: str, images: list[tuple[bytes, str]] | None = None) -> tuple[list[dict], float]:
    full_prompt = prompt + '\nReturn ONLY valid JSON: {"ocr_confidence": float, "actions": [{"action": "create|assign|complete|ignore", "task": str|null, "assignee": str|null, "deadline": str|null, "prerequisite_task": str|null}]}'
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
    raw = r.json()["choices"][0]["message"]["content"].strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    data = json.loads(raw)
    actions = [a for a in data.get("actions", []) if a.get("action") != "ignore"]
    conf = float(data.get("ocr_confidence", 1.0)) if images else 1.0
    return actions, conf


def analyze_message(text: str, sender: str = "", images: list[tuple[bytes, str]] | None = None, chat_id: str | None = None) -> tuple[list[dict], float]:
    """Returns (actions, ocr_confidence)."""
    today = date.today()
    cal = ", ".join((today + timedelta(days=i)).strftime("%a %Y-%m-%d") for i in range(14))
    recent = _get_recent(chat_id or bot_teams.CHAT_ID)
    recent_str = ", ".join(
        f"'{t}' (assignee: {a or 'none'}, by: {s or 'unknown'})" for t, a, s in list(recent)[-5:]
    )
    last_sender_task = next(
        (t for t, _, s in reversed(recent) if s == sender),
        recent[-1][0] if recent else None,
    )
    prompt = (
        f"Today is {today.isoformat()} ({today.strftime('%A')}). "
        f"Message sender: {sender or 'Unknown'}.\n"
        f"Upcoming 14 days: {cal}.\n"
        f"Recent tasks (oldest -> newest): {recent_str or 'none'}.\n"
        "Extract ALL task actions from the message and/or any attached whiteboard/notes image (split into separate items across lines OR conjunctions like 'and', 'then', 'also', ',', ';'):\n"
        "- ocr_confidence: float 0.0-1.0 for attached images (1.0 if no image). Be strict: 0.95-1.0 ONLY for crisp typed/printed text; 0.75-0.89 for clear handwriting; 0.50-0.74 for messy/blurry/skewed handwriting; <0.50 if mostly illegible. Lower the score if ANY word, name, or date had to be guessed.\n"
        "- action:\n"
        "  * 'assign': someone takes or is assigned a task (e.g. '[Name] will [task]', 'I will/can [task]', 'assign [task] to [Name]', '[task] - [Name] - [date]').\n"
        "  * 'complete': task is finished, done, or stated in past/present-perfect tense as already performed (e.g. 'I have gone to class', 'went to school', 'already reviewed the PR', 'pushed the changes').\n"
        "  * 'create': new task is created without assignment.\n"
        "  * 'ignore': no task action.\n"
        f"- assignee: Person assigned. If first-person ('I', 'me', 'myself'), use sender '{sender}'. Only set if explicitly stated, else null.\n"
        "- task: The task name.\n"
        "- deadline: ISO 8601 (YYYY-MM-DD) or null. Resolve relative phrases using the calendar above: "
        "'this <day>' or '<day>' = nearest upcoming <day>; "
        "'next <day>' = <day> in the following calendar week (or +7 days from this <day>); "
        "each extra 'next' (e.g. 'next next <day>') adds +7 days; "
        "'this week' = upcoming Friday (or Sunday if past Fri); 'next week' = Friday of next week; "
        "'in N days/weeks' = exact offset from today.\n"
        "- prerequisite_task: Clean underlying task name only (e.g. if 'after Steven is done eating', prerequisite_task is 'eat'; if 'after reviewing PR', prerequisite_task is 'review PR'). "
        f"Sequential/conditional words like 'then', 'once', 'before', 'after', 'afterwards', 'after that' link tasks: in 'A then B' or 'once A is done, B', B's prerequisite_task is A; if no task is named after 'after'/'then', use '{last_sender_task or 'null'}' (or the previous task in this message). "
        "If it says 'after [Person]' (e.g. 'after justin'), use [Person]'s most recent task from Recent tasks above or earlier in this message. "
        "If tasks are numbered in order (e.g. '1. A', '2. B', '3. C' in text or image), strip the numbers from task names and chain each item's prerequisite_task to the previous numbered task (unless another prerequisite is stated). "
        "Do NOT include person names or status phrases like 'is done'/'after', and NEVER set prerequisite_task to the task itself.\n"
        f"Example: 'I will go to school tomorrow and steven will bring the laptop after' -> 2 items: "
        f"(1) task='go to school', assignee='{sender or 'sender'}', deadline=tomorrow, prerequisite_task=null; "
        "(2) task='bring the laptop', assignee='steven', deadline=null, prerequisite_task='go to school'.\n"
        f"Message:\n{text or '(see attached image)'}"
    )
    last_err = None
    if GROQ_API_KEY:
        groq_models = ("qwen/qwen3.8-27b",) if images else ("openai/gpt-oss-20b", "qwen/qwen3.8-27b")
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
            data = json.loads(res.text)
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
    raise last_err

# ================= NOTION DB =================


def find_task(name: str, ds_id: str | None = None):
    if not name:
        return None
    ds = ds_id or DATA_SOURCE_ID
    # 1. Exact match
    res = notion.data_sources.query(data_source_id=ds, filter={
                                    "property": "Task Name", "title": {"equals": name}})
    if res.get("results"):
        return res["results"][0]

    # 2. NLP fuzzy match across active (uncompleted) tasks
    open_tasks = notion.data_sources.query(data_source_id=ds, filter={
                                           "property": "Completed", "checkbox": {"equals": False}})
    match = match_task_candidate(name, open_tasks.get("results", []))
    if match:
        return match

    # 3. Fallback: check all tasks
    all_tasks = notion.data_sources.query(data_source_id=ds)
    return match_task_candidate(name, all_tasks.get("results", []))


def sync_flowchart(cfg: dict | None = None):
    """Renders all Notion tasks + prerequisites into a native Notion Mermaid block."""
    global _flowchart_block_id
    try:
        ds = (cfg or {}).get("data_source_id") or DATA_SOURCE_ID
        block_id = (cfg or {}).get("flowchart_block_id") or (None if cfg else _flowchart_block_id)
        tasks = notion.data_sources.query(data_source_id=ds).get("results", [])
        lines = [
            "flowchart LR",
            "    classDef done fill:#dcfce7,stroke:#111827,stroke-width:1.5px,color:#111827,rx:8,ry:8;",
            "    classDef todo fill:#e5e7eb,stroke:#111827,stroke-width:1.5px,color:#111827,rx:8,ry:8;",
        ]
        if not tasks:
            lines.append('    empty["No tasks yet"]')
        for t in tasks:
            nid = "n" + t["id"].replace("-", "")
            props = t["properties"]
            title_list = props.get("Task Name", {}).get("title", [])
            title = title_list[0]["text"]["content"].replace('"', "'") if title_list else "Untitled"
            if title.lower() == "task flowchart":
                continue

            assignee_list = props.get("Assignee", {}).get("rich_text", [])
            raw_assignee = assignee_list[0]["text"]["content"].strip() if assignee_list else ""
            assignee = raw_assignee.split()[0].capitalize() if raw_assignee else "Unassigned"

            deadline = (props.get("Deadline", {}).get("date") or {}).get("start", "")
            dl_str = ""
            if deadline:
                try:
                    dl_str = date.fromisoformat(deadline[:10]).strftime("%m/%d (%a)")
                except ValueError:
                    dl_str = deadline

            cls = "done" if props.get("Completed", {}).get("checkbox", False) else "todo"
            footer = f"{assignee} - {dl_str}" if dl_str else assignee
            footer = footer.replace(" ", "\u00a0").replace("-", "\u2011")
            card = f"<b>&nbsp;&nbsp;{title}&nbsp;&nbsp;</b><br/><hr/>{footer}"
            lines.append(f'    {nid}["{card}"]:::{cls}')
            for pre in props.get("Prerequisite", {}).get("relation", []):
                pid = "n" + pre["id"].replace("-", "")
                lines.append(f"    {pid} --> {nid}")

        code_str = "\n".join(lines)
        body = {
            "language": "mermaid",
            "rich_text": [{"type": "text", "text": {"content": code_str[i:i + 2000]}} for i in range(0, len(code_str), 2000)],
        }

        if not block_id:
            page_id = os.getenv("notion_flowchart_id", "").strip()
            if not page_id:
                res = notion.search(query="Task Flowchart").get("results", [])
                pages = [r for r in res if r.get("object") == "page"]
                page_id = pages[0]["id"] if pages else _DB_INFO.get("parent", {}).get("page_id")
            if not page_id:
                print("[Skip] Flowchart: create a Notion page 'Task Flowchart' (shared with integration) or set notion_flowchart_id in .env")
                return

            for b in notion.blocks.children.list(block_id=page_id).get("results", []):
                if b.get("type") == "code" and b.get("code", {}).get("language") == "mermaid":
                    block_id = b["id"]
                    break
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

        notion.blocks.update(block_id=block_id, code=body)
        print("[Done] Synced Notion flowchart")
    except Exception as e:
        print(f"[Flowchart Error] {e}")


def handle_action(data: dict, sender: str = "", sync: bool = True, chat_id: str | None = None, cfg: dict | None = None) -> bool:
    task_name = data.get("task")
    action = data.get("action")
    db_id = (cfg or {}).get("db_id") or NOTION_DB_ID
    ds_id = (cfg or {}).get("data_source_id") or DATA_SOURCE_ID

    if action in ("create", "assign"):
        props = {}
        if data.get("assignee"):
            props["Assignee"] = {"rich_text": [{"text": {"content": data["assignee"]}}]}
        if data.get("deadline") and len(data["deadline"]) == 10 and data["deadline"][4] == "-" and data["deadline"][7] == "-":
            props["Deadline"] = {"date": {"start": data["deadline"]}}
        if data.get("prerequisite_task") and data["prerequisite_task"].lower() != (task_name or "").lower():
            prereq = find_task(data["prerequisite_task"], ds_id)
            if prereq:
                props["Prerequisite"] = {"relation": [{"id": prereq["id"]}]}

        page = find_task(task_name, ds_id)
        if page:
            if not props:
                print(f"[Skip] Task already exists: {task_name}")
                return False
            notion.pages.update(page["id"], properties=props)
            print(f"[Done] Updated task {task_name}: {', '.join(props.keys())}")
        else:
            props["Task Name"] = {"title": [{"text": {"content": task_name}}]}
            notion.pages.create(parent={"database_id": db_id}, properties=props)
            assigned_str = f" assigned to {data['assignee']}" if data.get('assignee') else ""
            print(f"[Done] Created task: {task_name}{assigned_str}")
        _get_recent(chat_id or bot_teams.CHAT_ID).append((task_name, data.get("assignee"), sender))
        if sync:
            sync_flowchart(cfg)
        return True

    elif action == "complete":
        page = find_task(task_name, ds_id)
        if page:
            notion.pages.update(page["id"], properties={
                                "Completed": {"checkbox": True}})
            print(f"[Done] Completed task: {task_name}")
            if sync:
                sync_flowchart(cfg)

            # Check prerequisites and notify
            prev_rt = page["properties"].get("Assignee", {}).get("rich_text", [])
            prev_raw = (prev_rt[0]["text"]["content"].strip() if prev_rt else "") or sender or "Someone"
            prev_assignee = prev_raw.split()[0].capitalize()
            prev_title = page["properties"]["Task Name"]["title"][0]["text"]["content"]

            deps = notion.data_sources.query(data_source_id=ds_id, filter={
                                             "property": "Prerequisite", "relation": {"contains": page["id"]}})
            for dep in deps["results"]:
                dep_rt = dep["properties"].get("Assignee", {}).get("rich_text", [])
                dep_raw = dep_rt[0]["text"]["content"].strip() if dep_rt else "Someone"
                assignee = dep_raw.split()[0].capitalize()
                dep_name = dep["properties"]["Task Name"]["title"][0]["text"]["content"]
                bot_teams.send(
                    f'@{assignee}, {prev_assignee} has completed the task "{prev_title}". You can start with your task "{dep_name}"',
                    chat_id=chat_id,
                )
                print(f"[Notification] Reminded {assignee} for {dep_name}")
            return True
        else:
            print(f"[Skip] Task not found to complete: {task_name}")
    return False


def check_deadlines():
    """Send @mention reminders for open tasks across all active chats."""
    today = date.today()
    for cid, cfg in load_chats().items():
        if cfg.get("ignored") or not cfg.get("data_source_id"):
            continue
        open_tasks = notion.data_sources.query(
            data_source_id=cfg["data_source_id"],
            filter={"property": "Completed", "checkbox": {"equals": False}},
        ).get("results", [])

        for t in open_tasks:
            props = t["properties"]
            deadline = (props.get("Deadline", {}).get("date") or {}).get("start", "")
            if not deadline:
                continue
            try:
                days_left = (date.fromisoformat(deadline[:10]) - today).days
            except ValueError:
                continue

            if days_left == 7:
                suffix = "is due in 7 days."
            elif days_left == 1:
                suffix = "is due tomorrow."
            elif days_left == 0:
                suffix = "is due today!"
            elif days_left < 0:
                suffix = "is past due!!!"
            else:
                continue

            title_list = props.get("Task Name", {}).get("title", [])
            title = title_list[0]["text"]["content"] if title_list else "Untitled"
            if title.lower() == "task flowchart":
                continue
            assignee_list = props.get("Assignee", {}).get("rich_text", [])
            assignee = assignee_list[0]["text"]["content"].strip() if assignee_list else "Someone"
            first_name = assignee.split()[0].capitalize()

            msg = f'@{first_name}, just a reminder, your task "{title}" {suffix}'
            if bot_teams.send(msg, chat_id=cid):
                print(f"[Reminder] Sent for '{title}' ({suffix}) to {first_name}")


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
    seen_ids = set()
    pending_confirm: dict[str, dict] = {}  # chat_id -> pending low-confidence OCR actions

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
        sync_flowchart(cfg)
        try:
            for msg_id, _, _, _ in get_chat_text(10, chat_id=cid):
                seen_ids.add(msg_id)
        except Exception as e:
            print(f"[Init Error] {cid}: {e}")

    poll_count = 0
    while True:
        try:
            # Check for newly joined group chats every ~30s
            if NOTION_ROOT_PAGE_ID and poll_count % 6 == 0:
                for cid, topic in bot_teams.list_group_chats():
                    if cid not in chats:
                        chats[cid] = provision_chat(cid, topic)
                        save_chats(chats)
                        for msg_id, _, _, _ in get_chat_text(10, chat_id=cid):
                            seen_ids.add(msg_id)
            poll_count += 1

            for cid, cfg in list(chats.items()):
                if cfg.get("ignored"):
                    continue

                # Check if a pending low-confidence OCR confirmation received 👍
                if cid in pending_confirm and bot_teams.has_thumbs_up(pending_confirm[cid]["client_msg_id"], chat_id=cid):
                    pend = pending_confirm.pop(cid)
                    print(f"\n[Confirmed 👍] Committing {len(pend['actions'])} pending task(s) to Notion...")
                    if any([handle_action(a, pend["sender"], sync=False, chat_id=cid, cfg=cfg) for a in pend["actions"]]):
                        threading.Thread(target=sync_flowchart, args=(cfg,), daemon=True).start()
                        bot_teams.react(pend["orig_msg_id"], chat_id=cid)

                for msg_id, text, sender, img_urls in get_chat_text(10, chat_id=cid):      # oldest -> newest
                    if msg_id in seen_ids:
                        continue

                    print(f"\n[{cfg.get('name', cid[:12])} | {sender or 'unknown'}] {text or '(image)'}" + (f" [{len(img_urls)} img]" if img_urls else ""))
                    if not img_urls and not looks_like_task(text):
                        seen_ids.add(msg_id)
                        print("[Skip] No task signal")
                        continue

                    images = [d for u in img_urls if (d := bot_teams.download_image(u))]
                    actions, confidence = analyze_message(text, sender, images, chat_id=cid)
                    if len(re.findall(r"^\s*\d+[.)]\s+", text, re.M)) >= 2 and len(actions) >= 2:
                        for i in range(1, len(actions)):
                            if not actions[i].get("prerequisite_task"):
                                actions[i]["prerequisite_task"] = actions[i - 1].get("task")
                    seen_ids.add(msg_id)
                    if len(seen_ids) > 500:
                        seen_ids.pop()
                    needs_confirm = bool(images and confidence < OCR_CONFIDENCE_THRESHOLD)
                    conf_tag = f" [OCR conf: {confidence:.2f}/{OCR_CONFIDENCE_THRESHOLD:.2f}]" if images else ""
                    print(f"[Parsed actions] {actions}{conf_tag}")

                    if needs_confirm and actions:
                        first_name = (sender or "there").split()[0].capitalize()
                        summary = "<br/>".join(
                            f"{a['task']}"
                            + (f" ({a['assignee'].split()[0].capitalize()})" if a.get("assignee") else "")
                            + (f" - due {a['deadline']}" if a.get("deadline") else "")
                            + (f" [after {a['prerequisite_task']}]" if a.get("prerequisite_task") else "")
                            for a in actions
                        )
                        ask = (
                            f"Hey {first_name}, that image was a bit hard to read! Here's what I caught:<br/>"
                            f"{summary}<br/>"
                            f"Drop a 👍 on this message if it looks right and I'll add it to Notion, otherwise just reply with the corrected version!"
                        )
                        if confirm_cid := bot_teams.send(ask, chat_id=cid):
                            pending_confirm[cid] = {"client_msg_id": confirm_cid, "orig_msg_id": msg_id, "actions": actions, "sender": sender}
                        continue
                    elif needs_confirm:
                        first_name = (sender or "there").split()[0].capitalize()
                        bot_teams.send(f"Hey {first_name}, I couldn't quite make out the writing in that image, could you type out the task(s) or send a clearer shot?", chat_id=cid)
                        continue

                    if actions:
                        pending_confirm.pop(cid, None)  # User replied with corrected tasks
                    if any([handle_action(a, sender, sync=False, chat_id=cid, cfg=cfg) for a in actions]):
                        threading.Thread(target=sync_flowchart, args=(cfg,), daemon=True).start()
                        bot_teams.react(msg_id, chat_id=cid)
        except Exception as e:
            print(f"Error: {e}")

        time.sleep(5)
