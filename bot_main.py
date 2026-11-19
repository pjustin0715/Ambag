import re
import sys
import time
import json
import base64
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
from dotenv import load_dotenv

load_dotenv()

# ================= CONFIG =================
GEMINI_API_KEY = os.getenv("gemini")
OPENROUTER_API_KEY = os.getenv("openrouter") or os.getenv("OPENROUTER_API_KEY")
NOTION_TOKEN = os.getenv("notion")
NOTION_DB_ID = os.getenv("notion_db_id", "...")  # Add this to your .env


llm = genai.Client(api_key=GEMINI_API_KEY, http_options=types.HttpOptions(timeout=20000))
notion = Client(auth=NOTION_TOKEN)
_DB_INFO = notion.databases.retrieve(NOTION_DB_ID)
DATA_SOURCE_ID = _DB_INFO["data_sources"][0]["id"]
_flowchart_block_id = None
recent_tasks = deque(maxlen=20)  # (task_name, assignee, sender)


class TaskAction(BaseModel):
    action: str  # create | assign | complete | ignore
    task: str | None
    assignee: str | None
    deadline: str | None
    prerequisite_task: str | None


class TaskList(BaseModel):
    actions: list[TaskAction]

# ================= LLM BRAIN =================


def _call_openrouter(prompt: str, model: str, images: list[tuple[bytes, str]] | None = None) -> list[dict]:
    full_prompt = prompt + '\nReturn ONLY valid JSON: {"actions": [{"action": "create|assign|complete|ignore", "task": str|null, "assignee": str|null, "deadline": str|null, "prerequisite_task": str|null}]}'
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
    }
    if model == "openai/gpt-4o-mini":
        body["response_format"] = {"type": "json_object"}
    r = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
        json=body,
        timeout=15 if images else 10,
    )
    r.raise_for_status()
    raw = r.json()["choices"][0]["message"]["content"].strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    data = json.loads(raw)
    return [a for a in data.get("actions", []) if a.get("action") != "ignore"]


def analyze_message(text: str, sender: str = "", images: list[tuple[bytes, str]] | None = None) -> list[dict]:
    """Returns list of JSON actions (empty if irrelevant)."""
    today = date.today()
    cal = ", ".join((today + timedelta(days=i)).strftime("%a %Y-%m-%d") for i in range(14))
    recent_str = ", ".join(
        f"'{t}' (assignee: {a or 'none'}, by: {s or 'unknown'})" for t, a, s in list(recent_tasks)[-5:]
    )
    last_sender_task = next(
        (t for t, _, s in reversed(recent_tasks) if s == sender),
        recent_tasks[-1][0] if recent_tasks else None,
    )
    prompt = (
        f"Today is {today.isoformat()} ({today.strftime('%A')}). "
        f"Message sender: {sender or 'Unknown'}.\n"
        f"Upcoming 14 days: {cal}.\n"
        f"Recent tasks (oldest -> newest): {recent_str or 'none'}.\n"
        "Extract ALL task actions from the message and/or any attached whiteboard/notes image (split into separate items across lines OR conjunctions like 'and', 'then', 'also', ',', ';'):\n"
        "- action:\n"
        "  * 'assign': someone takes or is assigned a task (e.g. '[Name] will [task]', 'I will/can [task]', 'assign [task] to [Name]', '[task] - [Name] - [date]').\n"
        "  * 'complete': task is marked finished or done.\n"
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
    if OPENROUTER_API_KEY:
        try:
            return _call_openrouter(prompt, "openrouter/free", images)
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
            return [a for a in data.get("actions", []) if a.get("action") != "ignore"]
        except Exception as e:
            last_err = e
            print(f"[LLM Fallback] {model} failed ({e}), trying next...")

    if OPENROUTER_API_KEY:
        try:
            return _call_openrouter(prompt, "openai/gpt-4o-mini", images)
        except Exception as e:
            last_err = e
            print(f"[LLM Fallback] openai/gpt-4o-mini failed ({e})")
    raise last_err

# ================= NOTION DB =================


def find_task(name: str):
    if not name:
        return None
    # 1. Exact match
    res = notion.data_sources.query(data_source_id=DATA_SOURCE_ID, filter={
                                    "property": "Task Name", "title": {"equals": name}})
    if res.get("results"):
        return res["results"][0]

    # 2. NLP fuzzy match across active (uncompleted) tasks
    open_tasks = notion.data_sources.query(data_source_id=DATA_SOURCE_ID, filter={
                                           "property": "Completed", "checkbox": {"equals": False}})
    match = match_task_candidate(name, open_tasks.get("results", []))
    if match:
        return match

    # 3. Fallback: check all tasks
    all_tasks = notion.data_sources.query(data_source_id=DATA_SOURCE_ID)
    return match_task_candidate(name, all_tasks.get("results", []))


def sync_flowchart():
    """Renders all Notion tasks + prerequisites into a native Notion Mermaid block."""
    global _flowchart_block_id
    try:
        tasks = notion.data_sources.query(data_source_id=DATA_SOURCE_ID).get("results", [])
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

        body = {"language": "mermaid", "rich_text": [{"type": "text", "text": {"content": "\n".join(lines)}}]}

        if not _flowchart_block_id:
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
                    _flowchart_block_id = b["id"]
                    break
            if not _flowchart_block_id:
                created = notion.blocks.children.append(
                    block_id=page_id,
                    children=[{"object": "block", "type": "code", "code": body}],
                )
                _flowchart_block_id = created["results"][0]["id"]
                print("[Done] Created Mermaid flowchart in Notion")
                return

        notion.blocks.update(block_id=_flowchart_block_id, code=body)
        print("[Done] Synced Notion flowchart")
    except Exception as e:
        print(f"[Flowchart Error] {e}")


def handle_action(data: dict, sender: str = "", sync: bool = True) -> bool:
    task_name = data.get("task")
    action = data.get("action")

    if action in ("create", "assign"):
        props = {}
        if data.get("assignee"):
            props["Assignee"] = {"rich_text": [{"text": {"content": data["assignee"]}}]}
        if data.get("deadline") and len(data["deadline"]) == 10 and data["deadline"][4] == "-" and data["deadline"][7] == "-":
            props["Deadline"] = {"date": {"start": data["deadline"]}}
        if data.get("prerequisite_task") and data["prerequisite_task"].lower() != (task_name or "").lower():
            prereq = find_task(data["prerequisite_task"])
            if prereq:
                props["Prerequisite"] = {"relation": [{"id": prereq["id"]}]}

        page = find_task(task_name)
        if page:
            if not props:
                print(f"[Skip] Task already exists: {task_name}")
                return False
            notion.pages.update(page["id"], properties=props)
            print(f"[Done] Updated task {task_name}: {', '.join(props.keys())}")
        else:
            props["Task Name"] = {"title": [{"text": {"content": task_name}}]}
            notion.pages.create(parent={"database_id": NOTION_DB_ID}, properties=props)
            assigned_str = f" assigned to {data['assignee']}" if data.get('assignee') else ""
            print(f"[Done] Created task: {task_name}{assigned_str}")
        recent_tasks.append((task_name, data.get("assignee"), sender))
        if sync:
            sync_flowchart()
        return True

    elif action == "complete":
        page = find_task(task_name)
        if page:
            notion.pages.update(page["id"], properties={
                                "Completed": {"checkbox": True}})
            print(f"[Done] Completed task: {task_name}")
            if sync:
                sync_flowchart()

            # Check prerequisites and notify
            prev_rt = page["properties"].get("Assignee", {}).get("rich_text", [])
            prev_raw = (prev_rt[0]["text"]["content"].strip() if prev_rt else "") or sender or "Someone"
            prev_assignee = prev_raw.split()[0].capitalize()
            prev_title = page["properties"]["Task Name"]["title"][0]["text"]["content"]

            deps = notion.data_sources.query(data_source_id=DATA_SOURCE_ID, filter={
                                             "property": "Prerequisite", "relation": {"contains": page["id"]}})
            for dep in deps["results"]:
                dep_rt = dep["properties"].get("Assignee", {}).get("rich_text", [])
                dep_raw = dep_rt[0]["text"]["content"].strip() if dep_rt else "Someone"
                assignee = dep_raw.split()[0].capitalize()
                dep_name = dep["properties"]["Task Name"]["title"][0]["text"]["content"]
                bot_teams.send(
                    f'@{assignee}, {prev_assignee} has completed the task "{prev_title}". You can start with your task "{dep_name}"'
                )
                print(f"[Notification] Reminded {assignee} for {dep_name}")
            return True
        else:
            print(f"[Skip] Task not found to complete: {task_name}")
    return False


def check_deadlines():
    """Send @mention reminders for open tasks due in 7 days, 1 day, today, or past due."""
    today = date.today()
    open_tasks = notion.data_sources.query(
        data_source_id=DATA_SOURCE_ID,
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
        if bot_teams.send(msg):
            print(f"[Reminder] Sent for '{title}' ({suffix}) to {first_name}")


# ================= LOOP =================
if __name__ == "__main__":
    if "--remind" in sys.argv:
        check_deadlines()
        sys.exit(0)

    print("Bot running... Polling for messages.")
    sync_flowchart()
    seen_ids = set()

    # Mark existing history as seen so a restart doesn't re-process old messages
    for msg_id, _, _, _ in get_chat_text(10):
        seen_ids.add(msg_id)

    while True:
        try:
            for msg_id, text, sender, img_urls in get_chat_text(10):      # oldest -> newest
                if msg_id in seen_ids:
                    continue

                print(f"\n[Incoming message from {sender or 'unknown'}] {text or '(image)'}" + (f" [{len(img_urls)} img]" if img_urls else ""))
                if not img_urls and not looks_like_task(text):
                    seen_ids.add(msg_id)
                    print("[Skip] No task signal")
                    continue

                images = [d for u in img_urls if (d := bot_teams.download_image(u))]
                actions = analyze_message(text, sender, images)
                if len(re.findall(r"^\s*\d+[.)]\s+", text, re.M)) >= 2 and len(actions) >= 2:
                    for i in range(1, len(actions)):
                        if not actions[i].get("prerequisite_task"):
                            actions[i]["prerequisite_task"] = actions[i - 1].get("task")
                seen_ids.add(msg_id)
                if len(seen_ids) > 100:
                    seen_ids.pop()
                print(f"[Parsed actions] {actions}")
                if any([handle_action(a, sender, sync=False) for a in actions]):
                    sync_flowchart()
                    bot_teams.react(msg_id)
        except Exception as e:
            print(f"Error: {e}")

        time.sleep(5)
