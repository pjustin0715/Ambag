import time
import json
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
NOTION_TOKEN = os.getenv("notion")
NOTION_DB_ID = os.getenv("notion_db_id", "...")  # Add this to your .env


llm = genai.Client(api_key=GEMINI_API_KEY)
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

# ================= LLM BRAIN =================


def analyze_message(text: str, sender: str = "") -> dict | None:
    """Returns JSON action or None if irrelevant."""
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
        "Extract structured task action from the message:\n"
        "- action:\n"
        "  * 'assign': someone takes or is assigned a task (e.g. '[Name] will [task]', 'I will/can [task]', 'assign [task] to [Name]').\n"
        "  * 'complete': task is marked finished or done.\n"
        "  * 'create': new task is created without assignment.\n"
        "  * 'ignore': no task action.\n"
        f"- assignee: Person assigned. If first-person ('I', 'me', 'myself'), use sender '{sender}'. Only set if explicitly stated, else null.\n"
        "- task: The task name.\n"
        "- deadline: ISO 8601 (YYYY-MM-DD) or null. Resolve relative phrases using the calendar above: "
        "'this <day>' or '<day>' = nearest upcoming <day>; "
        "'next <day>' = <day> in the following calendar week (or +7 days from this <day>); "
        "each extra 'next' (e.g. 'next next <day>') adds +7 days; "
        "'this week' = upcoming Friday (or Sunday if past Fri); 'next week' = next Monday; "
        "'in N days/weeks' = exact offset from today.\n"
        "- prerequisite_task: Clean underlying task name only (e.g. if 'after Steven is done eating', prerequisite_task is 'eat'; if 'after reviewing PR', prerequisite_task is 'review PR'). "
        f"If the message says 'after'/'afterwards'/'after that' with NO specific task named, use '{last_sender_task or 'null'}'. "
        "If it says 'after [Person]' (e.g. 'after justin'), use [Person]'s most recent task from Recent tasks above. "
        "Do NOT include person names or status phrases like 'is done'/'after'.\n"
        f"Message: {text}"
    )
    cfg = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=TaskAction,
        temperature=0,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    last_err = None
    for model in ("gemini-3.5-flash-lite", "gemini-3.8-flash", "gemini-4-hydrogen"):
        try:
            res = llm.models.generate_content(model=model, contents=prompt, config=cfg)
            data = json.loads(res.text)
            return data if data.get("action") != "ignore" else None
        except Exception as e:
            last_err = e
            print(f"[LLM Fallback] {model} failed ({e}), trying next...")
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


def handle_action(data: dict, sender: str = "") -> bool:
    task_name = data.get("task")
    action = data.get("action")

    if action in ("create", "assign"):
        page = find_task(task_name)
        if page:
            if not data.get("assignee"):
                print(f"[Skip] Task already exists: {task_name}")
                return False
            notion.pages.update(page["id"], properties={"Assignee": {
                                "rich_text": [{"text": {"content": data["assignee"]}}]}})
            print(f"[Done] Assigned existing task {task_name} to {data['assignee']}")
        else:
            # Task does not exist -> create it with assignee if recognized
            props = {"Task Name": {"title": [{"text": {"content": task_name}}]}}
            if data.get("assignee"):
                props["Assignee"] = {"rich_text": [{"text": {"content": data["assignee"]}}]}
            if data.get("deadline") and len(data["deadline"]) == 10 and data["deadline"][4] == "-" and data["deadline"][7] == "-":
                props["Deadline"] = {"date": {"start": data["deadline"]}}
            if data.get("prerequisite_task"):
                prereq = find_task(data["prerequisite_task"])
                if prereq:
                    props["Prerequisite"] = {"relation": [{"id": prereq["id"]}]}
            notion.pages.create(
                parent={"database_id": NOTION_DB_ID}, properties=props)
            assigned_str = f" assigned to {data['assignee']}" if data.get('assignee') else ""
            print(f"[Done] Created task: {task_name}{assigned_str}")
        recent_tasks.append((task_name, data.get("assignee"), sender))
        sync_flowchart()
        return True

    elif action == "complete":
        page = find_task(task_name)
        if page:
            notion.pages.update(page["id"], properties={
                                "Completed": {"checkbox": True}})
            print(f"[Done] Completed task: {task_name}")
            sync_flowchart()

            # Check prerequisites and notify
            deps = notion.data_sources.query(data_source_id=DATA_SOURCE_ID, filter={
                                             "property": "Prerequisite", "relation": {"contains": page["id"]}})
            for dep in deps["results"]:
                assignee = dep["properties"].get("Assignee", {}).get("rich_text", [{}])[
                    0].get("text", {}).get("content", "Someone")
                dep_name = dep["properties"]["Task Name"]["title"][0]["text"]["content"]
                bot_teams.send(
                    f"🔔 @{assignee}, {task_name} is done. You can start {dep_name}!")
                print(f"[Notification] Reminded {assignee} for {dep_name}")
            return True
        else:
            print(f"[Skip] Task not found to complete: {task_name}")
    return False


# ================= LOOP =================
if __name__ == "__main__":
    print("Bot running... Polling for messages.")
    sync_flowchart()
    seen_ids = set()

    # Mark existing history as seen so a restart doesn't re-process old messages
    for msg_id, _, _ in get_chat_text(10):
        seen_ids.add(msg_id)

    while True:
        try:
            for msg_id, text, sender in get_chat_text(10):      # oldest -> newest
                if msg_id in seen_ids:
                    continue

                print(f"\n[Incoming message from {sender or 'unknown'}] {text}")
                if not looks_like_task(text):
                    seen_ids.add(msg_id)
                    print("[Skip] No task signal")
                    continue

                action_data = analyze_message(text, sender)
                seen_ids.add(msg_id)
                if len(seen_ids) > 100:
                    seen_ids.pop()
                print(f"[Parsed action] {action_data}")
                if action_data and handle_action(action_data, sender):
                    bot_teams.react(msg_id)
        except Exception as e:
            print(f"Error: {e}")

        time.sleep(5)
