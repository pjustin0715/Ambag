import time
import json
from datetime import date
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
DATA_SOURCE_ID = notion.databases.retrieve(
    NOTION_DB_ID)["data_sources"][0]["id"]


class TaskAction(BaseModel):
    action: str  # create | assign | complete | ignore
    task: str | None
    assignee: str | None
    deadline: str | None
    prerequisite_task: str | None

# ================= LLM BRAIN =================


def analyze_message(text: str, sender: str = "") -> dict | None:
    """Returns JSON action or None if irrelevant."""
    prompt = (
        f"Today is {date.today().isoformat()} ({date.today().strftime('%A')}). "
        f"Message sender: {sender or 'Unknown'}.\n"
        "Extract structured task action from the message:\n"
        "- action:\n"
        "  * 'assign': someone takes or is assigned a task (e.g. '[Name] will [task]', 'I will/can [task]', 'assign [task] to [Name]').\n"
        "  * 'complete': task is marked finished or done.\n"
        "  * 'create': new task is created without assignment.\n"
        "  * 'ignore': no task action.\n"
        f"- assignee: Person assigned. If first-person ('I', 'me', 'myself'), use sender '{sender}'. Only set if explicitly stated, else null.\n"
        "- task: The task name.\n"
        "- deadline: ISO 8601 (YYYY-MM-DD) or null.\n"
        "- prerequisite_task: Clean underlying task name only (e.g. if 'after Steven is done eating', prerequisite_task is 'eat'; if 'after reviewing PR', prerequisite_task is 'review PR'). Do NOT include person names or status phrases like 'is done'/'after'.\n"
        f"Message: {text}"
    )
    res = llm.models.generate_content(
        model='gemini-3.5-flash-lite',
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=TaskAction,
            temperature=0,
        ),
    )
    data = json.loads(res.text)
    return data if data.get("action") != "ignore" else None

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


def handle_action(data: dict):
    task_name = data.get("task")
    action = data.get("action")

    if action in ("create", "assign"):
        page = find_task(task_name)
        if page:
            if data.get("assignee"):
                notion.pages.update(page["id"], properties={"Assignee": {
                                    "rich_text": [{"text": {"content": data["assignee"]}}]}})
                print(f"[Done] Assigned existing task {task_name} to {data['assignee']}")
            else:
                print(f"[Skip] Task already exists: {task_name}")
            return

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

    elif action == "complete":
        page = find_task(task_name)
        if page:
            notion.pages.update(page["id"], properties={
                                "Completed": {"checkbox": True}})
            print(f"[Done] Completed task: {task_name}")

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
        else:
            print(f"[Skip] Task not found to complete: {task_name}")


# ================= LOOP =================
if __name__ == "__main__":
    print("Bot running... Polling for messages.")
    seen_ids = set()

    # Mark existing history as seen so a restart doesn't re-process old messages
    for msg_id, _, _ in get_chat_text(10):
        seen_ids.add(msg_id)

    while True:
        try:
            for msg_id, text, sender in get_chat_text(10):      # oldest -> newest
                if msg_id in seen_ids:
                    continue
                seen_ids.add(msg_id)
                if len(seen_ids) > 100:
                    seen_ids.pop()

                print(f"\n[Incoming message from {sender or 'unknown'}] {text}")
                if not looks_like_task(text):
                    print("[Skip] No task signal")
                    continue

                action_data = analyze_message(text, sender)
                print(f"[Parsed action] {action_data}")
                if action_data:
                    handle_action(action_data)
        except Exception as e:
            print(f"Error: {e}")

        time.sleep(5)
