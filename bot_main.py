import time
import json
from pydantic import BaseModel
from google import genai
from google.genai import types
from notion_client import Client
import bot_teams

import os
from dotenv import load_dotenv

load_dotenv()

# ================= CONFIG =================
GEMINI_API_KEY = os.getenv("gemini")
NOTION_TOKEN = os.getenv("notion")
NOTION_DB_ID = os.getenv("notion_db_id", "...") # Add this to your .env


llm = genai.Client(api_key=GEMINI_API_KEY)
notion = Client(auth=NOTION_TOKEN)
DATA_SOURCE_ID = notion.databases.retrieve(NOTION_DB_ID)["data_sources"][0]["id"]

class TaskAction(BaseModel):
    action: str  # create | assign | complete | ignore
    task: str | None
    assignee: str | None
    deadline: str | None
    prerequisite_task: str | None

# ================= LLM BRAIN =================
def analyze_message(text: str) -> dict | None:
    """Returns JSON action or None if irrelevant."""
    res = llm.models.generate_content(
        model='gemini-3.5-flash-lite',
        contents=f"Analyze the message. If it creates, assigns, or completes a task, return structured data. Deadlines must be ISO 8601 (YYYY-MM-DD) or null.\nMessage: {text}",
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
    res = notion.data_sources.query(data_source_id=DATA_SOURCE_ID, filter={"property": "Task Name", "title": {"equals": name}})
    return res["results"][0] if res["results"] else None

def handle_action(data: dict):
    task_name = data.get("task")
    action = data.get("action")
    
    if action == "create":
        if find_task(task_name):
            print(f"[Skip] Task already exists: {task_name}")
            return
        props = {"Task Name": {"title": [{"text": {"content": task_name}}]}}
        if data.get("deadline") and len(data["deadline"]) == 10 and data["deadline"][4] == "-" and data["deadline"][7] == "-":
            props["Deadline"] = {"date": {"start": data["deadline"]}}
        if data.get("prerequisite_task"):
            prereq = find_task(data["prerequisite_task"])
            if prereq: props["Prerequisite"] = {"relation": [{"id": prereq["id"]}]}
        notion.pages.create(parent={"database_id": NOTION_DB_ID}, properties=props)
        print(f"[Done] Created task: {task_name}")

    elif action == "assign":
        page = find_task(task_name)
        if page:
            notion.pages.update(page["id"], properties={"Assignee": {"rich_text": [{"text": {"content": data["assignee"]}}]}})
            print(f"[Done] Assigned {task_name} to {data['assignee']}")
        else:
            print(f"[Skip] Task not found to assign: {task_name}")

    elif action == "complete":
        page = find_task(task_name)
        if page:
            notion.pages.update(page["id"], properties={"Completed": {"checkbox": True}})
            print(f"[Done] Completed task: {task_name}")
            
            # Check prerequisites and notify
            deps = notion.data_sources.query(data_source_id=DATA_SOURCE_ID, filter={"property": "Prerequisite", "relation": {"contains": page["id"]}})
            for dep in deps["results"]:
                assignee = dep["properties"].get("Assignee", {}).get("rich_text", [{}])[0].get("text", {}).get("content", "Someone")
                dep_name = dep["properties"]["Task Name"]["title"][0]["text"]["content"]
                bot_teams.send(f"🔔 @{assignee}, {task_name} is done. You can start {dep_name}!")
                print(f"[Notification] Reminded {assignee} for {dep_name}")
        else:
            print(f"[Skip] Task not found to complete: {task_name}")

# ================= LOOP =================
if __name__ == "__main__":
    print("Bot running... Polling for messages.")
    last_seen = set()
    
    while True:
        try:
            for msg_html in bot_teams.read(limit=1):
                # Extremely lazy HTML strip
                text = msg_html.replace("<p>", "").replace("</p>", "").strip()
                if text and text not in last_seen and not text.startswith("🔔"):
                    last_seen.add(text)
                    if len(last_seen) > 50: last_seen.pop() # Prevent leak
                    
                    print(f"\n[Incoming message] {text}")
                    action_data = analyze_message(text)
                    print(f"[Parsed action] {action_data}")
                    
                    if action_data:
                        handle_action(action_data)
        except Exception as e:
            print(f"Error: {e}")
        
        time.sleep(5)
