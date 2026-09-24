# Ambag 🤖

An AI-powered project management bot for Microsoft Teams that turns group chat conversations into structured tasks in Notion and automatically notifies members when prerequisite tasks are completed.

Built with **Gemini**, **Notion API**, and a lightweight **MS Teams Web API connector** that requires zero Azure bot registrations or tenant admin privileges.

---

## Features

- **Natural Language Parsing**: Understands conversational project updates (powered by Gemini Structured Outputs).
- **Notion Sync**: Automatically creates tasks, assigns members, sets deadlines, and marks tasks complete.
- **Dependency Chaining**: Link tasks with prerequisites (`do Task B after Task A`). When Task A is completed, the bot automatically pings Task B's assignee in Teams.
- **Zero Admin Privileges Required**: Works in restricted enterprise / university tenants (`.edu`) by using web session tokens.
- **Silent & Clean**: Does not spam confirmation messages; only sends actionable notifications when dependent tasks are unlocked.

---

## Notion Database Setup

Create a database in Notion with the following properties:

| Property Name | Type | Notes |
|---|---|---|
| `Task Name` | **Title** | The name of the task |
| `Assignee` | **Text** | Member name / handle |
| `Deadline` | **Date** | ISO date (`YYYY-MM-DD`) |
| `Completed` | **Checkbox** | Checked when task is finished |
| `Prerequisite` | **Relation** | Related to the **same database** (self-relation) |

Connect your Notion integration to this database (`...` top-right $\rightarrow$ **Connections** $\rightarrow$ select your integration).

---

## Installation & Setup

### 1. Clone & Install Dependencies

```bash
git clone <your-repo-url>
cd ambag

# Create and activate virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate
# Mac / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Environment Variables

Create a `.env` file in the root directory:

```env
gemini=YOUR_GEMINI_API_KEY
notion=YOUR_NOTION_INTEGRATION_TOKEN
notion_db_id=YOUR_NOTION_DATABASE_ID
teams_chat_id=YOUR_TEAMS_CHAT_ID
teams_token=YOUR_TEAMS_BEARER_TOKEN
```

#### How to get `teams_chat_id` and `teams_token`:
1. Open [Teams on Web](https://teams.cloud.microsoft) in Chrome or Edge.
2. Press `F12` to open Developer Tools and switch to the **Network** tab.
3. In the filter box, type `messages`.
4. Send a test message in your target group chat.
5. Click the `messages` request in DevTools:
   - **`teams_chat_id`**: Look at the Request URL:  
     `https://teams.cloud.microsoft/api/chatsvc/.../conversations/<CHAT_ID>/messages`  
     Copy the ID between `conversations/` and `/messages` (e.g. `19:...@unq.gbl.spaces`).
   - **`teams_token`**: In **Request Headers**, find `Authorization: Bearer <TOKEN>` and copy everything after `Bearer `.

---

## Running the Bot

```bash
python bot_main.py
```

The console will display real-time logs for:
- Incoming messages detected in Teams
- Extracted JSON actions from Gemini
- Notion database operations (`[Done] Created task`, `[Done] Assigned`, `[Done] Completed`)
- Sent prerequisite reminder notifications

---

## Example Chat Commands

| Intent | Example Teams Message | Result |
|---|---|---|
| **Create Task** | `do sprint 1 on 2026-09-25` | Creates "sprint 1" with deadline `2026-09-25` |
| **Assign Task** | `steven will do sprint 1` | Sets Assignee of "sprint 1" to `steven` |
| **Chained Task** | `do sprint 4 after sprint 3 on 2026-10-26` | Creates "sprint 4" with "sprint 3" as prerequisite |
| **Complete Task** | `sprint 3 done` | Marks "sprint 3" complete and alerts: `🔔 @steven, sprint 3 is done. You can start sprint 4!` |
