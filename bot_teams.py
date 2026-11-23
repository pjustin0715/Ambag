import json
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import os
from dotenv import load_dotenv
load_dotenv()

_http = requests.Session()
_http.mount("https://", HTTPAdapter(max_retries=Retry(total=3, connect=3, read=2, backoff_factor=0.3, allowed_methods=frozenset({"GET", "PUT", "POST"}))))
_last_client_id = 0
TOKEN = os.getenv('teams_token', '')
MEDIA_TOKEN = os.getenv('teams_media_token', '').strip()
CHAT_ID = os.getenv('teams_chat_id', '').strip()
BASE_URL = "https://teams.cloud.microsoft/api/chatsvc/apac/v1/users/ME/conversations"
URL = f"{BASE_URL}/{CHAT_ID}/messages"
if TOKEN.lower().startswith('skypetoken='):
    HEADERS = {"Authentication": TOKEN}
else:
    AUTHORIZATION = TOKEN if TOKEN.lower().startswith('bearer ') else f"Bearer {TOKEN}"
    HEADERS = {"Authorization": AUTHORIZATION}

def _msg_url(chat_id: str | None = None) -> str:
    return f"{BASE_URL}/{chat_id or CHAT_ID}/messages"

def list_group_chats(limit: int = 30) -> list[tuple[str, str]]:
    """Return [(chat_id, topic)] for group chats (@thread.v2) the bot account belongs to."""
    res = _http.get(f"{BASE_URL}?view=msnp24Equivalent&pageSize={limit}", headers=HEADERS, timeout=15)
    if not res.ok:
        hint = " (token expired? re-run with --setup)" if res.status_code == 401 else ""
        raise RuntimeError(f"Teams conversations request failed with HTTP {res.status_code}{hint}")
    chats = []
    for c in res.json().get("conversations", []):
        cid = c.get("id", "")
        if cid.endswith("@thread.v2"):
            topic = (c.get("threadProperties") or {}).get("topic") or "Group Project"
            chats.append((cid, topic))
    return chats

# ponytail: minimal payload; Teams only strictly requires content, type, and a unique client ID
def send(text: str, chat_id: str | None = None, mentions: list[tuple[str, str]] | None = None) -> str | None:
    global _last_client_id
    _last_client_id = max(int(time.time() * 1000), _last_client_id + 1)
    client_id = str(_last_client_id)
    payload = {
        "content": f"<p>{text}</p>",
        "messagetype": "RichText/Html",
        "contenttype": "Text",
        "clientmessageid": client_id,
    }
    if mentions:
        payload["properties"] = {
            "mentions": json.dumps([
                {
                    "@type": "http://schema.skype.com/Mention",
                    "itemid": str(i),
                    "mri": mri,
                    "mentionType": "person",
                    "displayName": name,
                }
                for i, (mri, name) in enumerate(mentions)
            ])
        }
    res = _http.post(_msg_url(chat_id), headers=HEADERS, json=payload, timeout=15)
    if res.status_code not in (200, 201):
        hint = " (token expired? re-run with --setup)" if res.status_code == 401 else ""
        print(f"[Send Error] HTTP {res.status_code}{hint}: {res.text[:120]}")
        return None
    return client_id

def check_confirmation_reaction(client_msg_id: str, chat_id: str | None = None) -> str | None:
    """Return 'yes' for 👍, 'no' for 👎/❌, else None on our message matching client_msg_id."""
    for m in read(15, chat_id=chat_id):
        if m.get("clientmessageid") == client_msg_id or m.get("id") == client_msg_id:
            emotions = (m.get("properties") or {}).get("emotions") or []
            for e in emotions:
                if not e.get("users"):
                    continue
                if e.get("key") in ("like", "👍", "1f44d_thumbsup"):
                    return "yes"
                if e.get("key") in ("dislike", "👎", "1f44e_thumbsdown", "❌", "274c_crossmark"):
                    return "no"
    return None

def has_thumbs_up(client_msg_id: str, chat_id: str | None = None) -> bool:
    return check_confirmation_reaction(client_msg_id, chat_id=chat_id) == "yes"

def react(msg_id: str, key: str = "2705_whiteheavycheckmark", chat_id: str | None = None) -> bool:
    """Add an emoji reaction to a Teams message via chatsvc."""
    url = f"{_msg_url(chat_id)}/{msg_id}/properties?name=emotions"
    payload = {"emotions": {"key": key, "value": int(time.time() * 1000)}}
    res = _http.put(url, headers=HEADERS, json=payload, timeout=15)
    if not res.ok:
        print(f"[React Error] HTTP {res.status_code}: {res.text[:120]}")
    return res.ok

def download_image(url: str) -> tuple[bytes, str] | None:
    """Download an image attachment from Teams ASM using teams_media_token."""
    if not MEDIA_TOKEN:
        print("[Image Error] Missing teams_media_token in .env (copy from imgo request in DevTools)")
        return None
    if MEDIA_TOKEN.lower().startswith("skypetoken_asm="):
        hdrs = {"Cookie": MEDIA_TOKEN}
    else:
        val = MEDIA_TOKEN if MEDIA_TOKEN.lower().startswith("skype_token ") else f"skype_token {MEDIA_TOKEN}"
        hdrs = {"Authorization": val}
    res = _http.get(url, headers=hdrs, timeout=15)
    if not res.ok:
        print(f"[Image Error] HTTP {res.status_code}")
        return None
    mime = res.headers.get("Content-Type", "image/jpeg").split(";")[0].strip()
    return (res.content, mime)

def read(limit: int = 1, chat_id: str | None = None) -> list[dict]:
    res = _http.get(f"{_msg_url(chat_id)}?pageSize={limit}", headers=HEADERS, timeout=15)
    if not res.ok:
        hint = " (token expired? re-run with --setup)" if res.status_code == 401 else ""
        raise RuntimeError(f"Teams messages request failed with HTTP {res.status_code}{hint}")
    return res.json().get("messages", [])

if __name__ == "__main__":
    for msg in read():
        # Clean print safe for Windows terminal
        print("Msg:", msg.encode("ascii", "replace").decode())
    print("Sent:", send("ping"))

