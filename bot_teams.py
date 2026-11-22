import time
import requests
import os
from dotenv import load_dotenv
load_dotenv()

_http = requests.Session()
TOKEN = os.getenv('teams_token', '')
MEDIA_TOKEN = os.getenv('teams_media_token', '').strip()
CHAT_ID = os.getenv('teams_chat_id', '19:253e041a-5f7f-4f21-88b1-2b349a351a53_4c107e33-2a08-4a96-8c8d-f081a3c2bab8@unq.gbl.spaces')
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
        raise RuntimeError(f"Teams conversations request failed with HTTP {res.status_code}")
    chats = []
    for c in res.json().get("conversations", []):
        cid = c.get("id", "")
        if cid.endswith("@thread.v2"):
            topic = (c.get("threadProperties") or {}).get("topic") or "Group Project"
            chats.append((cid, topic))
    return chats

# ponytail: minimal payload; Teams only strictly requires content, type, and a unique client ID
def send(text: str, chat_id: str | None = None) -> str | None:
    client_id = str(int(time.time() * 1000))
    payload = {
        "content": f"<p>{text}</p>",
        "messagetype": "RichText/Html",
        "contenttype": "Text",
        "clientmessageid": client_id,
    }
    ok = _http.post(_msg_url(chat_id), headers=HEADERS, json=payload, timeout=15).status_code in (200, 201)
    return client_id if ok else None

def has_thumbs_up(client_msg_id: str, chat_id: str | None = None) -> bool:
    """Return True if anyone reacted 👍 ('like') to our message matching client_msg_id."""
    for m in read(15, chat_id=chat_id):
        if m.get("clientmessageid") == client_msg_id or m.get("id") == client_msg_id:
            emotions = (m.get("properties") or {}).get("emotions") or []
            return any(e.get("key") in ("like", "👍", "1f44d_thumbsup") and e.get("users") for e in emotions)
    return False

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
        raise RuntimeError(f"Teams messages request failed with HTTP {res.status_code}")
    return res.json().get("messages", [])

if __name__ == "__main__":
    for msg in read():
        # Clean print safe for Windows terminal
        print("Msg:", msg.encode("ascii", "replace").decode())
    print("Sent:", send("ping"))

