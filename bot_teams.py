import time
import requests
import os
from dotenv import load_dotenv
load_dotenv()

_http = requests.Session()
TOKEN = os.getenv('teams_token', '')
MEDIA_TOKEN = os.getenv('teams_media_token', '').strip()
CHAT_ID = os.getenv('teams_chat_id', '19:253e041a-5f7f-4f21-88b1-2b349a351a53_4c107e33-2a08-4a96-8c8d-f081a3c2bab8@unq.gbl.spaces')
URL = f"https://teams.cloud.microsoft/api/chatsvc/apac/v1/users/ME/conversations/{CHAT_ID}/messages"
if TOKEN.lower().startswith('skypetoken='):
    HEADERS = {"Authentication": TOKEN}
else:
    AUTHORIZATION = TOKEN if TOKEN.lower().startswith('bearer ') else f"Bearer {TOKEN}"
    HEADERS = {"Authorization": AUTHORIZATION}

# ponytail: minimal payload; Teams only strictly requires content, type, and a unique client ID
def send(text: str) -> bool:
    payload = {
        "content": f"<p>{text}</p>",
        "messagetype": "RichText/Html",
        "contenttype": "Text",
        "clientmessageid": str(int(time.time() * 1000)),
    }
    return _http.post(URL, headers=HEADERS, json=payload, timeout=15).status_code in (200, 201)

def react(msg_id: str, key: str = "2705_whiteheavycheckmark") -> bool:
    """Add an emoji reaction to a Teams message via chatsvc."""
    url = f"{URL}/{msg_id}/properties?name=emotions"
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

def read(limit: int = 1) -> list[dict]:
    res = _http.get(f"{URL}?pageSize={limit}", headers=HEADERS, timeout=15)
    if not res.ok:
        raise RuntimeError(f"Teams messages request failed with HTTP {res.status_code}")
    return res.json().get("messages", [])

if __name__ == "__main__":
    for msg in read():
        # Clean print safe for Windows terminal
        print("Msg:", msg.encode("ascii", "replace").decode())
    print("Sent:", send("ping"))

