import time
import requests
import os
from dotenv import load_dotenv
load_dotenv()

TOKEN = os.getenv('teams_token', '')
CHAT_ID = os.getenv('teams_chat_id', '19:253e041a-5f7f-4f21-88b1-2b349a351a53_4c107e33-2a08-4a96-8c8d-f081a3c2bab8@unq.gbl.spaces')
URL = f"https://teams.cloud.microsoft/api/chatsvc/apac/v1/users/ME/conversations/{CHAT_ID}/messages"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}

# ponytail: minimal payload; Teams only strictly requires content, type, and a unique client ID
def send(text: str) -> bool:
    payload = {
        "content": f"<p>{text}</p>",
        "messagetype": "RichText/Html",
        "contenttype": "Text",
        "clientmessageid": str(int(time.time() * 1000)),
    }
    return requests.post(URL, headers=HEADERS, json=payload).status_code in (200, 201)

def read(limit: int = 1) -> list[dict]:
    res = requests.get(f"{URL}?pageSize={limit}", headers=HEADERS)
    return res.json().get("messages", [])

if __name__ == "__main__":
    for msg in read():
        # Clean print safe for Windows terminal
        print("Msg:", msg.encode("ascii", "replace").decode())
    print("Sent:", send("ping"))

