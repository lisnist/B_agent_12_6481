# 测试千问api能不能连上
import json
import urllib.error
import urllib.request


BASE_URL = "http://127.0.0.1:8012"
GENERATE_URL = f"{BASE_URL}/generate"
HEALTH_URL = f"{BASE_URL}/health"


def request_json(url, payload=None, timeout=600):
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url=url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = body
        return exc.code, parsed


payload = {
    "messages": [
        {
            "role": "system",
            "content": (
                "You are a local tool-using agent. Return exactly one valid JSON object. "
                'Use keys "content" and "tool_calls".'
            ),
        },
        {
            "role": "user",
            "content": (
                "你好，用一句中文回复我。"
                '必须严格返回 JSON，例如 {"content":"一句中文回复","tool_calls":[]}'
            ),
        },
    ],
    "generation": {
        "max_new_tokens": 128,
        "do_sample": False,
    },
}


health_status, health_data = request_json(HEALTH_URL, timeout=10)
print("health_status:", health_status)
print(json.dumps(health_data, ensure_ascii=False, indent=2))

status, data = request_json(GENERATE_URL, payload, timeout=600)
print("generate_status:", status)
print(json.dumps(data, ensure_ascii=False, indent=2))

if status == 200:
    print("raw_text:")
    print(data.get("raw_text", ""))
