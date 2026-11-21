#!/usr/bin/env python3
"""A small OpenAI-compatible server that proxies to ollama.

POST /v1/chat/completions: the received text (the last user message) is
sent to ollama's `qwen3.5:2b` with thinking disabled (`think: false`, the
API form of `ollama run qwen3.5:2b --think=false`). With `stream: true`
the model's reply is streamed back as standard OpenAI chunks ending with
`data: [DONE]`; otherwise it returns a normal `chat.completion` JSON.
"""

import json
import sys
import time
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

OLLAMA_HOST = "http://127.0.0.1:11434"
OLLAMA_MODEL = "qwen3.5:2b"
CHAT_COMPLETIONS_PATHS = {"/v1/chat/completions", "/chat/completions"}
DEFAULT_PORT = 8971


def extract_text(body):
    """The text to send to ollama: the last user message of the request."""
    for message in reversed(body.get("messages") or []):
        if isinstance(message, dict) and message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "".join(
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict)
                )
    return ""


def ollama_request(text):
    return urllib.request.Request(
        OLLAMA_HOST + "/api/chat",
        data=json.dumps(
            {
                "model": OLLAMA_MODEL,
                "messages": [{"role": "user", "content": text}],
                "stream": True,
                "think": False,
            }
        ).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )


def iter_ollama_chunks(text):
    """Yield the model's reply as it streams from ollama."""
    with urllib.request.urlopen(ollama_request(text), timeout=300) as resp:
        for line in resp:
            line = line.strip()
            if not line:
                continue
            event = json.loads(line)
            message = event.get("message") or {}
            content = message.get("content") or ""
            if content:
                yield content
            if event.get("done"):
                return


def make_chunk(id_, created, model, delta, finish_reason=None):
    return {
        "id": id_,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _write_chunk(self, data):
        self.wfile.write(f"{len(data):X}\r\n".encode("ascii") + data + b"\r\n")
        self.wfile.flush()

    def _send_sse(self, event):
        data = json.dumps(event, separators=(",", ":")).encode("utf-8")
        self._write_chunk(b"data: " + data + b"\n\n")

    def do_POST(self):
        if self.path not in CHAT_COMPLETIONS_PATHS:
            self.send_error(404, "Not Found")
            return

        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, OSError):
            self.send_error(400, "Bad Request")
            return

        model = str(body.get("model") or OLLAMA_MODEL)
        stream = bool(body.get("stream"))
        text = extract_text(body)

        if stream:
            self._stream_from_ollama(model, text)
        else:
            self._completion_from_ollama(model, text)

    def _stream_from_ollama(self, model, text):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        rid = "chatcmpl-" + uuid.uuid4().hex
        created = int(time.time())
        try:
            self._send_sse(make_chunk(rid, created, model, {"role": "assistant"}))
            for part in iter_ollama_chunks(text):
                self._send_sse(make_chunk(rid, created, model, {"content": part}))
            self._send_sse(make_chunk(rid, created, model, {}, "stop"))
            self._write_chunk(b"data: [DONE]\n\n")
        except (OSError, urllib.error.URLError):
            pass
        finally:
            try:
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except OSError:
                pass

    def _completion_from_ollama(self, model, text):
        try:
            content = "".join(iter_ollama_chunks(text))
        except (urllib.error.URLError, OSError) as error:
            self.send_error(500, f"ollama call failed: {error}")
            return

        completion = {
            "id": "chatcmpl-" + uuid.uuid4().hex,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }
        data = json.dumps(completion).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PORT
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(
        f"OpenAI-compatible streaming server on http://127.0.0.1:{port}/v1/chat/completions "
        f"(sends text to ollama {OLLAMA_MODEL} with thinking disabled)",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
