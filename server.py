#!/usr/bin/env python3
"""A tiny OpenAI-compatible server that streams one of three canned sentences.

Serves POST /v1/chat/completions in the standard OpenAI streaming format:
chat.completion.chunk events sent as SSE, ending with `data: [DONE]`.
The sentence is emitted as a few parts with ~200 ms between them to
simulate a ~5 token/s LLM. Non-streaming requests get the full sentence
back in a normal chat.completion JSON.
"""

import json
import random
import sys
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SENTENCES = [
    "Hello, how are you doing?",
    "Hi, I'm happy to be of service.",
    "Hi there, how can I help you today?",
]

PART_INTERVAL = 0.2  # ~5 tokens per second
CHAT_COMPLETIONS_PATHS = {"/v1/chat/completions", "/chat/completions"}
DEFAULT_PORT = 8971


def split_into_parts(sentence):
    """Split a sentence into a few parts, like an LLM emitting a few tokens at a time."""
    words = sentence.split(" ")
    return [(" " if i else "") + word for i, word in enumerate(words)] or [sentence]


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

        model = str(body.get("model") or "test-model")
        stream = bool(body.get("stream"))
        sentence = random.choice(SENTENCES)

        if not stream:
            completion = {
                "id": "chatcmpl-" + uuid.uuid4().hex,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": sentence},
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
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        rid = "chatcmpl-" + uuid.uuid4().hex
        created = int(time.time())
        parts = split_into_parts(sentence)

        try:
            self._send_sse(make_chunk(rid, created, model, {"role": "assistant"}))
            for i, part in enumerate(parts):
                self._send_sse(make_chunk(rid, created, model, {"content": part}))
                if i != len(parts) - 1:
                    time.sleep(PART_INTERVAL)
            self._send_sse(make_chunk(rid, created, model, {}, "stop"))
            self._write_chunk(b"data: [DONE]\n\n")
        except OSError:
            pass
        finally:
            try:
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except OSError:
                pass


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PORT
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(
        f"OpenAI-compatible streaming server on http://127.0.0.1:{port}/v1/chat/completions",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
