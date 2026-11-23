# home-buddy

A small OpenAI-compatible chat API that answers questions about your home and controls Home Assistant through a local `qwen3.5:2b` model. It talks to llama.cpp directly, streams replies token-by-token, and exposes HA state plus light/disco controls as model tools.

## Requirements

- Python 3 (standard library only)
- Home Assistant at `http://127.0.0.1:8123` with a long-lived access token
- A llama.cpp `llama-server` serving `qwen3.5:2b`

## Setup

1. Put your Home Assistant long-lived token in a `ha_token` file in this directory (gitignored), or set the `HA_TOKEN` environment variable.
2. Start llama.cpp with the tuned command in [api.md](api.md).
3. Start the server (port 8971 by default):

   ```bash
   python3 server.py [port]
   ```

## Usage

```bash
curl -X POST http://127.0.0.1:8971/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.5:2b","messages":[{"role":"user","content":"It is a bit dark here"}],"stream":true}'
```

Returns a standard OpenAI completion, or SSE chunks ending in `data: [DONE]` when `stream: true`.

## Configuration

| Variable | Default | Description |
| --- | --- | --- |
| `LLAMA_HOST` | `http://127.0.0.1:18080` | llama.cpp server URL |
| `LLAMA_MODEL` | `qwen3.5:2b` | model alias used with llama.cpp |
| `LLAMA_TEMPERATURE` | `0` | sampling temperature for llama.cpp |
| `HA_TOKEN` | `ha_token` file | Home Assistant long-lived access token |

Relevant HA states are cached for 10 minutes. Model timings and tool calls are appended to `llm.log` (gitignored).

See [api.md](api.md) for the full request/response format and backend setup.
