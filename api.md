# API

One endpoint:

- `POST /v1/chat/completions` (also `POST /chat/completions`)

## Request

Send a JSON body in the standard OpenAI chat completions shape:

```json
{
  "model": "your-model-name",
  "messages": [{"role": "user", "content": "What's the solar battery state?"}],
  "stream": true
}
```

The last user message is sent to `ling-3.0-tiny` with thinking disabled, along with a short static instruction: the state block is reference data, use the tools to carry out the user's request, refer to entities by the exact ids or names shown, target a named room or light (use All lights only for whole-home or unnamed requests), apply a bare brightness to All lights, and never repeat a tool call. The model acts on whatever comes last, so the state block is placed before the user's request. The server talks to a local llama.cpp `llama-server`; configure it with `LLAMA_HOST` (default `http://127.0.0.1:18080`), `LLAMA_MODEL` (default `ling-3.0-tiny`) and `LLAMA_TEMPERATURE` (default `0`).

The model gets Home Assistant state two ways:

1. **Prompt states** — the server fetches the states of the relevant HA entities from `GET http://127.0.0.1:8123/api/states` and prepends them to the last user message as a `<home_state>` block, refreshed every 10 minutes. Each entry shows the real entity id alongside its friendly label (e.g. `sensor.home_energy_manager_state (state of the heatpump): STANDBY`), so the model can address entities correctly. The block ends with the local time as `HH:MM` (24-hour), refreshed together with the state every 10 minutes so the prompt prefix stays stable between refreshes. Keeping the block and the request in the same trailing user message lets llama.cpp reuse the cached prefix when the state refreshes.
2. **Tool** — the server offers the model the tool `get_home_assistant_state(entity_id)`; when the model returns a tool call, the server fetches that entity's state from `GET /api/states/{entity_id}`. The tool description lists common untracked sensors by friendly name and exact entity id so the model can select them without guessing; near-miss ids and friendly labels (e.g. `state of the heatpump`) also resolve to real entity ids, and the returned JSON includes the `entity_id`.

The HA API is authenticated with the long-lived token in `ha_token` (or the `HA_TOKEN` environment variable).

`messages` and any other extra fields are optional and ignored.

A suitable llama.cpp backend can be started with:

```bash
~/llm/llama.cpp/build/bin/llama-server \
  -m ~/llm/models/Ling-3.0-tiny-bartowski-Q4_K_M.gguf \
  --host 127.0.0.1 --port 18080 --alias ling-3.0-tiny \
  --ctx-size 4096 --parallel 1 --reasoning off --no-webui \
  --threads 8 --threads-batch 8 -b 2048 -ub 512 \
  --cache-prompt --cache-ram 8192 --load-mode mmap+mlock
```

`--load-mode mmap+mlock` keeps the model locked in RAM so it is never swapped out; the included `llama-server.service` runs this command under systemd with automatic restart. A cold uncached prompt costs several seconds, so the server should be left running.

`set_light` replies report the requested brightness percentage, since Home Assistant's state can briefly lag behind the service call. Repeated identical tool calls are detected and stopped to prevent loops; if no answer is produced, a short fallback reply is returned. Tool results are handled directly for the action tools `set_light` and `set_disco`: the server turns the Home Assistant result into a short confirmation without asking the model for a second reply. Informational tool results (`get_home_assistant_state`) are fed back to the model for a final answer. Each model call's backend, model, finish reason and timings are appended to `llm.log` as `STATS` records. Llama.cpp also logs its build fingerprint, cached-token count and throughput.

## Streaming response (`stream: true`)

`Content-Type: text/event-stream`, standard OpenAI-compatible stream of the model's reply:

1. A chunk with `delta: {"role": "assistant"}` and `finish_reason: null`.
2. One or more content chunks with `delta: {"content": "<part>"}` — the reply is streamed from the model as it generates.
3. A final chunk with an empty `delta: {}` and `finish_reason: "stop"`.
4. `data: [DONE]`.

Each event is sent as `data: <json>\n\n`, where each JSON object is a `chat.completion.chunk` (with `id`, `object`, `created`, `model`, and `choices: [{index, delta, finish_reason}]`).

Content is forwarded token-by-token as it arrives. If the model writes a preamble before a tool call, that text is streamed too, followed by the tool's deterministic confirmation.

## Non-streaming response (default)

A standard `chat.completion` JSON object; the model's reply is in `choices[0].message.content` as a single string.
