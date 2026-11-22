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

The last user message is sent to the model — ollama's `qwen3.5:2b`, thinking disabled.

The model gets Home Assistant state two ways:

1. **Prompt states** — the server fetches the states of the relevant HA entities (those whose entity_id or friendly name contains "solar", "saj_battery" or "Battery 1 SOC" — e.g. `sensor.saj_battery_1_soc`) from `GET http://127.0.0.1:8123/api/states` and adds them to the prompt as context, refreshed every 10 minutes.
2. **Tool** — the server offers the model the tool `get_home_assistant_state(entity_id)`; when the model returns a tool call, the server fetches that entity's state from `GET /api/states/{entity_id}` and gives it back to the model.

The HA API is authenticated with the long-lived token in `ha_token` (or the `HA_TOKEN` environment variable).

`messages` and any other extra fields are optional and ignored.

Tool results are handled directly for the action tools `set_light` and `set_disco`: the server turns the Home Assistant result into a short confirmation without asking the model for a second reply. Informational tool results (`get_home_assistant_state`) are fed back to the model for a final answer. Each model call's Ollama cache counters and timings are appended to `llm.log` as `STATS` records.

## Streaming response (`stream: true`)

`Content-Type: text/event-stream`, standard OpenAI-compatible stream of the model's reply:

1. A chunk with `delta: {"role": "assistant"}` and `finish_reason: null`.
2. One or more content chunks with `delta: {"content": "<part>"}` — the reply is streamed from the model as it generates.
3. A final chunk with an empty `delta: {}` and `finish_reason: "stop"`.
4. `data: [DONE]`.

Each event is sent as `data: <json>\n\n`, where each JSON object is a `chat.completion.chunk` (with `id`, `object`, `created`, `model`, and `choices: [{index, delta, finish_reason}]`).

## Non-streaming response (default)

A standard `chat.completion` JSON object; the model's reply is in `choices[0].message.content` as a single string.
