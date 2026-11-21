# API

One endpoint:

- `POST /v1/chat/completions` (also `POST /chat/completions`)

## Request

Send a JSON body in the standard OpenAI chat completions shape:

```json
{
  "model": "your-model-name",
  "messages": [{"role": "user", "content": "hi"}],
  "stream": true
}
```

The last user message in `messages` is the text sent to the model — ollama's `qwen3.5:2b`, with thinking disabled. `messages` and any other extra fields are optional and ignored.

## Streaming response (`stream: true`)

`Content-Type: text/event-stream`, standard OpenAI-compatible stream of the model's reply:

1. A chunk with `delta: {"role": "assistant"}` and `finish_reason: null`.
2. One or more content chunks with `delta: {"content": "<part>"}` — the reply is streamed from the model as it generates.
3. A final chunk with an empty `delta: {}` and `finish_reason: "stop"`.
4. `data: [DONE]`.

Each event is sent as `data: <json>\n\n`, where each JSON object is a `chat.completion.chunk` (with `id`, `object`, `created`, `model`, and `choices: [{index, delta, finish_reason}]`).

## Non-streaming response (default)

A standard `chat.completion` JSON object; the model's reply is in `choices[0].message.content` as a single string.
