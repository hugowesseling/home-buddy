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

`messages` and any other extra fields are optional and ignored.

## Streaming response (`stream: true`)

`Content-Type: text/event-stream`, standard OpenAI-compatible stream:

1. A chunk with `delta: {"role": "assistant"}` and `finish_reason: null`.
2. One or more content chunks with `delta: {"content": "<part>"}` — the reply is emitted as a few word-sized parts, with about 200 ms between them.
3. A final chunk with an empty `delta: {}` and `finish_reason: "stop"`.
4. `data: [DONE]`.

Each event is sent as `data: <json>\n\n`, where each JSON object is a `chat.completion.chunk` (with `id`, `object`, `created`, `model`, and `choices: [{index, delta, finish_reason}]`).

## Non-streaming response (default)

A standard `chat.completion` JSON object; the reply is in `choices[0].message.content` as a single string.
