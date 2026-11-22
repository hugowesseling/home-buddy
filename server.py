#!/usr/bin/env python3
"""A small OpenAI-compatible server that proxies to llama.cpp and Home Assistant.

POST /v1/chat/completions: the received text is sent to `qwen3.5:2b` with
thinking disabled. By default llama.cpp's OpenAI-compatible endpoint is used,
falling back to ollama if it is unavailable. The model gets Home Assistant state
two ways:

1. Prompt states: the server fetches the states of relevant HA entities
   (`GET http://127.0.0.1:8123/api/states`) and adds them to the prompt as
   context (refreshed every HA_REFRESH_INTERVAL seconds, 10 minutes).
2. Tools: the server offers the model the tools
   `get_home_assistant_state(entity_id)` (fetch a state),
   `set_light(entity_id, state, brightness, color)` (control a light) and
   `set_disco(state)` (cycle light colors). When the model returns tool
   calls, the server runs them. Informational results are fed back to the
   model for another round; action results are turned into a confirmation
   directly, avoiding a second (slow) model call.

With `stream: true` the final reply streams back as standard OpenAI chunks
ending with `data: [DONE]`; otherwise it returns a `chat.completion` JSON.
"""

import json
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

def _endpoint(value, default):
    """Accept bare host:port env values as well as full URLs."""
    value = (value or "").strip()
    if not value or value in ("0.0.0.0", "::", "[::]"):
        value = default
    if "://" not in value:
        value = "http://" + value
    return value.rstrip("/")


OLLAMA_HOST = _endpoint(os.environ.get("OLLAMA_HOST"), "http://127.0.0.1:11434")
OLLAMA_MODEL = "qwen3.5:2b"
LLAMA_HOST = _endpoint(os.environ.get("LLAMA_HOST"), "http://127.0.0.1:18080")
LLAMA_MODEL = os.environ.get("LLAMA_MODEL", OLLAMA_MODEL)
# auto: try llama.cpp first, then ollama. llama/ollama force one backend.
LLM_BACKEND = os.environ.get("LLM_BACKEND", "auto").strip().lower()
HA_BASE_URL = "http://127.0.0.1:8123"
HA_TOKEN_FILE = Path("ha_token")  # HA long-lived access token for the API
HA_REFRESH_INTERVAL = 600  # 10 minutes
HA_TRACKED_ENTITIES = {
    "sensor.saj_battery_1_soc": "battery SoC",
    "sensor.power_production_now": "current solar power coming in",
    "sensor.saj_total_load_power": "current power used",
    "sensor.home_energy_manager_state": "state of the heatpump",
}
HA_UNITS = {
    "battery SoC": "percent",
    "current solar power coming in": "Watt",
    "current power used": "Watt",
}
HA_TRACKED_LIGHTS = {
    "light.all_lights": "All lights",
    "light.piano_lights": "Piano lights",
    "light.kajplats_e27_cws_globe_1055lm": "Living room light",
    "light.kajplats_e27_cws_globe_1055lm_2": "Dining room light",
}
DEFAULT_ENTITY = "Battery 1 SOC"
MAX_TOOL_ROUNDS = 5
CHAT_COMPLETIONS_PATHS = {"/v1/chat/completions", "/chat/completions"}
DEFAULT_PORT = 8971
LLM_LOG_FILE = Path("llm.log")


def _normalize(text):
    return "".join(c.lower() for c in str(text) if c.isalnum())


def entity_matches(entity, names):
    """True when the entity's id or friendly name contains any of the (normalized) names."""
    haystack = _normalize(entity.get("entity_id", "")) + " " + _normalize(
        (entity.get("attributes") or {}).get("friendly_name", "")
    )
    return any(_normalize(name) in haystack for name in names)


def _strong_name_match(entity, name):
    """Match an entity against a name, ignoring digits and spacing.

    So 'battery_soc' or 'battery SoC' matches sensor.saj_battery_1_soc.
    """
    if entity_matches(entity, [name]):
        return True
    key = "".join(c for c in _normalize(name) if not c.isdigit())
    if not key:
        return False
    for text in (
        _normalize(entity.get("entity_id", "")),
        _normalize((entity.get("attributes") or {}).get("friendly_name", "")),
    ):
        compact = "".join(c for c in text if not c.isdigit())
        if compact and (key in compact or compact in key):
            return True
    return False


# --- Home Assistant ---------------------------------------------------------

def ha_token():
    token = os.environ.get("HA_TOKEN")
    if token:
        return token
    try:
        return HA_TOKEN_FILE.read_text().strip() or None
    except OSError:
        return None


def fetch_all_ha_states():
    """GET http://127.0.0.1:8123/api/states -> list of entity state dicts."""
    token = ha_token()
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(HA_BASE_URL + "/api/states", headers=headers)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def _ha_service_call(service, data):
    """Call an HA service: POST {HA_BASE_URL}/api/services/{service} with the data.

    Returns the service response, or None on failure.
    """
    token = ha_token()
    if token is None:
        return None
    req = urllib.request.Request(
        HA_BASE_URL + "/api/services/" + service,
        data=json.dumps(data).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read())
    except (urllib.error.URLError, OSError, ValueError):
        return None


def fetch_relevant_ha_states():
    """The HA entities in HA_TRACKED_ENTITIES, or None if there's no token."""
    if ha_token() is None:
        return None
    try:
        states = fetch_all_ha_states()
    except (urllib.error.URLError, OSError, ValueError):
        return None
    return [entity for entity in states if entity_matches(entity, HA_TRACKED_ENTITIES)]


_ha_state_cache = {"fetched_at": 0.0, "states": []}


def cached_ha_states():
    if time.time() - _ha_state_cache["fetched_at"] > HA_REFRESH_INTERVAL:
        states = fetch_relevant_ha_states()
        if states is not None:
            _ha_state_cache["states"] = states
        _ha_state_cache["fetched_at"] = time.time()
    return _ha_state_cache["states"]


def ha_context():
    """Prompt context: the current states of the relevant HA entities."""
    states = cached_ha_states()
    if not states:
        return ""
    lines = ["Home Assistant state (current):"]
    for entity in states:
        name = HA_TRACKED_ENTITIES.get(entity.get("entity_id", ""))
        if name:
            unit = HA_UNITS.get(name)
            value = entity.get("state", "")
            if unit:
                value = f"{value} {unit}"
            lines.append(f"- {name}: {value}")
    return "\n".join(lines)


def with_ha_context(messages, context):
    """Append HA context to the last user message.

    Keeping these fast-changing values at the very end of the prompt lets
    llama.cpp reuse the cached prefix when the state refreshes.
    """
    if not context:
        return messages
    result = [dict(message) for message in messages]
    for index in range(len(result) - 1, -1, -1):
        if result[index].get("role") == "user":
            content = result[index].get("content") or ""
            result[index]["content"] = f"{content}\n\n{context}" if content else context
            return result
    return result + [{"role": "user", "content": context}]


def _entity_json(entity):
    entity_id = entity.get("entity_id")
    name = HA_TRACKED_ENTITIES.get(entity_id, entity_id)
    attributes = {
        key: value
        for key, value in (entity.get("attributes") or {}).items()
        if key != "friendly_name"
    }
    return json.dumps(
        {"name": name, "state": entity.get("state"), "attributes": attributes}
    )


def get_home_assistant_state(entity_id=None):
    """Tool executor: return the JSON of the requested HA entity's state."""
    token = ha_token()
    if token is None:
        return "Error: no Home Assistant access token available."
    name = (entity_id or "").strip() or DEFAULT_ENTITY
    try:
        req = urllib.request.Request(
            HA_BASE_URL + "/api/states/" + urllib.parse.quote(name, safe=""),
            headers={"Authorization": f"Bearer {token}"},
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            return _entity_json(json.loads(resp.read()))
    except urllib.error.HTTPError as error:
        if error.code != 404:
            return f"Error: fetching {name!r} failed (HTTP {error.code})."
    except (urllib.error.URLError, OSError, ValueError) as error:
        return f"Error: fetching {name!r} failed: {error}"
    # The name isn't a live entity id; resolve it against the full states list.
    try:
        matches = [
            entity
            for entity in fetch_all_ha_states()
            if _strong_name_match(entity, name)
        ]
    except (urllib.error.URLError, OSError, ValueError):
        return "Error: could not fetch Home Assistant states."
    for entity in matches:
        if entity.get("entity_id") in HA_TRACKED_ENTITIES:
            return _entity_json(entity)
    if matches:
        return _entity_json(matches[0])
    return f"Entity {name!r} not found in Home Assistant."


def _resolve_light_entity(name):
    """Resolve a light name or entity id to a tracked HA light entity id."""
    name = (name or "").strip()
    if not name:
        return None
    for entity_id, user_name in HA_TRACKED_LIGHTS.items():
        if _strong_name_match(
            {"entity_id": entity_id, "attributes": {"friendly_name": user_name}}, name
        ):
            return entity_id
    try:
        states = fetch_all_ha_states()
    except (urllib.error.URLError, OSError, ValueError):
        return None
    matches = [
        entity
        for entity in states
        if entity.get("entity_id", "").startswith("light.")
        and _strong_name_match(entity, name)
    ]
    for entity in matches:
        if entity.get("entity_id") in HA_TRACKED_LIGHTS:
            return entity.get("entity_id")
    if matches:
        return matches[0].get("entity_id")
    return None


def _light_result(entity):
    """Tool result for a light: its user-facing name and current state."""
    entity_id = entity.get("entity_id")
    attributes = entity.get("attributes") or {}
    result = {
        "name": HA_TRACKED_LIGHTS.get(entity_id, entity_id),
        "state": entity.get("state"),
    }
    if attributes.get("brightness") is not None:
        result["brightness"] = attributes.get("brightness")
    return json.dumps(result)


_COLOR_NAMES = {
    "red": (255, 0, 0),
    "green": (0, 128, 0),
    "blue": (0, 0, 255),
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "yellow": (255, 255, 0),
    "orange": (255, 165, 0),
    "purple": (128, 0, 128),
    "pink": (255, 192, 203),
    "cyan": (0, 255, 255),
    "magenta": (255, 0, 255),
    "lime": (0, 255, 0),
    "teal": (0, 128, 128),
    "navy": (0, 0, 128),
    "maroon": (128, 0, 0),
    "olive": (128, 128, 0),
    "silver": (192, 192, 192),
    "gray": (128, 128, 128),
    "grey": (128, 128, 128),
    "brown": (165, 42, 42),
    "gold": (255, 215, 0),
    "violet": (238, 130, 238),
    "indigo": (75, 0, 130),
    "coral": (255, 127, 80),
    "turquoise": (64, 224, 208),
    "salmon": (250, 128, 114),
    "lavender": (230, 230, 250),
    "khaki": (240, 230, 140),
    "beige": (245, 245, 220),
    "crimson": (220, 20, 60),
    "skyblue": (135, 206, 235),
    "palegreen": (152, 251, 152),
}


def _parse_color(color):
    """Parse a color string ('red', '#ff0000', '#f00', 'rgb(255, 0, 0)') to an [r, g, b] list."""
    text = str(color).strip().lower()
    if not text:
        return None
    if text.startswith("#"):
        hexpart = text[1:]
        if len(hexpart) == 3:
            hexpart = "".join(c * 2 for c in hexpart)
        if len(hexpart) >= 6:
            try:
                return [
                    int(hexpart[0:2], 16),
                    int(hexpart[2:4], 16),
                    int(hexpart[4:6], 16),
                ]
            except ValueError:
                return None
        return None
    if text.startswith("rgb"):
        try:
            parts = [int(v) for v in text[3:].strip("()").split(",")]
        except ValueError:
            return None
        if len(parts) >= 3:
            return [
                max(0, min(255, parts[0])),
                max(0, min(255, parts[1])),
                max(0, min(255, parts[2])),
            ]
        return None
    rgb = _COLOR_NAMES.get(text.replace(" ", "").replace("_", "").replace("-", ""))
    return list(rgb) if rgb else None


def set_light(entity_id=None, state=None, brightness=None, color=None):
    """Tool executor: turn a light on/off, set brightness or color via the HA service."""
    if ha_token() is None:
        return "Error: no Home Assistant access token available."
    resolved = _resolve_light_entity(entity_id)
    if resolved is None:
        return f"Error: light {(entity_id or '')!r} not found in Home Assistant."
    if (state or "").strip().lower() == "off":
        service = "light/turn_off"
        data = {"entity_id": resolved}
    else:
        service = "light/turn_on"
        data = {"entity_id": resolved}
        if brightness is None:
            data["brightness"] = 255  # full brightness
        else:
            try:
                level = float(brightness)
            except (TypeError, ValueError):
                level = None
            if level is not None:
                level = max(0.0, min(100.0, level))
                data["brightness"] = int(round(level * 2.55))
        if color is not None and str(color).strip():
            rgb = _parse_color(color)
            if rgb is None:
                return f"Error: unrecognized color {color!r}."
            data["rgb_color"] = rgb
    # Capture the current state before the call (to poll for the change).
    previous = None
    try:
        for entity in fetch_all_ha_states():
            if entity.get("entity_id") == resolved:
                previous = entity.get("state")
                break
    except (urllib.error.URLError, OSError, ValueError):
        pass
    if _ha_service_call(service, data) is None:
        return f"Error: could not call the HA service for {resolved!r}."
    # Poll briefly for the new state (HA applies service calls asynchronously).
    expected = "off" if service == "light/turn_off" else "on"
    for _ in range(20):
        time.sleep(0.5)
        try:
            states = fetch_all_ha_states()
        except (urllib.error.URLError, OSError, ValueError):
            break
        for entity in states:
            if entity.get("entity_id") != resolved:
                continue
            state = entity.get("state")
            if state == expected or state != previous:
                return _light_result(entity)
            break
    # Fall back to the current state.
    try:
        states = fetch_all_ha_states()
    except (urllib.error.URLError, OSError, ValueError):
        return f"Called {service} for {resolved!r}, but the state could not be fetched."
    for entity in states:
        if entity.get("entity_id") == resolved:
            return _light_result(entity)
    return f"Called {service} for {resolved!r}."


disco_state = {
    "enabled": False,
    "stop_event": None,
    "original_states": {},
}


def _individual_lights():
    """All `light.*` entities that are not groups (the disco doesn't use the groups)."""
    try:
        states = fetch_all_ha_states()
    except (urllib.error.URLError, OSError, ValueError):
        return []
    return [
        entity
        for entity in states
        if entity.get("entity_id", "").startswith("light.")
        and not isinstance((entity.get("attributes") or {}).get("entity_id"), list)
    ]


def _random_rgb_colors(count):
    """Generate `count` distinct random [r, g, b] colors."""
    colors = []
    seen = set()
    for _ in range(count * 50):
        if len(colors) >= count:
            break
        rgb = random.randrange(0x1000000)
        if rgb in seen:
            continue
        seen.add(rgb)
        colors.append([(rgb >> 16) & 0xFF, (rgb >> 8) & 0xFF, rgb & 0xFF])
    return colors


def _start_disco():
    """Start the disco: remember the lights' states, then change colors every ~0.5s."""
    if disco_state["enabled"]:
        return
    original = {}
    for entity in _individual_lights():
        attributes = entity.get("attributes") or {}
        original[entity.get("entity_id")] = {
            "state": entity.get("state"),
            "rgb_color": attributes.get("rgb_color"),
        }
    disco_state["original_states"] = original
    stop = threading.Event()
    disco_state["stop_event"] = stop
    disco_state["enabled"] = True

    def loop():
        while not stop.is_set():
            try:
                lights = _individual_lights()
                colors = _random_rgb_colors(len(lights))
                for entity, rgb in zip(lights, colors):
                    _ha_service_call(
                        "light/turn_on",
                        {"entity_id": entity.get("entity_id"), "rgb_color": rgb},
                    )
            except Exception:
                pass
            stop.wait(0.5)

    threading.Thread(target=loop, daemon=True).start()


def _stop_disco():
    """Stop the disco and restore the lights to their original states. Returns restored ids."""
    if not disco_state["enabled"]:
        return []
    stop = disco_state["stop_event"]
    disco_state["enabled"] = False
    if stop is not None:
        stop.set()
    original = disco_state["original_states"]
    disco_state["original_states"] = {}
    restored = []
    for entity_id, info in original.items():
        try:
            if info.get("state") == "off":
                _ha_service_call("light/turn_off", {"entity_id": entity_id})
            else:
                data = {"entity_id": entity_id}
                if info.get("rgb_color"):
                    data["rgb_color"] = info["rgb_color"]
                _ha_service_call("light/turn_on", data)
            restored.append(entity_id)
        except Exception:
            pass
    return restored


def set_disco(state=None):
    """Tool executor: turn disco mode on/off (changing light colors continuously)."""
    action = (state or "").strip().lower()
    if action in ("on", "start", "enable"):
        lights = _individual_lights()
        _start_disco()
        return json.dumps(
            {
                "disco": "on",
                "individual_lights": [entity.get("entity_id") for entity in lights],
            }
        )
    if action in ("off", "stop", "disable"):
        restored = _stop_disco()
        return json.dumps({"disco": "off", "restored": restored})
    return "Error: state must be 'on' or 'off'."


LIGHT_NAMES = ", ".join(f"'{name}'" for name in HA_TRACKED_LIGHTS.values())

# Tool schemas are part of every prompt Ollama evaluates. Keep them small:
# uncached prompt tokens are evaluated on the CPU at roughly 18 ms/token here.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_home_assistant_state",
            "description": "Fetch a Home Assistant entity's current state.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_id": {"type": "string", "description": "Entity id or name."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_light",
            "description": f"Control a light (on/off, brightness 0-100, color). Lights: {LIGHT_NAMES}.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_id": {"type": "string", "description": "Name from the list or an entity id."},
                    "state": {"type": "string", "description": "'on' or 'off'."},
                    "brightness": {"type": "number", "description": "0-100."},
                    "color": {"type": "string", "description": "Color name or hex."},
                },
                "required": ["entity_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_disco",
            "description": "Turn disco mode on/off (cycles light colors every ~0.5s; off restores them).",
            "parameters": {
                "type": "object",
                "properties": {
                    "state": {"type": "string", "description": "'on' or 'off'."},
                },
            },
        },
    },
]


def execute_tool(tool_call):
    function = tool_call.get("function") or {}
    name = function.get("name", "")
    arguments = function.get("arguments") or {}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except ValueError:
            arguments = {}
    if name == "get_home_assistant_state":
        return get_home_assistant_state(arguments.get("entity_id"))
    if name == "set_light":
        return set_light(
            arguments.get("entity_id"),
            arguments.get("state"),
            arguments.get("brightness"),
            arguments.get("color"),
        )
    if name == "set_disco":
        return set_disco(arguments.get("state"))
    return f"Error: unknown tool {name!r}."


def _brightness_percent(value):
    """Home Assistant reports brightness on 0-255; users think in 0-100."""
    try:
        return int(round(float(value) / 2.55))
    except (TypeError, ValueError):
        return None


def tool_result_reply(results):
    """Build a reply from action-tool results, or None if the model must run again.

    Asking the 2B model to rephrase a successful set_light/set_disco result costs
    another full prompt evaluation (~6 s here), so answer directly. Informational
    tools still need a second model round to interpret their results.
    """
    parts = []
    for tool_call, result in results:
        name = (tool_call.get("function") or {}).get("name", "")
        if name not in ("set_light", "set_disco"):
            return None
        try:
            data = json.loads(result)
        except (TypeError, ValueError):
            data = None
        if not isinstance(data, dict):
            parts.append(result)
            continue
        if name == "set_disco":
            if data.get("disco") == "on":
                parts.append("Disco mode is on.")
            else:
                parts.append("Disco mode is off.")
                if data.get("restored"):
                    parts[-1] += f" Restored {len(data['restored'])} light(s)."
            continue
        light = data.get("name") or "Light"
        if data.get("state") == "off":
            parts.append(f"Turned {light} off.")
            continue
        brightness = _brightness_percent(data.get("brightness"))
        if brightness is not None:
            parts.append(f"Set {light} to {brightness}%.")
        elif data.get("state") == "on":
            parts.append(f"Turned {light} on.")
        else:
            parts.append(f"Set {light}.")
    return " ".join(parts) if parts else None


# --- model backends ---------------------------------------------------------

def log_llm(tag, data):
    """Append an LLM interaction record (tag + data as JSON) to llm.log."""
    try:
        with LLM_LOG_FILE.open("a") as log:
            log.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {tag}\n")
            log.write(json.dumps(data, ensure_ascii=False, indent=2))
            log.write("\n\n")
    except OSError:
        pass


def iter_ollama_events(messages, tools=None):
    """Call ollama /api/chat (streaming) and yield normalized message events."""
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": True,
        "think": False,
    }
    if tools:
        payload["tools"] = tools
    log_llm("REQUEST", {"backend": "ollama", **payload})
    req = urllib.request.Request(
        OLLAMA_HOST + "/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        for line in resp:
            line = line.strip()
            if not line:
                continue
            event = json.loads(line)
            if event.get("done"):
                stats = {
                    key: event[key]
                    for key in (
                        "prompt_eval_count",
                        "prompt_eval_cached_count",
                        "eval_count",
                    )
                    if key in event
                }
                for key in (
                    "total_duration",
                    "load_duration",
                    "prompt_eval_duration",
                    "eval_duration",
                ):
                    if key in event:
                        stats[key.replace("_duration", "_ms")] = round(
                            event[key] / 1_000_000, 1
                        )
                yield {"done": True, "stats": {"backend": "ollama", **stats}}
                return
            yield event


def _llama_stats(body):
    """Map llama-server's usage/timings onto the Ollama stat names."""
    usage = body.get("usage") or {}
    timings = body.get("timings") or {}
    return {
        "backend": "llama",
        "prompt_eval_count": usage.get("prompt_tokens")
        or timings.get("prompt_n"),
        "prompt_eval_cached_count": timings.get("cache_n"),
        "eval_count": usage.get("completion_tokens")
        or timings.get("predicted_n"),
        "prompt_eval_ms": round(timings.get("prompt_ms", 0), 1),
        "eval_ms": round(timings.get("predicted_ms", 0), 1),
    }


def iter_llama_events(messages, tools=None):
    """Call llama-server's OpenAI-compatible endpoint and yield message events.

    The response is requested non-streaming; one content chunk is enough for
    this server's short replies and avoids having to reassemble fragmented
    tool-call arguments from llama.cpp's SSE stream.
    """
    payload = {
        "model": LLAMA_MODEL,
        "messages": messages,
        "stream": False,
        # Greedy decoding: Q4_K_M at temperature 1 sometimes answers in prose
        # instead of emitting the expected tool call. Set LLAMA_TEMPERATURE to
        # override if a different sampling behaviour is wanted.
        "temperature": float(os.environ.get("LLAMA_TEMPERATURE", "0")),
    }
    if tools:
        payload["tools"] = tools
    log_llm("REQUEST", {"backend": "llama", **payload})
    req = urllib.request.Request(
        LLAMA_HOST.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        body = json.load(resp)
    choice = (body.get("choices") or [{}])[0]
    yield {"message": choice.get("message") or {}}
    yield {"done": True, "stats": _llama_stats(body)}


def iter_model_events(messages, tools=None):
    """Use llama.cpp when available, falling back to ollama in auto mode."""
    if LLM_BACKEND in ("auto", "llama"):
        try:
            yield from iter_llama_events(messages, tools)
            return
        except (OSError, urllib.error.URLError) as error:
            if LLM_BACKEND == "llama":
                raise
            log_llm("FALLBACK", {"from": "llama", "to": "ollama", "error": str(error)})
    yield from iter_ollama_events(messages, tools)


def run_model(messages, sink):
    """Call the model; return the tool calls.

    Content is buffered until the round finishes: when the model emits tool
    calls, its "I will now..." preamble is not part of the answer. The caller
    supplies the final reply (deterministically for action tools, or via the
    next model round for informational tools).
    """
    tool_calls = []
    content_parts = []
    for event in iter_model_events(messages, TOOLS):
        if event.get("done"):
            if event.get("stats"):
                log_llm("STATS", event["stats"])
            continue
        message = event.get("message") or {}
        content = (message.get("content") or "").replace("**", "")
        if content:
            content_parts.append(content)
        if message.get("tool_calls"):
            tool_calls.extend(message["tool_calls"])
    log_llm("RESPONSE", {"content": "".join(content_parts), "tool_calls": tool_calls})
    if not tool_calls and sink:
        for part in content_parts:
            sink(part)
    return tool_calls


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
        user_messages = [
            message
            for message in (body.get("messages") or [])
            if isinstance(message, dict)
        ]

        messages = with_ha_context(user_messages, ha_context())

        rid = "chatcmpl-" + uuid.uuid4().hex
        created = int(time.time())
        collected = []

        def sink(part):
            if stream:
                self._send_sse(make_chunk(rid, created, model, {"content": part}))
            else:
                collected.append(part)

        if stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            self._send_sse(make_chunk(rid, created, model, {"role": "assistant"}))

        try:
            for _ in range(MAX_TOOL_ROUNDS):
                tool_calls = run_model(messages, sink)
                if not tool_calls:
                    break
                messages = messages + [
                    {"role": "assistant", "content": "", "tool_calls": tool_calls}
                ]
                results = []
                for tool_call in tool_calls:
                    result = execute_tool(tool_call)
                    results.append((tool_call, result))
                    messages.append({"role": "tool", "content": result})
                    log_llm("TOOL", {"call": tool_call, "result": result})
                reply = tool_result_reply(results)
                if reply is not None:
                    sink(reply)
                    log_llm("REPLY", {"content": reply})
                    break
        except (OSError, urllib.error.URLError) as error:
            log_llm("ERROR", {"error": str(error)})

        if not stream:
            completion = {
                "id": rid,
                "object": "chat.completion",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "".join(collected)},
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

        try:
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
    try:
        cached_ha_states()  # warm the HA state cache at startup
    except Exception:
        pass
    backend = {
        "auto": f"llama.cpp ({LLAMA_HOST}) -> ollama ({OLLAMA_HOST})",
        "llama": f"llama.cpp ({LLAMA_HOST})",
        "ollama": f"ollama ({OLLAMA_HOST})",
    }.get(LLM_BACKEND, f"unknown LLM_BACKEND={LLM_BACKEND!r}")
    print(
        f"OpenAI-compatible streaming server on http://127.0.0.1:{port}/v1/chat/completions "
        f"(model {OLLAMA_MODEL}, thinking disabled; backend: {backend}; HA states from {HA_BASE_URL})",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
