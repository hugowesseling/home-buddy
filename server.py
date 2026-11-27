#!/usr/bin/env python3
"""A small OpenAI-compatible server that proxies to llama.cpp and Home Assistant.

POST /v1/chat/completions: the received text is sent to `ling-3.0-tiny` with
thinking disabled through llama.cpp's OpenAI-compatible endpoint. The model gets
Home Assistant state two ways:

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


LLAMA_HOST = _endpoint(os.environ.get("LLAMA_HOST"), "http://127.0.0.1:18080")
LLAMA_MODEL = os.environ.get("LLAMA_MODEL", "ling-3.0-tiny")
HA_BASE_URL = "http://127.0.0.1:8123"
HA_TOKEN_FILE = Path("ha_token")  # HA long-lived access token for the API
HA_REFRESH_INTERVAL = 600  # 10 minutes
HA_TRACKED_ENTITIES = {
    "sensor.saj_battery_1_soc": "battery SoC",
    "sensor.power_production_now": "current solar power coming in",
    "sensor.saj_total_load_power": "current power used",
    "sensor.home_energy_manager_state": "state of the heatpump",
    "sensor.energy_production_today": "total solar generated today",
    "sensor.energy_production_tomorrow": "solar forecast for tomorrow",
    "sensor.saj_sum_all_phases_feed_in_today": "total solar exported today",
    "sensor.saj_total_today_load": "total power used today",
}
HA_UNITS = {
    "battery SoC": "percent",
    "current solar power coming in": "Watt",
    "current power used": "Watt",
    "total solar generated today": "kWh",
    "solar forecast for tomorrow": "kWh",
    "total solar exported today": "kWh",
    "total power used today": "kWh",
}
HA_TRACKED_LIGHTS = {
    "light.all_lights": "All lights",
    "light.piano_lights": "Piano lights",
    "light.kajplats_e27_cws_globe_1055lm": "Living room light",
    "light.kajplats_e27_cws_globe_1055lm_2": "Dining room light",
}
# Friendly names for sensors the model may be asked about. Included in the
# get_home_assistant_state tool description so it can pass exact entity ids
# instead of guessing, and used below to resolve near-miss ids.
KNOWN_SENSORS = {
    "living room temperature": "sensor.living_room_tuya_temperature",
    "living room humidity": "sensor.living_room_tuya_humidity",
    "upstairs temperature": "sensor.upstairs_tuya_temperature",
    "upstairs humidity": "sensor.upstairs_tuya_humidity",
    "outdoor temperature": "sensor.home_outdoor_temperature",
    "hot water tank temperature": "sensor.home_domestic_hot_water_0_tank_temperature",
    "battery temperature": "sensor.saj_battery_temperature",
    "inverter temperature": "sensor.saj_inverter_temperature",
    "system flow temperature": "sensor.home_system_flow_temperature",
    "weather": "weather.forecast_home",
    "sun status": "sun.sun",
    "sunrise": "sensor.sun_next_rising",
    "sunset": "sensor.sun_next_setting",
    "water pressure": "sensor.home_system_water_pressure",
    "heating efficiency": "sensor.home_heating_energy_efficiency",
    "hot water setpoint": "sensor.home_domestic_hot_water_0_setpoint",
    "hot water mode": "sensor.home_domestic_hot_water_0_operation_mode",
}
# HA_TRACKED_ENTITIES are deliberately omitted: their values are already in
# home_state, so listing them here tempts the model into a redundant tool round.
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


def _canonical_name(text):
    """Stable key for matching near-miss entity ids (drops vendor/prefix words).

    e.g. both 'sensor.upstairs_humidity' and 'sensor.upstairs_tuya_humidity'
    reduce to 'upstairs humidity', so a small model's invented id still resolves.
    """
    value = _normalize(text)
    for word in ("sensor", "binary", "tuya", "saj", "home", "light"):
        value = value.replace(word, "")
    return value


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


_ha_state_cache = {"fetched_at": 0.0, "states": [], "time": ""}


def cached_ha_states():
    if time.time() - _ha_state_cache["fetched_at"] > HA_REFRESH_INTERVAL:
        states = fetch_relevant_ha_states()
        if states is not None:
            _ha_state_cache["states"] = states
        _ha_state_cache["fetched_at"] = time.time()
        # Refreshed with the state, not every minute: changing the prompt every
        # minute would invalidate llama.cpp's cached prefix needlessly.
        _ha_state_cache["time"] = time.strftime("%H:%M")
    return _ha_state_cache["states"]


def ha_context():
    """Prompt context: the current states of the relevant HA entities."""
    states = cached_ha_states()
    if not states:
        return ""
    lines = ["<home_state>"]
    for entity in states:
        entity_id = entity.get("entity_id", "")
        name = HA_TRACKED_ENTITIES.get(entity_id)
        if name:
            unit = HA_UNITS.get(name)
            value = entity.get("state", "")
            if unit:
                value = f"{value} {unit}"
            # Include the real entity id: the friendly label alone does not
            # resolve to the HA entity (e.g. "state of the heatpump" lives at
            # sensor.home_energy_manager_state).
            lines.append(f"{entity_id} ({name}): {value}")
    # Updated with the 10-minute state refresh, so the prompt stays byte-for-byte
    # stable between refreshes and llama.cpp can reuse the cached prefix.
    lines.append(f"Current time: {_ha_state_cache['time'] or time.strftime('%H:%M')}")
    lines.append("</home_state>")
    return "\n".join(lines)


def with_ha_context(messages, context):
    """Prepend HA context to the last user message, leaving the request last.

    The model acts on whatever comes last in the prompt, so the user's request
    must follow the state block. Keeping both in the same trailing user message
    still lets llama.cpp reuse the cached prefix when the state refreshes.
    """
    if not context:
        return messages
    result = [dict(message) for message in messages]
    for index in range(len(result) - 1, -1, -1):
        if result[index].get("role") == "user":
            content = result[index].get("content") or ""
            result[index]["content"] = f"{context}\n\n{content}" if content else context
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
        {
            "entity_id": entity_id,
            "name": name,
            "state": entity.get("state"),
            "attributes": attributes,
        }
    )


def _get_entity_by_id(entity_id):
    """Fetch one entity by id; return JSON, an error string, or None if absent."""
    token = ha_token()
    try:
        req = urllib.request.Request(
            HA_BASE_URL + "/api/states/" + urllib.parse.quote(entity_id, safe=""),
            headers={"Authorization": f"Bearer {token}"},
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            return _entity_json(json.loads(resp.read()))
    except urllib.error.HTTPError as error:
        if error.code != 404:
            return f"Error: fetching {entity_id!r} failed (HTTP {error.code})."
    except (urllib.error.URLError, OSError, ValueError) as error:
        return f"Error: fetching {entity_id!r} failed: {error}"
    return None


def get_home_assistant_state(entity_id=None):
    """Tool executor: return the JSON of the requested HA entity's state."""
    if ha_token() is None:
        return "Error: no Home Assistant access token available."
    name = (entity_id or "").strip() or DEFAULT_ENTITY
    result = _get_entity_by_id(name)
    if result is not None:
        return result
    # Labels from our own state block (e.g. "state of the heatpump") are not
    # HA entity ids. Resolve them through the tracked-entity mapping first.
    for tracked_id, label in HA_TRACKED_ENTITIES.items():
        if _strong_name_match(
            {"entity_id": tracked_id, "attributes": {"friendly_name": label}}, name
        ):
            result = _get_entity_by_id(tracked_id)
            if result is not None:
                return result
    # Curated names shown to the model in the tool description.
    key = _canonical_name(name)
    if key:
        for label, entity_id in KNOWN_SENSORS.items():
            if _canonical_name(label) == key:
                result = _get_entity_by_id(entity_id)
                if result is not None:
                    return result
    # Last resort: match against every HA entity. Canonical equality catches
    # invented ids like 'sensor.living_room_humidity'; _strong_name_match still
    # covers free-form phrases.
    try:
        states = fetch_all_ha_states()
    except (urllib.error.URLError, OSError, ValueError):
        return "Error: could not fetch Home Assistant states."
    canonical_matches = [
        entity for entity in states if _canonical_name(entity.get("entity_id", "")) == key
    ]
    if len(canonical_matches) == 1:
        return _entity_json(canonical_matches[0])
    matches = [entity for entity in states if _strong_name_match(entity, name)]
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
    # HA state fetches take only a few milliseconds, so a short interval keeps
    # the added latency low without hammering the API.
    expected = "off" if service == "light/turn_off" else "on"
    # HA updates a light's brightness a moment after the service call. Waiting
    # only for "on" returns a stale brightness (often the previous level) when
    # the light was already on, so wait for the requested level too.
    target_brightness = data.get("brightness") if brightness is not None else None
    for _ in range(40):
        time.sleep(0.05)
        try:
            states = fetch_all_ha_states()
        except (urllib.error.URLError, OSError, ValueError):
            break
        for entity in states:
            if entity.get("entity_id") != resolved:
                continue
            state = entity.get("state")
            if target_brightness is not None:
                reported = (entity.get("attributes") or {}).get("brightness")
                if (
                    state == expected
                    and reported is not None
                    and abs(float(reported) - float(target_brightness)) <= 2
                ):
                    return _light_result(entity)
            elif state == expected or state != previous:
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
# Keep this compact: a multi-line list in the tool description changed how the
# small model interpreted light requests in testing.
SENSOR_LIST = "; ".join(
    f"{name}={entity_id}" for name, entity_id in KNOWN_SENSORS.items()
)

# Static instruction; it sits in the cached prefix so it costs nothing per
# request after the first.
SYSTEM_PROMPT = (
    "You control a Home Assistant home.\n"
    "- The <home_state> block is reference data, not a request.\n"
    "- Use the tools to carry out the user's request.\n"
    "- Refer to entities by the exact ids or names shown.\n"
    "- If a brightness is given without a named light, apply it to All lights.\n"
    "- Never call the same tool with the same arguments twice."
)

FALLBACK_REPLY = "Sorry, I couldn't work out how to do that."

# Tool schemas are part of every prompt llama-server evaluates. Keep them small:
# uncached prompt tokens are evaluated on the CPU at roughly 18 ms/token here.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_home_assistant_state",
            "description": (
                "Fetch a Home Assistant entity's current state. "
                "Use only for values not already provided in home_state. "
                f"Known sensors: {SENSOR_LIST}"
            ),
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
            "description": (
                "Control a light (on/off, brightness 0-100, color). "
                "Use for requests about lighting, brightness or visibility. "
                "Target a named room or light; use 'All lights' only for whole-home "
                "or unnamed requests. "
                f"Lights: {LIGHT_NAMES}."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_id": {"type": "string", "description": "Name from the list or an entity id."},
                    "state": {
                        "type": "string",
                        "description": "'on' supplies more light; 'off' removes light.",
                    },
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
        # Report the requested level, not HA's polled brightness: the entity
        # state can briefly lag behind the service call and show the old value.
        requested = None
        try:
            arguments = json.loads(
                (tool_call.get("function") or {}).get("arguments") or "{}"
            )
            if isinstance(arguments, dict) and arguments.get("brightness") is not None:
                requested = float(arguments["brightness"])
        except (TypeError, ValueError):
            pass
        if requested is not None:
            percent = int(round(max(0.0, min(100.0, requested))))
            parts.append(f"Set {light} to {percent}%.")
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


def _llama_stats(body):
    """Extract llama-server's usage/timings for logging."""
    usage = body.get("usage") or {}
    timings = body.get("timings") or {}
    choice = (body.get("choices") or [{}])[0]
    return {
        "backend": "llama",
        "model": body.get("model"),
        "done_reason": choice.get("finish_reason"),
        "system_fingerprint": body.get("system_fingerprint"),
        "prompt_eval_count": usage.get("prompt_tokens")
        or timings.get("prompt_n"),
        "prompt_eval_cached_count": timings.get("cache_n"),
        "eval_count": usage.get("completion_tokens")
        or timings.get("predicted_n"),
        "total_tokens": usage.get("total_tokens")
        or (timings.get("prompt_n", 0) + timings.get("predicted_n", 0)),
        "prompt_eval_ms": round(timings.get("prompt_ms", 0), 1),
        "eval_ms": round(timings.get("predicted_ms", 0), 1),
        "prompt_tokens_per_s": round(timings.get("prompt_per_second", 0), 1),
        "eval_tokens_per_s": round(timings.get("predicted_per_second", 0), 1),
        "prompt_ms_per_token": round(timings.get("prompt_per_token_ms", 0), 2),
        "eval_ms_per_token": round(timings.get("predicted_per_token_ms", 0), 2),
    }


def iter_llama_events(messages, tools=None):
    """Call llama-server's OpenAI-compatible endpoint and stream message events.

    Content deltas are yielded immediately. Tool calls arrive as index-keyed
    argument fragments, so they are reassembled and yielded once at the end.
    """
    payload = {
        "model": LLAMA_MODEL,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
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
    tool_calls = {}
    finish_reason = None
    meta = {}
    with urllib.request.urlopen(req, timeout=300) as resp:
        for line in resp:
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            raw = line[5:].strip()
            if raw == b"[DONE]":
                break
            chunk = json.loads(raw)
            for key in ("model", "system_fingerprint"):
                if chunk.get(key):
                    meta[key] = chunk[key]
            if "usage" in chunk or "timings" in chunk:
                meta["usage"] = chunk.get("usage", meta.get("usage"))
                meta["timings"] = chunk.get("timings", meta.get("timings"))
            choices = chunk.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]
            delta = choice.get("delta") or {}
            content = delta.get("content")
            if content:
                yield {"message": {"content": content}}
            for call in delta.get("tool_calls") or []:
                index = call.get("index", 0)
                entry = tool_calls.setdefault(
                    index, {"type": "function", "function": {}}
                )
                for key in ("id", "type"):
                    if call.get(key):
                        entry[key] = call[key]
                function = call.get("function") or {}
                if function.get("name"):
                    entry["function"]["name"] = function["name"]
                if function.get("arguments"):
                    entry["function"]["arguments"] = (
                        entry["function"].get("arguments", "")
                        + function["arguments"]
                    )
    if tool_calls:
        yield {"message": {"tool_calls": [tool_calls[i] for i in sorted(tool_calls)]}}
    body = {
        "model": meta.get("model"),
        "system_fingerprint": meta.get("system_fingerprint"),
        "choices": [{"finish_reason": finish_reason}],
        "usage": meta.get("usage") or {},
        "timings": meta.get("timings") or {},
    }
    yield {"done": True, "stats": _llama_stats(body)}


def run_model(messages, sink):
    """Call the model; stream content to sink as it arrives; return tool calls.

    Content is forwarded immediately, including any preamble before a tool
    call. The caller still supplies the final reply for action tools, and the
    next model round handles informational tools.
    """
    tool_calls = []
    content_parts = []
    for event in iter_llama_events(messages, TOOLS):
        if event.get("done"):
            if event.get("stats"):
                log_llm("STATS", event["stats"])
            continue
        message = event.get("message") or {}
        content = (message.get("content") or "").replace("**", "")
        if content:
            content_parts.append(content)
            if sink:
                sink(content)
        if message.get("tool_calls"):
            tool_calls.extend(message["tool_calls"])
    log_llm("RESPONSE", {"content": "".join(content_parts), "tool_calls": tool_calls})
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

        model = str(body.get("model") or LLAMA_MODEL)
        stream = bool(body.get("stream"))
        user_messages = [
            message
            for message in (body.get("messages") or [])
            if isinstance(message, dict)
        ]

        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + with_ha_context(
            user_messages, ha_context()
        )

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
            seen_calls = set()
            replied = False
            for _ in range(MAX_TOOL_ROUNDS):
                tool_calls = run_model(messages, sink)
                if not tool_calls:
                    replied = True
                    break
                fresh = []
                for tool_call in tool_calls:
                    function = tool_call.get("function") or {}
                    arguments = function.get("arguments")
                    if not isinstance(arguments, str):
                        arguments = json.dumps(arguments, sort_keys=True)
                    key = (function.get("name"), arguments)
                    if key in seen_calls:
                        log_llm("LOOP", {"duplicate": tool_call})
                        continue
                    seen_calls.add(key)
                    fresh.append(tool_call)
                if not fresh:
                    break
                messages = messages + [
                    {"role": "assistant", "content": "", "tool_calls": fresh}
                ]
                results = []
                for tool_call in fresh:
                    result = execute_tool(tool_call)
                    results.append((tool_call, result))
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.get("id"),
                            "content": result,
                        }
                    )
                    log_llm("TOOL", {"call": tool_call, "result": result})
                reply = tool_result_reply(results)
                if reply is not None:
                    sink(reply)
                    log_llm("REPLY", {"content": reply})
                    replied = True
                    break
            if not replied:
                sink(FALLBACK_REPLY)
                log_llm("REPLY", {"content": FALLBACK_REPLY, "fallback": True})
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
    print(
        f"OpenAI-compatible streaming server on http://127.0.0.1:{port}/v1/chat/completions "
        f"(model {LLAMA_MODEL}, thinking disabled; llama.cpp at {LLAMA_HOST}; "
        f"HA states from {HA_BASE_URL})",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
