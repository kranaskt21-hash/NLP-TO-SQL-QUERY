"""Allowlisted diagnostics: never return raw provider messages, URLs or locals."""

import re
from pathlib import Path


def exception_chain(error):
    chain = []
    seen = set()
    pending = [error]
    while pending and len(chain) < 10:
        node = pending.pop(0)
        if id(node) in seen:
            continue
        seen.add(id(node))
        chain.append(node)
        nested = node.__cause__ or node.__context__
        if nested is not None:
            pending.append(nested)
    return chain


def provider_status(error):
    for node in exception_chain(error):
        values = [getattr(node, "code", None), getattr(node, "status_code", None)]
        response = getattr(node, "response", None)
        if response is not None:
            values.append(getattr(response, "status_code", None))
        for value in values:
            if isinstance(value, int) and 100 <= value <= 599:
                return value
            if isinstance(value, str) and re.fullmatch(r"[1-5][0-9]{2}", value):
                return int(value)
    return None


def safe_failure(error):
    chain = exception_chain(error)
    status = provider_status(error)
    # Messages are inspected in memory only to classify fixed, actionable hints.
    # No excerpts, request data, API keys or URLs are returned or logged.
    messages = " ".join(str(e).lower() for e in chain)
    types = [type(e).__name__ for e in chain]
    hints = {
        "authentication": "Check GEMINI_API_KEY in Streamlit Cloud Settings > Secrets and the API key restrictions. Save secrets and reboot the app.",
        "model_unavailable": "The configured Gemini model is unavailable to this key/API endpoint. Use the Gemini connection check and select an accessible generation model.",
        "quota": "Gemini rate limit or quota was exhausted. Check this project/model quota and billing in Google AI Studio, then retry later.",
        "request_configuration": "Gemini rejected the request settings. Check the selected model and its support for structured output/thinking settings.",
        "network": "The SDK could not reach Gemini reliably. Check outbound HTTPS, DNS, proxy settings and TLS certificates on the host.",
        "credentials_backend": "The SDK attempted cloud credential resolution. This application requires Gemini Developer API authentication with GEMINI_API_KEY, not Vertex/enterprise credentials.",
        "runtime_event_loop": "An SDK event-loop/client lifecycle error occurred. Reboot the app and use the pinned dependencies. Connection diagnostics identify the failing adapter.",
        "structured_output": "Gemini did not return valid structured SQL/abstention output. The response may be blocked or incomplete; check the model and output budget.",
        "service": "Gemini reported a server-side failure. Retry later within the call budget.",
        "timeout": "Gemini did not respond before the request deadline. Increase Model request timeout in the sidebar, or try fewer examples/a smaller context. If it persists, try another accessible model or retry later. A timed-out request may still consume provider quota.",
        "runtime": "An SDK/runtime error occurred. Run the Gemini connection check in the sidebar and share its safe diagnostics (never your API key).",
    }
    if "could not resolve api token" in messages or "default credentials" in messages:
        category = "credentials_backend"
    elif status in (401, 403) or "api key not valid" in messages:
        category = "authentication"
    elif status in (404, 410):
        category = "model_unavailable"
    elif status == 429:
        category = "quota"
    elif status == 400:
        category = "request_configuration"
    elif status in (408, 504) or any("timeout" in name.lower() for name in types):
        category = "timeout"
    elif status is not None and status >= 500:
        category = "service"
    elif any(
        word in messages
        for word in (
            "certificate verify",
            "ssl handshake",
            "connection refused",
            "name resolution",
        )
    ) or any(
        any(
            word in name.lower()
            for word in ("timeout", "connecterror", "proxyerror", "transporterror")
        )
        for name in types
    ):
        category = "network"
    elif any(word in messages for word in ("event loop", "client has been closed")):
        category = "runtime_event_loop"
    elif (
        any(name in ("ValidationError", "JSONDecodeError") for name in types)
        or "structured output parse" in messages
    ):
        category = "structured_output"
    else:
        category = "runtime"
    locations = []
    for node in chain:
        tb = node.__traceback__
        frames = []
        while tb is not None:
            name = Path(tb.tb_frame.f_code.co_filename).name
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]{0,79}\.py", name):
                frames.append(
                    {
                        "file": name,
                        "line": tb.tb_lineno,
                        "function": tb.tb_frame.f_code.co_name,
                    }
                )
            tb = tb.tb_next
        locations.extend(frames[-3:])
    return {
        "category": category,
        "http_status": status,
        "exception_types": types,
        "locations": locations,
        "hint": hints[category],
    }


def failure_message(details):
    label = str(details["http_status"] or details["exception_types"][0])
    return (
        "Model request failed ("
        + label
        + "; "
        + details["category"]
        + "). "
        + details["hint"]
    )
