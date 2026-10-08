import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from fastapi.testclient import TestClient

import llm
from main import app, limiter
from prompts import build_system_prompt


@pytest.fixture(autouse=True)
def reset_limiter():
    limiter.reset()


client = TestClient(app)


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model"] == llm.MODEL


def test_translate_ok(monkeypatch):
    seen = {}

    def fake(text, direction, intensity):
        seen.update(text=text, direction=direction, intensity=intensity)
        return "Business-Text", 42

    monkeypatch.setattr(llm, "translate", fake)
    r = client.post("/api/translate", json={"text": "  Hallo  ", "direction": "to_business"})
    assert r.status_code == 200
    assert r.json() == {"result": "Business-Text", "direction": "to_business", "tokens": 42}
    assert seen == {"text": "Hallo", "direction": "to_business", "intensity": "medium"}


@pytest.mark.parametrize(
    "payload",
    [
        {"text": "", "direction": "to_plain"},
        {"text": "   ", "direction": "to_plain"},
        {"text": "x" * 501, "direction": "to_plain"},
        {"text": "ok", "direction": "sideways"},
        {"text": "ok", "direction": "to_plain", "intensity": "extreme"},
    ],
)
def test_translate_validation(payload):
    assert client.post("/api/translate", json=payload).status_code == 422


def test_translate_llm_error(monkeypatch):
    def boom(*_):
        raise llm.LLMError("kaputt")

    monkeypatch.setattr(llm, "translate", boom)
    r = client.post("/api/translate", json={"text": "hi", "direction": "to_plain"})
    assert r.status_code == 502
    assert r.json()["detail"] == "kaputt"


def test_translate_timeout(monkeypatch):
    def slow(*_):
        raise llm.LLMTimeout("zu langsam")

    monkeypatch.setattr(llm, "translate", slow)
    r = client.post("/api/translate", json={"text": "hi", "direction": "to_plain"})
    assert r.status_code == 504


def test_prompts_differ_by_direction_and_intensity():
    assert "Klartext" in build_system_prompt("to_plain", "medium")
    assert build_system_prompt("to_business", "low") != build_system_prompt("to_business", "high")


def _fake_client(content, finish_reason="stop", choices=True):
    from types import SimpleNamespace as NS

    response = NS(
        choices=[NS(finish_reason=finish_reason, message=NS(content=content))] if choices else [],
        usage=NS(total_tokens=7),
    )
    return NS(chat=NS(completions=NS(create=lambda **_: response)))


def test_llm_strips_echoed_tags(monkeypatch):
    monkeypatch.setattr(llm, "_get_client", lambda: _fake_client(" <text>Klartext</text> "))
    assert llm.translate("x", "to_plain", "medium") == ("Klartext", 7)


@pytest.mark.parametrize(
    "client_kwargs",
    [
        {"content": "halber Sat", "finish_reason": "length"},
        {"content": None, "choices": False},
        {"content": "  "},
    ],
)
def test_llm_rejects_unusable_responses(monkeypatch, client_kwargs):
    monkeypatch.setattr(llm, "_get_client", lambda: _fake_client(**client_kwargs))
    with pytest.raises(llm.LLMError):
        llm.translate("x", "to_plain", "medium")


def test_llm_requires_api_key(monkeypatch):
    monkeypatch.setattr(llm, "_client", None)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(llm.LLMError, match="OPENROUTER_API_KEY"):
        llm.translate("x", "to_plain", "medium")
