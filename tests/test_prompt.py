"""Canonical prompt + target serialization tests."""

import json

from docutune.config import PROMPT_VERSION as CONFIG_PROMPT_VERSION
from docutune.inference.prompt import (
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    build_messages,
    build_prompt_text,
    build_user_message,
    canonicalize_target,
    serialize_target,
)


def test_prompt_version_shared():
    assert PROMPT_VERSION == "1.0"
    assert PROMPT_VERSION == CONFIG_PROMPT_VERSION  # single source of truth


def test_system_prompt_mentions_null_and_lists():
    assert "null" in SYSTEM_PROMPT
    assert "[]" in SYSTEM_PROMPT
    assert "Do not invent" in SYSTEM_PROMPT


def test_user_message_wraps_resume():
    message = build_user_message("John Doe\nEngineer")
    assert "<RESUME>" in message
    assert "</RESUME>" in message
    assert "John Doe" in message
    assert message.index("Extract the structured information") < message.index("<RESUME>")


def test_build_messages_roles():
    messages = build_messages("some resume")
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "some resume" in messages[1]["content"]


def test_serialize_target_deterministic_and_canonical(small_dataset):
    target = small_dataset["examples"][0]["target"]
    first = serialize_target(target)
    second = serialize_target(target)
    assert first == second
    # Key order matches the schema contract
    assert list(json.loads(first).keys()) == [
        "name", "email", "phone", "location", "summary", "skills",
        "education", "experience", "projects", "certifications",
    ]


def test_canonicalize_fills_defaults(small_dataset):
    raw = {"name": "X", "skills": ["Python"]}
    canonical = canonicalize_target(raw)
    assert canonical["education"] == []
    assert canonical["projects"] == []
    assert canonical["email"] is None


def test_prompt_text_uses_chat_template():
    class StubTokenizer:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
            parts = [f"<|{msg['role']}|>{msg['content']}" for msg in messages]
            if add_generation_prompt:
                parts.append("<|assistant|>")
            return "".join(parts)

    text = build_prompt_text(StubTokenizer(), "resume body")
    assert text.startswith("<|system|>")
    assert text.endswith("<|assistant|>")
    assert "resume body" in text
