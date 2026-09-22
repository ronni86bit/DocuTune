"""Tokenization + label masking with a stub tokenizer (no torch needed for
build_features; SFTDataset only needs torch at construction time)."""

from docutune.training.dataset import build_features


class StubTokenizer:
    """Word-level tokenizer mimicking the interface build_features uses.

    The leading space in eos_token keeps it a separate word-level token
    (mirroring how real tokenizers emit a standalone EOS id).
    """

    eos_token = " <eos>"

    def __init__(self):
        self.vocab: dict[str, int] = {}

    def _encode(self, text: str) -> list[int]:
        ids = []
        for word in text.split():
            if word not in self.vocab:
                self.vocab[word] = len(self.vocab) + 1
            ids.append(self.vocab[word])
        return ids

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        parts = [f"<|{m['role']}|>\n{m['content']}\n" for m in messages]
        if add_generation_prompt:
            parts.append("<|assistant|>\n")
        return "".join(parts)

    def __call__(self, text: str, add_special_tokens: bool = True):
        return {"input_ids": self._encode(text)}


def _gold():
    return {
        "name": "Jane Doe",
        "email": None,
        "phone": None,
        "location": "Pune, Maharashtra",
        "summary": "Engineer.",
        "skills": ["Python"],
        "education": [],
        "experience": [{
            "title": "Engineer", "company": "Acme", "location": None,
            "start_date": "2020-01", "end_date": None, "current": True,
            "responsibilities": ["Built things"],
        }],
        "projects": [],
        "certifications": [],
    }


def test_prompt_tokens_masked():
    tokenizer = StubTokenizer()
    features = build_features(tokenizer, "Jane Doe\nEngineer resume text", _gold(),
                              max_length=1024)
    labels = features["labels"]
    input_ids = features["input_ids"]
    assert len(labels) == len(input_ids)
    prompt_len = features["prompt_tokens"]
    assert all(label == -100 for label in labels[:prompt_len])
    assert all(label != -100 for label in labels[prompt_len:])
    # labels are aligned with input ids
    assert labels[prompt_len:] == input_ids[prompt_len:]


def test_target_includes_eos():
    tokenizer = StubTokenizer()
    features = build_features(tokenizer, "resume text", _gold(), max_length=1024)
    eos_id = tokenizer._encode(" <eos>")[0]
    assert features["input_ids"][-1] == eos_id
    assert features["labels"][-1] == eos_id  # the eos token is supervised


def test_truncation_to_max_length():
    tokenizer = StubTokenizer()
    long_text = " ".join(["word"] * 5000)
    features = build_features(tokenizer, long_text, _gold(), max_length=128)
    assert len(features["input_ids"]) == 128
    assert len(features["labels"]) == 128
    assert features["truncated"] is True


def test_attention_mask():
    tokenizer = StubTokenizer()
    features = build_features(tokenizer, "resume", _gold(), max_length=1024)
    assert features["attention_mask"] == [1] * len(features["input_ids"])


def test_masks_are_loss_ready(small_dataset):
    """Real generated examples must have supervised tokens after masking."""
    tokenizer = StubTokenizer()
    for ex in small_dataset["examples"][:4]:
        features = build_features(tokenizer, ex["text"], ex["target"], max_length=2048)
        supervised = [label for label in features["labels"] if label != -100]
        assert len(supervised) > 10
