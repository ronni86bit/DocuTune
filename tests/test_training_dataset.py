"""Tokenization + label masking with a stub tokenizer (no torch needed for
build_features; SFTDataset only needs torch at construction time)."""

import pytest

from docutune.training.dataset import build_features
from docutune.utils.io import write_jsonl


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


# ---------------------------------------------------------------------------
# SFTDataset regression tests (torch-gated; CI installs no torch and skips)
# Regression guard for: "TypeError: Dataset() takes no arguments" - the
# smoke-test path must never try to instantiate torch.utils.data.Dataset.
# ---------------------------------------------------------------------------


def _tiny_jsonl(tmp_path):
    rows = [
        {"id": "r1", "text": "Jane Doe\nEngineer resume", "target": _gold()},
        {"id": "r2", "text": "John Smith\nAnalyst resume", "target": _gold()},
    ]
    path = tmp_path / "train.jsonl"
    write_jsonl(path, rows)
    return str(path)


class TestSFTDatasetRegression:
    def test_construction_len_getitem(self, tmp_path):
        pytest.importorskip("torch")

        from docutune.training.dataset import SFTDataset

        dataset = SFTDataset(_tiny_jsonl(tmp_path), StubTokenizer(), max_length=1024)
        assert len(dataset) == 2
        item = dataset[0]
        assert set(item.keys()) == {"input_ids", "attention_mask", "labels"}
        assert len(item["input_ids"]) == len(item["attention_mask"]) == len(item["labels"])
        # prompt masked, target supervised (same contract as build_features)
        first_supervised = next(i for i, v in enumerate(item["labels"]) if v != -100)
        assert all(v == -100 for v in item["labels"][:first_supervised])
        assert any(v != -100 for v in item["labels"])

    def test_as_torch_dataset_returns_itself(self, tmp_path):
        pytest.importorskip("torch")

        from docutune.training.dataset import SFTDataset

        dataset = SFTDataset(_tiny_jsonl(tmp_path), StubTokenizer(), max_length=1024)
        # must be the SAME instance - never torch.utils.data.Dataset(features)
        assert dataset.as_torch_dataset() is dataset

    def test_torch_dataloader_accepts_it(self, tmp_path):
        """DataLoader is what the Trainer drives underneath; if it iterates
        every sample, the Trainer path works without ever instantiating
        Dataset. (Collation/padding into tensors is DataCollatorForSeq2Seq's
        job in the real pipeline, so only protocol-level guarantees are
        asserted here.)"""
        pytest.importorskip("torch")

        from torch.utils.data import DataLoader

        from docutune.training.dataset import SFTDataset

        dataset = SFTDataset(_tiny_jsonl(tmp_path), StubTokenizer(), max_length=1024)
        batches = list(DataLoader(dataset, batch_size=1))
        assert len(batches) == 2
        assert set(batches[0].keys()) == {"input_ids", "attention_mask", "labels"}
        assert set(batches[1].keys()) == {"input_ids", "attention_mask", "labels"}

    def test_max_samples_cap(self, tmp_path):
        pytest.importorskip("torch")

        from docutune.training.dataset import SFTDataset

        dataset = SFTDataset(_tiny_jsonl(tmp_path), StubTokenizer(), max_length=1024,
                             max_samples=1)
        assert len(dataset) == 1

    def test_smoke_path_never_instantiates_torch_dataset(self):
        """CI-friendly structural tripwire (no torch required): as_torch_dataset
        must return self, and nothing in the module may call Dataset(...) as a
        constructor. AST-based, so docstring prose mentioning the old bug is
        fine."""
        import ast
        import inspect

        from docutune.training import dataset as dataset_module

        tree = ast.parse(inspect.getsource(dataset_module))
        as_torch = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "as_torch_dataset"
        )
        returns = [n for n in ast.walk(as_torch) if isinstance(n, ast.Return)]
        assert len(returns) == 1 and isinstance(returns[0].value, ast.Name)
        assert returns[0].value.id == "self"

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id == "Dataset":
                    raise AssertionError("module instantiates Dataset(...) directly")
                if isinstance(func, ast.Attribute) and func.attr == "Dataset":
                    raise AssertionError("module instantiates <...>.Dataset(...)")
