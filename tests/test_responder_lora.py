from __future__ import annotations

import hashlib

from scripts.train_responder_lora import (
    DEV_SESSION_HASHES,
    MAX_RESPONSE_TOKENS,
    MAX_SEQUENCE_TOKENS,
    MAX_USER_TOKENS,
    SPLIT_SEED,
    collate_dialogues,
    eligible_complete_response_rows,
    encode_dialogue,
    first_response_sentence,
    session_order_key,
    split_development_rows,
)


class FakeTokenizer:
    pad_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [int(part) for part in text.split()]

    def decode(self, values: list[int], skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return " ".join(str(value) for value in values)

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        return_dict: bool,
    ) -> dict[str, list[int]]:
        del tokenize, return_dict
        ids = [900, 901]
        ids.extend(int(part) for part in messages[1]["content"].split())
        ids.append(902)
        if len(messages) == 3:
            ids.extend(int(part) for part in messages[2]["content"].split())
            ids.append(903)
        elif not add_generation_prompt:
            raise AssertionError("expected generation prompt for user-only messages")
        return {"input_ids": ids}


def _find_preimage(prefix: str, target: str) -> str:
    for index in range(1_000_000):
        candidate = f"session-{index}"
        if hashlib.sha256(candidate.encode()).hexdigest() == target:
            return candidate
    raise AssertionError(f"fixture has no preimage for {prefix}")


def test_encode_dialogue_masks_prompt_and_honors_caps() -> None:
    tokenizer = FakeTokenizer()
    row = {
        "text": " ".join(str(index) for index in range(MAX_USER_TOKENS + 5)),
        "response_text": " ".join(str(index + 1000) for index in range(MAX_RESPONSE_TOKENS + 5)),
    }
    encoded = encode_dialogue(tokenizer, row)
    assert len(encoded["input_ids"]) <= MAX_SEQUENCE_TOKENS
    assert encoded["input_ids"][2] == 5
    first_target = encoded["labels"].index(next(v for v in encoded["labels"] if v != -100))
    assert all(value == -100 for value in encoded["labels"][:first_target])
    assert encoded["labels"][first_target] == 1000
    assert sum(value != -100 for value in encoded["labels"]) == MAX_RESPONSE_TOKENS + 1


def test_first_response_sentence_uses_first_nonempty_sentence() -> None:
    assert first_response_sentence("  پاسخ نخست؟ ادامه پاسخ.  ") == "پاسخ نخست؟"
    assert first_response_sentence("پاسخ بدون نشانه") == "پاسخ بدون نشانه"


def test_complete_response_filter_excludes_overlong_target() -> None:
    tokenizer = FakeTokenizer()
    rows = [
        {"response_text": "1 2 3"},
        {"response_text": " ".join(str(index) for index in range(MAX_RESPONSE_TOKENS + 1))},
    ]
    assert eligible_complete_response_rows(tokenizer, rows) == [rows[0]]


def test_collator_masks_padding() -> None:
    batch = collate_dialogues(
        [
            {"input_ids": [1, 2, 3], "labels": [-100, 2, 3]},
            {"input_ids": [4, 5], "labels": [-100, 5]},
        ],
        pad_token_id=0,
    )
    assert batch["input_ids"].tolist() == [[1, 2, 3], [4, 5, 0]]
    assert batch["labels"].tolist() == [[-100, 2, 3], [-100, 5, -100]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]


def test_development_selection_is_hash_ordered_and_grouped(monkeypatch) -> None:
    sessions = [f"session-{index}" for index in range(5)]
    ordered = sorted(sessions, key=session_order_key)
    expected_hashes = tuple(hashlib.sha256(value.encode()).hexdigest() for value in ordered[:2])
    monkeypatch.setattr("scripts.train_responder_lora.DEV_SESSION_HASHES", expected_hashes)
    rows = [
        {"session_id": session, "utt_id": f"u-{index}"} for index, session in enumerate(sessions)
    ]
    selected = split_development_rows(rows)
    assert {row["session_id"] for row in selected} == set(ordered[:2])
    assert len(DEV_SESSION_HASHES) == 2
    assert SPLIT_SEED == "20260906-responder-lora-v1"
