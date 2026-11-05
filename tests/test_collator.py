from peft_bench.data import CausalLabelCollator, encode_row, get_label_token_ids


class MockTokenizer:
    pad_token_id = 0
    def encode(self, text, add_special_tokens=False):
        return {"A": [10], "B": [20]}.get(text, [1, 2])
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        return [2, 3, 4, 5]


def test_masked_answer_token_and_collation():
    tok = MockTokenizer()
    ids = get_label_token_ids(tok)
    item = encode_row(tok, "sst2", {"sentence": "Fine", "label": 1}, ids, max_length=32)
    assert item["labels"] == [-100, -100, -100, -100, 20]
    another = {"input_ids": [2, 3, 10], "attention_mask": [1, 1, 1], "labels": [-100, -100, 10]}
    batch = CausalLabelCollator(0)([item, another])
    assert batch["input_ids"].shape == (2, 5)
    assert batch["labels"][1].tolist() == [-100, -100, 10, -100, -100]
    assert batch["attention_mask"][1].tolist() == [1, 1, 1, 0, 0]


def test_deterministic_head_tail_truncation():
    class LongPrompt(MockTokenizer):
        def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
            return list(range(30))
    tok = LongPrompt()
    ex = encode_row(tok, "sst2", {"sentence": "Fine", "label": 0}, (10, 20), max_length=16)
    assert ex["truncated"]
    assert len(ex["input_ids"]) == 16
    assert ex["input_ids"][-2:] == [29, 10]
