import pytest
from peft_bench.tasks import make_prompt_text, label_letter


def test_sst2_prompt():
    x = make_prompt_text("sst2", {"sentence": "I loved this film"})
    assert "sentiment" in x.lower()
    assert "I loved this film" in x
    assert "A = negative. B = positive" in x


def test_qnli_prompt():
    x = make_prompt_text("qnli", {"question": "Who?", "sentence": "Alice."})
    assert "Question: Who?" in x
    assert "Sentence: Alice." in x


def test_mrpc_label_mapping():
    assert label_letter(0) == "A"
    assert label_letter(1) == "B"
    with pytest.raises(ValueError):
        label_letter(-1)
