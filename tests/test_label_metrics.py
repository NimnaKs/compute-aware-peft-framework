import numpy as np
import torch
from peft_bench.evaluation import evaluate_class_scores, make_logit_preprocessor


def test_preprocess_gathers_previous_token_position():
    # Two examples. Answers occur at token positions 3 and 2.
    labels = torch.tensor([[-100, -100, -100, 20], [-100, -100, 10, -100]])
    scores = torch.zeros(2, 4, 30)
    scores[0, 2, 20] = 9.0  # Correct B for row 0
    scores[1, 1, 10] = 7.0  # Correct A for row 1
    # Distractors at answer positions must NOT affect the prediction.
    scores[0, 3, 10] = 99.0
    scores[1, 2, 20] = 99.0
    result = make_logit_preprocessor((10, 20))(scores, labels)
    assert result.shape == (2, 2)
    assert result.argmax(-1).tolist() == [1, 0]
    metrics = evaluate_class_scores(result.numpy(), labels.numpy(), (10, 20), "sst2")
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0


def test_mrpc_f1():
    labels = np.asarray([[-100, 20], [-100, 20], [-100, 10], [-100, 10]])
    scores = np.asarray([[0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    metrics = evaluate_class_scores(scores, labels, (10, 20), "mrpc")
    assert metrics["accuracy"] == .75
    assert abs(metrics["f1"] - 2/3) < 1e-8
