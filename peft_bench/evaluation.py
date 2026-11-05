"""Metrics for forced binary label-token classification.

Outputs are two next-token scores for A/B, NOT unrestricted generated strings.
Every prediction has a well-defined category.
"""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def make_logit_preprocessor(label_token_ids: tuple[int, int]):
    def preprocess_logits_for_metrics(logits, labels):
        import torch
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        # labels contain one non--100 value at the final (answer) position.
        answer_pos = (labels != -100).long().argmax(dim=1)
        idx = torch.arange(labels.shape[0], device=labels.device)
        # Causal LM predicts the target answer at the immediately PREVIOUS position.
        scores = logits[idx, answer_pos - 1, :]
        return scores[:, [label_token_ids[0], label_token_ids[1]]]
    return preprocess_logits_for_metrics


def evaluate_class_scores(predictions, labels, label_token_ids, task):
    scores = np.asarray(predictions)
    if isinstance(predictions, (tuple, list)):
        scores = np.asarray(predictions[0])
    labels = np.asarray(labels)
    if scores.ndim != 2 or scores.shape[1] != 2:
        raise ValueError(f"Expected [n_examples, 2] scores; got {scores.shape}")
    answer_pos = (labels != -100).argmax(axis=1)
    true_ids = labels[np.arange(len(labels)), answer_pos]
    classes = {int(label_token_ids[0]): 0, int(label_token_ids[1]): 1}
    if any(int(tid) not in classes for tid in true_ids):
        raise ValueError("Unexpected label token ID in evaluation")
    y_true = np.asarray([classes[int(tid)] for tid in true_ids])
    y_pred = scores.argmax(axis=-1)
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    if task == "mrpc":
        result["f1"] = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
    return result


def make_compute_metrics(label_token_ids, task):
    def compute_metrics(eval_pred):
        return evaluate_class_scores(eval_pred.predictions, eval_pred.label_ids, label_token_ids, task)
    return compute_metrics
