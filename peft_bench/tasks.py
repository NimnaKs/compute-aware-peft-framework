"""GLUE task schemas and deterministic two-choice prompts.

Important: labels A/B are *code words*, not sentiment words; this avoids
synonym parsing and produces a precise, identical inference protocol for all runs.
"""
from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class TaskSpec:
    name: str
    field_names: tuple[str, ...]
    instruction: str
    labels: tuple[str, str]  # dataset label 0 -> A; dataset label 1 -> B


TASKS: Mapping[str, TaskSpec] = {
    "sst2": TaskSpec(
        "sst2", ("sentence",),
        "Classify the sentiment of the movie review. A = negative. B = positive.",
        ("negative", "positive"),
    ),
    "mrpc": TaskSpec(
        "mrpc", ("sentence1", "sentence2"),
        "Do both sentences have the same meaning? A = no. B = yes.",
        ("not_equivalent", "equivalent"),
    ),
    "qnli": TaskSpec(
        "qnli", ("question", "sentence"),
        "Does the sentence contain the answer to the question? "
        "A = yes (entailment). B = no (not entailment).",
        ("entailment", "not_entailment"),
    ),
}


def make_prompt_text(task: str, row: dict) -> str:
    spec = TASKS[task]
    lines = [spec.instruction, "Reply using exactly one letter: A or B."]
    for field in spec.field_names:
        lines.append(f"{field.replace('_', ' ').capitalize()}: {row[field]}")
    return "\n".join(lines)


def label_letter(label: int) -> str:
    if int(label) not in (0, 1):
        raise ValueError(f"Expected GLUE binary label 0 or 1; received {label}")
    return ("A", "B")[int(label)]
