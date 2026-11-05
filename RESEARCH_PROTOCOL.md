# Experimental Protocol (LoRA / QLoRA Phase 1)

**Title:** A Compute-Aware Decision Framework for Parameter-Efficient Fine-Tuning in LLMs.

## Research hypotheses

- **H1 (memory):** At identical rank/targets, QLoRA requires less peak allocated GPU training memory than 16-bit LoRA.
- **H2 (quality):** QLoRA can achieve classification performance comparable to LoRA under a matched example/epoch training budget.
- **H3 (rank):** Increasing rank does not always provide a practically significant performance improvement relative to additional trainable parameters.

These are hypotheses to test; none is assumed to be true.

## Independent variables

1. Method: LoRA, QLoRA.
2. Rank: 4, 8, 16, 32.
3. GLUE task: SST-2, MRPC, QNLI.
4. Random seed: 42, 123, 456.
5. Model family: primary Llama 3.2 3B Instruct, secondary Gemma 2 2B IT (validation stage).

## Controlled variables

Within an apples-to-apples pair: model ID and revision; dataset revision and split; task prompt; next-token label objective; target modules `all-linear`; `alpha = 2r`; dropout 0.05; learning rate 2e-4; AdamW; batch size, gradient accumulation, sequence length, 3-epoch budget, selected GPU hardware. The planned method difference is **16-bit frozen base versus NF4 4-bit frozen base with double quantization**.

BF16 is used for computation if supported, otherwise FP16. QLoRA storage is 4-bit; adapter training is not 4-bit.

## Dataset partitioning

- GLUE `train` -> internal training 90% + checkpoint-selection tuning 10%, stratified by label; fixed split seed 2026.
- GLUE labeled `validation` -> final local evaluation, not used for hyperparameter selection.
- GLUE hidden `test` -> excluded, since official labels are not available for local metric calculation.
- Pilot uses capped examples; primary comparisons use full selected splits.

## Primary outcomes

| Task | Quality | Efficiency |
| --- | --- | --- |
| SST-2 | Accuracy, Macro-F1 | VRAM GiB, train seconds, tokens/sec estimate, trainable params, checkpoint MiB |
| MRPC | Accuracy, positive-class F1, Macro-F1 | Same |
| QNLI | Accuracy, Macro-F1 | Same |

Correlations are outside scope until STS-B regression is implemented.

## Reporting

1. For each model/task/method/rank, mean and standard deviation over three training seeds.
2. Provide both accuracy percentage-point differences and absolute F1 differences.
3. Provide paired rank-matched LoRA/QLoRA memory ratios and time ratios.
4. Report total number of training examples and fraction truncated.
5. Report all failures, OOM, timeouts, and dropped runs.
6. Explain that `peak_cuda_allocated_gib` is PyTorch process-managed allocation, not overall device peak memory.
7. State AWS hourly price and included billing components when cost is reported.
8. Avoid claiming QLoRA is faster or more accurate without measured evidence.

## Next stage after the benchmark

Build a small decision engine: filter configurations that violate VRAM/time/budget constraints; among feasible candidates satisfying a validation-quality threshold, select a Pareto-efficient method. Evaluate the decision rules on held-out GLUE tasks/ranks or a second model architecture. This recommender is **not implemented in phase 1**.
