# PR course project A01 about LLM privacy.
This project is an experiment for detecting and evaluating privacy leakage of large language model agents.

## File Structure
```text
metrics.py       # Calculate evaluation metrics, confusion matrix and statistical indicators
prompts.py       # Prompt templates used in experiments
rejudge.py       # Script for re-judging results
run.py           # Main entry of the experiment: run tests, call models and generate outputs
results/         # Directory for evaluation outputs
├─ confusion_matrix.csv  # Confusion matrix
├─ leak_labels.csv       # Leakage labels of samples
├─ metadata.json         # Experiment metadata
└─ summary.csv           # Summary statistics of the experiment
runs/            # Directory for running logs
├─ generations.jsonl     # Model generation records
└─ judgments.jsonl       # Judgment records
