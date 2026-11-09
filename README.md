# PR course project A01 about LLM privacy.
This project is an experiment for detecting and evaluating privacy leakage of large language model agents.

## File Structure
├── metrics.py       # Calculate evaluation metrics, confusion matrix and statistical indicators \n
├── prompts.py       # Prompt templates used in experiments \n
├── rejudge.py       # Script for re-judging results \n
├── run.py           # Main entry of the experiment: run tests, call models and generate outputs \n
├── results/         # Directory for evaluation outputs \n
│   ├── confusion_matrix.csv  # Confusion matrix \n
│   ├── leak_labels.csv       # Leakage labels of samples \n
│   ├── metadata.json         # Experiment metadata \n 
│   └── summary.csv           # Summary statistics of the experiment \n
└── runs/            # Directory for running logs \n
├── generations.jsonl     # Model generation records \n
└── judgments.jsonl       # Judgment records \n
