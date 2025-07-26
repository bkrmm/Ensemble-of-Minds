# Ensemble of Minds

A LangChain-based framework that leverages an ensemble of four roles—**Proposer**, **Verifier**, **Refiner**, and **Synthesizer**—to collaboratively solve problems. The framework is designed for benchmarking on reasoning and code generation datasets such as GSM8K and HumanEval.

## Architecture

- **Proposer**: Generates an initial solution to the problem.
- **Verifier**: Critically evaluates the proposed solution for correctness.
- **Refiner**: Improves or corrects the solution based on verifier feedback.
- **Synthesizer**: Extracts and presents the final answer or solution.

The workflow is orchestrated using [LangGraph](https://github.com/langchain-ai/langgraph), with each role implemented as a node in the graph. The state is passed and updated between nodes, allowing for iterative refinement and validation.

## Core Technologies
- [LangChain](https://github.com/langchain-ai/langchain)
- [LangGraph](https://github.com/langchain-ai/langgraph)
- [LangChain Google GenAI](https://github.com/langchain-ai/langchain-google-genai) (Gemini 2.5 Flash)
- [HuggingFace Datasets](https://github.com/huggingface/datasets)
- Python 3.9+

## Installation

1. **Clone the repository**
2. **Install dependencies** (using pip or poetry):

```bash
pip install langchain langgraph langchain-google-genai datasets typing-extensions google-generativeai
```

Or use the provided `pyprojects.toml` with your preferred tool.

## Google Gemini API Key

Set your Google Gemini API key in the code or as an environment variable. The framework uses Gemini 2.5 Flash for all four roles.

## Usage

### Benchmark on GSM8K
Runs the ensemble on the GSM8K math word problem dataset and reports accuracy.

```bash
python Ensemble_Framework.py
```

### Benchmark on HumanEval
Runs the ensemble on the HumanEval code generation dataset and reports exact match accuracy.

```bash
python Ensemble_Framework.py humaneval
```

## File Overview

- `Ensemble_Framework.py`: Main framework, role definitions, LangGraph workflow, and benchmarking scripts.
- `pyprojects.toml`: Dependency specification.

## How the Code Works

1. **State Definition**: The state dictionary tracks user input, messages, and intermediate/final solutions.
2. **Role Nodes**: Each role is a function with a unique system prompt, using Gemini to generate or critique solutions.
3. **LangGraph Workflow**: Nodes are connected in sequence: Proposer → Verifier → Refiner → Synthesizer.
4. **Benchmarking**: For each dataset, the framework loads problems, runs them through the workflow, and compares the final answer to the gold/reference answer.
5. **Reporting**: Prints per-sample results and overall accuracy.

## Extending
- To add new datasets, implement a loader and benchmarking function similar to GSM8K and HumanEval.
- To use different LLMs or prompts for each role, modify the node functions and system prompts.

## License
MIT 