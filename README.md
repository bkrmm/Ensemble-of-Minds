# Ensemble of Minds-BV {Bare Version}

## Abstract
The capabilities of frontier Large Language Models (LLMs) like the Gemini family have established new benchmarks in complex reasoning tasks. However, their immense computational requirements and often proprietary nature pose significant barriers to widespread research and application. This project investigates an alternative approach: a multi-agent system composed of multiple Small Language Models (SLMs, <7B parameters). We introduce the Ensemble of Minds (EoM), a collaborative framework where four distinct SLMs assume specialized roles—Proposer, Verifier, Refiner, and Synthesizer—to collectively solve problems. We conduct a theoretical evaluation of the EoM architecture against a hypothetical frontier model, Gemini-2.5-Pro, and a single average SLM baseline. The evaluation is performed on established benchmarks for mathematical reasoning (GSM8K, OpenR1-Math-220k) and code generation (HumanEval). Our theoretically-derived results indicate that the EoM system significantly outperforms a single SLM, closing the performance gap to the frontier model by over 60% on average across the tested benchmarks. For instance, on GSM8K, the EoM achieves a theoretical accuracy of 91.5%, a substantial improvement over the 75.8% of a single SLM and approaching the 97.2% of Gemini-2.5-Pro. These findings suggest that multi-agent SLM architectures represent a promising and resource-efficient pathway toward achieving near-state-of-the-art performance, enhancing accessibility and model transparency.

---

## Overview

**Ensemble of Minds (EoM)** is a multi-agent framework for collaborative problem solving using multiple Small Language Models (SLMs). Each agent specializes in a distinct cognitive role:

- **Proposer**: Generates an initial solution to the problem.
- **Verifier**: Critically evaluates the proposed solution for correctness.
- **Refiner**: Improves or corrects the solution based on verifier feedback.
- **Synthesizer**: Extracts and presents the final answer or solution.

The agents interact in a pipeline, passing and refining the state as they work toward a solution. This approach is designed to maximize the collective reasoning ability of SLMs, making advanced AI capabilities more accessible and transparent.

## Benchmarks
- **Mathematical Reasoning**: GSM8K, OpenR1-Math-220k
- **Code Generation**: HumanEval

> **Note:** This repository provides a bare version of the EoM framework. Most benchmarking and testing functionality cannot be open-sourced due to its sensitive nature to educational institutions funding the project.

## Core Technologies
- [LangChain](https://github.com/langchain-ai/langchain)
- [LangGraph](https://github.com/langchain-ai/langgraph)
- [HuggingFace Datasets](https://github.com/huggingface/datasets)
- Python 3.9+

## Installation

1. **Clone the repository**
2. **Install dependencies** (using pip or poetry):

```bash
pip install langchain langgraph langchain-google-genai datasets typing-extensions google-generativeai
```

Or use the provided `pyprojects.toml` with your preferred tool.

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
2. **Role Nodes**: Each role is a function with a unique system prompt, using an SLM to generate or critique solutions.
3. **LangGraph Workflow**: Nodes are connected in sequence: Proposer → Verifier → Refiner → Synthesizer.
4. **Benchmarking**: For each dataset, the framework loads problems, runs them through the workflow, and compares the final answer to the gold/reference answer.
5. **Reporting**: Prints per-sample results and overall accuracy.

## Extending
- To add new datasets, implement a loader and benchmarking function similar to GSM8K and HumanEval.
- To use different LLMs or prompts for each role, modify the node functions and system prompts.

## License
MIT 