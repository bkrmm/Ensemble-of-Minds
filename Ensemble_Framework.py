from typing_extensions import TypedDict, Annotated
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from langchain_google_genai import ChatGoogleGenerativeAI
import os
import datasets

class State(TypedDict):
    user_input: str 
    messages: list[HumanMessage | AIMessage]
    feedback: str
    proposed_solution: str
    verification: str
    refined_solution: str
    final_answer: str

builder = StateGraph(State)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0, google_api_key="")

# System prompts for each role
PROPOSER_PROMPT = "You are the Proposer. Given a math word problem, propose a detailed step-by-step solution."
VERIFIER_PROMPT = "You are the Verifier. Critically evaluate the proposed solution for correctness. Point out any errors or confirm its validity."
REFINER_PROMPT = "You are the Refiner. If the Verifier found issues, revise and improve the solution. If not, restate the solution more clearly."
SYNTHESIZER_PROMPT = "You are the Synthesizer. Given the refined solution, extract and present the final answer in a clear, concise sentence."

# Role functions
def proposer_node(state: State):
    user_input = state["user_input"]
    messages = [HumanMessage(content=user_input)]
    system_message = HumanMessage(content=PROPOSER_PROMPT)
    response = llm.invoke([system_message] + messages)
    return {
        **state,
        "messages": state["messages"] + [response],
        "proposed_solution": response.content
    }

def verifier_node(state: State):
    proposed_solution = state["proposed_solution"]
    system_message = HumanMessage(content=VERIFIER_PROMPT)
    response = llm.invoke([system_message, HumanMessage(content=proposed_solution)])
    return {
        **state,
        "messages": state["messages"] + [response],
        "verification": response.content
    }

def refiner_node(state: State):
    verification = state["verification"]
    proposed_solution = state["proposed_solution"]
    system_message = HumanMessage(content=REFINER_PROMPT)
    response = llm.invoke([
        system_message,
        HumanMessage(content=f"Proposed Solution: {proposed_solution}\nVerification: {verification}")
    ])
    return {
        **state,
        "messages": state["messages"] + [response],
        "refined_solution": response.content
    }

def synthesizer_node(state: State):
    refined_solution = state["refined_solution"]
    system_message = HumanMessage(content=SYNTHESIZER_PROMPT)
    response = llm.invoke([system_message, HumanMessage(content=refined_solution)])
    return {
        **state,
        "messages": state["messages"] + [response],
        "final_answer": response.content
    }

# Load GSM8K dataset
def load_gsm8k(split="test"):
    dataset = datasets.load_dataset("gsm8k", "main", split=split)
    return dataset

# Load HumanEval dataset
def load_humaneval(split="test"):
    dataset = datasets.load_dataset("openai_humaneval", split=split)
    return dataset

# Build the LangGraph workflow
builder.add_node("proposer", proposer_node)
builder.add_node("verifier", verifier_node)
builder.add_node("refiner", refiner_node)
builder.add_node("synthesizer", synthesizer_node)

builder.add_edge(START, "proposer")
builder.add_edge("proposer", "verifier")
builder.add_edge("verifier", "refiner")
builder.add_edge("refiner", "synthesizer")
builder.add_edge("synthesizer", END)

graph = builder.compile()

def benchmark_gsm8k(graph, num_samples=20):
    dataset = load_gsm8k(split="test")
    correct = 0
    total = 0
    for i, sample in enumerate(dataset):
        if i >= num_samples:
            break
        problem = sample["question"]
        gold_answer = sample["answer"]
        # Initialize state
        state = {
            "user_input": problem,
            "messages": [],
            "feedback": "",
            "proposed_solution": "",
            "verification": "",
            "refined_solution": "",
            "final_answer": ""
        }
        # Run through the graph
        result = graph.invoke(state)
        pred_answer = result["final_answer"]
        # Extract the numeric answer from both gold and pred for fair comparison
        import re
        def extract_number(s):
            match = re.search(r"[-+]?[.]?\d+([.,]\d+)?", s.replace(",", ""))
            return match.group(0) if match else None
        gold_num = extract_number(gold_answer)
        pred_num = extract_number(pred_answer)
        is_correct = gold_num == pred_num
        print(f"Sample {i+1}:")
        print(f"Q: {problem}")
        print(f"Gold: {gold_answer}")
        print(f"Pred: {pred_answer}")
        print(f"Correct: {is_correct}\n{'-'*40}")
        if is_correct:
            correct += 1
        total += 1
    accuracy = correct / total if total > 0 else 0
    print(f"\nAccuracy on {total} samples: {accuracy*100:.2f}%")

def benchmark_humaneval(graph, num_samples=10):
    dataset = load_humaneval()
    correct = 0
    total = 0
    for i, sample in enumerate(dataset):
        if i >= num_samples:
            break
        problem = sample["prompt"]
        gold_solution = sample["canonical_solution"].strip()
        # Initialize state
        state = {
            "user_input": problem,
            "messages": [],
            "feedback": "",
            "proposed_solution": "",
            "verification": "",
            "refined_solution": "",
            "final_answer": ""
        }
        # Run through the graph
        result = graph.invoke(state)
        pred_solution = result["final_answer"].strip()
        is_correct = pred_solution == gold_solution
        print(f"Sample {i+1}:")
        print(f"Prompt:\n{problem}")
        print(f"Gold Solution:\n{gold_solution}")
        print(f"Predicted Solution:\n{pred_solution}")
        print(f"Exact Match: {is_correct}\n{'-'*40}")
        if is_correct:
            correct += 1
        total += 1
    accuracy = correct / total if total > 0 else 0
    print(f"\nExact match accuracy on {total} HumanEval samples: {accuracy*100:.2f}%")

# Update main to allow running either benchmark
def main():
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "humaneval":
        benchmark_humaneval(graph, num_samples=10)
    else:
        benchmark_gsm8k(graph, num_samples=20)

if __name__ == "__main__":
    main()

