import pandas as pd
import time
from rag_core import BaseballRAGSystem
from typing import List
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def run_evaluation(
        questions: List[str],
        docs_folder: str = "Docs",
        api_key: str = None,
        output_file: str = "evaluation_results.csv"
):
    """
    Runs a set of questions through all three reasoning modes and saves
    the answers to a CSV for manual comparison/grading.
    """

    print(f"Initializing RAG System...")
    # Initialize the system
    rag = BaseballRAGSystem()

    results = []

    print(f"\nStarting evaluation on {len(questions)} questions.\n" + "=" * 60)

    for i, q in enumerate(questions):
        print(f"Processing Question {i + 1}: {q}")

        # --- Mode 1: Direct RAG ---
        start_time = time.time()
        res_direct = rag.answer_direct(q)
        time_direct = time.time() - start_time

        # --- Mode 2: Chain-of-Thought ---
        start_time = time.time()
        res_cot = rag.cot_refine(q)
        time_cot = time.time() - start_time


        # Collect data
        row = {
            "Question": q,
            "Direct_Answer": res_direct["answer"],
            "Direct_Time": round(time_direct, 2),
            "CoT_Answer": res_cot["answer"],
            "CoT_Time": round(time_cot, 2),
        }
        results.append(row)
        print("...Done.")

    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print("=" * 60)
    print(f"Evaluation complete! Results saved to {output_file}")
    print("Open this CSV to manually grade or compare the answers for your report.")


# -------------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------------
if __name__ == "__main__":

    # DEFINE TEST QUESTIONS
    # Use a mix of factual (easy) and subjective/complex (hard) questions
    test_questions = [
        "What is wOBA and how is it calculated?",  # Factual
        "Explain what BABIP is and how it is related to luck.",  # Complex causality
        "Is clutch hitting a sustainable skill?",  # High debate potential
        "What are the key differences between FIP and ERA?"  # Technical comparison
    ]
    run_evaluation(test_questions,)