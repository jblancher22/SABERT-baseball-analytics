# main_cli.py

import os
from rag_core import BaseballRAGSystem


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    print("Initializing advanced baseball RAG system...")

    # Initialize the system. It will load the embedding model and connect to Chroma.
    rag = BaseballRAGSystem()

    print("\nType baseball analytics questions. Ctrl+C to exit.\n")

    try:
        while True:
            q = input("Q: ").strip()
            if not q:
                continue

            mode = input("Select mode [direct/cot]: ").strip().lower()
            if mode == "cot":
                res = rag.cot_refine(q)
            else:
                res = rag.answer_direct(q)

            print("\n" + "=" * 20 + " RAG SYSTEM RESPONSE " + "=" * 20)
            print(f"[Mode: {res['mode']}]")
            print(f"\nAnswer: {res['answer']}")

            if res["sources"]:
                print("\nSources:")
                for s in res["sources"]:
                    print(" -", s)

            print("=" * 60 + "\n")

    except KeyboardInterrupt:
        print("\nGoodbye!")