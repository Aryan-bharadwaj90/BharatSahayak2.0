import argparse
import os
import json
import re
import time
from pathlib import Path
from urllib.parse import urlsplit
import pandas as pd

from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.embeddings import HuggingFaceEmbeddings
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)

from src.retrieval import retrieve
from src.rag import generate_answer


TEST_FILE = Path(__file__).with_name("test_questions.json")
OUTPUT_FILE = Path(__file__).with_name("results.csv")
TOP_K = 5


def normalized_text(text):
    return " ".join(text.casefold().split())


def source_url_key(url):
    parts = urlsplit(url.strip())
    return (parts.scheme.lower(), parts.netloc.lower(), parts.path.rstrip("/"), parts.query, parts.fragment)


def retrieval_metrics(item, chunks):
    contexts = [normalized_text(chunk["text"]) for chunk in chunks]
    evidence = item.get("reference_evidence") or []
    missing = [s for s in evidence if not any(normalized_text(s) in c for c in contexts)]
    title = item.get("reference_source_title")
    url = item.get("reference_source_url")
    matches = [
        normalized_text(chunk.get("source_title", "")) == normalized_text(title)
        and source_url_key(chunk.get("source_url", "")) == source_url_key(url)
        for chunk in chunks
    ] if title and url else None
    return {
        "evidence_coverage": (len(evidence) - len(missing)) / len(evidence) if evidence else None,
        "missing_reference_evidence": json.dumps(missing, ensure_ascii=False),
        "reference_source_title": title,
        "reference_source_url": url,
        "source_hit_at_5": int(any(matches)) if matches is not None else None,
        "source_precision_at_5": sum(matches) / max(1, len(chunks)) if matches is not None else None,
    }


def setup_ragas():
    api_key = os.getenv("LLM_API_KEY")
    base_url = os.getenv("LLM_BASE_URL", "").rstrip("/").removesuffix("/chat/completions")
    model_name = os.getenv("LLM_MODEL")

    if not api_key:
        raise ValueError("LLM_API_KEY environment variable is missing.")

    if not model_name:
        raise ValueError("LLM_MODEL environment variable is missing.")

    if not base_url:
        raise ValueError("LLM_BASE_URL environment variable is missing.")

    client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url
    )

    evaluator_llm = llm_factory(
        model_name,
        client=client,
        temperature=0.0
    )

    evaluator_embeddings = HuggingFaceEmbeddings(
        model="intfloat/multilingual-e5-small",
        device="cpu",
        normalize_embeddings=True
    )

    metrics = {
        "faithfulness": Faithfulness(llm=evaluator_llm),
        "answer_relevancy": AnswerRelevancy(
            llm=evaluator_llm,
            embeddings=evaluator_embeddings
        ),
        "context_precision": ContextPrecision(llm=evaluator_llm),
        "context_recall": ContextRecall(llm=evaluator_llm),
    }

    return metrics


def get_score(metric, **kwargs):
    try:
        result = metric.score(**kwargs)
        return round(float(result.value), 4)

    except Exception as error:
        print(
            f"Metric skipped: {type(error).__name__}: {error}",
            flush=True
        )
        return None


def evaluate_questions(retrieval_only=False, output_file=None):
    if output_file is None:
        output_file = OUTPUT_FILE.with_name("retrieval_results.csv") if retrieval_only else OUTPUT_FILE
    with open(TEST_FILE, encoding="utf-8") as file:
        tests = json.load(file)

    limit = int(os.getenv("EVAL_LIMIT", "0"))

    if limit > 0:
        tests = tests[:limit]

    metrics = {} if retrieval_only else setup_ragas()
    rows = []

    print(f"\nEvaluating {len(tests)} questions...\n")

    for index, item in enumerate(tests, start=1):
        question = item["question"]
        reference = item["reference_answer"]
        language = item.get("language", "en")

        print(f"\n[{index}/{len(tests)}] {question}")

        start = time.time()
        row = {
            "question": question,
            "language": language,
            "expected_scheme": item.get("expected_scheme"),
            "reference_answer": reference,
        }

        try:
            retrieved_chunks = retrieve(question, top_k=TOP_K)

            retrieved_contexts = [
                chunk.get("text", "")
                for chunk in retrieved_chunks
                if chunk.get("text")
            ]

            found_schemes = [
                chunk.get("scheme_id")
                for chunk in retrieved_chunks
            ]

            expected_scheme = item.get("expected_scheme")

            hit_at_5 = (
                int(expected_scheme in found_schemes)
                if expected_scheme else None
            )
            row.update(retrieval_metrics(item, retrieved_chunks))
            row.update(retrieved_scheme_ids=json.dumps(found_schemes), hit_at_5=hit_at_5)

            generated = {} if retrieval_only else generate_answer(
                question,
                retrieved_chunks,
                language=language
            )

            answer = generated.get("answer", "")

            scores = {}

            if generated.get("error"):
                row["error"] = answer
            elif answer.strip() and retrieved_contexts:
                if item.get("reference_source_url"):
                    urls = re.findall(r"https?://[^\s<>)\]]+", answer)
                    row["answer_cites_reference_url"] = int(any(
                        source_url_key(url.rstrip(".,;")) == source_url_key(item["reference_source_url"])
                        for url in urls
                    ))
                scores["faithfulness"] = get_score(
                    metrics["faithfulness"],
                    user_input=question,
                    response=answer,
                    retrieved_contexts=retrieved_contexts
                )

                scores["answer_relevancy"] = get_score(
                    metrics["answer_relevancy"],
                    user_input=question,
                    response=answer
                )

                scores["context_precision"] = get_score(
                    metrics["context_precision"],
                    user_input=question,
                    reference=reference,
                    retrieved_contexts=retrieved_contexts
                )

                scores["context_recall"] = get_score(
                    metrics["context_recall"],
                    user_input=question,
                    reference=reference,
                    retrieved_contexts=retrieved_contexts
                )
            else:
                scores = {
                    "faithfulness": None,
                    "answer_relevancy": None,
                    "context_precision": None,
                    "context_recall": None
                }

            row.update(scores)
            if not retrieval_only:
                row["generated_answer"] = answer

        except Exception as error:
            print(f"Question evaluation failed: {error}")

            row["error"] = str(error)

        row["latency_seconds"] = round(time.time() - start, 2)
        rows.append(row)

        pd.DataFrame(rows).to_csv(output_file, index=False)

        print("Scores:", {
            key: row.get(key)
            for key in (
                "evidence_coverage",
                "source_precision_at_5",
                "faithfulness",
                "answer_relevancy",
                "context_precision",
                "context_recall"
            )
        })

    df = pd.DataFrame(rows)

    print("\n========== EVALUATION SUMMARY ==========")

    for metric in (
        "hit_at_5",
        "evidence_coverage",
        "source_hit_at_5",
        "source_precision_at_5",
        "answer_cites_reference_url",
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall"
    ):
        if metric in df.columns:
            average = pd.to_numeric(
                df[metric], errors="coerce"
            ).mean()

            if pd.notna(average):
                print(f"{metric}: {average:.4f}")

    print(f"\nResults saved to: {output_file}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval-only", action="store_true", help="Score retrieval without an LLM.")
    parser.add_argument("--output", type=Path, help="Override the results CSV path.")
    args = parser.parse_args()
    evaluate_questions(args.retrieval_only, args.output)
