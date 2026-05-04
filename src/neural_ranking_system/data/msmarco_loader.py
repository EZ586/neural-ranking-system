import os
os.environ["PYTHONUTF8"] = "1"

import ir_datasets
import random
from typing import Dict, List


def load_msmarco_dev_subset(
    dataset_name: str = "msmarco-passage/dev/small",
    max_queries: int = 100,
    max_docs: int = 50000,
    seed: int = 42,
):
    random.seed(seed)

    dataset = ir_datasets.load(dataset_name)

    docs = []
    doc_lookup = {}

    print("[INFO] Loading documents...")

    for i, doc in enumerate(dataset.docs_iter()):
        if i >= max_docs:
            break

        item = {
            "id": str(doc.doc_id),
            "text": str(doc.text),
        }

        docs.append(item)
        doc_lookup[item["id"]] = item

        if i % 10000 == 0:
            print(f"[INFO] Loaded {i} docs")

    print(f"[INFO] Finished loading {len(docs)} docs")

    qrels_by_query: Dict[str, List[str]] = {}

    print("[INFO] Loading qrels...")

    for qrel in dataset.qrels_iter():
        if qrel.relevance > 0:
            qrels_by_query.setdefault(
                str(qrel.query_id),
                []
            ).append(str(qrel.doc_id))

    queries = []

    print("[INFO] Loading queries...")

    for query in dataset.queries_iter():

        query_id = str(query.query_id)

        if query_id not in qrels_by_query:
            continue

        relevant_docs = [
            doc_id
            for doc_id in qrels_by_query[query_id]
            if doc_id in doc_lookup
        ]

        if relevant_docs:
            queries.append({
                "query_id": query_id,
                "query": str(query.text),
                "relevant_doc_ids": relevant_docs,
            })

        if len(queries) >= max_queries:
            break

    print(f"[INFO] Finished loading {len(queries)} queries")

    return docs, queries