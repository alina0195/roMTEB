"""Shared BEIR-format push for retrieval / reranking datasets.

A retrieval repo has three configs (corpus, queries, default/qrels).
A reranking repo adds ``top_ranked``. If ``top_ranked`` is present, MTEB
restricts search to that candidate pool; omit it for full-corpus retrieval.
"""

from __future__ import annotations

from romteb._hf_datasets import Dataset, DatasetDict
from huggingface_hub import HfApi

from romteb.data_prep._common import _record_revision


def push_beir_repo(
    corpus: Dataset,
    queries: Dataset,
    qrels: Dataset,
    repo_id: str,
    top_ranked: Dataset | None = None,
    commit_message: str | None = None,
) -> str:
    """Push corpus / queries / qrels [/ top_ranked] and pin the revision SHA."""
    api = HfApi()
    print(f"Logged in as: {api.whoami()['name']}")
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)

    print(f"Pushing corpus     ({len(corpus):,} docs) -> {repo_id}")
    corpus.push_to_hub(
        repo_id,
        config_name="corpus",
        split="corpus",
        commit_message=commit_message or f"corpus {repo_id}",
    )

    print(f"Pushing queries    ({len(queries):,} queries)")
    queries.push_to_hub(
        repo_id,
        config_name="queries",
        split="queries",
        commit_message=commit_message or f"queries {repo_id}",
    )

    print(f"Pushing qrels      ({len(qrels):,} judgements)")
    DatasetDict({"test": qrels}).push_to_hub(
        repo_id,
        config_name="default",
        commit_message=commit_message or f"qrels {repo_id}",
    )

    if top_ranked is not None:
        print(f"Pushing top_ranked ({len(top_ranked):,} candidate sets)")
        DatasetDict({"test": top_ranked}).push_to_hub(
            repo_id,
            config_name="top_ranked",
            commit_message=commit_message or f"top_ranked {repo_id}",
        )

    refs = api.list_repo_refs(repo_id, repo_type="dataset")
    sha = refs.branches[0].target_commit if refs.branches else "main"
    _record_revision(repo_id, sha)
    print(f"Pushed: https://huggingface.co/datasets/{repo_id}  (rev {sha})")
    return sha


def print_beir_stats(
    name: str,
    corpus: Dataset,
    queries: Dataset,
    qrels: Dataset,
    top_ranked: Dataset | None = None,
    extra: dict | None = None,
) -> None:
    print(f"\n=== {name} ===")
    print(
        f"  corpus={len(corpus):,}  queries={len(queries):,}  "
        f"qrels={len(qrels):,}"
        + (f"  top_ranked={len(top_ranked):,}" if top_ranked is not None else "  (no top_ranked)")
    )
    if extra:
        for k, v in extra.items():
            print(f"  {k}: {v}")
