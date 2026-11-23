"""Evaluation: retrieval ablation, abstention tuning on a dev split, and honest test-split results."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .answer import ExtractiveAnswerer, grounded_fraction
from .data import normalize
from .pipeline import Pipeline
from .retrieval import MODES


@dataclass
class Question:
    id: str
    question: str
    answerable: bool
    gold: list[tuple[str, str]]
    must_contain: list[str]
    split: str


def load_questions(path: str | Path) -> list[Question]:
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        out.append(Question(r["id"], r["question"], r["answerable"], [tuple(g) for g in r["gold"]],
                            r["must_contain"], r["split"]))
    return out


def rank_of_gold(hit_keys: list[tuple[str, str]], gold: list[tuple[str, str]]) -> int | None:
    for i, k in enumerate(hit_keys, start=1):
        if k in gold:
            return i
    return None


def retrieval_metrics(pipe: Pipeline, qs: list[Question], mode: str) -> dict:
    """Section-level hit@k and MRR over answerable questions."""
    per_q = []
    for q in qs:
        if not q.answerable:
            continue
        keys = [h.key for h in pipe.retrieve(q.question, k=10, mode=mode).hits]
        r = rank_of_gold(keys, q.gold)
        per_q.append({"id": q.id, "rank": r})
    n = len(per_q)
    hit = lambda k: sum(1 for p in per_q if p["rank"] and p["rank"] <= k) / n
    mrr = sum(1 / p["rank"] for p in per_q if p["rank"]) / n
    return {"n": n, "hit@1": hit(1), "hit@3": hit(3), "hit@5": hit(5), "mrr": mrr, "per_q": per_q}


def bootstrap_ci(values: list[float], n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    m = len(values)
    means = sorted(sum(values[rng.randrange(m)] for _ in range(m)) / m for _ in range(n_boot))
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]


def paired_bootstrap_diff(a: list[float], b: list[float], n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Mean of (a - b) with a 95% percentile interval, resampling questions together."""
    rng = random.Random(seed)
    d = [x - y for x, y in zip(a, b)]
    m = len(d)
    means = sorted(sum(d[rng.randrange(m)] for _ in range(m)) / m for _ in range(n_boot))
    return sum(d) / m, means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]


def judge(answer, q: Question) -> dict:
    """Score one answer. 'correct' means: answered, top citation in gold, and the key fact is in the text."""
    if q.answerable:
        cited_ok = (not answer.abstained) and bool(answer.citations) and answer.citations[0] in q.gold
        fact = (not answer.abstained) and all(normalize(p) in normalize(answer.text) for p in q.must_contain)
        return {"answered": not answer.abstained, "cited_ok": cited_ok, "fact": fact,
                "correct": cited_ok and fact, "wrong_answer": (not answer.abstained) and not (cited_ok and fact)}
    return {"answered": not answer.abstained, "cited_ok": False, "fact": False, "correct": False,
            "wrong_answer": not answer.abstained}


def evaluate_answers(pipe: Pipeline, qs: list[Question], tau_dense: float, tau_cov: float, n_sentences: int = 2) -> dict:
    ans = ExtractiveAnswerer(pipe.retriever, pipe.sections, tau_dense, tau_cov, n_sentences)
    rows = []
    for q in qs:
        a = pipe.ask(q.question, answerer=ans)
        j = judge(a, q)
        j["id"] = q.id
        j["grounded"] = (grounded_fraction(a.text, [pipe.sections[c].text for c in a.citations])
                         if (not a.abstained and a.citations) else None)
        rows.append((q, a, j))
    return summarize(rows)


def summarize(rows) -> dict:
    ans_q = [(q, a, j) for q, a, j in rows if q.answerable]
    un_q = [(q, a, j) for q, a, j in rows if not q.answerable]
    answered = [j for _, _, j in rows if j["answered"]]
    out = {
        "n_answerable": len(ans_q), "n_unanswerable": len(un_q),
        "answerable_correct": sum(j["correct"] for _, _, j in ans_q) / max(1, len(ans_q)),
        "answerable_answered": sum(j["answered"] for _, _, j in ans_q) / max(1, len(ans_q)),
        "citation_accuracy_when_answered": (sum(j["cited_ok"] for _, _, j in ans_q if j["answered"])
                                            / max(1, sum(j["answered"] for _, _, j in ans_q))),
        "unanswerable_abstained": sum(not j["answered"] for _, _, j in un_q) / max(1, len(un_q)),
        "unanswerable_wrongly_answered": sum(j["answered"] for _, _, j in un_q),
        "answered_total": len(answered),
        "wrong_among_answered": (sum(j["wrong_answer"] for j in answered) / max(1, len(answered))),
        "grounded_among_answered": (sum(1 for _, _, j in rows if j.get("grounded") == 1.0)
                                    / max(1, len(answered))),
        "rows": [dict(j) for _, _, j in rows],
    }
    return out


def tune_thresholds(pipe: Pipeline, dev: list[Question], wrong_penalty: float = 2.0) -> tuple[float, float, int, dict]:
    """Grid-search the two abstention thresholds and the sentence count on the dev split only.

    Objective: correct answers minus wrong_penalty times wrong answers, because a fluent wrong
    answer is worse than no answer.
    """
    cached = []
    for q in dev:
        r = pipe.retrieve(q.question)
        cached.append((q, r))
    best = (-1e9, 0.0, 0.0, 2)
    for ns in (1, 2, 3):
        for td in np.linspace(0.0, 0.9, 19):
            for tc in np.linspace(0.0, 1.0, 21):
                ans = ExtractiveAnswerer(pipe.retriever, pipe.sections, float(td), float(tc), ns)
                score = 0.0
                for q, r in cached:
                    j = judge(ans.answer(r), q)
                    score += j["correct"] - wrong_penalty * j["wrong_answer"]
                if score > best[0] + 1e-9:
                    best = (score, float(td), float(tc), ns)
    return best[1], best[2], best[3], {"dev_objective": best[0], "wrong_penalty": wrong_penalty}


def run(data_dir: str, questions: str, out_dir: str, seed: int = 42) -> dict:
    qs = load_questions(questions)
    dev = [q for q in qs if q.split == "dev"]
    test = [q for q in qs if q.split == "test"]
    pipe = Pipeline.from_dir(data_dir, seed=seed)
    res: dict = {"corpus": {"sections": len(pipe.sections), "chunks": len(pipe.chunks),
                            "acts": sorted({s.act for s in pipe.sections.values()})},
                 "questions": {"total": len(qs), "dev": len(dev), "test": len(test),
                               "answerable": sum(q.answerable for q in qs)},
                 "embedder": pipe.retriever.embedder.name, "seed": seed}
    # 1) retrieval ablation on the answerable questions of ALL splits (no tuning involved)
    allq = [q for q in qs]
    ret = {m: retrieval_metrics(pipe, allq, m) for m in MODES}
    res["retrieval_all"] = {m: {k: v for k, v in r.items() if k != "per_q"} for m, r in ret.items()}
    for m, r in ret.items():
        h3 = [1.0 if p["rank"] and p["rank"] <= 3 else 0.0 for p in r["per_q"]]
        lo, hi = bootstrap_ci(h3)
        res["retrieval_all"][m]["hit@3_ci95"] = [lo, hi]
    rt = {m: retrieval_metrics(pipe, test, m) for m in MODES}
    res["retrieval_test"] = {m: {k: v for k, v in r.items() if k != "per_q"} for m, r in rt.items()}
    h3 = {m: [1.0 if p["rank"] and p["rank"] <= 3 else 0.0 for p in ret[m]["per_q"]] for m in MODES}
    rr = {m: [1 / p["rank"] if p["rank"] else 0.0 for p in ret[m]["per_q"]] for m in MODES}
    res["paired"] = {
        "hybrid_minus_bm25_hit@3": paired_bootstrap_diff(h3["hybrid"], h3["bm25"]),
        "hybrid_minus_dense_hit@3": paired_bootstrap_diff(h3["hybrid"], h3["dense"]),
        "hybrid_minus_bm25_mrr": paired_bootstrap_diff(rr["hybrid"], rr["bm25"]),
    }
    # 2) tune abstention thresholds on dev, report on test
    td, tc, ns, info = tune_thresholds(pipe, dev)
    res["thresholds"] = {"tau_dense": td, "tau_coverage": tc, "n_sentences": ns, **info}
    res["answers_test"] = evaluate_answers(pipe, test, td, tc, ns)
    res["answers_test_no_abstention"] = evaluate_answers(pipe, test, 0.0, 0.0, ns)
    # 3) chunk-size ablation (hybrid hit@3, all answerable questions)
    res["chunk_size_ablation"] = {}
    for mc in (300, 700, 1500):
        p2 = Pipeline(list(pipe.sections.values()), max_chars=mc, seed=seed)
        m = retrieval_metrics(p2, allq, "hybrid")
        res["chunk_size_ablation"][str(mc)] = {"chunks": len(p2.chunks), "hit@3": m["hit@3"], "mrr": m["mrr"]}
    # 4) failure list on test
    fails = []
    for q in test:
        r = pipe.retrieve(q.question, k=3)
        keys = [f"{a} s.{s}" for a, s in (h.key for h in r.hits)]
        if q.answerable:
            rk = rank_of_gold([h.key for h in pipe.retrieve(q.question, k=10).hits], q.gold)
            if rk is None or rk > 3:
                fails.append({"id": q.id, "question": q.question, "gold": [f"{a} s.{s}" for a, s in q.gold],
                              "top3": keys, "gold_rank": rk})
    res["retrieval_failures_test"] = fails
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(res, indent=2))
    (out / "report.md").write_text(render_report(res))
    (out / "failures.md").write_text(render_failures(res, pipe, test, td, tc, ns))
    return res


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def render_report(res: dict) -> str:
    c, q = res["corpus"], res["questions"]
    L = [f"# Results\n",
         f"Corpus: {', '.join(c['acts'])}, {c['sections']} sections, {c['chunks']} chunks. Embedder: {res['embedder']} "
         f"(offline LSA, not a neural model). Seed {res['seed']}.\n",
         f"Questions: {q['total']} ({q['answerable']} answerable, {q['total'] - q['answerable']} unanswerable); "
         f"{q['dev']} dev (used only to tune abstention thresholds), {q['test']} test.\n",
         "## Retrieval (all answerable questions, section-level)\n",
         "| Retriever | hit@1 | hit@3 | hit@3 95% CI | hit@5 | MRR |", "|---|---|---|---|---|---|"]
    for m, r in res["retrieval_all"].items():
        lo, hi = r["hit@3_ci95"]
        L.append(f"| {m} | {pct(r['hit@1'])} | {pct(r['hit@3'])} | {pct(lo)} to {pct(hi)} | {pct(r['hit@5'])} | {r['mrr']:.3f} |")
    L += ["\nTest split only (answerable questions, never used for tuning):\n", "| Retriever | n | hit@1 | hit@3 | hit@5 | MRR |", "|---|---|---|---|---|---|"]
    for m, r in res["retrieval_test"].items():
        L.append(f"| {m} | {r['n']} | {pct(r['hit@1'])} | {pct(r['hit@3'])} | {pct(r['hit@5'])} | {r['mrr']:.3f} |")
    p = res["paired"]
    for name, (d, lo, hi) in p.items():
        L.append(f"\nPaired bootstrap, {name}: {d:+.3f} (95% interval {lo:+.3f} to {hi:+.3f}).")
    t = res["thresholds"]
    L += ["\n## Chunk size (hybrid)\n", "| max chars | chunks | hit@3 | MRR |", "|---|---|---|---|"]
    for mc, r in res["chunk_size_ablation"].items():
        L.append(f"| {mc} | {r['chunks']} | {pct(r['hit@3'])} | {r['mrr']:.3f} |")
    L.append(f"\n## Answers on the test split (hybrid retrieval, extractive answerer)\n")
    L.append(f"Abstention thresholds tuned on dev only: dense similarity >= {t['tau_dense']:.2f}, "
             f"query-term coverage >= {t['tau_coverage']:.2f}, {t['n_sentences']} sentence(s) returned (objective: correct minus {t['wrong_penalty']:.0f} x wrong).\n")
    L += ["| Metric | With abstention | Without abstention |", "|---|---|---|"]
    a, b = res["answers_test"], res["answers_test_no_abstention"]
    rows = [("Answerable questions answered correctly (right section and key fact)", "answerable_correct"),
            ("Answerable questions answered at all", "answerable_answered"),
            ("Right section cited, among answered answerable", "citation_accuracy_when_answered"),
            ("Unanswerable questions correctly refused", "unanswerable_abstained"),
            ("Wrong answers among all answers given", "wrong_among_answered")]
    for label, key in rows:
        L.append(f"| {label} | {pct(a[key])} | {pct(b[key])} |")
    L.append(f"\nTest set: {a['n_answerable']} answerable, {a['n_unanswerable']} unanswerable. "
             f"Unanswerable questions wrongly answered: {a['unanswerable_wrongly_answered']} with abstention, "
             f"{b['unanswerable_wrongly_answered']} without. Samples are small, so treat differences of a few points as noise.\n")
    return "\n".join(L)


def render_failures(res: dict, pipe: Pipeline, test: list[Question], td: float, tc: float, ns: int = 2) -> str:
    L = ["# Failure analysis (test split)\n", "## Answerable questions whose gold section was not in the top 3\n"]
    if not res["retrieval_failures_test"]:
        L.append("None.\n")
    for f in res["retrieval_failures_test"]:
        L.append(f"- {f['id']}: {f['question']}\n  - gold: {', '.join(f['gold'])} (rank {f['gold_rank']})\n  - top 3: {', '.join(f['top3'])}")
    ans = ExtractiveAnswerer(pipe.retriever, pipe.sections, td, tc, ns)
    L.append("\n## Wrong answers given (after abstention)\n")
    n = 0
    for q in test:
        a = pipe.ask(q.question, answerer=ans)
        j = judge(a, q)
        if j["wrong_answer"]:
            n += 1
            kind = "unanswerable question answered" if not q.answerable else (
                "wrong section" if not j["cited_ok"] else "right section, key fact missing from the sentences picked")
            cite = ", ".join(f"{x} s.{y}" for x, y in a.citations)
            L.append(f"- {q.id} ({kind}): {q.question}\n  - cited: {cite}\n  - signals: dense {a.signals['dense_top']:.2f}, coverage {a.signals['coverage']:.2f}")
    if n == 0:
        L.append("None.")
    return "\n".join(L) + "\n"
