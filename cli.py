"""Command line: ask a question or run the evaluation."""
from __future__ import annotations

import argparse
import json

from .evaluate import run
from .pipeline import Pipeline


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="actqa")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ask", help="ask one question")
    a.add_argument("question")
    a.add_argument("--data", default="data")
    a.add_argument("--k", type=int, default=5)
    e = sub.add_parser("eval", help="run the full evaluation")
    e.add_argument("--data", default="data")
    e.add_argument("--questions", default="eval/questions.jsonl")
    e.add_argument("--out", default="results")
    args = p.parse_args(argv)
    if args.cmd == "ask":
        pipe = Pipeline.from_dir(args.data)
        print(json.dumps(pipe.ask(args.question, k=args.k).as_dict(), indent=2))
    else:
        res = run(args.data, args.questions, args.out)
        print(json.dumps({k: res[k] for k in ("corpus", "questions", "thresholds")}, indent=2))
        print(open(f"{args.out}/report.md").read())


if __name__ == "__main__":
    main()
