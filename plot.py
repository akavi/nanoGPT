"""Plot metrics from training logs.

Usage: uv run plot.py 88:ce_loss,log(loss),scale(mfu,1000),smooth(loss,50) 90:loss

Transforms:
  metric        — raw values
  log(metric)   — log scale (natural log)
  scale(metric,factor) — multiply by factor
  smooth(metric,window) — rolling mean with given window size
"""

import json
import math
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def split_args(s: str) -> list[str]:
    """Split on top-level commas (not inside parens)."""
    args = []
    depth = 0
    current: list[str] = []
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            args.append("".join(current).strip())
            current = []
            continue
        current.append(ch)
    args.append("".join(current).strip())
    return args


def eval_expr(expr: str, data: list[dict]) -> tuple[str, list[float]]:
    """Recursively evaluate a metric expression, returning (label, values)."""
    m = re.fullmatch(r"(\w+)\((.+)\)", expr, re.DOTALL)
    if not m:
        # Base case: raw metric name
        return expr, [d[expr] for d in data]

    func = m.group(1)
    args = split_args(m.group(2))

    if func == "log":
        label, values = eval_expr(args[0], data)
        return f"log({label})", [math.log(v) if v > 0 else float("-inf") for v in values]
    elif func == "scale":
        label, values = eval_expr(args[0], data)
        factor = float(args[1])
        return f"scale({label},{args[1]})", [v * factor for v in values]
    elif func == "smooth":
        label, values = eval_expr(args[0], data)
        window = int(args[1])
        out = []
        for i in range(len(values)):
            start = max(0, i - window + 1)
            out.append(sum(values[start : i + 1]) / (i + 1 - start))
        return f"smooth({label},{args[1]})", out
    else:
        raise ValueError(f"Unknown transform: {func}")


def parse_spec(spec: str) -> tuple[str, list[str]]:
    run_id, rest = spec.split(":", 1)
    # Split on commas not inside parentheses
    exprs = []
    depth = 0
    current = []
    for ch in rest:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            exprs.append("".join(current))
            current = []
            continue
        current.append(ch)
    exprs.append("".join(current))
    return run_id, exprs


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <run_id>:<expr>[,<expr>] ...")
        print("Expressions: metric, log(metric), scale(metric,factor), smooth(metric,window)")
        sys.exit(1)

    # Extract --range=start:finish if present
    iter_start, iter_end = None, None
    raw_args = []
    for a in sys.argv[1:]:
        if a.startswith("--range="):
            parts = a[len("--range="):].split(":")
            if parts[0]:
                iter_start = int(parts[0])
            if len(parts) > 1 and parts[1]:
                iter_end = int(parts[1])
        else:
            raw_args.append(a)

    # Split on spaces, but not inside parentheses
    args = []
    for a in raw_args:
        depth = 0
        current: list[str] = []
        for ch in a:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            elif ch == " " and depth == 0:
                if current:
                    args.append("".join(current))
                    current = []
                continue
            current.append(ch)
        if current:
            args.append("".join(current))
    specs = [parse_spec(s) for s in args]

    fig, ax = plt.subplots()
    for run_id, exprs in specs:
        path = Path(f"outputs/{run_id}/log.json")
        data = json.loads(path.read_text())
        if iter_start is not None:
            data = [d for d in data if d["iter"] >= iter_start]
        if iter_end is not None:
            data = [d for d in data if d["iter"] <= iter_end]
        iters = [d["iter"] for d in data]
        for expr in exprs:
            label, values = eval_expr(expr, data)
            ax.plot(iters, values, label=f"{run_id}:{label}")

    ax.set_xlabel("iter")
    ax.legend()
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
