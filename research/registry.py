"""Run registry: an append-only log of every backtest and validation run.

Search enough configurations and one will look good by luck. The only defence is to know how many
were tried, so every run that is saved (and every validation) is appended here with what was run,
on which code, and how it came out. The trial count it reports is what a result must be judged
against: a p-value of 0.01 means something different after 3 looks than after 300.

The log is a JSON-lines file under DATA_DIR/registry, one run per line, written by append only.

    python -m research.registry                 # list the runs and count the trials
    python -m research.registry --import-run data/runs/sweep_fvg_all_plan_spread
"""
import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from data_pipeline import config

KINDS = ("backtest", "validation", "crosssection")


def registry_path(directory: Optional[Path] = None) -> Path:
    return Path(directory or config.DATA_DIR / "registry") / "runs.jsonl"


def git_state() -> tuple[Optional[str], Optional[bool]]:
    """(commit, has uncommitted changes). A dirty tree means the run cannot be reproduced from
    the commit alone, which is worth knowing later."""
    root = Path(__file__).resolve().parent.parent
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                cwd=root, timeout=10).stdout.strip() or None
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                                    text=True, cwd=root, timeout=10).stdout.strip())
        return commit, dirty
    except (OSError, subprocess.SubprocessError):
        return None, None


def record(kind: str, strategy: str, exit: str, costs: str, start: str,
           instruments: list[str], results: dict, slippage_pips: float = 0.0,
           params: Optional[dict] = None, note: str = "", directory: Optional[Path] = None,
           created_at: Optional[str] = None, extra: Optional[dict] = None) -> dict:
    """Appends one run to the log and returns the entry."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    commit, dirty = git_state()
    created = created_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
    entry = {
        "id": hashlib.sha1(f"{created}|{kind}|{strategy}|{exit}|{costs}|{len(instruments)}".encode())
              .hexdigest()[:10],
        "created_at": created, "kind": kind, "git_commit": commit, "git_dirty": dirty,
        "strategy": strategy, "params": params or {}, "exit": exit, "costs": costs,
        "slippage_pips": slippage_pips, "start": start, "instruments": sorted(instruments),
        "data_dir": str(config.DATA_DIR), "results": results, "note": note, **(extra or {}),
    }
    path = registry_path(directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")
    return entry


def load_runs(directory: Optional[Path] = None) -> list[dict]:
    """Every run logged so far. A line that cannot be read is skipped, not fatal: an unreadable
    entry must not stop a researcher seeing the rest."""
    path = registry_path(directory)
    if not path.exists():
        return []
    runs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            runs.append(json.loads(line))
        except ValueError:
            print(f"registry: skipping an unreadable line in {path.name}", file=sys.stderr)
    return runs


def config_key(run: dict) -> tuple:
    """What makes two runs the same hypothesis: strategy, its parameters, exit rule and costs."""
    return (run["strategy"], json.dumps(run.get("params") or {}, sort_keys=True), run["exit"],
            run["costs"], run.get("slippage_pips", 0.0))


def trial_counts(directory: Optional[Path] = None, strategy: Optional[str] = None) -> dict:
    """{'runs', 'configs', 'pair_tests'}: how many runs, how many distinct configurations, and
    how many configuration-by-instrument tests have been looked at. Re-running a configuration
    is not a new hypothesis; testing it on a new instrument is."""
    # A control (a check that the machinery works) is not a hypothesis about an edge, so it is
    # not a look at the data in the sense the trial count exists to track.
    runs = [r for r in load_runs(directory)
            if (strategy is None or r["strategy"] == strategy) and not r.get("control")]
    tested: dict[tuple, set] = {}
    for run in runs:
        tested.setdefault(config_key(run), set()).update(run.get("instruments", []))
    return {"runs": len(runs), "configs": len(tested),
            "pair_tests": sum(len(instruments) for instruments in tested.values())}


def import_run(out_dir: Path, directory: Optional[Path] = None, note: str = "") -> dict:
    """Logs a run saved before the registry existed, from its run.json."""
    data = json.loads((Path(out_dir) / "run.json").read_text(encoding="utf-8"))
    pooled = data.get("pooled") or {}
    return record("backtest", data["strategy"], data["exit"], data["costs"],
                  data.get("start", "auto"), data["instruments"], pooled,
                  slippage_pips=data.get("slippage_pips", 0.0), directory=directory,
                  created_at=data.get("created_at"),
                  note=note or f"imported from {Path(out_dir).name}",
                  extra={"imported": True, "git_commit_at_run": data.get("git_commit")})


def format_runs(runs: list[dict]) -> str:
    lines = [f"{'id':10} {'when':16} {'kind':10} {'strategy':10} {'exit':6} {'costs':6} "
             f"{'pairs':>5}  note"]
    for r in runs:
        lines.append(f"{r['id']:10} {r['created_at'][:16]:16} {r['kind']:10} {r['strategy']:10} "
                     f"{r['exit']:6} {r['costs']:6} {len(r.get('instruments', [])):>5}  "
                     f"{r.get('note', '')[:60]}")
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--import-run", metavar="DIR",
                        help="Log a saved run (its run.json) that predates the registry.")
    parser.add_argument("--note", default="")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if args.import_run:
        entry = import_run(Path(args.import_run), note=args.note)
        print(f"logged {entry['id']}: {entry['note']}")
    runs = load_runs()
    print(format_runs(runs) if runs else "no runs logged yet")
    counts = trial_counts()
    print(f"\n{counts['runs']} runs, {counts['configs']} distinct configurations, "
          f"{counts['pair_tests']} configuration-by-instrument tests")
    return 0


if __name__ == "__main__":
    sys.exit(main())
