"""Trial-alignment checks for the clip-log (2026-09/10) recordings.

1) After a run -- print the alignment QC + label-shift audit of every enabled condition:

       python check_alignment.py

   Healthy session: n_on_sound == n_trials (180), shift_margin >= 1, ambiguous False,
   audit_suspect False. Anything else: open that session's *_onsets.png before trusting it.

2) Audit an OLD output folder (first/second pass) without touching any WAV: re-align each
   cached envelope (*_align.npz) against its clip log and report by how many whole trials the
   old pipeline's onsets were shifted (0 = correct, 10 = every position mislabelled):

       python check_alignment.py --audit-old "Z:\\...\\newdata_second_pass_output\\results"
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

import run_analysis as ra
from zyldoa import onsets, unity

EXP_DIR = Path(__file__).resolve().parent


def report_current(conditions_file: str = "conditions.json") -> None:
    reg = json.loads((EXP_DIR / conditions_file).read_text(encoding="utf-8"))
    print(f"{'condition':22s} {'session':58s} {'on_sound':>9s} {'margin':>6s} {'ambig':>5s} {'audit':>6s}")
    for name, c in reg["conditions"].items():
        if not c.get("enabled", True):
            continue
        summ = ra.load_condition(name, conditions_file)["results_dir"] / "summary.json"
        if not summ.exists():
            print(f"{name:22s} (not run yet)")
            continue
        s = json.loads(summ.read_text(encoding="utf-8"))
        audit = s.get("label_shift_audit", {})
        for sess, q in (s.get("alignment_qc") or {}).items():
            a = audit.get(sess, {})
            flag = "  <-- CHECK" if (q.get("ambiguous") or a.get("suspect")
                                     or q.get("n_on_sound", 0) < 0.9 * q.get("n_trials", 1)) else ""
            print(f"{name:22s} {sess[:58]:58s} {q.get('n_on_sound'):>4}/{q.get('n_trials'):<4} "
                  f"{q.get('shift_margin_trials'):>6} {str(q.get('ambiguous'))[0]:>5s} "
                  f"{str(a.get('suspect'))[0]:>6s}{flag}")


def audit_old(results_root: Path, data_root: Path) -> None:
    slug = lambda n: re.sub(r"[^A-Za-z0-9]+", "_", n).strip("_")
    folders = {slug(p.name): p for p in data_root.iterdir() if p.is_dir()}
    seen = set()
    print(f"{'session':58s} {'on_sound':>9s} {'margin':>6s}  old pipeline shift (trials: count)")
    for npz_path in sorted(results_root.rglob("*_align.npz")):
        tag = npz_path.name[:-len("_align.npz")]
        if tag in seen or tag not in folders:
            continue
        seen.add(tag)
        z = np.load(npz_path)
        log = unity.parse_clip_log(folders[tag])
        res = onsets.align_schedule_env(z["env"], float(z["env_rate"]), log["onset_unity_s"],
                                        blocks=log["block"], clip_dur_s=log["clip_dur_s"])
        q = res["qc"]
        shift = np.round((z["onset_s"] - res["onset_s"]) / q["iti_s"]).astype(int)
        vals, cnt = np.unique(shift, return_counts=True)
        note = "" if q["n_on_sound"] >= 0.9 * q["n_trials"] else "   (weak cached envelope: re-run to verify)"
        print(f"{tag[:58]:58s} {q['n_on_sound']:>4}/{q['n_trials']:<4} {q['shift_margin_trials']:>6}  "
              f"{dict(zip(vals.tolist(), cnt.tolist()))}{note}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audit-old", default=None, help="results folder of an old (first/second pass) run")
    ap.add_argument("--data-root", default=r"Z:\JordanWashecheck\Objective Localization")
    ap.add_argument("--conditions-file", default="conditions.json")
    a = ap.parse_args()
    if a.audit_old:
        audit_old(Path(a.audit_old), Path(a.data_root))
    else:
        report_current(a.conditions_file)


if __name__ == "__main__":
    main()
