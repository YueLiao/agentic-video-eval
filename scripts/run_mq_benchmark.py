#!/usr/bin/env python3
"""MQ benchmark: do the CV motion signals alone carry any human-aligned signal?

Runs the tools-only ablation against real human preference data for motion
quality. No VLM involved, so it answers a narrow question honestly: before
spending anything on a judge, is there signal in the hand-designed motion
features at all, and in which of them?

Protocol follows the existing in-house MQ analysis so the numbers are directly
comparable to the trained reward model: per-model means and ranking, forced win
rate (sign of the score difference) against the human forced win rate, and a
tie-threshold three-way split cross-validated across the two model pairs -- the
threshold is fitted on one pair to reproduce its human tie count, then applied
to the other pair.

Crucially **nothing is fitted to the human labels**. Only aggregate win/tie/loss
counts exist, so fitting a combined score on them and then reporting agreement
with them would be circular. Each feature is evaluated on its own, as a
predictor it had no chance to tune against.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

MODELS = {
    "WAN": "WAN2.2_14B_Elite150_WAN2.2_14B_Elite150_21052026",
    "Cosmo": "Cosmo3_super_150_elite_150_elite",
    "pangu": "pangu_5B_720p_sft_ema_iter_59000",
}
# 观星台 15-rater AB on 运动合理性, vs WAN: (ours win, tie, WAN win)
HUMAN = {"pangu": (12, 63, 75), "Cosmo": (22, 101, 27)}
HUMAN_ORDER = ["WAN", "Cosmo", "pangu"]

#: Sign convention: +1 means a larger value should mean *better* motion.
FEATURES: dict[str, int] = {
    "speed_median": +1,      # motion present at all
    "speed_max": +1,
    "jerk_rms": -1,          # judder
    "freeze_frac": -1,       # stalls
    "accel_spike_frac": -1,  # speed discontinuities
    "mc_residual_p95": -1,   # change motion cannot explain
    "flow_anomaly_p95": -1,  # tearing / popping
    "freeze_sig_p95": -1,    # two-sided anomaly: too little change
    "shake_index": -1,       # camera instability
    "track_survival": +1,    # trackable = coherent
}


def extract(path: str) -> dict[str, float]:
    import tempfile
    from agenteval.media.clip import VideoHandle
    from agenteval.signals import suspicion
    from agenteval.tools import camera as C
    from agenteval.tools import physics as P
    from agenteval.tools import renders as R

    out: dict[str, float] = {}
    try:
        v = VideoHandle(path)
        tmp = Path(tempfile.mkdtemp())
        mc = R.motion_curves(v, tmp)
        val = mc.value
        n = max(1, int(val.get("n_pairs", 1)))
        out["speed_median"] = float(val.get("speed_median", 0.0))
        out["speed_max"] = float(val.get("speed_max", 0.0))
        out["jerk_rms"] = float(val.get("jerk_rms", 0.0))
        out["freeze_frac"] = len(val.get("near_freeze_frames") or []) / n
        out["accel_spike_frac"] = len(val.get("accel_spike_frames") or []) / n

        maps = suspicion.compute_maps(path)
        for k, name in (("mc_residual", "mc_residual_p95"),
                        ("flow_anomaly", "flow_anomaly_p95"),
                        ("freeze", "freeze_sig_p95")):
            out[name] = float(np.percentile(maps[k], 95)) if k in maps else 0.0

        cm = C.camera_motion(v, stride=3)
        out["shake_index"] = float(cm.value.get("shake_index", 0.0))

        tr = P.track_points(v, bbox=(0.2, 0.2, 0.6, 0.6),
                            t_span=(0, min(v.total, 48)))
        out["track_survival"] = float(tr.value.get("survival_rate", 0.0))
    except Exception as e:  # noqa: BLE001
        out["error"] = 1.0
        out["_msg"] = str(e)[:80]  # type: ignore[assignment]
    return out


def _job(args):
    model, path = args
    return model, Path(path).name, extract(path)


def three_way(delta: np.ndarray, tau: float) -> tuple[int, int, int]:
    return (int((delta > tau).sum()), int((np.abs(delta) <= tau).sum()),
            int((delta < -tau).sum()))


def tau_for_ties(delta: np.ndarray, n_tie: int) -> float:
    a = np.sort(np.abs(delta))
    return float(a[min(n_tie, len(a)) - 1]) if n_tie > 0 else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/pub/evaluation_group/cy/datasets/models")
    ap.add_argument("--out", default="bench/mq")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--cache", default="")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache) if args.cache else out / "features.json"

    if cache.exists():
        feats = json.loads(cache.read_text())
        print(f"loaded cached features from {cache}")
    else:
        jobs = []
        for m, d in MODELS.items():
            vids = sorted((Path(args.root) / d).glob("*.mp4"))
            if args.limit:
                vids = vids[: args.limit]
            jobs += [(m, str(p)) for p in vids]
        print(f"extracting {len(jobs)} videos with {args.workers} workers ...")
        t0 = time.time()
        feats: dict[str, dict[str, dict[str, float]]] = {m: {} for m in MODELS}
        done = 0
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for m, name, f in ex.map(_job, jobs, chunksize=2):
                feats[m][name] = f
                done += 1
                if done % 60 == 0:
                    print(f"  {done}/{len(jobs)}  ({time.time()-t0:.0f}s)", flush=True)
        cache.write_text(json.dumps(feats, ensure_ascii=False), encoding="utf-8")
        print(f"extracted in {time.time()-t0:.0f}s -> {cache}")

    ids = sorted(set(feats["WAN"]) & set(feats["Cosmo"]) & set(feats["pangu"]))
    n_err = sum(1 for m in MODELS for i in ids if feats[m][i].get("error"))
    print(f"\n{len(ids)} prompts common to all 3 models; {n_err} extraction errors")

    print(f"\n人评 (观星台 15 人, 运动合理性): 排序 WAN > Cosmo > pangu")
    for k, (w, t, l) in HUMAN.items():
        print(f"  {k:6s} vs WAN = {w}/{t}/{l}   WAN 强制胜率 {100*l/(w+l):.1f}%")

    rows = []
    for feat, sign in FEATURES.items():
        S = {m: np.array([feats[m][i].get(feat, np.nan) for i in ids], float)
             for m in MODELS}
        if any(np.isnan(v).all() for v in S.values()):
            continue
        for m in MODELS:
            v = S[m]
            v[np.isnan(v)] = np.nanmedian(v) if not np.isnan(v).all() else 0.0
            S[m] = v * sign                      # orient so larger == better
        means = {m: float(S[m].mean()) for m in MODELS}
        order = sorted(means, key=lambda k: -means[k])
        order_ok = order == HUMAN_ORDER

        wr = {}
        for k in ("pangu", "Cosmo"):
            d = S[k] - S["WAN"]
            wr[k] = float((d < 0).mean())        # WAN wins
        hw = {k: HUMAN[k][2] / (HUMAN[k][0] + HUMAN[k][2]) for k in HUMAN}
        err = np.mean([abs(wr[k] - hw[k]) for k in wr])

        # cross-validated tie threshold
        cross = {}
        for fit, app in (("pangu", "Cosmo"), ("Cosmo", "pangu")):
            tau = tau_for_ties(S[fit] - S["WAN"], HUMAN[fit][1])
            cross[app] = three_way(S[app] - S["WAN"], tau)
        rows.append((feat, order_ok, order, wr, err, cross))

    rows.sort(key=lambda r: r[4])
    print(f"\n{'feature':20s} {'排序对':5s} {'WAN vs pangu':>13s} {'WAN vs Cosmo':>13s} "
          f"{'|Δ胜率|':>8s}   交叉阈值三分 (RM vs 人评)")
    print("  " + "-" * 108)
    print(f"{'[人评]':20s} {'--':5s} {100*hw['pangu']:12.1f}% {100*hw['Cosmo']:12.1f}%"
          f" {'--':>8s}   pangu {HUMAN['pangu']}  Cosmo {HUMAN['Cosmo']}")
    for feat, ok, order, wr, err, cross in rows:
        print(f"{feat:20s} {'✓' if ok else '✗':5s} {100*wr['pangu']:12.1f}% "
              f"{100*wr['Cosmo']:12.1f}% {100*err:7.1f}%   "
              f"pangu {cross.get('pangu')}  Cosmo {cross.get('Cosmo')}")
    print("  " + "-" * 108)
    print(f"  排序对 = 是否复现人评的 {' > '.join(HUMAN_ORDER)}；"
          f"|Δ胜率| = 与人评强制胜率的平均绝对偏差（越小越好）")

    json.dump({"ids": len(ids), "human": HUMAN,
               "rows": [{"feature": f, "order_ok": o, "order": od,
                         "win_rate": w, "winrate_err": e,
                         "cross_threeway": {k: list(v) for k, v in c.items()}}
                        for f, o, od, w, e, c in rows]},
              open(out / "mq_result.json", "w"), ensure_ascii=False, indent=1)
    print(f"\nwrote {out/'mq_result.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
