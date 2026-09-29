#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

import flybrain_evolution_unified as core

STATE_DIR = Path(os.environ.get("FLY_STATE_DIR", "flybrain-state"))
STATE_DIR.mkdir(parents=True, exist_ok=True)
GENS = int(os.environ.get("GENERATIONS_PER_CYCLE", "120"))
POP = int(os.environ.get("CONTINUOUS_POPULATION", "32"))
BASE_SEED = int(os.environ.get("CONTINUOUS_SEED", str(core.SEED + 9000)))


def load_genome(path: Path):
    d = np.load(path)
    return {
        "scale": d["scale"].astype(np.float32),
        "delta": d["delta"].astype(np.float32),
        "cap": d["cap"].astype(np.float32),
        "exc": d["exc"].astype(np.float32),
        "oi": d["oi"].astype(np.float32),
        "oo": d["oo"].astype(np.float32),
        "rr": d["rr"].astype(np.float32),
        "leak": float(d["leak"][0]),
        "sigma": float(d["sigma"][0]),
    }


def save_genome(path: Path, g):
    np.savez_compressed(
        path,
        scale=g["scale"],
        delta=g["delta"],
        cap=g["cap"],
        exc=g["exc"],
        oi=g["oi"],
        oo=g["oo"],
        rr=g["rr"],
        leak=np.array([g["leak"]], dtype=np.float32),
        sigma=np.array([g["sigma"]], dtype=np.float32),
    )


def multi_score(g, base, task_sets):
    ev = [core.score(g, base, t) for t in task_sets]
    values = np.array([x[0] for x in ev], dtype=np.float64)
    # Reward average ability but punish fragile individuals that fail one environment.
    robust = float(values.mean() - 0.25 * values.std() + 0.15 * values.min())
    m = {}
    keys = ev[0][1].keys()
    for k in keys:
        vals = [float(x[1][k]) for x in ev]
        m[k] = float(np.mean(vals))
    m["environment_score_mean"] = float(values.mean())
    m["environment_score_std"] = float(values.std())
    m["environment_score_min"] = float(values.min())
    return robust, m


def main():
    t0 = time.time()
    n, e, names, base = core.load_connectome()
    state_json = STATE_DIR / "state.json"
    genome_file = STATE_DIR / "best_genome.npz"

    if state_json.exists() and genome_file.exists():
        state = json.loads(state_json.read_text())
        root = load_genome(genome_file)
        cycle = int(state.get("cycle", 0)) + 1
        total_before = int(state.get("total_generations", 0))
        print("RESUME", json.dumps({"cycle": cycle, "total_generations": total_before}))
    else:
        state = {}
        cycle = 1
        total_before = 0
        root = core.init_genome(np.random.default_rng(BASE_SEED))
        print("START_NEW_LINEAGE")

    rng = np.random.default_rng(BASE_SEED + cycle * 104729)

    # Three noisy environments + one held-out environment per cycle.
    train_sets = [
        core.tasks(BASE_SEED + cycle * 101 + offset, n=64)
        for offset in (1, 17, 53)
    ]
    heldout = core.tasks(BASE_SEED + cycle * 101 + 9973, n=128)

    population = [core.clone(root)]
    population += [core.mutate(root, rng, 0.55) for _ in range(POP - 1)]

    best = core.clone(root)
    best_score, best_metrics = multi_score(best, base, train_sets)
    history = []
    stagnant = 0

    for local_gen in range(GENS + 1):
        evaluations = [multi_score(g, base, train_sets) for g in population]
        scores = np.array([x[0] for x in evaluations], dtype=np.float64)
        bi = int(scores.argmax())
        if scores[bi] > best_score + 1e-10:
            best_score = float(scores[bi])
            best = core.clone(population[bi])
            best_metrics = evaluations[bi][1]
            stagnant = 0
        else:
            stagnant += 1

        if local_gen % 10 == 0 or local_gen == GENS:
            held_score, held_metrics = core.score(best, base, heldout)
            row = {
                "cycle": cycle,
                "local_generation": local_gen,
                "total_generation": total_before + local_gen,
                "train_robust_score": best_score,
                "heldout_score": held_score,
                "heldout_accuracy": held_metrics["accuracy"],
                "new_edges": best_metrics["new_edges"],
                "low_capacity_groups": best_metrics["low_capacity_groups"],
                "mutation_sigma": best["sigma"],
            }
            history.append(row)
            print("CONTINUOUS_EVOLVE", json.dumps(row))

        if local_gen == GENS:
            break

        elite_n = max(3, POP // 8)
        elite_idx = np.argsort(scores)[-elite_n:][::-1]
        nxt = [core.clone(population[int(i)]) for i in elite_idx]
        mutation_rate = min(0.95, 0.30 + 0.012 * stagnant)

        def tournament():
            cand = rng.integers(0, POP, 5)
            return int(cand[np.argmax(scores[cand])])

        while len(nxt) < POP:
            a, b = population[tournament()], population[tournament()]
            child = core.cross(a, b, rng)
            child = core.mutate(child, rng, mutation_rate)
            nxt.append(child)
        population = nxt

    final_held_score, final_held = core.score(best, base, heldout)
    total_after = total_before + GENS
    save_genome(genome_file, best)

    previous_best = float(state.get("lifetime_best_heldout_score", -1e30))
    lifetime_best = max(previous_best, float(final_held_score))
    out = {
        "status": "ok",
        "cycle": cycle,
        "generations_this_cycle": GENS,
        "total_generations": total_after,
        "connectome": {"neurons": n, "edges": e, "groups": len(names)},
        "train_robust_score": best_score,
        "train_metrics": best_metrics,
        "heldout_score": float(final_held_score),
        "heldout_metrics": final_held,
        "lifetime_best_heldout_score": lifetime_best,
        "structural_genome": {
            "new_edges": int(np.count_nonzero(np.abs(best["delta"]) > 1e-3)),
            "low_capacity_groups": int(np.count_nonzero(best["cap"] < 0.08)),
            "mutation_sigma": float(best["sigma"]),
            "leak": float(best["leak"]),
        },
        "runtime_sec": time.time() - t0,
        "history": history,
    }
    state_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    (STATE_DIR / f"cycle_{cycle:06d}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("CONTINUOUS_FINAL=" + json.dumps(out))


if __name__ == "__main__":
    main()
