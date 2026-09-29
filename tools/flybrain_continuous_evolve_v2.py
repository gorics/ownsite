#!/usr/bin/env python3
from __future__ import annotations

import gzip
import json
import math
import os
import struct
import time
from pathlib import Path

import numpy as np

SEED = 20260929
FLY = Path(os.environ.get("FLY_REPO", "flybrain-src"))
STATE_DIR = Path(os.environ.get("FLY_STATE_DIR", "flybrain-state"))
GENS = int(os.environ.get("GENERATIONS_PER_CYCLE", "120"))
POP = int(os.environ.get("CONTINUOUS_POPULATION", "32"))
BASE_SEED = int(os.environ.get("CONTINUOUS_SEED", str(SEED + 9000)))
STATE_DIR.mkdir(parents=True, exist_ok=True)

SENS = [0, 6, 10, 14]
MOTOR = np.array([56, 57, 58, 35], dtype=np.int64)
CENTRAL = np.array([17, 19, 21, 23, 25, 26, 27, 28, 35, 60], dtype=np.int64)


def load_connectome():
    meta = json.loads((FLY / "data/neuron_meta.json").read_text())
    names = [x["name"] for x in meta["groups"]]
    with gzip.open(FLY / "data/connectome.bin.gz", "rb") as f:
        raw = f.read()
    n, e = struct.unpack_from("<II", raw, 0)
    edge_dtype = np.dtype([("pre", "<u4"), ("post", "<u4"), ("w", "<f4")])
    edges = np.frombuffer(raw, dtype=edge_dtype, count=e, offset=8)
    meta_dtype = np.dtype([("region", "u1"), ("group", "<u2")])
    neurons = np.frombuffer(raw, dtype=meta_dtype, count=n, offset=8 + e * 12)
    groups = neurons["group"].astype(np.int64)
    G = len(names)
    base = np.zeros((G, G), dtype=np.float64)
    np.add.at(base, (groups[edges["pre"]], groups[edges["post"]]), edges["w"].astype(np.float64))
    nz = np.abs(base[base != 0])
    scale = np.percentile(nz, 90) if nz.size else 1.0
    base = np.tanh(base / max(scale, 1e-6))
    base /= np.maximum(np.sum(np.abs(base), axis=1, keepdims=True), 1.0)
    return n, e, names, base.astype(np.float32)


def init_genome(rng, G=63):
    return {
        "scale": np.ones((G, G), np.float32),
        "delta": np.zeros((G, G), np.float32),
        "cap": np.ones(G, np.float32),
        "exc": np.ones(G, np.float32),
        "oi": rng.normal(0, 0.02, (16, G)).astype(np.float32),
        "oo": rng.normal(0, 0.02, (G, 16)).astype(np.float32),
        "rr": rng.normal(0, 0.01, (16, 16)).astype(np.float32),
        "leak": 0.62,
        "sigma": 0.12,
    }


def clone(g):
    return {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in g.items()}


def mutate(g, rng, rate=0.4):
    c = clone(g)
    G = len(c["cap"])
    c["sigma"] = float(np.clip(c["sigma"] * math.exp(rng.normal(0, 0.05)), 0.02, 0.7))
    s = c["sigma"]
    k = max(3, int(G * G * rate * 0.012))
    i = rng.integers(0, G, k)
    j = rng.integers(0, G, k)
    c["scale"][i, j] *= np.exp(rng.normal(0, s, k)).astype(np.float32)
    np.clip(c["scale"], 0, 4, out=c["scale"])
    for _ in range(max(5, int(7 + 18 * rate))):
        a, b = int(rng.integers(G)), int(rng.integers(G))
        r = rng.random()
        if r < 0.65:
            c["delta"][a, b] += float(rng.normal(0, 1.7 * s))
        elif r < 0.90:
            c["delta"][a, b] *= float(rng.uniform(0, 0.4))
        else:
            c["delta"][a, b] = -c["delta"][a, b] + float(rng.normal(0, s))
    np.clip(c["delta"], -3, 3, out=c["delta"])
    for _ in range(max(2, int(8 * rate))):
        q = int(rng.integers(G))
        c["cap"][q] *= float(math.exp(rng.normal(0, 0.6 * s)))
        if q not in MOTOR and rng.random() < 0.07:
            c["cap"][q] *= float(rng.uniform(0.03, 0.25))
    np.clip(c["cap"], 0.02, 3.5, out=c["cap"])
    idx = rng.choice(G, max(2, int(G * 0.08)), replace=False)
    c["exc"][idx] *= np.exp(rng.normal(0, 0.35 * s, len(idx))).astype(np.float32)
    np.clip(c["exc"], 0.25, 3, out=c["exc"])
    for key, p, mult in [("oi", 0.03, 0.3), ("oo", 0.03, 0.3), ("rr", 0.05, 0.25)]:
        arr = c[key]
        mask = rng.random(arr.shape) < p
        arr[mask] += rng.normal(0, s * mult, int(mask.sum())).astype(np.float32)
        np.clip(arr, -1, 1, out=arr)
    c["leak"] = float(np.clip(c["leak"] + rng.normal(0, 0.04 * s), 0.2, 0.95))
    return c


def cross(a, b, rng):
    c = clone(a)
    G = len(c["cap"])
    rows = rng.random(G) < 0.5
    nodes = rng.random(G) < 0.5
    cols = rng.random(G) < 0.5
    c["scale"][rows] = b["scale"][rows]
    c["delta"][rows] = b["delta"][rows]
    c["cap"][nodes] = b["cap"][nodes]
    c["exc"][nodes] = b["exc"][nodes]
    c["oi"][:, cols] = b["oi"][:, cols]
    c["oo"][cols] = b["oo"][cols]
    rm = rng.random(16) < 0.5
    c["rr"][rm] = b["rr"][rm]
    if rng.random() < 0.5:
        c["leak"] = b["leak"]
    c["sigma"] = (a["sigma"] + b["sigma"]) / 2
    return c


def effective(base, g):
    m = (base * g["scale"] + 0.35 * g["delta"]) * np.sqrt(g["cap"][:, None] * g["cap"][None, :])
    return (m / np.maximum(np.sum(np.abs(m), axis=1, keepdims=True), 1.0)).astype(np.float32)


def make_tasks(seed, n=64):
    rng = np.random.default_rng(seed)
    out = []
    for k in range(n):
        y = k % 4
        cue = np.zeros(63, np.float32)
        cue[SENS[y]] = 1.0
        cue[SENS[(y + 1) % 4]] = 0.1
        cue += rng.normal(0, 0.035, 63).astype(np.float32)
        llm = np.zeros(8, np.float32)
        llm[y] = 1.0
        llm[4 + (y + 1) % 4] = 0.4
        llm += rng.normal(0, 0.035, 8).astype(np.float32)
        sd = np.zeros(8, np.float32)
        sd[(2 * y) % 8] = 1.0
        sd[(2 * y + 1) % 8] = -0.5
        sd += rng.normal(0, 0.035, 8).astype(np.float32)
        out.append((y, cue, llm, sd))
    return out


def score(g, base, tasks):
    m = effective(base, g)
    correct = 0
    margins, memories, crossmodal, energies = [], [], [], []
    for y, cue, llm, sd in tasks:
        brain = np.zeros(63, np.float32)
        organ = np.r_[llm, sd].astype(np.float32)
        early = None
        for t in range(10):
            ext = cue if t < 2 else 0
            organ_drive = organ @ g["oi"] if 1 <= t <= 6 else 0
            brain = np.tanh(g["leak"] * brain + (brain @ m) * g["exc"] + (ext + organ_drive) * g["exc"])
            organ = np.tanh(0.55 * organ + brain @ g["oo"] + organ @ g["rr"])
            if t == 2:
                early = brain.copy()
        q = brain[MOTOR]
        correct += int(np.argmax(q) == y)
        margins.append(float(q[y] - np.max(np.delete(q, y))))
        memories.append(float(np.dot(early[CENTRAL], brain[CENTRAL]) / len(CENTRAL)))
        crossmodal.append(float(np.dot(organ, np.r_[llm, sd]) / 16))
        energies.append(float(np.mean(np.abs(brain))))
    acc = correct / len(tasks)
    new_edges = int(np.count_nonzero(np.abs(g["delta"]) > 1e-3))
    low_capacity = int(np.count_nonzero(g["cap"] < 0.08))
    complexity = new_edges / (63 * 63) + 0.003 * low_capacity
    fit = 5 * acc + 1.15 * np.mean(margins) + 0.45 * np.mean(memories) + 0.35 * np.mean(crossmodal) - 0.1 * np.mean(energies) - 0.18 * complexity
    return float(fit), {
        "accuracy": acc,
        "margin": float(np.mean(margins)),
        "memory": float(np.mean(memories)),
        "crossmodal": float(np.mean(crossmodal)),
        "energy": float(np.mean(energies)),
        "new_edges": new_edges,
        "low_capacity_groups": low_capacity,
        "complexity": float(complexity),
    }


def load_genome(path):
    d = np.load(path)
    return {"scale": d["scale"], "delta": d["delta"], "cap": d["cap"], "exc": d["exc"], "oi": d["oi"], "oo": d["oo"], "rr": d["rr"], "leak": float(d["leak"][0]), "sigma": float(d["sigma"][0])}


def save_genome(path, g):
    np.savez_compressed(path, scale=g["scale"], delta=g["delta"], cap=g["cap"], exc=g["exc"], oi=g["oi"], oo=g["oo"], rr=g["rr"], leak=np.array([g["leak"]], np.float32), sigma=np.array([g["sigma"]], np.float32))


def robust_score(g, base, task_sets):
    results = [score(g, base, t) for t in task_sets]
    s = np.array([x[0] for x in results])
    robust = float(s.mean() - 0.25 * s.std() + 0.15 * s.min())
    metrics = {k: float(np.mean([x[1][k] for x in results])) for k in results[0][1]}
    metrics.update(environment_score_mean=float(s.mean()), environment_score_std=float(s.std()), environment_score_min=float(s.min()))
    return robust, metrics


def main():
    t0 = time.time()
    n, e, names, base = load_connectome()
    meta_path = STATE_DIR / "state.json"
    genome_path = STATE_DIR / "best_genome.npz"
    if meta_path.exists() and genome_path.exists():
        prior = json.loads(meta_path.read_text())
        root = load_genome(genome_path)
        cycle = int(prior.get("cycle", 0)) + 1
        before = int(prior.get("total_generations", 0))
        print("RESUME", json.dumps({"cycle": cycle, "total_generations": before}))
    else:
        prior = {}
        cycle, before = 1, 0
        root = init_genome(np.random.default_rng(BASE_SEED))
        print("START_NEW_LINEAGE")

    rng = np.random.default_rng(BASE_SEED + cycle * 104729)
    train_sets = [make_tasks(BASE_SEED + cycle * 101 + x, 64) for x in (1, 17, 53)]
    heldout = make_tasks(BASE_SEED + cycle * 101 + 9973, 128)
    pop = [clone(root)] + [mutate(root, rng, 0.55) for _ in range(POP - 1)]
    best = clone(root)
    best_score, best_metrics = robust_score(best, base, train_sets)
    history, stagnant = [], 0

    for gen in range(GENS + 1):
        ev = [robust_score(g, base, train_sets) for g in pop]
        scores = np.array([x[0] for x in ev])
        bi = int(scores.argmax())
        if scores[bi] > best_score + 1e-10:
            best_score, best, best_metrics, stagnant = float(scores[bi]), clone(pop[bi]), ev[bi][1], 0
        else:
            stagnant += 1
        if gen % 10 == 0 or gen == GENS:
            hs, hm = score(best, base, heldout)
            row = {"cycle": cycle, "local_generation": gen, "total_generation": before + gen, "train_robust_score": best_score, "heldout_score": hs, "heldout_accuracy": hm["accuracy"], "new_edges": best_metrics["new_edges"], "low_capacity_groups": best_metrics["low_capacity_groups"], "mutation_sigma": best["sigma"]}
            history.append(row)
            print("CONTINUOUS_EVOLVE", json.dumps(row))
        if gen == GENS:
            break
        elite_n = max(3, POP // 8)
        elite = np.argsort(scores)[-elite_n:][::-1]
        nxt = [clone(pop[int(i)]) for i in elite]
        mutation_rate = min(0.95, 0.30 + 0.012 * stagnant)
        def tournament():
            q = rng.integers(0, POP, 5)
            return int(q[np.argmax(scores[q])])
        while len(nxt) < POP:
            nxt.append(mutate(cross(pop[tournament()], pop[tournament()], rng), rng, mutation_rate))
        pop = nxt

    held_score, held_metrics = score(best, base, heldout)
    total = before + GENS
    save_genome(genome_path, best)
    result = {
        "status": "ok", "cycle": cycle, "generations_this_cycle": GENS, "total_generations": total,
        "connectome": {"neurons": n, "edges": e, "groups": len(names)},
        "train_robust_score": best_score, "train_metrics": best_metrics,
        "heldout_score": held_score, "heldout_metrics": held_metrics,
        "lifetime_best_heldout_score": max(float(prior.get("lifetime_best_heldout_score", -1e30)), held_score),
        "structural_genome": {"new_edges": int(np.count_nonzero(np.abs(best["delta"]) > 1e-3)), "low_capacity_groups": int(np.count_nonzero(best["cap"] < 0.08)), "mutation_sigma": float(best["sigma"]), "leak": float(best["leak"])},
        "runtime_sec": time.time() - t0, "history": history,
    }
    meta_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    (STATE_DIR / f"cycle_{cycle:06d}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("CONTINUOUS_FINAL=" + json.dumps(result))


if __name__ == "__main__":
    main()
