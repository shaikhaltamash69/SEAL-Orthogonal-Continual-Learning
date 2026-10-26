import json, os, re
from datetime import datetime

results_dir = 'results'
now = datetime.now()

CONFIG_PREFIX = {
    'C2': 'C2_vanilla_lora',
    'C3': 'C3_olora_planned',
    'C4': 'C4_olora_selfgen',
    'C5': 'C5_lori_selfgen',
    'C6': 'C6_lori_planned',
}
CONFIG_LABEL = {
    'C2': 'Vanilla LoRA',
    'C3': 'O-LoRA Planned',
    'C4': 'O-LoRA Self-Gen',
    'C5': 'LoRI Self-Gen',
    'C6': 'LoRI Planned',
}
ALL_CONFIGS = ['C2','C3','C4','C5','C6']
ALL_SEEDS   = [42, 7, 123]
TOTAL_RUNS  = len(ALL_CONFIGS) * len(ALL_SEEDS)

# Build a map: (cfg_key, seed) -> run info
run_map = {}

for d in os.listdir(results_dir):
    d_path = os.path.join(results_dir, d)
    if not os.path.isdir(d_path):
        continue
    log_path = os.path.join(d_path, 'run_log.json')
    if not os.path.exists(log_path):
        continue
    try:
        with open(log_path) as f:
            data = json.load(f)
    except Exception:
        continue

    # Match which config this is
    cfg_key = None
    for k, prefix in CONFIG_PREFIX.items():
        if d.startswith(prefix):
            cfg_key = k
            break
    if cfg_key is None:
        continue

    seed = data.get('config', {}).get('seed')
    if seed not in ALL_SEEDS:
        continue

    steps   = data.get('steps', {})
    n_done  = len(steps)
    finished = 'end_time' in data

    final_acc = None
    if n_done > 0:
        last_key = str(max(int(k) for k in steps.keys()))
        final_acc = steps[last_key].get('mean_accuracy', 0)

    mtime = os.path.getmtime(log_path)
    last_update = datetime.fromtimestamp(mtime)
    secs_ago = int((now - last_update).total_seconds())

    run_map[(cfg_key, seed)] = {
        'dir': d,
        'n_done': n_done,
        'finished': finished,
        'final_acc': final_acc,
        'secs_ago': secs_ago,
    }

# ── Print report ─────────────────────────────────────────────────
print("=" * 65)
print(f"  SEAL Multi-Seed Run — Status at {now.strftime('%H:%M:%S')}")
print("=" * 65)

completed = [(k, v) for k, v in run_map.items() if v['finished']]
running   = [(k, v) for k, v in run_map.items() if not v['finished']]
pending_pairs = [
    (cfg, seed) for cfg in ALL_CONFIGS for seed in ALL_SEEDS
    if (cfg, seed) not in run_map
]

print(f"\n  Progress: {len(completed)}/{TOTAL_RUNS} complete, "
      f"{len(running)} running, {len(pending_pairs)} pending\n")

# Completed
for (cfg, seed), v in sorted(completed, key=lambda x: x[0]):
    print(f"  [DONE]    {CONFIG_PREFIX[cfg]:<28} seed={seed}  "
          f"{v['n_done']:2}/19 steps  acc={v['final_acc']:.4f}")

# Running
for (cfg, seed), v in running:
    steps_left = 19 - v['n_done']
    print(f"  [RUNNING] {CONFIG_PREFIX[cfg]:<28} seed={seed}  "
          f"{v['n_done']:2}/19 steps  ~{steps_left} left  "
          f"(last update {v['secs_ago']}s ago)")

# Pending
if pending_pairs:
    print(f"\n  Pending ({len(pending_pairs)} runs):")
    for cfg, seed in pending_pairs:
        print(f"    - {CONFIG_PREFIX[cfg]}  seed={seed}")

# Matrix
print(f"\n  {'Config':<18}", end='')
for s in ALL_SEEDS:
    print(f"  seed={s:3d}", end='')
print()
print("  " + "-" * 52)
for cfg in ALL_CONFIGS:
    print(f"  {CONFIG_LABEL[cfg]:<18}", end='')
    for seed in ALL_SEEDS:
        info = run_map.get((cfg, seed))
        if info is None:
            cell = "  pending "
        elif not info['finished']:
            cell = f"  {info['n_done']:2}/19 RUN"
        else:
            cell = f"  {info['final_acc']:.4f}  "
        print(cell, end='')
    print()

# Quick ETA estimate (based on ~2min/step observed)
mins_per_step_c2   = 2.1
mins_per_step_c345 = 3.5   # O-LoRA/LoRI slower due to penalty
steps_remaining = 0
for cfg, seed in pending_pairs:
    factor = mins_per_step_c2 if cfg == 'C2' else mins_per_step_c345
    steps_remaining += 19 * factor
for (cfg, seed), v in running:
    factor = mins_per_step_c2 if cfg == 'C2' else mins_per_step_c345
    steps_remaining += (19 - v['n_done']) * factor

print(f"\n  Estimated time remaining: ~{steps_remaining/60:.1f} hours "
      f"({int(steps_remaining)} min)")
print()
