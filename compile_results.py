import json, os, numpy as np

CONFIG_LABELS = {
    'C2_vanilla_lora':   ('C2', 'Vanilla LoRA (baseline)'),
    'C3_olora_planned':  ('C3', 'O-LoRA + Planned CTG'),
    'C4_olora_selfgen':  ('C4', 'O-LoRA + Self-Gen CTG'),
    'C5_lori_selfgen':   ('C5', 'LoRI + Self-Gen CTG'),
    'C6_lori_planned':   ('C6', 'LoRI + Planned CTG'),
}
SEEDS = [42, 7, 123]

results = {k: {} for k in CONFIG_LABELS}

for d in os.listdir('results'):
    d_path = os.path.join('results', d)
    log = os.path.join(d_path, 'run_log.json')
    if not os.path.isdir(d_path) or not os.path.exists(log):
        continue
    with open(log) as f:
        data = json.load(f)
    if 'end_time' not in data:
        continue
    seed = data.get('config', {}).get('seed')
    if seed not in SEEDS:
        continue
    cfg_key = next((k for k in CONFIG_LABELS if d.startswith(k)), None)
    if not cfg_key:
        continue
    steps = data.get('steps', {})
    if not steps:
        continue
    last = str(max(int(k) for k in steps.keys()))
    acc = steps[last].get('mean_accuracy', 0)
    # Keep latest completed run per (cfg, seed)
    if seed not in results[cfg_key] or d > results[cfg_key][seed][0]:
        results[cfg_key][seed] = (d, acc)

# Extract only accuracies
for cfg_key in results:
    results[cfg_key] = {s: v[1] for s, v in results[cfg_key].items()}

print()
print('='*72)
print('  SEAL FINAL RESULTS  (3 seeds: 42, 7, 123)')
print('='*72)
print(f"  {'Config':<6}  {'Method':<28}  {'s=42':>7}  {'s=7':>7}  {'s=123':>7}  {'Mean':>7}  {'Std':>7}")
print('  ' + '-'*68)

summary = {}
for cfg_key, (cfg_id, label) in CONFIG_LABELS.items():
    vals = [results[cfg_key].get(s, None) for s in SEEDS]
    vals_clean = [v for v in vals if v is not None]
    mean = float(np.mean(vals_clean)) if vals_clean else 0.0
    std  = float(np.std(vals_clean))  if vals_clean else 0.0
    summary[cfg_id] = {'mean': mean, 'std': std, 'vals': vals}
    row = f"  {cfg_id:<6}  {label:<28}"
    for v in vals:
        row += f"  {v:.4f}" if v is not None else "     --"
    row += f"  {mean:.4f}  {std:.4f}"
    print(row)

print()
print('  CURRICULUM TRANSFER GAPS (CTG):')
print('  ' + '-'*50)
c3 = summary['C3']['mean']
c4 = summary['C4']['mean']
c5 = summary['C5']['mean']
c6 = summary['C6']['mean']
olora_ctg = c3 - c4
lori_ctg  = c6 - c5
print(f"  O-LoRA CTG  (C3 - C4): {olora_ctg:+.4f}  ({'planned BETTER' if olora_ctg>0 else 'selfgen BETTER'})")
print(f"  LoRI   CTG  (C6 - C5): {lori_ctg:+.4f}  ({'planned BETTER' if lori_ctg>0 else 'selfgen BETTER'})")
print()
c2_mean = summary['C2']['mean']
print(f"  Improvement over baseline (C2 mean={c2_mean:.4f}):")
for cfg_id in ['C3', 'C4', 'C5', 'C6']:
    delta = summary[cfg_id]['mean'] - c2_mean
    pct   = (delta / c2_mean * 100) if c2_mean > 0 else 0
    print(f"    {cfg_id}: {delta:+.4f} ({pct:+.1f}%)")
print('='*72)

# Also write a clean summary JSON
out = {}
for cfg_id in ['C2','C3','C4','C5','C6']:
    out[cfg_id] = summary[cfg_id]
with open('results/final_results_summary.json', 'w') as f:
    json.dump(out, f, indent=2)
print("\n  Saved: results/final_results_summary.json")
