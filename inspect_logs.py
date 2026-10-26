import json

# Check C4 run log - curriculum ordering comparison vs C3
with open('results/C4_olora_selfgen_seed42_20260901_104901/run_log.json') as f:
    c4 = json.load(f)

with open('results/C3_olora_planned_seed42_20260901_103037/run_log.json') as f:
    c3 = json.load(f)

with open('results/C2_vanilla_lora_seed42_20260901_101429/run_log.json') as f:
    c2 = json.load(f)

with open('results/C5_lori_selfgen_seed42_20260901_111437/run_log.json') as f:
    c5 = json.load(f)

print("=== CURRICULUM ORDERS ===")
c2_order = c2['config']['curriculum_order_used']
c3_order = c3['config']['curriculum_order_used']
c4_order = c4['config']['curriculum_order_used']
c5_order = c5['config']['curriculum_order_used']
print(f"C2: {c2_order}")
print(f"C3: {c3_order}")
print(f"C4: {c4_order}")
print(f"C5: {c5_order}")
print()
print(f"C2 == C4? {c2_order == c4_order}")
print(f"C3 == C4? {c3_order == c4_order}")
print()

# Check model reasoning for C4 steps
print("=== C4 CURRICULUM CHOICES (all 19 steps) ===")
fallback_count = 0
for step_key, step_data in c4['steps'].items():
    if 'curriculum_choice' in step_data:
        cc = step_data['curriculum_choice']
        reasoning = cc.get('model_reasoning', '')
        # Check if it actually parsed a number from start
        import re
        match = re.match(r'^\s*NUMBER', reasoning.strip())
        if match:
            fallback_count += 1
        print(f"Step {step_key}: chose={cc.get('model_chose')} | starts_with_NUMBER={bool(match)} | reasoning={reasoning[:60]!r}")

print(f"\nTotal fallback (NUMBER at start) count: {fallback_count}/19")

print()
print("=== FINAL ACCURACY COMPARISON ===")
for config_name, log in [("C2", c2), ("C3", c3), ("C4", c4), ("C5", c5)]:
    steps = log['steps']
    last_step = str(max(int(k) for k in steps.keys()))
    if 'accuracy_matrix' in steps[last_step]:
        acc_dict = steps[last_step]['accuracy_matrix']
        mean_acc = sum(acc_dict.values()) / len(acc_dict)
        print(f"{config_name} final mean accuracy: {mean_acc:.4f} (n={len(acc_dict)})")
