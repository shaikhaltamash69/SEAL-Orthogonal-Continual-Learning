import json, os

runs = [
    'results/C2_vanilla_lora_seed42_20261001_080302',
    'results/C2_vanilla_lora_seed7_20261001_082423',
]

for run_dir in runs:
    log_path = os.path.join(run_dir, 'run_log.json')
    if not os.path.exists(log_path):
        continue
    with open(log_path) as f:
        data = json.load(f)
    run_id = data['run_id']
    steps = data.get('steps', {})

    print(f'================== {run_id} ==================')
    print(f'Total steps: {len(steps)}')
    print()

    # Show synthetic data quality for ALL steps
    bad_steps = []
    good_steps = []
    for sk in sorted(steps.keys(), key=int):
        sd = steps[sk]
        synth = sd.get('synthetic_data', '')
        loss = sd.get('loss', {}).get('sft_loss', '?')
        # Quality check: repetition is a sign of hallucination/degeneration
        words = synth.split()
        if len(words) > 0:
            unique_ratio = len(set(words)) / len(words)
        else:
            unique_ratio = 0

        quality = 'GOOD' if unique_ratio > 0.4 else 'DEGENERATE'
        if unique_ratio <= 0.4:
            bad_steps.append(int(sk))
        else:
            good_steps.append(int(sk))

        acc = sd.get('accuracy_matrix', {})
        mean_acc = sum(acc.values()) / len(acc) if acc else 0

        print(f'  Step {sk}: loss={loss:.4f} | unique_word_ratio={unique_ratio:.2f} | quality={quality} | mean_acc={mean_acc:.4f}')
        if quality == 'DEGENERATE':
            print(f'    First 150 chars: {synth[:150]!r}')

    print()
    print(f'  SUMMARY: {len(good_steps)} good steps, {len(bad_steps)} DEGENERATE steps')
    print(f'  Degenerate at steps: {bad_steps}')
    print()
