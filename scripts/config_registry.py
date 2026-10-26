"""
scripts/config_registry.py

WP1 / WP3: Single source of truth for config ids, file paths, and display labels.
Imported by compile_results.py and run_multi_seeds.py to eliminate duplicate dicts.
"""
from __future__ import annotations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Protocol 1 configs (v1 experiments — original 5-config suite)
# ---------------------------------------------------------------------------
CONFIGS_V1 = {
    "C2": {
        "label": "C2 Vanilla LoRA",
        "config_file": ROOT / "configs" / "config_c2_vanilla.json",
        "loop_type": "vanilla",
        "curriculum": "self_generated",
        "protocol": 1,
    },
    "C3": {
        "label": "C3 O-LoRA Planned",
        "config_file": ROOT / "configs" / "config_c3_olora_planned.json",
        "loop_type": "olora",
        "curriculum": "planned",
        "protocol": 1,
    },
    "C4": {
        "label": "C4 O-LoRA Self-Gen",
        "config_file": ROOT / "configs" / "config_c4_olora_selfgen.json",
        "loop_type": "olora",
        "curriculum": "self_generated",
        "protocol": 1,
    },
    "C5": {
        "label": "C5 LoRI Self-Gen",
        "config_file": ROOT / "configs" / "config_c5_lori_selfgen.json",
        "loop_type": "lori",
        "curriculum": "self_generated",
        "protocol": 1,
    },
    "C6": {
        "label": "C6 LoRI Planned",
        "config_file": ROOT / "configs" / "config_c6_lori_planned.json",
        "loop_type": "lori",
        "curriculum": "planned",
        "protocol": 1,
    },
}

# ---------------------------------------------------------------------------
# Protocol 2 configs (v2 experiments — new keys, control curricula C7-C10)
# ---------------------------------------------------------------------------
CONFIGS_V2 = {
    "C2": {
        "label": "C2 Vanilla LoRA (v2)",
        "config_file": ROOT / "configs" / "v2" / "config_c2_vanilla.json",
        "loop_type": "vanilla",
        "curriculum": "self_generated",
        "protocol": 2,
    },
    "C3": {
        "label": "C3 O-LoRA Planned (v2)",
        "config_file": ROOT / "configs" / "v2" / "config_c3_olora_planned.json",
        "loop_type": "olora",
        "curriculum": "planned",
        "protocol": 2,
    },
    "C4": {
        "label": "C4 O-LoRA Self-Gen (v2)",
        "config_file": ROOT / "configs" / "v2" / "config_c4_olora_selfgen.json",
        "loop_type": "olora",
        "curriculum": "self_generated",
        "protocol": 2,
    },
    "C5": {
        "label": "C5 LoRI Self-Gen (v2)",
        "config_file": ROOT / "configs" / "v2" / "config_c5_lori_selfgen.json",
        "loop_type": "lori",
        "curriculum": "self_generated",
        "protocol": 2,
    },
    "C6": {
        "label": "C6 LoRI Planned (v2)",
        "config_file": ROOT / "configs" / "v2" / "config_c6_lori_planned.json",
        "loop_type": "lori",
        "curriculum": "planned",
        "protocol": 2,
    },
    "C7": {
        "label": "C7 O-LoRA Random (v2)",
        "config_file": ROOT / "configs" / "v2" / "config_c7_olora_random.json",
        "loop_type": "olora",
        "curriculum": "random",
        "protocol": 2,
    },
    "C8": {
        "label": "C8 O-LoRA Dispersed (v2)",
        "config_file": ROOT / "configs" / "v2" / "config_c8_olora_dispersed.json",
        "loop_type": "olora",
        "curriculum": "dispersed",
        "protocol": 2,
    },
    "C9": {
        "label": "C9 LoRI Random (v2)",
        "config_file": ROOT / "configs" / "v2" / "config_c9_lori_random.json",
        "loop_type": "lori",
        "curriculum": "random",
        "protocol": 2,
    },
    "C10": {
        "label": "C10 LoRI Dispersed (v2)",
        "config_file": ROOT / "configs" / "v2" / "config_c10_lori_dispersed.json",
        "loop_type": "lori",
        "curriculum": "dispersed",
        "protocol": 2,
    },
}

# ---------------------------------------------------------------------------
# All seeds used in multi-seed runs
# ---------------------------------------------------------------------------
SEEDS_V1 = [42, 7, 123]
SEEDS_V2 = [42, 7, 123]

# ---------------------------------------------------------------------------
# Folder prefix -> config id mapping (for auto-detecting runs)
# ---------------------------------------------------------------------------
FOLDER_PREFIX_TO_CONFIG = {
    "C2_vanilla_lora": "C2",
    "C3_olora_planned": "C3",
    "C4_olora_selfgen": "C4",
    "C5_lori_selfgen": "C5",
    "C6_lori_planned": "C6",
    "C7_olora_random": "C7",
    "C8_olora_dispersed": "C8",
    "C9_lori_random": "C9",
    "C10_lori_dispersed": "C10",
}


def config_for_run_id(run_id: str) -> str | None:
    """Return the config id (e.g. 'C3') for a run_id, or None if unrecognised."""
    for prefix, cfg_id in FOLDER_PREFIX_TO_CONFIG.items():
        if run_id.startswith(prefix):
            return cfg_id
    return None


def all_configs(protocol: int = 1) -> dict:
    """Return the config registry for the given protocol version."""
    if protocol == 1:
        return CONFIGS_V1
    elif protocol == 2:
        return CONFIGS_V2
    raise ValueError(f"Unknown protocol: {protocol}")
