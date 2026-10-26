"""
scripts/config_registry.py

WP1: Single source of truth for config ids, file paths, and display labels.
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
        "curriculum": "planned",
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
# Protocol 2 configs (v2 experiments — new keys, control curricula)
# ---------------------------------------------------------------------------
CONFIGS_V2: dict = {}  # populated by WP3 when v2 configs exist

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
        return {**CONFIGS_V1, **CONFIGS_V2}
    raise ValueError(f"Unknown protocol: {protocol}")
