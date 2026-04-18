"""Utilities for group-aware data mixing in LightGCN."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

import world


def _parse_alpha_mix(alpha_mix_str: str, aug_labels: list[str]) -> dict[str, float]:
    if not alpha_mix_str:
        raise ValueError("alpha_mix must be provided when group mixing is enabled.")

    # Preferred format: JSON object mapping label -> weight.
    if alpha_mix_str.strip().startswith("{"):
        parsed = json.loads(alpha_mix_str)
        if set(parsed.keys()) != set(aug_labels):
            raise ValueError(
                "alpha_mix dict keys must exactly match augmentation labels. "
                f"Expected {sorted(aug_labels)}, got {sorted(parsed.keys())}."
            )
        mix = {str(k): float(v) for k, v in parsed.items()}
    else:
        values = [float(x.strip()) for x in alpha_mix_str.split(",") if x.strip()]
        if len(values) != len(aug_labels):
            raise ValueError(
                "alpha_mix list length must equal number of augmentation groups. "
                f"Expected {len(aug_labels)}, got {len(values)}."
            )
        ordered_labels = sorted(aug_labels)
        mix = {label: values[i] for i, label in enumerate(ordered_labels)}

    total = float(sum(mix.values()))
    if total <= 0:
        raise ValueError("alpha_mix weights must sum to a positive value.")
    mix = {k: v / total for k, v in mix.items()}
    return mix


def build_group_mixing_state(dataset_n_users: int):
    """Return mixing state dict (weights and source users), or None if disabled."""
    enabled = bool(world.group_mixing_enabled)
    if not enabled:
        return None

    if not world.group_labels_path:
        raise ValueError("--group_labels_pkl is required when --group_mixing is enabled.")
    if not world.feature_name:
        raise ValueError("--feature_name is required when --group_mixing is enabled.")
    if not world.source_group:
        raise ValueError("--source_group is required when --group_mixing is enabled.")

    labels_path = Path(world.group_labels_path).expanduser()
    if not labels_path.is_file():
        raise FileNotFoundError(f"Group labels pickle not found: {labels_path}")

    with labels_path.open("rb") as f:
        all_labels = pickle.load(f)

    if world.feature_name not in all_labels:
        raise KeyError(
            f"Feature '{world.feature_name}' not found in labels pickle. "
            f"Available: {list(all_labels.keys())}"
        )
    label_to_users_raw = all_labels[world.feature_name]

    # Normalize labels and user lists.
    label_to_users = {
        str(label): np.array(sorted({int(u) for u in users}), dtype=np.int64)
        for label, users in label_to_users_raw.items()
    }

    source_label = str(world.source_group)
    if source_label not in label_to_users:
        raise KeyError(
            f"Source label '{source_label}' not found for feature '{world.feature_name}'. "
            f"Available labels: {sorted(label_to_users.keys())}"
        )

    # Keep users in-range for the current dataset.
    label_to_users = {
        label: users[(users >= 0) & (users < dataset_n_users)]
        for label, users in label_to_users.items()
    }
    for label, users in label_to_users.items():
        if users.size == 0:
            raise ValueError(f"Label '{label}' has no users in range [0, {dataset_n_users}).")

    aug_labels = [label for label in label_to_users.keys() if label != source_label]
    if not aug_labels:
        raise ValueError("Need at least one augmentation group (g-1 >= 1).")

    alpha_mix = _parse_alpha_mix(world.alpha_mix, aug_labels)
    alpha_aug = float(world.alpha_aug)
    if alpha_aug < 0:
        raise ValueError("alpha_aug must be non-negative.")

    n_g = {label: int(label_to_users[label].size) for label in label_to_users}
    n_aug = int(sum(n_g[label] for label in aug_labels))
    if n_aug <= 0:
        raise ValueError("n_aug must be positive.")

    # Weight each group's bpr(g) term in: bpr(s) + n_aug*alpha_aug*sum_{g!=s}(alpha_g/n_g)*bpr(g)
    group_weight = {source_label: 1.0}
    for label in aug_labels:
        group_weight[label] = n_aug * alpha_aug * (alpha_mix[label] / n_g[label])

    user_weights = np.zeros(dataset_n_users, dtype=np.float32)
    user_to_group = np.array([""] * dataset_n_users, dtype=object)
    for label, users in label_to_users.items():
        user_to_group[users] = label
        user_weights[users] = float(group_weight[label])

    uncovered = np.where(user_to_group == "")[0]
    if uncovered.size > 0:
        raise ValueError(
            f"{uncovered.size} users are not covered by group labels for "
            f"feature '{world.feature_name}'."
        )

    source_users = set(label_to_users[source_label].tolist())
    world.cprint(
        "group mixing enabled | "
        f"feature={world.feature_name}, source={source_label}, alpha_aug={alpha_aug}"
    )
    print(f"group sizes: {n_g}")
    print(f"alpha_mix (normalized): {alpha_mix}")
    print(f"group weights: {group_weight}")

    return {
        "user_weights": user_weights,
        "source_users": source_users,
        "label_to_users": label_to_users,
        "alpha_mix": alpha_mix,
        "alpha_aug": alpha_aug,
        "source_label": source_label,
    }
