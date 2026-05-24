"""
Deterministic dataset splitter with small-dataset protection.

Split policy
------------
Documents are shuffled using Python's random.Random seeded with `split_seed`
(default 42), then sliced sequentially:
  [0 : train_end]        → train
  [train_end : val_end]  → validation  (optimization objective)
  [val_end : ]           → test         (held-out, evaluated once at the end)

Small-dataset protection
------------------------
When the dataset is too small for the requested ratios to produce at least
1 document per non-empty split, the splitter redistributes automatically:
  - val  always gets ≥ 1 doc (required for the optimization loop to function)
  - test always gets ≥ 1 doc (required for the final evaluation)
  - train gets the remainder (may be 0 on a 2-document dataset)

A warning is printed when redistribution occurs so the user is aware.
"""

import random
from typing import Any, Dict, List, Tuple


def deterministic_split(
    data: List[Dict[str, Any]],
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> Tuple[List, List, List]:
    """
    Shuffle `data` with a fixed seed and return (train, val, test) splits.

    Parameters
    ----------
    data        : List of document dicts.
    seed        : RNG seed for full reproducibility.
    train_ratio : Target fraction for training.
    val_ratio   : Target fraction for validation.

    Returns
    -------
    (train, val, test) — each is a non-empty list when len(data) >= 2.
    """
    if not data:
        return [], [], []

    rng = random.Random(seed)
    shuffled = data.copy()
    rng.shuffle(shuffled)

    total = len(shuffled)

    # Ideal split sizes
    train_end = int(total * train_ratio)
    val_end   = train_end + int(total * val_ratio)

    # ---------- Small-dataset protection ----------
    # We need at least 1 val doc and 1 test doc for the pipeline to function.
    # Redistribute when the dataset is too small for the requested ratios.
    if total >= 3:
        # Normal case: honour ratios, clamp to valid indices
        train_end = max(0, min(train_end, total - 2))
        val_end   = max(train_end + 1, min(val_end, total - 1))

    elif total == 2:
        # 2 documents: 0 train | 1 val | 1 test
        train_end = 0
        val_end   = 1
        print(
            f"  ⚠️  SMALL DATASET: only {total} documents loaded. "
            f"Using 0 train | 1 val | 1 test (ratios ignored)."
        )

    else:  # total == 1
        # 1 document: use it for both val and test (degenerate case)
        train_end = 0
        val_end   = 1
        print(
            f"  ⚠️  TINY DATASET: only {total} document loaded. "
            f"Using the same document for val and test."
        )
        return [], shuffled[:1], shuffled[:1]

    train = shuffled[:train_end]
    val   = shuffled[train_end:val_end]
    test  = shuffled[val_end:]

    # If redistribution changed the ratio significantly, warn the user
    actual_train = len(train) / total
    actual_val   = len(val)   / total
    if abs(actual_train - train_ratio) > 0.15 or abs(actual_val - val_ratio) > 0.15:
        print(
            f"  ⚠️  Split ratios adjusted for small dataset ({total} docs). "
            f"Actual: {len(train)} train | {len(val)} val | {len(test)} test"
        )

    return train, val, test