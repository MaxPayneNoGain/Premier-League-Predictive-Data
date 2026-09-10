"""Pulling predictions back toward the base rate.

The goals model discriminates far better than the base rates it is scored
against and states its beliefs less honestly. Blending the two trades the
resolution it has to spare for the reliability it lacks.
"""

from plpd.models.poisson import Vector

# Swept over the 2024-2025 folds. Brier, log loss and RPS all bottom out between
# 0.20 and 0.25, so no metric is being favoured over another here.
SHRINKAGE_WEIGHT = 0.20


def shrink(probabilities: Vector, towards: Vector, weight: float) -> Vector:
    """Blend predictions with a reference, `weight` of 1.0 discarding the model."""
    if not 0.0 <= weight <= 1.0:
        raise ValueError(f"weight must be between 0 and 1, got {weight}")

    blended: Vector = (1.0 - weight) * probabilities + weight * towards
    return blended
