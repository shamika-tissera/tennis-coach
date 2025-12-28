from __future__ import annotations
from .stroke_types import PoseStroke
from .similarity_dtw import form_distance


def form_similarity_score(d_form: float, d_max: float) -> float:
    """Maps distance to similarity score in [0, 1]."""
    return max(0.0, 1.0 - d_form / d_max)


def score_stroke_against_template(
    user: PoseStroke,
    expert: PoseStroke,
    d_max: float,
) -> tuple[float, float]:
    """Returns (s_form, d_form) for a user stroke versus expert."""
    d_form = form_distance(user, expert)
    s_form = form_similarity_score(d_form, d_max)
    return s_form, d_form
