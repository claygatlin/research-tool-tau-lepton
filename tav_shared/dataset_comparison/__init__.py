"""
Standardized dataset comparison for MC vs empirical and model vs data pipelines.

Primary-key alignment, feature scaling, and tolerance-aware numeric diffs.
"""

from tav_shared.dataset_comparison.pipeline import (
    DatasetComparisonReport,
    compare_aggregate_metrics,
    compare_dimuon_kinematics,
    compare_histogram_pair,
    compare_model_predictions_to_observed,
    prepare_datasets_for_comparison,
)
from tav_shared.dataset_comparison.scaling import MinMaxScaler, StandardScaler
from tav_shared.dataset_comparison.tolerance import (
    compare_mass_gap_mev,
    tolerant_allclose,
    tolerant_diff,
)

__all__ = [
    "DatasetComparisonReport",
    "MinMaxScaler",
    "StandardScaler",
    "compare_aggregate_metrics",
    "compare_dimuon_kinematics",
    "compare_histogram_pair",
    "compare_mass_gap_mev",
    "compare_model_predictions_to_observed",
    "prepare_datasets_for_comparison",
    "tolerant_allclose",
    "tolerant_diff",
]