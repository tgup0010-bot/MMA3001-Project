"""occupancy: room-occupancy pattern modelling for MMA3001.

Predicts short-term room occupancy from historical sensor event logs, to
support energy-aware HVAC/lighting control decisions in the Monash Smart
Infrastructure building.

See the project README and ``docs/report`` for the full engineering context,
validation methodology and results.
"""

from occupancy.baselines import MarkovBaseline, persistence_predict_proba
from occupancy.data_loading import load_occupancy_log
from occupancy.evaluation import chronological_split, evaluate_predictions
from occupancy.features import build_supervised_dataset
from occupancy.model import build_logistic_pipeline
from occupancy.preprocessing import resample_occupancy

__all__ = [
    "load_occupancy_log",
    "resample_occupancy",
    "build_supervised_dataset",
    "MarkovBaseline",
    "persistence_predict_proba",
    "build_logistic_pipeline",
    "chronological_split",
    "evaluate_predictions",
]

__version__ = "0.1.0"
