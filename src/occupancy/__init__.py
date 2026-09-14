"""occupancy: room-occupancy pattern modelling for MMA3001.

Predicts short-term room occupancy from historical sensor event logs, to
support energy-aware HVAC/lighting control decisions in the Monash Smart
Infrastructure building.

See the project README and ``docs/report`` for the full engineering context,
validation methodology and results.
"""

from occupancy.baselines import MarkovBaseline, persistence_predict_proba
from occupancy.data_loading import load_occupancy_log
from occupancy.env_sensors import load_env_sensor_log, pivot_variable
from occupancy.evaluation import chronological_split, evaluate_predictions
from occupancy.external_weather import load_bom_weather
from occupancy.features import build_supervised_dataset
from occupancy.model import build_logistic_pipeline, build_random_forest_pipeline
from occupancy.preprocessing import resample_occupancy
from occupancy.rooms import room_label
from occupancy.sensor_matching import build_correlation_matrix, lagged_correlation
from occupancy.sensors import sensor_label
from occupancy.weather_comparison import compare_to_bom, daily_indoor_series

__all__ = [
    "load_occupancy_log",
    "resample_occupancy",
    "build_supervised_dataset",
    "MarkovBaseline",
    "persistence_predict_proba",
    "build_logistic_pipeline",
    "build_random_forest_pipeline",
    "chronological_split",
    "evaluate_predictions",
    "room_label",
    "load_env_sensor_log",
    "pivot_variable",
    "sensor_label",
    "build_correlation_matrix",
    "lagged_correlation",
    "load_bom_weather",
    "compare_to_bom",
    "daily_indoor_series",
]

__version__ = "0.1.0"
