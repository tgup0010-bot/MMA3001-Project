"""occupancy: room-occupancy pattern modelling for MMA3001.

Predicts short-term room occupancy from historical sensor event logs, to
support energy-aware HVAC/lighting control decisions in the Monash Smart
Infrastructure building.

See the project README and ``docs/report`` for the full engineering context,
validation methodology and results.
"""

from occupancy.baselines import MarkovBaseline, persistence_predict_proba
from occupancy.co2_models import (
    build_decision_tree_regression_pipeline,
    build_linear_regression_pipeline,
    build_neural_network_regression_pipeline,
    build_svr_pipeline,
)
from occupancy.co2_regression import build_co2_supervised_dataset, building_occupancy_series, co2_series
from occupancy.data_loading import load_occupancy_log
from occupancy.env_sensors import load_env_sensor_log, pivot_variable
from occupancy.evaluation import chronological_split, evaluate_predictions, evaluate_regression_predictions
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
    "build_linear_regression_pipeline",
    "build_decision_tree_regression_pipeline",
    "build_svr_pipeline",
    "build_neural_network_regression_pipeline",
    "building_occupancy_series",
    "co2_series",
    "build_co2_supervised_dataset",
    "evaluate_regression_predictions",
]

__version__ = "0.1.0"
