"""Friendly labels for the environmental sensor identifiers used in the data.

Cross-referencing ``Sensor ID and Locations.xlsx`` against the environmental
sensor log's ``sensorid`` values resolved only 1 of the 5 sensors to a named
physical space (see the report, Limitations). The other 4 are left as their
raw identifier -- inventing a plausible-sounding name for them would
misrepresent what is actually known about the data. Mirrors
:mod:`occupancy.rooms`, which does the same for the occupancy zones.
"""

from __future__ import annotations

#: sensorid -> physical room name, for the 1 sensor that could be traced
#: through the location spreadsheet.
SENSOR_LABELS: dict[str, str] = {
    "6012002000869": "G.38 (Compactus_upper)",
}


def sensor_label(sensorid: str) -> str:
    """Return a human-friendly label for an environmental sensor identifier.

    Args:
        sensorid: The raw sensor identifier from the environmental sensor log.

    Returns:
        The known physical room name if ``sensorid`` is in
        :data:`SENSOR_LABELS`; otherwise ``"Sensor <id> (unidentified)"`` --
        never a made-up room name.
    """
    if sensorid in SENSOR_LABELS:
        return SENSOR_LABELS[sensorid]
    return f"Sensor {sensorid} (unidentified)"
