# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Sentinel-1 external .EOF orbit file reader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from lxml import etree
from perseo_core.timing import PreciseDateTime

from eo_products.common.utilities import StateVectors
from eo_products.sentinel1.common import InvalidTimeBoundaries, S1OrbitType, S1ReferenceFrameType


def read_external_orbit(
    xml_path: str | Path,
    time_boundaries: tuple[PreciseDateTime, PreciseDateTime] | None = None,
) -> StateVectors:
    """Read Sentinel-1 external .EOF orbit file.

    External orbits can be very large and span over several hours so
    time boundaries can be provided to limit the orbit extent just around the acquisition time interval.

    Parameters
    ----------
    xml_path : str | Path
        path to the external orbit file
    time_boundaries : tuple[PreciseDateTime, PreciseDateTime] | None, optional
        time boundaries as start time [0] and stop time [1], by default None

    Returns
    -------
    StateVectors
        StateVectors dataclass from orbit data
    """

    xml_path = Path(xml_path)

    # loading the xml file
    root = etree.parse(xml_path).getroot()

    # finding nodes
    fixed_header = root.find("Earth_Explorer_Header/Fixed_Header")
    variable_header = root.find("Earth_Explorer_Header/Variable_Header")
    validity_period = fixed_header.find("Validity_Period")
    state_vectors_list = root.find("Data_Block/List_of_OSVs")

    # assessing reference frame
    reference_frame = variable_header.find("Ref_Frame").text
    if reference_frame == "EARTH_FIXED":
        reference_frame = S1ReferenceFrameType.EARTH_FIXED

    # retrieving datetimes information
    validity_start = PreciseDateTime.fromisoformat(validity_period.find("Validity_Start").text.split("=")[-1])
    validity_stop = PreciseDateTime.fromisoformat(validity_period.find("Validity_Stop").text.split("=")[-1])

    if time_boundaries is not None:
        if time_boundaries[0] < validity_start or time_boundaries[1] > validity_stop:
            raise InvalidTimeBoundaries(
                f"Time boundaries {time_boundaries} are out of validity period {(validity_start, validity_stop)}"
            )

    # retrieving state vectors data
    state_vectors_num = int(state_vectors_list.get("count"))
    positions = np.zeros((state_vectors_num, 3))
    velocities = np.zeros((state_vectors_num, 3))
    times_for_assessing_delta = []
    for idx, item in enumerate(state_vectors_list):
        positions[idx] = [
            float(item.find("X").text),
            float(item.find("Y").text),
            float(item.find("Z").text),
        ]
        velocities[idx] = [
            float(item.find("VX").text),
            float(item.find("VY").text),
            float(item.find("VZ").text),
        ]
        if idx < 2:
            # taking the first two state vectors reference times to compute the time delta for the time axis
            times_for_assessing_delta.append(PreciseDateTime.from_utc_string(item.find("UTC").text.replace("UTC=", "")))

    # computing time axis
    time_delta = times_for_assessing_delta[1] - times_for_assessing_delta[0]
    # using the first state vector reference time as start time for the time axis
    time_axis = np.arange(state_vectors_num) * time_delta + times_for_assessing_delta[0]

    # orbit type
    orbit_type = fixed_header.find("File_Type").text
    if orbit_type == "AUX_PREORB":
        orbit_type = S1OrbitType.PREDICTED
    elif orbit_type == "AUX_RESORB":
        orbit_type = S1OrbitType.RESTITUTED
    elif orbit_type == "AUX_POEORB":
        orbit_type = S1OrbitType.PRECISE

    if time_boundaries is None:
        return StateVectors(
            num=state_vectors_num,
            time_axis=time_axis,
            positions=positions,
            velocities=velocities,
            reference_frame=reference_frame,
            time_step=time_delta,
            orbit_type=orbit_type,
        )

    # cropping orbit only between provided time boundaries
    start_index = np.where(time_axis > time_boundaries[0])[0][0] - 1
    stop_index = np.where(time_axis > time_boundaries[1])[0][0] + 1
    time_axis = time_axis[start_index:stop_index]
    positions = positions[start_index:stop_index]
    velocities = velocities[start_index:stop_index]
    state_vectors_num = stop_index - start_index

    return StateVectors(
        num=state_vectors_num,
        time_axis=time_axis,
        positions=positions,
        velocities=velocities,
        reference_frame=reference_frame,
        time_step=time_delta,
        orbit_type=orbit_type,
    )
