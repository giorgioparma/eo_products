# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Sentinel-1 ZARR metadata parsing functionalities."""

from __future__ import annotations

import numpy as np
import xarray as xr
from numpy.polynomial import Polynomial
from perseo_core.timing import PreciseDateTime

from eo_products.common.utilities import (
    AcquisitionTimeline,
    ConversionFunction,
    CoordinatesConversions,
    DopplerEvaluator,
)
from eo_products.sentinel1.common import (
    S1AntennaPattern,
    S1DCEstimateMethod,
    S1Noise,
)


def doppler_centroid_from_metadata(
    doppler_centroid: xr.DataTree, estimate_method: S1DCEstimateMethod
) -> DopplerEvaluator:
    """Creating a DopplerEvaluator doppler centroid polynomial wrapper from metadata.

    Parameters
    ----------
    doppler_centroid : xr.DataTree
        doppler centroid group
    estimate_method : S1DCEstimateMethod
        doppler centroid estimate method

    Returns
    -------
    DopplerEvaluator
        DopplerEvaluator dataclass for Doppler Centroid polynomial
    """
    if estimate_method == S1DCEstimateMethod.DATA:
        dc_poly = doppler_centroid["data_dc_polynomial"].values
    else:
        dc_poly = doppler_centroid["geometry_dc_polynomial"].values
    az_times = doppler_centroid["azimuth_time"].values
    origins = doppler_centroid["t0"].values

    doppler_poly_list = []
    for coeff, az_time, origin in zip(dc_poly, az_times, origins, strict=True):
        doppler_poly_list.append(
            ConversionFunction(
                azimuth_reference_time=PreciseDateTime.fromisoformat(str(az_time)),
                origin=origin,
                function=Polynomial(coeff),
            )
        )

    return DopplerEvaluator(
        functions=doppler_poly_list,
        azimuth_reference_times=np.array([c.azimuth_reference_time for c in doppler_poly_list]),
    )


def doppler_rate_from_metadata(azimuth_fm_rate_list: list[dict[str]]) -> DopplerEvaluator:
    """Creating a DopplerEvaluator doppler rate vector polynomial wrapper from metadata.

    Parameters
    ----------
    azimuth_fm_rate_node : etree._Element
        azimuthFmRateList metadata xml node

    Returns
    -------
    DopplerEvaluator
        DopplerEvaluator dataclass for Doppler Rate polynomial
    """
    doppler_poly_list = []
    for item in azimuth_fm_rate_list:
        coefficients = [float(x) for x in next(iter(item["azimuth_fm_rate_polynomial"].values())).split()]
        doppler_poly_list.append(
            ConversionFunction(
                azimuth_reference_time=PreciseDateTime.from_utc_string(item["azimuth_time"]),
                origin=item["t_0"],
                function=Polynomial(coefficients),
            )
        )
    return DopplerEvaluator(
        functions=doppler_poly_list,
        azimuth_reference_times=np.array([c.azimuth_reference_time for c in doppler_poly_list]),
    )


def coordinates_conversions_from_metadata(coord_conversion_list) -> CoordinatesConversions:
    """Generating CoordinatesConversions object from metadata.

    About coefficients annotated in metadata:

    Coefficients of polynomial fit to the "Ground to Slant Range" transform applied. Fixed along all image slices.
    Defined with respect to (slant range in m)/(pixel no.)^n where first pixel in line is 0.

    Values output in order A0, A1, ...,  An, in order of increasing degree.
    Polynomial to be evaluated is:  A_0+ A_1 x + ... + A_n x^n
    where x is pixel number in the line, starting from 0.

    Evaluated polynomial gives Slant Range in meters.

    Parameters
    ----------
    coord_conversion_node : etree._Element
        coordinateConversion xml node

    Returns
    -------
    CoordinatesConversions
        polynomial for coordinate conversion dataclass
    """
    # polynomials written in metadata are supposed to be always already sorted in time, i.e. azimuth validity
    # reference time will be strictly increasing (no need to be sorted afterwards)

    return CoordinatesConversions()


def antenna_pattern_from_metadata(antenna_pattern: xr.DataTree) -> dict[str, S1AntennaPattern]:
    """Creating a dictionary of S1AntennaPattern dataclass for each swath from safe xml node.

    Parameters
    ----------
    antenna_pattern_list_node : xr.DataTree
        antenna pattern metadata group

    Returns
    -------
    S1AntennaPattern
        antenna pattern data for the current swath
    """
    return S1AntennaPattern(
        swath=antenna_pattern.swath.values[0],
        time_axis=np.array([PreciseDateTime.fromisoformat(str(t)) for t in antenna_pattern.azimuth_time.values]),
        slant_range_time_array=antenna_pattern.slant_range_time_ap.values,
        elevation_angle_array=antenna_pattern.elevation_angle.values,
        # elevation_pattern_array=antenna_pattern.elevation_pattern.values,
        elevation_pattern_array=None,
        incidence_angle=antenna_pattern.incidence_angle.values,
        terrain_height=antenna_pattern.terrain_height.values,
        roll=antenna_pattern.roll.values,
    )


def noise_from_metadata(noise_list: list[dict[str]], swath: str) -> S1Noise:
    """Generating S1Noise object directly from metadata xml nodes.

    Parameters
    ----------
    noise_list_node : etree._Element
        noiseList metadata xml node
    swath : str
        swath of interest

    Returns
    -------
    S1Noise
        noise correction factor dataclass
    """

    # taking only the nodes corresponding to the selected swath
    filtered_nodes = filter(lambda x: x["swath"] == swath, noise_list)

    time_axis = []
    noise_power_correction_factor = []
    noise_lines_num = []
    for item in filtered_nodes:
        time_axis.append(PreciseDateTime.fromisoformat(item["azimuth_time"]))
        noise_power_correction_factor.append(item["noise_power_correction_factor"])
        noise_lines_num.append(item["number_of_noise_lines"])

    return S1Noise(
        swath=swath,
        time_axis=np.array(time_axis),
        noise_power_correction_factor=np.array(noise_power_correction_factor),
        noise_lines_num=np.array(noise_lines_num),
    )


def acquisition_timeline_from_metadata(downlink_information_list) -> AcquisitionTimeline:
    """Creating a AcquisitionTimeline metadata element from safe xml nodes.

    Parameters
    ----------
    downlink_info_node : etree._Element
        downlinkInformation metadata xml node

    Returns
    -------
    AcquisitionTimeline
        AcquisitionTimeline metadata object
    """
    swst_list = [c["downlink_values"]["swst_list"] for c in downlink_information_list]
    swl_list = [c["downlink_values"]["swl_list"] for c in downlink_information_list]

    # missing elements set with these dummy values
    missing_lines_number = 0
    missing_lines_azimuth_times = None
    noise_packets_number = 0
    noise_packets_azimuth_times = None
    internal_calibration_number = 0
    internal_calibration_azimuth_times = None

    swst_azimuth_times = []
    swst_values = []
    for swst in swst_list:
        swst_azimuth_times.extend([PreciseDateTime.fromisoformat(s["azimuth_time"]) for s in swst])
        swst_values.extend([s["value"] for s in swst])
    swst_changes_number = len(swst_values)

    swl_azimuth_times = []
    swl_values = []
    for swl in swl_list:
        swl_azimuth_times.extend([PreciseDateTime.fromisoformat(s["azimuth_time"]) for s in swl])
        swl_values.extend([s["value"] for s in swl])
    swl_changes_number = len(swl_values)

    return AcquisitionTimeline(
        missing_lines_number=missing_lines_number,
        missing_lines_azimuth_times=missing_lines_azimuth_times,
        swst_changes_number=swst_changes_number,
        swst_changes_azimuth_times=swst_azimuth_times,
        swst_changes_values=swst_values,
        noise_packets_number=noise_packets_number,
        noise_packets_azimuth_times=noise_packets_azimuth_times,
        internal_calibration_number=internal_calibration_number,
        internal_calibration_azimuth_times=internal_calibration_azimuth_times,
        swl_changes_number=swl_changes_number,
        swl_changes_azimuth_times=swl_azimuth_times,
        swl_changes_values=swl_values,
    )
