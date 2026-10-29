# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Sentinel-1 ZARR reader support module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr
from numpy.polynomial import Polynomial
from perseo_core.timing import PreciseDateTime

from eo_products.common.utilities import (
    ConversionFunction,
    CoordinatesConversions,
    DopplerEvaluator,
)
from eo_products.sentinel1.common import S1AntennaPattern, S1DCEstimateMethod, S1Noise

PRODUCT_FORMAT_EXTENSION = ".zarr"


class S1ZarrProduct:
    """Sentinel-1 Zarr product object"""

    def __init__(self, path: str | Path) -> None:
        self._product_path = Path(path)
        self._product_name = self._product_path.name.replace(PRODUCT_FORMAT_EXTENSION, "")

        dt = xr.open_datatree(self._product_path, engine="zarr", chunks={})

        self.sensor = dt.attrs["stac_discovery"]["properties"]["platform"].replace("sentinel-", "s")

        # acquisition time
        self._acq_time = PreciseDateTime.fromisoformat(dt.attrs["stac_discovery"]["properties"]["datetime"])

        # footprint
        bbox_lat = [
            dt.attrs["stac_discovery"]["bbox"][1],
            dt.attrs["stac_discovery"]["bbox"][3],
        ]
        bbox_lon = [
            dt.attrs["stac_discovery"]["bbox"][0],
            dt.attrs["stac_discovery"]["bbox"][2],
        ]
        self._footprint = [min(bbox_lat), max(bbox_lat), min(bbox_lon), max(bbox_lon)]

        # TODO: check with Wave products
        self._internal_structure_map = self._internal_structure_mapping(groups=dt.groups)

        first_key = next(iter(self._internal_structure_map))
        first_item = self._internal_structure_map[first_key][0]
        processing_history: dict[str, list[dict[str]]] = dt[first_item].attrs["processing_history"]

        adf = next(
            (item["adfs"] for history in processing_history.values() for item in history if "adfs" in item),
            None,
        )
        self._adf = {
            "AUX_PP1": next((p for p in adf if "AUX_PP1" in p), None),
            "AUX_CAL": next((p for p in adf if "AUX_CAL" in p), None),
            "AUX_INS": next((p for p in adf if "AUX_INS" in p), None),
            "ORBIT": next((p for p in adf if "ORB_" in p), None),
        }

        dt.close()

    def _channel_pattern_from_group(self, path: str) -> str:
        """Get the channel pattern pol_swath from from full group name."""
        name_parts = path.replace("/measurements", "").split("_")
        return "_".join([name_parts[-3], name_parts[-2]]).lower()

    def _internal_structure_mapping(self, groups: list[str]) -> dict[str, list[str]]:
        """Defining internal channel to groups association structure."""
        group_names = {
            path.replace("/measurements", "").replace("/", "") for path in groups if path.endswith("/measurements")
        }
        unique_channels = sorted(
            {self._channel_pattern_from_group(g) for g in group_names},
            key=lambda x: x.split("_")[-1],
        )

        return {
            "_".join([self.sensor, ch.split("_")[1], ch.split("_")[0]]): sorted(
                [g for g in group_names if ch.replace(self.sensor, "").upper() in g]
            )
            for ch in unique_channels
        }

    @property
    def auxiliary_data_files(self) -> dict[str, str]:
        """Auxiliary data files used to generate this product."""
        return self._adf

    @property
    def internal_structure_map(self) -> dict[str, list[str]]:
        """Zarr channels internal structure map."""
        return self._internal_structure_map

    @property
    def product_path(self) -> Path:
        """Zarr product path."""
        return self._product_path

    @property
    def acquisition_time(self) -> PreciseDateTime:
        """Acquisition start time for this product"""
        return self._acq_time

    @property
    def channels_number(self) -> int:
        """Returning the number of channels in the product"""
        return len(self._internal_structure_map)

    @property
    def channels_list(self) -> list[str]:
        """Returning the list of channels in terms of SwathID (swath-polarization)"""
        return list(self._internal_structure_map.keys())

    @property
    def footprint(self) -> tuple[float, float, float, float]:
        """Product footprint as tuple of (min lat, max lat, min lon, max lon)"""
        return self._footprint


def is_s1_zarr_product(product: str | Path) -> bool:
    """Check if input path corresponds to a valid S1 Zarr product, basic version.

    Conditions to be met for basic validity:
        - path exists
        - path is a directory
        - path ends with Zarr
        - S1ZarrProduct can be instantiated

    Parameters
    ----------
    product : str | Path
        path to the product to be checked

    Returns
    -------
    bool
        True if it is a valid S1 zarr, else False
    """
    product = Path(product)

    if not product.exists() or not product.is_dir():
        return False

    if not product.name.endswith(PRODUCT_FORMAT_EXTENSION):
        return False

    try:
        S1ZarrProduct(product)
    except Exception:
        return False

    return True


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


def coordinates_conversions_from_metadata(coord_conversion_node) -> CoordinatesConversions:
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
    # TODO: implement for GRD
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
