# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Sentinel-1 ZARR reader support module."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt
import xarray as xr
from numpy.polynomial import Polynomial
from perseo_core.timing import PreciseDateTime

from eo_products.common.utilities import (
    AcquisitionTimeline,
    ConversionFunction,
    CoordinatesConversions,
    DopplerEvaluator,
    SARPolarization,
)
from eo_products.sentinel1.common import (
    S1AcquisitionMode,
    S1AntennaPattern,
    S1DCEstimateMethod,
    S1L1ProductType,
    S1Noise,
)

PRODUCT_FORMAT_EXTENSION = ".zarr"


@dataclass
class S1ZarrBurstStructure:
    """Internal definition of a burst for the Sentinel-1 Zarr product."""

    name: str
    swath: str
    polarization: SARPolarization
    lines: int
    samples: int
    azimuth_start_time: PreciseDateTime
    azimuth_stop_time: PreciseDateTime
    range_start: float
    range_stop: float


@dataclass
class S1ZarrChannelStructure:
    """Internal definition of a channel for the Sentinel-1 Zarr product."""

    name: str
    bursts_structures: list[S1ZarrBurstStructure] | None

    bursts: list[str] = field(init=False)
    swath: str = field(init=False)
    polarization: SARPolarization = field(init=False)
    lines: int = field(init=False)
    global_burst_start_lines: npt.NDArray[np.integer] = field(init=False)
    samples: int = field(init=False)
    lines_per_burst: int = field(init=False)
    azimuth_start_time: PreciseDateTime = field(init=False)
    azimuth_stop_time: PreciseDateTime = field(init=False)
    range_start: float = field(init=False)
    range_stop: float = field(init=False)

    _burst_map: dict[str, S1ZarrBurstStructure] = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if not self.bursts_structures:
            raise ValueError("At least one burst structure is required.")
        self.bursts_structures.sort(key=lambda burst: burst.azimuth_start_time)

        swath_id = {(b.swath, b.polarization) for b in self.bursts_structures}
        assert len(swath_id) == 1, "Bursts cannot have different polarizations/swaths in same channels"
        swath_id = swath_id.pop()

        lines_per_burst = {b.lines for b in self.bursts_structures}
        assert len(lines_per_burst) == 1, "Bursts with different lines_per_burst in same channel"

        samples = {b.samples for b in self.bursts_structures}
        assert len(samples) == 1, "Bursts with different samples_per_burst in same channel"

        self._burst_map = {burst.name: burst for burst in self.bursts_structures}

        if len(self._burst_map) != len(self.bursts_structures):
            raise ValueError("Duplicate burst names found.")

        self.swath = swath_id[0]
        self.polarization = swath_id[1]
        self.bursts = list(self._burst_map.keys())
        self.lines_per_burst = lines_per_burst.pop()
        self.lines = self.lines_per_burst * len(self.bursts_structures)
        self.global_burst_start_lines = np.r_[[0], np.cumsum([self.lines_per_burst] * len(self.bursts_structures))]
        self.samples = samples.pop()
        self.azimuth_start_time = self.bursts_structures[0].azimuth_start_time
        self.azimuth_stop_time = self.bursts_structures[-1].azimuth_stop_time
        self.range_start = self.bursts_structures[0].range_start
        self.range_stop = self.bursts_structures[-1].range_stop

    def get_burst_structure(self, burst: str) -> S1ZarrBurstStructure:
        try:
            return self._burst_map[burst]
        except KeyError:
            raise KeyError(f"Burst '{burst}' not found.") from None


@dataclass
class S1ZarrInternalStructureMap:
    """Definition of internal structure and channels of the Sentinel-1 Zarr product."""

    product_type: S1L1ProductType
    mode: S1AcquisitionMode
    channel_structures: list[S1ZarrChannelStructure]

    # Generated fields
    channels: list[str] = field(init=False)
    first_azimuth_time: PreciseDateTime = field(init=False)
    last_azimuth_time: PreciseDateTime = field(init=False)
    range_start: float = field(init=False)
    range_stop: float = field(init=False)

    _channel_map: dict[str, S1ZarrChannelStructure] = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if not self.channel_structures:
            raise ValueError("At least one channel structure is required.")

        self.channel_structures.sort(key=lambda channel: channel.azimuth_start_time)

        self._channel_map = {channel.name: channel for channel in self.channel_structures}

        if len(self._channel_map) != len(self.channel_structures):
            raise ValueError("Duplicate channel names found.")

        self.channels = list(self._channel_map)

        self.first_azimuth_time = self.channel_structures[0].azimuth_start_time
        self.last_azimuth_time = max(c.azimuth_stop_time for c in self.channel_structures)

        self.range_start = min(c.range_start for c in self.channel_structures)
        self.range_stop = max(c.range_stop for c in self.channel_structures)

    def get_channel_structure(self, channel: str) -> S1ZarrChannelStructure:
        try:
            return self._channel_map[channel]
        except KeyError:
            raise KeyError(f"Channel '{channel}' not found.") from None


class S1ZarrProduct:
    """Sentinel-1 Zarr product object"""

    def __init__(self, path: str | Path) -> None:
        self._product_path = Path(path)
        self._product_name = self._product_path.name.replace(PRODUCT_FORMAT_EXTENSION, "")

        try:
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
            self._internal_structure_map = zarr_internal_structure_mapping(prod=dt)

            first_channel = self._internal_structure_map.get_channel_structure(self._internal_structure_map.channels[0])
            processing_history: dict[str, list[dict[str]]] = dt[first_channel.bursts[0]].attrs["processing_history"]

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
        finally:
            dt.close()

    @property
    def auxiliary_data_files(self) -> dict[str, str]:
        """Auxiliary data files used to generate this product."""
        return self._adf

    @property
    def internal_structure_map(self) -> S1ZarrInternalStructureMap:
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
        return self._internal_structure_map.channels

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


def zarr_internal_structure_mapping(prod: xr.DataTree) -> S1ZarrInternalStructureMap:
    """Defining internal channel to groups association structure.

    Parameters
    ----------
    prod : xr.DataTree
        zarr product

    Returns
    -------
    S1ZarrInternalStructureMap
        internal zarr structure channels-groups mapping
    """
    prod_type = prod.attrs["stac_discovery"]["properties"]["product:type"]
    product_type = S1L1ProductType[prod_type[-3:]]
    sensor = prod.attrs["stac_discovery"]["properties"]["platform"].replace("sentinel-", "s")
    group_names = {
        path.replace("/measurements", "").replace("/", "") for path in prod.groups if path.endswith("/measurements")
    }

    match product_type:
        case S1L1ProductType.SLC:
            mode = S1AcquisitionMode(prod.attrs["stac_discovery"]["properties"]["sar:instrument_mode"])
            unique_channels = sorted(
                {_channel_pattern_from_group(g) for g in group_names},
                key=lambda x: x.split("_")[-1],
            )

            burst_channels_association = {
                "_".join([sensor, ch.split("_")[1], ch.split("_")[0]]): sorted(
                    [g for g in group_names if ch.replace(sensor, "").upper() in g]
                )
                for ch in unique_channels
            }
            channels_struct = []
            for channel, bursts in burst_channels_association.items():
                name_parts = channel.split("_")
                swath = name_parts[-2].upper()
                pol = SARPolarization[name_parts[-1].upper()]
                burst_struct = []
                for burst in bursts:
                    burst_struct.append(
                        S1ZarrBurstStructure(
                            name=burst,
                            swath=swath,
                            polarization=pol,
                            lines=prod[burst].attrs["other_metadata"]["swath_timing"]["lines_per_burst"],
                            samples=prod[burst].attrs["other_metadata"]["swath_timing"]["samples_per_burst"],
                            azimuth_start_time=PreciseDateTime.fromisoformat(
                                str(prod[burst]["measurements"].azimuth_time.data[0])
                            ),
                            azimuth_stop_time=PreciseDateTime.fromisoformat(
                                str(prod[burst]["measurements"].azimuth_time.data[-1])
                            ),
                            range_start=float(prod[burst]["measurements"].slant_range_time.data[0]),
                            range_stop=float(prod[burst]["measurements"].slant_range_time.data[-1]),
                        )
                    )
                channels_struct.append(S1ZarrChannelStructure(name=channel, bursts_structures=burst_struct))
        case S1L1ProductType.GRD:
            swath = prod.attrs["stac_discovery"]["properties"]["sar:instrument_mode"]
            mode = swath
            burst_channels_association = {
                "_".join(["grd", swath.lower(), ch.split("_")[-1].lower()]): ch for ch in group_names
            }
            channels_struct = []
            for channel, ch_group in burst_channels_association.items():
                burst_struct = [
                    S1ZarrBurstStructure(
                        name=ch_group,
                        swath=swath,
                        polarization=SARPolarization[prod[ch_group].attrs["other_metadata"]["polarisation"]],
                        lines=prod[ch_group].attrs["other_metadata"]["image_annotation"]["image_information"][
                            "number_of_lines"
                        ],
                        samples=prod[ch_group].attrs["other_metadata"]["image_annotation"]["image_information"][
                            "number_of_samples"
                        ],
                        azimuth_start_time=PreciseDateTime.fromisoformat(
                            str(prod[ch_group]["measurements"].azimuth_time.data[0])
                        ),
                        azimuth_stop_time=PreciseDateTime.fromisoformat(
                            str(prod[ch_group]["measurements"].azimuth_time.data[-1])
                        ),
                        range_start=float(prod[ch_group]["measurements"].ground_range.data[0]),
                        range_stop=float(prod[ch_group]["measurements"].ground_range.data[-1]),
                    )
                ]
                channels_struct.append(S1ZarrChannelStructure(name=channel, bursts_structures=burst_struct))

    return S1ZarrInternalStructureMap(mode=mode, product_type=product_type, channel_structures=channels_struct)


def _channel_pattern_from_group(path: str) -> str:
    """Get the channel pattern pol_swath from from full group name."""
    name_parts = path.replace("/measurements", "").split("_")
    return "_".join([name_parts[-3], name_parts[-2]]).lower()
