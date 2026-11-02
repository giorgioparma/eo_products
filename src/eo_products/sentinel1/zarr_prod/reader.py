# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Sentinel-1 Zarr product format reader."""

from __future__ import annotations

import numpy as np
import xarray as xr
from perseo_core.timing import PreciseDateTime

import eo_products.sentinel1.zarr_prod.utilities as zarr_utils
from eo_products.common.utilities import (
    BurstInfo,
    DatasetInfo,
    OrbitDirection,
    RasterInfo,
    RasterInfoAxis,
    SARPolarization,
    SARProjection,
    SARRadiometricQuantity,
    SARSamplingFrequencies,
    StateVectors,
    SwathInfo,
)
from eo_products.sentinel1.common import (
    S1_SIDE_LOOKING,
    S1AcquisitionMode,
    S1AzimuthNoiseVector,
    S1ChannelMetadata,
    S1DCEstimateMethod,
    S1GeneralChannelInfo,
    S1L1ProductType,
    S1Pulse,
    S1RangeNoiseVector,
    get_standard_acq_mode,
)
from eo_products.sentinel1.zarr_prod.utilities import (
    antenna_pattern_from_metadata,
    coordinates_conversions_from_metadata,
    doppler_centroid_from_metadata,
    doppler_rate_from_metadata,
    noise_from_metadata,
)


def read_channel_metadata(prod: zarr_utils.S1ZarrProduct, channel: str) -> S1ChannelMetadata:
    """Read channel metadata from the Zarr product.

    Parameters
    ----------
    prod : zarr_utils.S1ZarrProduct
        zarr product object
    channel : str
        selected channel

    Returns
    -------
    S1ChannelMetadata
        metadata for the selected channel
    """
    channel_structure = prod.internal_structure_map.get_channel_structure(channel)
    first_burst_name = channel_structure.bursts[0]
    try:
        dt = xr.open_datatree(prod.product_path, engine="zarr", chunks={})

        mode = S1AcquisitionMode(dt.attrs["stac_discovery"]["properties"]["sar:instrument_mode"])
        orbit_direction = OrbitDirection(
            dt[first_burst_name].attrs["other_metadata"]["general_annotation"]["product_information"]["pass"].lower()
        )
        swath = dt[first_burst_name].attrs["other_metadata"]["downlink_information"]["swath"]
        radar_frequency = dt[first_burst_name].attrs["other_metadata"]["general_annotation"]["product_information"][
            "radar_frequency"
        ]
        prod_type = dt[first_burst_name].attrs["stac_discovery"]["properties"]["product:type"]
        product_type = S1L1ProductType[prod_type[-3:]]
        if product_type.name != "SLC":
            raise RuntimeError("Only SLC support for now.")
        polarization = SARPolarization[dt[first_burst_name].attrs["other_metadata"]["polarisation"]]
        sensor_name = prod.sensor.upper()
        projection = SARProjection(dt[first_burst_name].attrs["other_metadata"]["projection"].upper())
        orbit_time_axis = np.asarray(
            [
                PreciseDateTime.fromisoformat(str(t))
                for t in dt[first_burst_name]["conditions/orbit/azimuth_time"].values
            ]
        )
        range_sampling_rate = dt[first_burst_name].attrs["other_metadata"]["range_sampling_rate"]

        general_info = S1GeneralChannelInfo(
            mission_id=sensor_name,
            swath=swath,
            product_type=product_type,
            polarization=polarization,
            projection=projection,
            mode=mode,
            acquisition_mode_std=get_standard_acq_mode(mode),
            orbit_direction=orbit_direction,
            signal_frequency=radar_frequency,
            start_time=PreciseDateTime.fromisoformat(
                dt[first_burst_name].attrs["other_metadata"]["downlink_information"]["first_line_sensing_time"]
            ),
            stop_time=PreciseDateTime.fromisoformat(
                dt[channel_structure.bursts[-1]].attrs["other_metadata"]["downlink_information"][
                    "first_line_sensing_time"
                ]
            ),
        )
        state_vectors = StateVectors(
            num=orbit_time_axis.size,
            orbit_direction=orbit_direction,
            time_axis=orbit_time_axis,
            time_step=orbit_time_axis[1] - orbit_time_axis[0],
            positions=dt[first_burst_name]["conditions/orbit/position"].values,
            velocities=dt[first_burst_name]["conditions/orbit/velocity"].values,
            reference_frame=dt[first_burst_name].attrs["other_metadata"]["general_annotation"]["orbit_list"][0][
                "frame"
            ],
            # orbit_type=1,
        )
        sampling_constants = SARSamplingFrequencies(
            azimuth_freq_hz=1
            / dt[first_burst_name].attrs["other_metadata"]["image_annotation"]["image_information"][
                "azimuth_time_interval"
            ],
            azimuth_bandwidth_freq_hz=dt[first_burst_name].attrs["other_metadata"]["image_annotation"][
                "processing_information"
            ]["swath_proc_params_list"][0]["azimuth_processing"]["processing_bandwidth"],
            range_freq_hz=range_sampling_rate,
            range_bandwidth_freq_hz=dt[first_burst_name].attrs["other_metadata"]["image_annotation"][
                "processing_information"
            ]["swath_proc_params_list"][0]["range_processing"]["processing_bandwidth"],
        )
        swath_info = SwathInfo(
            swath=swath,
            rank=dt[first_burst_name].attrs["other_metadata"]["downlink_information"]["rank"],
            azimuth_steering_rate_poly=(
                dt[first_burst_name].attrs["other_metadata"]["azimuth_steering_rate"] * np.pi / 180,
                0,
                0,
            ),
            prf=dt[first_burst_name].attrs["other_metadata"]["downlink_information"]["prf"],
        )
        dataset_info = DatasetInfo(
            fc_hz=radar_frequency,
            acquisition_mode=mode,
            image_type="MULTILOOK" if prod_type == S1L1ProductType.GRD else "AZIMUTH FOCUSED RANGE COMPENSATED",
            projection=projection,
            sensor_name=sensor_name,
            side_looking=S1_SIDE_LOOKING,
        )

        burst_info = BurstInfo(
            num=len(channel_structure.bursts),
            lines_per_burst=channel_structure.lines_per_burst,
            samples_per_burst=channel_structure.samples,
            azimuth_start_times=np.array(
                [channel_structure.get_burst_structure(b).azimuth_start_time for b in channel_structure.bursts]
            ),
            range_start_times=np.array(
                [channel_structure.get_burst_structure(b).slant_range_start_time for b in channel_structure.bursts]
            ),
        )
        burst_sensing_times = np.array(
            [
                PreciseDateTime.fromisoformat(c["sensing_time"])
                for c in dt[first_burst_name].attrs["other_metadata"]["swath_timing"]["burst_list"]
            ]
        )

        raster_lines = RasterInfoAxis(
            length=dt[first_burst_name].attrs["other_metadata"]["image_annotation"]["image_information"][
                "number_of_lines"
            ],
            start=PreciseDateTime.fromisoformat(str(dt[first_burst_name]["measurements"].azimuth_time.values[0])),
            step=dt[first_burst_name].attrs["other_metadata"]["image_annotation"]["image_information"][
                "azimuth_time_interval"
            ],
            step_unit="s",
        )
        # TODO: change for GRD
        raster_samples = RasterInfoAxis(
            length=dt[first_burst_name].attrs["other_metadata"]["image_annotation"]["image_information"][
                "number_of_samples"
            ],
            start=dt[first_burst_name]["measurements"].slant_range_time.values[0],
            step=1 / range_sampling_rate,
            step_unit="s",
        )
        raster_info = RasterInfo(lines=raster_lines, samples=raster_samples)

        doppler_centroid_poly = doppler_centroid_from_metadata(
            doppler_centroid=dt[first_burst_name]["conditions/doppler_centroid"],
            estimate_method=S1DCEstimateMethod.DATA,
        )
        doppler_rate_poly = doppler_rate_from_metadata(
            azimuth_fm_rate_list=dt[first_burst_name].attrs["other_metadata"]["general_annotation"][
                "azimuth_fm_rate_list"
            ],
        )

        # TODO: implement for GRD
        coordinate_conversions = coordinates_conversions_from_metadata(None)

        antenna_pattern = antenna_pattern_from_metadata(
            antenna_pattern=dt[first_burst_name]["conditions/antenna_pattern"]
        )

        noise = noise_from_metadata(
            noise_list=dt[first_burst_name].attrs["other_metadata"]["general_annotation"]["noise_list"], swath=swath
        )

        acquisition_timeline = 1

        pulse_info = S1Pulse(
            length=dt[first_burst_name].attrs["other_metadata"]["downlink_information"]["tx_pulse_length"],
            bandwidth=dt[first_burst_name].attrs["other_metadata"]["image_annotation"]["processing_information"][
                "swath_proc_params_list"
            ][0]["range_processing"]["total_bandwidth"],
            energy=1.0,
            start_frequency=dt[first_burst_name].attrs["other_metadata"]["downlink_information"][
                "tx_pulse_start_frequency"
            ],
            start_phase=0.0,
            direction="UP",
        )

        calibration_factor = dt[first_burst_name]["quality/calibration/dn"].values
        # calibration_factor_beta = dt[first_burst_name]["quality/calibration/beta_nought"].values
        # calibration_factor_sigma = dt[first_burst_name]["quality/calibration/sigma_nought"].values
        # calibration_factor_gamma = dt[first_burst_name]["quality/calibration/gamma"].values
        # ...
        # 1 / float(cal_vectors[0].find("betaNought").text.split(" ")[0])

    finally:
        dt.close()

    return S1ChannelMetadata(
        general_info=general_info,
        image_radiometric_quantity=SARRadiometricQuantity.BETA_NOUGHT,
        raster_info=raster_info,
        burst_info=burst_info,
        dataset_info=dataset_info,
        swath_info=swath_info,
        sampling_constants=sampling_constants,
        state_vectors=state_vectors,
        orbit=state_vectors.orbit,
        doppler_centroid_poly=doppler_centroid_poly,
        doppler_rate_poly=doppler_rate_poly,
        coordinate_conversions=coordinate_conversions,
        antenna_pattern=antenna_pattern,
        burst_sensing_times=burst_sensing_times,
        acquisition_timeline=acquisition_timeline,
        pulse=pulse_info,
        noise=noise,
        calibration_factor=calibration_factor,
        chirp_replica=None,
        attitude=None,
    )


def read_channel_noise_vectors(
    prod: zarr_utils.S1ZarrProduct, channel: str
) -> tuple[list[S1AzimuthNoiseVector] | None, list[S1RangeNoiseVector] | None]:
    """Reading channel azimuth and range noise vectors.

    Parameters
    ----------
    prod : zarr_utils.S1ZarrProduct
        zarr product object
    channel : str
        selected channel

    Returns
    -------
    list[S1AzimuthNoiseVector] | None
        list of azimuth noise vectors, if any else None
    list[S1RangeNoiseVector] | None
        list of range noise vectors, if any else None
    """
    channel_structure = prod.internal_structure_map.get_channel_structure(channel)
    first_burst_name = channel_structure.bursts[0]

    try:
        dt = xr.open_datatree(prod.product_path, engine="zarr", chunks={})
        first_group = dt[first_burst_name]
        quality_groups = [g.split("/")[-1] for g in first_group["quality"].groups]

        azimuth_vectors = None
        if "noise_azimuth" in quality_groups:
            azimuth_vectors = []
            for idx, swath in enumerate(first_group["quality/noise_azimuth"].swath.values):
                azimuth_vectors.append(
                    S1AzimuthNoiseVector(
                        swath=str(swath),
                        first_azimuth_line=int(first_group["quality/noise_azimuth"].first_azimuth_time.values[idx]),
                        last_azimuth_line=int(first_group["quality/noise_azimuth"].last_azimuth_time.values[idx]),
                        first_range_sample=int(first_group["quality/noise_azimuth"].first_range_sample.values[idx]),
                        last_range_sample=int(first_group["quality/noise_azimuth"].last_range_sample.values[idx]),
                        lines=first_group["quality/noise_azimuth"].line.values,
                        lut=first_group["quality/noise_azimuth"].noise_azimuth_lut.values,
                    )
                )

        if "noise_range" in quality_groups:
            range_vectors = []
            for idx, line in enumerate(first_group["quality/noise_range"].line.values):
                az_time = first_group["quality/noise_range"].azimuth_time[idx]
                range_vectors.append(
                    S1RangeNoiseVector(
                        azimuth_time=PreciseDateTime.fromisoformat(str(az_time.data)),
                        line=int(line),
                        pixels=first_group["quality/noise_range"].sel(azimuth_time=az_time).pixel.values,
                        lut=first_group["quality/noise_range"].sel(azimuth_time=az_time).noise_range_lut.values,
                    )
                )
    finally:
        dt.close()
    return azimuth_vectors, range_vectors


def read_channel_data(
    prod: zarr_utils.S1ZarrProduct,
    channel: str,
    block_to_read: list[int],
    scaling_conversion: float = 1,
) -> np.ndarray:
    """Read ROI from zarr product. ROI must be expressed in global lines and samples indexes.

    Parameters
    ----------
    prod : zarr_utils.S1ZarrProduct
        zarr product object
    channel : str
        selected channel
    block_to_read : list[int]
        data block to be read, to be specified as a list of 4 integers, in the form:

            0. first line to be read
            1. first sample to be read
            2. total number of lines to be read
            3. total number of samples to be read

    scaling_conversion : float, optional
        scaling conversion to be applied to the data read, by default 1

    Returns
    -------
    np.ndarray
        numpy array containing the data read from raster file, with shape (lines, samples)
    """

    channel_structure = prod.internal_structure_map.get_channel_structure(channel)
    num_bursts = len(channel_structure.bursts)

    start_line, start_sample, num_lines, num_samples = block_to_read
    stop_line = start_line + num_lines
    stop_sample = start_sample + num_samples

    first_burst = np.searchsorted(channel_structure.global_burst_start_lines, start_line, side="right") - 1
    last_burst = np.searchsorted(channel_structure.global_burst_start_lines, stop_line - 1, side="right") - 1

    if first_burst < 0 or last_burst > num_bursts:
        raise RuntimeError("Error in detecting ROI intersected bursts.")

    data_parts = []
    dt = xr.open_datatree(prod.product_path, engine="zarr", chunks={})
    var = list(dt[channel_structure.bursts[first_burst]]["measurements"].data_vars)[0]

    try:
        for burst_idx in range(first_burst, last_burst + 1):
            burst_start = channel_structure.global_burst_start_lines[burst_idx]
            local_start = max(0, start_line - burst_start)
            local_stop = min(
                channel_structure.lines_per_burst,
                stop_line - burst_start,
            )

            burst_group = dt[channel_structure.bursts[burst_idx]]["measurements"]

            data = (
                burst_group[var]
                .isel(
                    azimuth_time=slice(local_start, local_stop),
                    slant_range_time=slice(start_sample, stop_sample),
                )
                .values
            )

            data_parts.append(data)

    finally:
        dt.close()

    data_slice = np.concatenate(data_parts, axis=0)

    return data_slice * scaling_conversion
