# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Sentinel-1 SAFE product format reader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr
from lxml import etree
from tifffile import imread

import eo_products.sentinel1.safe_prod.utilities as safe_utils
from eo_products.common.utilities import SARRadiometricQuantity
from eo_products.sentinel1.auxiliaries.ext_orbit import read_external_orbit
from eo_products.sentinel1.common import (
    S1AzimuthNoiseVector,
    S1ChannelMetadata,
    S1DCEstimateMethod,
    S1OrbitType,
    S1RangeNoiseVector,
)


def read_channel_metadata(xml_path: str | Path, external_orbit_path: str | Path | None = None) -> S1ChannelMetadata:
    """Reading SAFE product channel metadata.

    Parameters
    ----------
    xml_path : str | Path
        path to the channel metadata xml file
    external_orbit_path : str | Path | None, optional
        path to the external orbit xml file, by default None

    Returns
    -------
    S1ChannelMetadata
        channel metadata dataclass
    """

    xml_path = Path(xml_path)
    external_orbit_path = Path(external_orbit_path) if external_orbit_path is not None else None

    # loading the xml file
    root = etree.parse(xml_path).getroot()

    # gathering nodes from metadata
    header = root.find("adsHeader")
    general_annotation = root.find("generalAnnotation")
    product_info = general_annotation.find("productInformation")
    general_info, range_sampling_rate = safe_utils.general_channel_info_from_metadata(
        header_node=header, product_info_node=product_info
    )
    image_annotation = root.find("imageAnnotation")
    image_information = image_annotation.find("imageInformation")
    antenna_pattern = root.find("antennaPattern")
    downlink_info = general_annotation.find("downlinkInformationList/downlinkInformation")
    dc_estimate = root.find("dopplerCentroid/dcEstimateList")
    swath_processing = root.find("imageAnnotation/processingInformation/swathProcParamsList/swathProcParams")
    swath_timing = root.find("swathTiming")
    orbit_list = general_annotation.find("orbitList")
    attitude_list = general_annotation.find("attitudeList")
    azimuth_fm_rate = general_annotation.find("azimuthFmRateList")
    coord_conversion_node = root.find("coordinateConversion/coordinateConversionList")
    replica_list_info = general_annotation.find("replicaInformationList")
    noise_list = general_annotation.find("noiseList")
    antenna_pattern_list = antenna_pattern.find("antennaPatternList")

    # orbit
    if external_orbit_path is None:
        orbit_type = None
        if image_annotation.find("processingInformation/orbitSource").text == "Extracted":
            orbit_type = S1OrbitType.DOWNLINK
        elif image_annotation.find("processingInformation/orbitSource").text == "Auxiliary":
            orbit_type = S1OrbitType.UNKNOWN
        state_vectors = safe_utils.state_vectors_from_metadata(orbit_node=orbit_list, orbit_type=orbit_type)
    else:
        # reducing orbit span around start and end time of acquisition
        # 5 mins before start time and after stop time
        state_vectors = read_external_orbit(
            xml_path=external_orbit_path,
            time_boundaries=[
                general_info.start_time - 300,
                general_info.stop_time + 300,
            ],
        )

    # updating orbit direction and type
    current_ind = int((general_info.start_time - state_vectors.time_axis[0]) / state_vectors.time_step)
    state_vectors.orbit_direction = (
        safe_utils.OrbitDirection.ASCENDING
        if state_vectors.velocities[current_ind][2] > 0
        else safe_utils.OrbitDirection.DESCENDING
    )

    # attitude
    attitude = safe_utils.attitude_from_metadata_node(attitude_node=attitude_list)

    # raster info
    if general_info.product_type == safe_utils.S1L1ProductType.GRD:
        # GRD
        raster_info = safe_utils.raster_info_from_metadata_node(image_information_node=image_information)
    else:
        # SLC
        raster_info = safe_utils.raster_info_from_metadata_node(
            image_information_node=image_information,
            samples_step=1 / range_sampling_rate,
        )

    # burst info
    burst_info = safe_utils.burst_info_from_metadata(burst_node=swath_timing, raster_info=raster_info)

    # burst sensing times
    burst_sensing_times = safe_utils.burst_sensing_times_from_metadata(burst_node=swath_timing)

    # dataset info
    dataset_info = safe_utils.dataset_info_from_metadata_nodes(header_node=header, product_info_node=product_info)

    # swath info
    swath_info = safe_utils.swath_info_from_metadata(
        product_info=product_info,
        downlink_info_node=downlink_info,
        swath=general_info.swath,
    )

    # sampling constants
    sampling_constants = safe_utils.sampling_constants_from_metadata_nodes(
        swath_processing_node=swath_processing,
        product_info_node=product_info,
        image_info_node=image_information,
    )

    # acquisition timeline
    acquisition_timeline = safe_utils.acquisition_timeline_from_metadata_nodes(downlink_info_node=downlink_info)

    # doppler centroid vector
    # TODO: check if forcing estimate method to be DATA is correct
    doppler_centroid_poly = safe_utils.doppler_centroid_evaluator_from_metadata_nodes(
        dc_estimate_node=dc_estimate, estimate_method=S1DCEstimateMethod.DATA
    )

    # doppler rate vector
    doppler_rate_poly = safe_utils.doppler_rate_evaluator_from_metadata_nodes(azimuth_fm_rate_node=azimuth_fm_rate)

    # pulse
    pulse = safe_utils.pulse_from_metadata_nodes(
        swath_processing_node=swath_processing, downlink_info_node=downlink_info
    )

    # coordinates conversion
    coords_conversion = safe_utils.coordinates_conversions_from_metadata(coord_conversion_node=coord_conversion_node)

    # chirp replica
    chirp_replicas = []
    for item in replica_list_info:
        chirp_replicas.append(safe_utils.chirp_replica_from_metadata_node(item))
    chirp_replicas = {c.swath: c for c in chirp_replicas}

    # noise
    noise = safe_utils.noise_from_metadata_nodes(noise_list_node=noise_list)

    # antenna pattern
    antenna = safe_utils.antenna_pattern_from_metadata_nodes(antenna_pattern_list_node=antenna_pattern_list)

    # composing output channel dataclass
    return S1ChannelMetadata(
        general_info=general_info,
        orbit=state_vectors.orbit,
        image_radiometric_quantity=SARRadiometricQuantity.BETA_NOUGHT,
        attitude=attitude,
        burst_info=burst_info,
        raster_info=raster_info,
        dataset_info=dataset_info,
        swath_info=swath_info,
        sampling_constants=sampling_constants,
        acquisition_timeline=acquisition_timeline,
        doppler_centroid_poly=doppler_centroid_poly,
        doppler_rate_poly=doppler_rate_poly,
        pulse=pulse,
        coordinate_conversions=coords_conversion,
        state_vectors=state_vectors,
        chirp_replica=chirp_replicas,
        noise=noise,
        antenna_pattern=antenna,
        burst_sensing_times=burst_sensing_times,
        calibration_factor=None,
    )


def read_channel_calibration(
    xml_path: str | Path,
    radiometric_quantity: SARRadiometricQuantity = SARRadiometricQuantity.BETA_NOUGHT,
) -> float:
    """Reading SAFE product channel calibration metadata.

    Parameters
    ----------
    xml_path : str | Path
        path to the channel calibration xml file
    radiometric_quantity : SARRadiometricQuantity, optional
        radiometric quantity selected, by default SARRadiometricQuantity.BETA_NOUGHT

    Returns
    -------
    float
        radiometric data scaling factor
    """
    radiometric_quantity = SARRadiometricQuantity(radiometric_quantity.value)

    xml_path = Path(xml_path)

    # loading the xml file
    root = etree.parse(xml_path).getroot()

    cal_vector_node = root.find("calibrationVectorList")
    return safe_utils.data_scaling_factor_from_metadata_node(
        calibration_vector_node=cal_vector_node,
        radiometric_quantity=radiometric_quantity,
    )


def read_channel_noise_vectors(
    noise_file: str | Path,
) -> tuple[list[S1AzimuthNoiseVector] | None, list[S1RangeNoiseVector]]:
    """Reading SAFE product channel noise vectors file in annotations folder.

    Parameters
    ----------
    noise_file : str | Path
        path to the channel noise vectors xml file

    Returns
    -------
    list[S1AzimuthNoiseVector] | None
        list of azimuth noise vectors, if any else None
    list[S1RangeNoiseVector]
        list of range noise vectors
    """
    # loading the xml file
    root = etree.parse(str(noise_file)).getroot()

    azimuth_vectors = [
        safe_utils.az_noise_vector_from_node(node=node)
        for node in root.findall("noiseAzimuthVectorList/noiseAzimuthVector")
    ]
    if not azimuth_vectors:
        azimuth_vectors = None

    range_vectors = [
        safe_utils.rng_noise_vector_from_node(node=node)
        for node in root.findall("noiseRangeVectorList/noiseRangeVector")
    ]

    return azimuth_vectors, range_vectors


def read_channel_data(
    raster_file: str | Path,
    block_to_read: list[int] = None,
    scaling_conversion: float = 1,
) -> np.ndarray:
    """Reading SAFE tiff channel data file.

    Parameters
    ----------
    raster_file : str | Path
        Path to Tiff raster file to be read
    block_to_read : list[int] | None, optional
        data block to be read, to be specified as a list of 4 integers, in the form:

            0. first line to be read
            1. first sample to be read
            2. total number of lines to be read
            3. total number of samples to be read

        if None, the whole raster is read, by default None
    scaling_conversion : float, optional
        scaling conversion to be applied to the data read

    Returns
    -------
    np.ndarray
        numpy array containing the data read from raster file, with shape (lines, samples)
    """
    img_store = imread(raster_file, aszarr=True)
    z = zarr.open(img_store, mode="r")
    if block_to_read is None:
        target_area = z[:]
    else:
        target_area = z[
            block_to_read[0] : block_to_read[0] + block_to_read[2],
            block_to_read[1] : block_to_read[1] + block_to_read[3],
        ]
    img_store.close()

    # applying input scaling factor
    target_area = target_area * scaling_conversion

    return target_area
