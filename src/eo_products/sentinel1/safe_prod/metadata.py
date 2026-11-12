# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Sentinel-1 SAFE metadata parsing functionalities."""

from __future__ import annotations

import re

import numpy as np
from lxml import etree
from numpy.polynomial.polynomial import Polynomial
from perseo_core.timing import PreciseDateTime

from eo_products.common.utilities import (
    AcquisitionTimeline,
    BurstInfo,
    ConversionFunction,
    CoordinatesConversions,
    DatasetInfo,
    DopplerEvaluator,
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
    S1AntennaPattern,
    S1Attitude,
    S1AzimuthNoiseVector,
    S1ChirpReplica,
    S1DCEstimateMethod,
    S1GeneralChannelInfo,
    S1L1ProductType,
    S1Noise,
    S1OrbitType,
    S1Pulse,
    S1RangeNoiseVector,
    S1ReferenceFrameType,
    get_standard_acq_mode,
)

# camel case pattern
_cc_pattern = re.compile(r"(?<!^)(?=[A-Z])")


def convert_camel2snake(text: str) -> str:
    """CamelCase to snake_case converter.

    Parameters
    ----------
    text : str
        CamelCase text

    Returns
    -------
    str
        snake_case text
    """
    return _cc_pattern.sub("_", text).lower()


def data_scaling_factor_from_metadata_node(
    calibration_vector_node: etree._Element,
    radiometric_quantity: SARRadiometricQuantity = SARRadiometricQuantity.BETA_NOUGHT,
) -> float:
    """Extract data scaling factor from safe Calibration xml metadata file.

    Parameters
    ----------
    calibration_vector_node : etree._Element
        calibrationVectorList xml node
    radiometric_quantity : SARRadiometricQuantity, optional
        selected radiometric quantity, by default SARRadiometricQuantity.BETA_NOUGHT

    Returns
    -------
    float
        calibration scaling factor to be applied to raster data
    """
    # NOTE Beta Nought LUT is supposed to be equal to a constant value
    cal_vectors = calibration_vector_node.findall("calibrationVector")
    if radiometric_quantity == SARRadiometricQuantity.BETA_NOUGHT:
        return 1 / float(cal_vectors[0].find("betaNought").text.split(" ")[0])


def raster_info_from_metadata_node(
    image_information_node: etree._Element, samples_step: float | None = None
) -> RasterInfo:
    """Creating a RasterInfo metadata element from safe xml node.

    Parameters
    ----------
    image_information_node : etree._Element
        imageInformation metadata xml node
    samples_step : float | None, optional
        samples step in seconds, if no None it means that the product is SLC, otherwise it's GRD, by default None

    Returns
    -------
    RasterInfo
        RasterInfo metadata object
    """
    image_information_dict = dict([(convert_camel2snake(p.tag), p.text) for p in image_information_node])
    if samples_step is None:
        # GRD
        samples_start = 0
        samples_step_ = float(image_information_dict["range_pixel_spacing"])
        samples_step_unit = "m"
        celltype = "FLOAT32"
    else:
        # SLC
        samples_start = float(image_information_dict["slant_range_time"])
        samples_step_ = samples_step
        samples_step_unit = "s"
        celltype = "FLOAT_COMPLEX"

    raster_lines = RasterInfoAxis(
        length=int(image_information_dict["number_of_lines"]),
        start=PreciseDateTime.from_utc_string(image_information_dict["product_first_line_utc_time"]),
        step=float(image_information_dict["azimuth_time_interval"]),
        step_unit="s",
    )
    raster_samples = RasterInfoAxis(
        length=int(image_information_dict["number_of_samples"]),
        start=samples_start,
        step=samples_step_,
        step_unit=samples_step_unit,
    )

    return RasterInfo(
        lines=raster_lines,
        samples=raster_samples,
        data_type=celltype,
    )


def burst_info_from_metadata(burst_node: etree._Element, raster_info: RasterInfo) -> BurstInfo:
    """Generating BurstInfo object directly from metadata.

    Parameters
    ----------
    burst_node : etree._Element
        swathTiming xml node
    raster_info : RasterInfo
        raster info

    Returns
    -------
    BurstInfo
        burst info dataclass
    """
    lines_per_burst = int(burst_node.find("linesPerBurst").text)
    samples_per_burst = int(burst_node.find("samplesPerBurst").text)
    bursts = burst_node.find("burstList")
    numerosity = int(bursts.values()[0])
    azimuth_start_times = np.array(
        [PreciseDateTime.from_utc_string(t.text) for t in bursts.findall("./burst/azimuthTime")]
    )

    if numerosity == 0:
        # stripmap case
        return BurstInfo(
            num=1,
            lines_per_burst=raster_info.lines.length,
            samples_per_burst=raster_info.samples.length,
            azimuth_start_times=raster_info.lines.start,
            range_start_times=raster_info.samples.start,
        )

    assert numerosity == azimuth_start_times.size
    return BurstInfo(
        num=numerosity,
        lines_per_burst=lines_per_burst,
        samples_per_burst=samples_per_burst,
        azimuth_start_times=azimuth_start_times,
        range_start_times=np.repeat(raster_info.samples.start, numerosity),
    )


def dataset_info_from_metadata_nodes(header_node: etree._Element, product_info_node: etree._Element) -> DatasetInfo:
    """Creating a DatasetInfo metadata element from safe xml nodes.

    Parameters
    ----------
    header_node : etree._Element
        adsHeader metadata xml node
    product_info_node : etree._Element
        productInformation metadata xml node

    Returns
    -------
    DatasetInfo
        DatasetInfo metadata object
    """
    product_info_dict = dict([(convert_camel2snake(p.tag), p.text) for p in product_info_node])
    header_dict = dict([(convert_camel2snake(p.tag), p.text) for p in header_node])
    return DatasetInfo(
        fc_hz=float(product_info_dict["radar_frequency"]),
        acquisition_mode=header_dict["mode"],
        image_type="MULTILOOK" if header_dict["product_type"].upper() == "GRD" else "AZIMUTH FOCUSED RANGE COMPENSATED",
        projection=product_info_dict["projection"].upper(),
        sensor_name=header_dict["mission_id"],
        side_looking=S1_SIDE_LOOKING,
    )


def swath_info_from_metadata(product_info: etree._Element, downlink_info_node: etree._Element, swath: str) -> SwathInfo:
    """Creating a SwathInfo metadata object from metadata file.

    Parameters
    ----------
    product_info : etree._Element
        productInformation xml node
    downlink_info_node : etree._Element
        downlinkInformation xml node
    swath : str
        swath name

    Returns
    -------
    SwathInfo
        SwathInfo metadata object
    """
    return SwathInfo(
        swath=swath,
        rank=int(downlink_info_node.find("downlinkValues/rank").text),
        azimuth_steering_rate_poly=(float(product_info.find("azimuthSteeringRate").text) * np.pi / 180, 0, 0),
        prf=float(downlink_info_node.find("prf").text),
    )


def state_vectors_from_metadata(orbit_node: etree._Element, orbit_type: S1OrbitType | None = None) -> StateVectors:
    """Generating StateVectors object directly from product metadata.

    Parameters
    ----------
    orbit_node : etree._Element
        orbitList xml node
    orbit_type : S1OrbitType, optional
        S1OrbitType orbit type, by default None

    Returns
    -------
    StateVectors
        orbit's state vectors dataclass
    """
    numerosity = int(orbit_node.values()[0])

    pos_x_component = np.array([float(p.text) for p in orbit_node.findall("./orbit/position/x")])
    pos_y_component = np.array([float(p.text) for p in orbit_node.findall("./orbit/position/y")])
    pos_z_component = np.array([float(p.text) for p in orbit_node.findall("./orbit/position/z")])
    positions = np.stack([pos_x_component, pos_y_component, pos_z_component], axis=1)

    vel_x_component = np.array([float(p.text) for p in orbit_node.findall("./orbit/velocity/x")])
    vel_y_component = np.array([float(p.text) for p in orbit_node.findall("./orbit/velocity/y")])
    vel_z_component = np.array([float(p.text) for p in orbit_node.findall("./orbit/velocity/z")])
    velocities = np.stack([vel_x_component, vel_y_component, vel_z_component], axis=1)

    frames = [frame.text for frame in orbit_node.findall("./orbit/frame")]
    time_axis = np.array([PreciseDateTime.from_utc_string(time.text) for time in orbit_node.findall("./orbit/time")])

    assert len(set(frames)) == 1
    assert positions.shape[0] == velocities.shape[0] == time_axis.size == len(frames) == numerosity

    return StateVectors(
        num=numerosity,
        positions=positions,
        velocities=velocities,
        time_axis=time_axis,
        time_step=time_axis[1] - time_axis[0],
        reference_frame=S1ReferenceFrameType(set(frames).pop()),
        orbit_type=orbit_type,
    )


def sampling_constants_from_metadata_nodes(
    swath_processing_node: etree._Element, product_info_node: etree._Element, image_info_node: etree._Element
) -> SARSamplingFrequencies:
    """Creating a SARSamplingFrequencies metadata element from safe xml nodes.

    Parameters
    ----------
    swath_processing_node : etree._Element
        swathProcParams metadata xml node
    product_info_node : etree._Element
        productInformation metadata xml node
    image_info_node : etree._Element
        imageInformation metadata xml node

    Returns
    -------
    SARSamplingFrequencies
        SARSamplingFrequencies metadata object
    """
    return SARSamplingFrequencies(
        azimuth_freq_hz=1 / float(image_info_node.find("azimuthTimeInterval").text),
        azimuth_bandwidth_freq_hz=float(swath_processing_node.find("azimuthProcessing/processingBandwidth").text),
        range_freq_hz=float(product_info_node.find("rangeSamplingRate").text),
        range_bandwidth_freq_hz=float(swath_processing_node.find("rangeProcessing/processingBandwidth").text),
    )


def acquisition_timeline_from_metadata_nodes(downlink_info_node: etree._Element) -> AcquisitionTimeline:
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
    swst_list = downlink_info_node.find("downlinkValues/swstList")
    swl_list = downlink_info_node.find("downlinkValues/swlList")
    swst_num = int(swst_list.get("count"))
    swl_num = int(swl_list.get("count"))

    # missing elements set with these dummy values
    missing_lines_number = 0
    missing_lines_azimuth_times = None
    noise_packets_number = 0
    noise_packets_azimuth_times = None
    internal_calibration_number = 0
    internal_calibration_azimuth_times = None

    if swst_num > 1:
        swst_changes_number = swst_num
        swst_changes_azimuth_times = [
            PreciseDateTime.from_utc_string(s.text) for s in swst_list.findall("swst/azimuthTime")
        ]
        swst_changes_values = [float(s.text) for s in swst_list.findall("swst/value")]
    else:
        swst_changes_number = 1
        swst_changes_azimuth_times = [PreciseDateTime.from_utc_string(swst_list.find("swst/azimuthTime").text)]
        swst_changes_values = [float(swst_list.find("swst/value").text)]

    if swl_num > 1:
        swl_changes_number = swl_num
        swl_changes_azimuth_times = [
            PreciseDateTime.from_utc_string(s.text) for s in swl_list.findall("swl/azimuthTime")
        ]
        swl_changes_values = [float(s.text) for s in swl_list.findall("swl/value")]
    else:
        swl_changes_number = 1
        swl_changes_azimuth_times = [PreciseDateTime.from_utc_string(swl_list.find("swl/azimuthTime").text)]
        swl_changes_values = [float(swl_list.find("swl/value").text)]

    return AcquisitionTimeline(
        missing_lines_number=missing_lines_number,
        missing_lines_azimuth_times=missing_lines_azimuth_times,
        swst_changes_number=swst_changes_number,
        swst_changes_azimuth_times=swst_changes_azimuth_times,
        swst_changes_values=swst_changes_values,
        noise_packets_number=noise_packets_number,
        noise_packets_azimuth_times=noise_packets_azimuth_times,
        internal_calibration_number=internal_calibration_number,
        internal_calibration_azimuth_times=internal_calibration_azimuth_times,
        swl_changes_number=swl_changes_number,
        swl_changes_azimuth_times=swl_changes_azimuth_times,
        swl_changes_values=swl_changes_values,
    )


def doppler_centroid_evaluator_from_metadata_nodes(
    dc_estimate_node: etree._Element, estimate_method: S1DCEstimateMethod
) -> DopplerEvaluator:
    """Creating a DopplerEvaluator doppler centroid polynomial wrapper from metadata.

    Parameters
    ----------
    dc_estimate_node : etree._Element
        dcEstimateList metadata xml node
    estimate_method : S1DCEstimateMethod
        doppler centroid estimate method

    Returns
    -------
    DopplerEvaluator
        DopplerEvaluator dataclass for Doppler Centroid polynomial
    """
    doppler_poly_list = []
    for item in dc_estimate_node:
        if estimate_method == S1DCEstimateMethod.DATA:
            coefficients = [float(c) for c in item.find("dataDcPolynomial").text.split(" ") if c]
        else:
            coefficients = [float(c) for c in item.find("geometryDcPolynomial").text.split(" ") if c]

        doppler_poly_list.append(
            ConversionFunction(
                azimuth_reference_time=PreciseDateTime.from_utc_string(item.find("azimuthTime").text),
                origin=float(item.find("t0").text),
                function=Polynomial(coefficients),
            )
        )

    return DopplerEvaluator(
        functions=doppler_poly_list,
        azimuth_reference_times=np.array([c.azimuth_reference_time for c in doppler_poly_list]),
    )


def doppler_rate_evaluator_from_metadata_nodes(azimuth_fm_rate_node: etree._Element) -> DopplerEvaluator:
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
    for item in azimuth_fm_rate_node:
        coefficients = [float(c) for c in item.find("azimuthFmRatePolynomial").text.split(" ") if c]
        doppler_poly_list.append(
            ConversionFunction(
                azimuth_reference_time=PreciseDateTime.from_utc_string(item.find("azimuthTime").text),
                origin=float(item.find("t0").text),
                function=Polynomial(coefficients),
            )
        )
    return DopplerEvaluator(
        functions=doppler_poly_list,
        azimuth_reference_times=np.array([c.azimuth_reference_time for c in doppler_poly_list]),
    )


def coordinates_conversions_from_metadata(coord_conversion_node: etree._Element) -> CoordinatesConversions:
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
    if coord_conversion_node.get("count") == "0":
        return CoordinatesConversions()

    # polynomials written in metadata are supposed to be always already sorted in time, i.e. azimuth validity
    # reference time will be strictly increasing (no need to be sorted afterwards)
    num_poly = int(coord_conversion_node.get("count"))
    slant_to_ground_coeff = []
    ground_to_slant_coeff = []
    az_time_axis = []
    rng_time_axis = []
    slant_range_origin = []
    ground_range_origin = []
    for item in coord_conversion_node:
        az_time_axis.append(PreciseDateTime.from_utc_string(item.find("azimuthTime").text))
        rng_time_axis.append(float(item.find("slantRangeTime").text))
        slant_range_origin.append(float(item.find("sr0").text))
        ground_range_origin.append(float(item.find("gr0").text))
        slant_to_ground_coeff.append([float(c) for c in item.find("srgrCoefficients").text.split(" ") if c])
        ground_to_slant_coeff.append([float(c) for c in item.find("grsrCoefficients").text.split(" ") if c])

    # creating polynomial dataclasses
    ground_to_slant_poly = [
        ConversionFunction(
            azimuth_reference_time=az_time_axis[p],
            origin=ground_range_origin[p],
            function=Polynomial(ground_to_slant_coeff[p]),
        )
        for p in range(num_poly)
    ]
    slant_to_ground_poly = [
        ConversionFunction(
            azimuth_reference_time=az_time_axis[p],
            origin=slant_range_origin[p],
            function=Polynomial(slant_to_ground_coeff[p]),
        )
        for p in range(num_poly)
    ]

    return CoordinatesConversions(
        azimuth_reference_times=np.array(az_time_axis),
        ground_to_slant=ground_to_slant_poly,
        slant_to_ground=slant_to_ground_poly,
    )


def noise_from_metadata_nodes(noise_list_node: etree._Element) -> dict[str, S1Noise]:
    """Creating a dictionary of S1Noise dataclass for each swath from safe xml node.

    Parameters
    ----------
    noise_list_node : etree._Element
        noiseList metadata xml node

    Returns
    -------
    dict[str, S1Noise]
        keys are the swaths, values are S1Noise dataclasses
    """

    dict_keys = {k.find("swath").text for k in noise_list_node}
    noise = {key: s1_noise_from_metadata_node(noise_list_node=noise_list_node, swath=key) for key in sorted(dict_keys)}

    return noise


def antenna_pattern_from_metadata_nodes(antenna_pattern_list_node: etree._Element) -> dict[str, S1AntennaPattern]:
    """Creating a dictionary of S1AntennaPattern dataclass for each swath from safe xml node.

    Parameters
    ----------
    antenna_pattern_list_node : etree._Element
        antennaPatternList metadata xml node

    Returns
    -------
    dict[str, S1AntennaPattern]
        keys are the swaths, values are S1AntennaPattern dataclasses
    """

    dict_keys = {k.find("swath").text for k in antenna_pattern_list_node}
    antenna = {
        key: s1_antenna_pattern_from_metadata_node(antenna_pattern_list_node=antenna_pattern_list_node, swath=key)
        for key in sorted(dict_keys)
    }

    return antenna


def burst_sensing_times_from_metadata(burst_node: etree._Element) -> np.ndarray | None:
    """Generating burst sensing times array from metadata xml node.

    Parameters
    ----------
    burst_node : etree._Element
        bursts xml node from metadata

    Returns
    -------
    np.ndarray | None
        burst sensing times array, if any else None
    """
    bursts = burst_node.find("burstList")
    if bursts.find("./burst") is not None:
        return np.array([PreciseDateTime.from_utc_string(t.text) for t in bursts.findall("./burst/sensingTime")])
    return None


def general_channel_info_from_metadata(
    header_node: etree._Element, product_info_node: etree._Element
) -> tuple[S1GeneralChannelInfo, float]:
    """Generate S1GeneralChannelInfo from xml metadata.

    Parameters
    ----------
    header_node : etree._Element
        xml header node
    product_info_node : etree._Element
        xml product_info node

    Returns
    -------
    S1GeneralChannelInfo
        S1GeneralChannelInfo object
    float
        range sampling rate
    """

    header_dict = dict([(convert_camel2snake(h.tag), h.text) for h in header_node])
    product_info_dict = dict([(convert_camel2snake(p.tag), p.text) for p in product_info_node])

    polarization = header_dict["polarisation"]
    # converting the polarization string value to a proper input format for SARPolarization enum
    if len(polarization) == 2:
        polarization = polarization[0] + "/" + polarization[1]

    mode = S1AcquisitionMode(header_dict["mode"])
    mode_std = get_standard_acq_mode(mode)

    channel_info = S1GeneralChannelInfo(
        mission_id=header_dict["mission_id"],
        swath=header_dict["swath"],
        product_type=S1L1ProductType(header_dict["product_type"]),
        polarization=SARPolarization(polarization),
        projection=SARProjection(product_info_dict["projection"].upper()),
        mode=mode,
        acquisition_mode_std=mode_std,
        orbit_direction=OrbitDirection(product_info_dict["pass"].lower()),
        signal_frequency=float(product_info_dict["radar_frequency"]),
        start_time=PreciseDateTime.from_utc_string(header_dict["start_time"]),
        stop_time=PreciseDateTime.from_utc_string(header_dict["stop_time"]),
    )
    sampling_rate = float(product_info_dict["range_sampling_rate"])
    return channel_info, sampling_rate


def pulse_from_metadata_nodes(swath_processing_node: etree._Element, downlink_info_node: etree._Element) -> S1Pulse:
    """Generating S1Pulse object directly from metadata xml nodes.

    Parameters
    ----------
    swath_processing_node : etree._Element
        swathProcParams xml node
    downlink_info_node : etree._Element
        downlinkInformation xml node

    Returns
    -------
    S1Pulse
        pulse info dataclass
    """
    # TODO: check this
    return S1Pulse(
        length=float(downlink_info_node.find("downlinkValues/txPulseLength").text),
        bandwidth=float(swath_processing_node.find("rangeProcessing/totalBandwidth").text),
        energy=1.0,
        start_frequency=float(downlink_info_node.find("downlinkValues/txPulseStartFrequency").text),
        start_phase=0.0,
        direction="UP",
    )


def attitude_from_metadata_node(attitude_node: etree._Element) -> S1Attitude:
    """Generating S1Attitude object directly from metadata xml node.

    Parameters
    ----------
    attitude_node : etree._Element
        attitudeList xml node

    Returns
    -------
    S1Attitude
        sensor's attitude dataclass
    """

    numerosity = int(attitude_node.values()[0])
    quaternions = _get_attitude_quaternions_from_node(node=attitude_node)
    angular_velocities = _get_attitude_angular_velocities_from_node(node=attitude_node)
    yaw = np.array([float(y.text) for y in attitude_node.findall("./attitude/yaw")])
    pitch = np.array([float(p.text) for p in attitude_node.findall("./attitude/pitch")])
    roll = np.array([float(r.text) for r in attitude_node.findall("./attitude/roll")])
    frames = [frame.text for frame in attitude_node.findall("./attitude/frame")]
    time_axis = np.array(
        [PreciseDateTime.from_utc_string(time.text) for time in attitude_node.findall("./attitude/time")]
    )

    assert len(set(frames)) == 1
    assert (
        quaternions.shape[0]
        == angular_velocities.shape[0]
        == yaw.size
        == pitch.size
        == roll.size
        == time_axis.size
        == len(frames)
        == numerosity
    )

    return S1Attitude(
        num=numerosity,
        quaternions=quaternions,
        angular_velocities=angular_velocities,
        frame=S1ReferenceFrameType(set(frames).pop()),
        yaw=yaw,
        pitch=pitch,
        roll=roll,
        time_axis=time_axis,
    )


def chirp_replica_from_metadata_node(replica_info_node: etree._Element) -> S1ChirpReplica:
    """Generating S1ChirpReplica object directly from metadata xml nodes.

    Parameters
    ----------
    replica_info_node : etree._Element
        replicaInformation metadata xml node

    Returns
    -------
    S1ChirpReplica
        chirp replica dataclass
    """

    # for item in replica_info_node:
    swath = replica_info_node.find("swath").text
    replica_list = replica_info_node.find("replicaList")
    time_axis = np.array([PreciseDateTime.from_utc_string(rep.find("azimuthTime").text) for rep in replica_list])
    cross_correlation_bandwidth = np.array([float(rep.find("crossCorrelationBandwidth").text) for rep in replica_list])
    cross_correlation_pslr = np.array([float(rep.find("crossCorrelationPslr").text) for rep in replica_list])
    cross_correlation_islr = np.array([float(rep.find("crossCorrelationIslr").text) for rep in replica_list])
    cross_correlation_peak_location = np.array(
        [float(rep.find("crossCorrelationPeakLocation").text) for rep in replica_list]
    )
    pg_product_amplitude = np.array([float(rep.find("pgProductAmplitude").text) for rep in replica_list])
    pg_product_phase = np.array([float(rep.find("pgProductPhase").text) for rep in replica_list])
    model_pg_product_amplitude = np.array([float(rep.find("modelPgProductAmplitude").text) for rep in replica_list])
    model_pg_product_phase = np.array([float(rep.find("modelPgProductPhase").text) for rep in replica_list])
    internal_time_delay = np.array([float(rep.find("internalTimeDelay").text) for rep in replica_list])
    reconstructed_replica_validity = [bool(rep.find("reconstructedReplicaValidFlag").text) for rep in replica_list]
    relative_validity = [bool(rep.find("relativePgProductValidFlag").text) for rep in replica_list]
    absolute_validity = [bool(rep.find("absolutePgProductValidFlag").text) for rep in replica_list]

    return S1ChirpReplica(
        swath=swath,
        time_axis=time_axis,
        cross_correlation_bandwidth=cross_correlation_bandwidth,
        cross_correlation_pslr=cross_correlation_pslr,
        cross_correlation_islr=cross_correlation_islr,
        cross_correlation_peak_location=cross_correlation_peak_location,
        pg_product_amplitude=pg_product_amplitude,
        pg_product_phase=pg_product_phase,
        model_pg_product_amplitude=model_pg_product_amplitude,
        model_pg_product_phase=model_pg_product_phase,
        internal_time_delay=internal_time_delay,
        reconstructed_replica_validity=reconstructed_replica_validity,
        relative_validity=relative_validity,
        absolute_validity=absolute_validity,
    )


def s1_antenna_pattern_from_metadata_node(antenna_pattern_list_node: etree._Element, swath: str) -> S1AntennaPattern:
    """Generating S1AntennaPattern object directly from metadata xml nodes.

    Parameters
    ----------
    antenna_pattern_list_node : etree._Element
        antennaPatternList metadata xml node
    swath : str
        swath of interest

    Returns
    -------
    S1Noise
        antenna pattern dataclass
    """

    # taking only the nodes corresponding to the selected swath
    filtered_nodes = filter(lambda x: x.find("swath").text == swath, antenna_pattern_list_node)

    def _parse_ap(array: np.ndarray) -> np.ndarray:
        return array[:, 0] + 1j * array[:, 1]

    time_axis = []
    slant_range_time_array = []
    elevation_angle_array = []
    elevation_pattern_array = []
    incidence_angle = []
    terrain_height = []
    roll = []
    for item in filtered_nodes:
        time_axis.append(PreciseDateTime.from_utc_string(item.find("azimuthTime").text))
        slant_range_time_array.append(np.array([float(c) for c in item.find("slantRangeTime").text.split(" ") if c]))
        elevation_angle_array.append(np.array([float(c) for c in item.find("elevationAngle").text.split(" ") if c]))
        elevation_pattern_array.append(
            _parse_ap(np.array([float(c) for c in item.find("elevationPattern").text.split(" ") if c]).reshape(-1, 2))
        )
        incidence_angle.append(np.array([float(c) for c in item.find("incidenceAngle").text.split(" ") if c]))
        terrain_height.append(float(item.find("terrainHeight").text))
        roll.append(float(item.find("roll").text))

    return S1AntennaPattern(
        swath=swath,
        time_axis=np.array(time_axis),
        slant_range_time_array=np.stack(slant_range_time_array),
        elevation_angle_array=np.stack(elevation_angle_array),
        elevation_pattern_array=np.stack(elevation_pattern_array),
        incidence_angle=np.stack(incidence_angle),
        terrain_height=np.array(terrain_height),
        roll=np.array(roll),
    )


def s1_noise_from_metadata_node(noise_list_node: etree._Element, swath: str) -> S1Noise:
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
    filtered_nodes = filter(lambda x: x.find("swath").text == swath, noise_list_node)

    time_axis = []
    noise_power_correction_factor = []
    noise_lines_num = []
    for item in filtered_nodes:
        time_axis.append(PreciseDateTime.from_utc_string(item.find("azimuthTime").text))
        noise_power_correction_factor.append(float(item.find("noisePowerCorrectionFactor").text))
        noise_lines_num.append(int(item.find("numberOfNoiseLines").text))

    return S1Noise(
        swath=swath,
        time_axis=np.array(time_axis),
        noise_power_correction_factor=np.array(noise_power_correction_factor),
        noise_lines_num=np.array(noise_lines_num),
    )


def rng_noise_vector_from_node(node: etree._Element) -> S1RangeNoiseVector:
    """Generating S1RangeNoiseVector object directly from metadata xml node.

    Parameters
    ----------
    node : etree._Element
        noiseRangeVectorList/noiseRangeVector xml node

    Returns
    -------
    S1RangeNoiseVector
        object corresponding to the input xml node
    """
    azimuth_time = PreciseDateTime.from_utc_string(node.find("azimuthTime").text)
    line = int(node.find("line").text)
    pixels = np.array([int(p) for p in node.find("pixel").text.split()])
    lut = np.array([float(p) for p in node.find("noiseRangeLut").text.split()])
    return S1RangeNoiseVector(azimuth_time=azimuth_time, line=line, pixels=pixels, lut=lut)


def az_noise_vector_from_node(node: etree._Element) -> S1AzimuthNoiseVector:
    """Generating S1AzimuthNoiseVector object directly from metadata xml node.

    Parameters
    ----------
    node : etree._Element
        noiseAzimuthVectorList/noiseAzimuthVector xml node

    Returns
    -------
    S1AzimuthNoiseVector
        object corresponding to the input xml node
    """
    swath = node.find("swath").text
    first_azimuth_line = int(node.find("firstAzimuthLine").text)
    last_azimuth_line = int(node.find("lastAzimuthLine").text)
    first_range_sample = int(node.find("firstRangeSample").text)
    last_range_sample = int(node.find("lastRangeSample").text)
    lines = np.array([int(p) for p in node.find("line").text.split()])
    lut = np.array([float(p) for p in node.find("noiseAzimuthLut").text.split()])
    return S1AzimuthNoiseVector(
        swath=swath,
        first_azimuth_line=first_azimuth_line,
        last_azimuth_line=last_azimuth_line,
        first_range_sample=first_range_sample,
        last_range_sample=last_range_sample,
        lines=lines,
        lut=lut,
    )


def _get_attitude_quaternions_from_node(node: etree._Element) -> np.ndarray:
    """Extracting quaternions array shape (N, 4) with q0, q1, q2 and q3 components from xml attitude node.

    Parameters
    ----------
    node : etree._Element
        attitudeList xml node

    Returns
    -------
    np.ndarray
        quaternions array, (N, 4)
    """
    q_0 = np.array([float(q.text) for q in node.findall("./attitude/q0")])
    q_1 = np.array([float(q.text) for q in node.findall("./attitude/q1")])
    q_2 = np.array([float(q.text) for q in node.findall("./attitude/q2")])
    q_3 = np.array([float(q.text) for q in node.findall("./attitude/q3")])
    return np.stack([q_0, q_1, q_2, q_3], axis=1)


def _get_attitude_angular_velocities_from_node(node: etree._Element) -> np.ndarray:
    """Extracting angular velocities array shape (N, 3) with wx, wy and wz components from xml attitude node.

    Parameters
    ----------
    node : etree._Element
        attitudeList xml node

    Returns
    -------
    np.ndarray
        angular velocities array, (N, 3)
    """
    w_x = np.array([float(q.text) for q in node.findall("./attitude/wx")])
    w_y = np.array([float(q.text) for q in node.findall("./attitude/wy")])
    w_z = np.array([float(q.text) for q in node.findall("./attitude/wz")])
    return np.stack([w_x, w_y, w_z], axis=1)
