# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Sentinel-1 SAFE reader support module."""

from __future__ import annotations

from pathlib import Path

from lxml import etree
from perseo_core.timing import PreciseDateTime

PRODUCT_FORMAT_EXTENSION = ".SAFE"
_IMAGE_FORMAT = ".tiff"
_METADATA_EXTENSION = ".xml"
_S1_TX_PULSE_LATCH_TIME = 1.439e-6  # [s]


class ChannelDataPairMismatch(RuntimeError):
    """Mismatch between channel data pair (raster data and corresponding metadata)"""


class InvalidChannelId(RuntimeError):
    """Invalid channel number"""


def _extract_channel_name_from_file(filename: str, is_wave: bool) -> str:
    """Extracting channel name from metadata file name.

    Parameters
    ----------
    filename : str
        metadata file name
    is_wave : bool
        if the product's acquisition mode is Wave, the number of the imagette must be added

    Returns
    -------
    str
        channel name
    """
    name_parts = filename.split("/")[-1].split("-")
    if is_wave:
        return "-".join(name_parts[:4] + [name_parts[-1].replace(_METADATA_EXTENSION, "")])
    return "-".join(name_parts[:4])


class S1Manifest:
    """Sentinel-1 SAFE manifest parser"""

    def __init__(self, manifest_path: Path) -> None:
        """Parsing the SAFE product manifest to gather information about data and metadata files.

        Parameters
        ----------
        manifest_path : Path
            path to the manifest file
        """
        self._path = manifest_path
        self._channel_list = []
        self._data_paths = []
        self._metadata_paths = []
        self._root = self._read_file()

    @staticmethod
    def _extract_relative_path_from_nodes(nodes: list[etree._Element]) -> list:
        """Extracting relative paths from location element for each input dataObject node.

        Parameters
        ----------
        nodes : list[etree._Element]
            list of dataObject nodes

        Returns
        -------
        list
            list of paths for each input node
        """
        file_locations = [n.find("byteStream").find("fileLocation") for n in nodes]
        relative_file_paths = [dict(loc.items())["href"] for loc in file_locations]
        return relative_file_paths

    def _read_file(self) -> etree._Element:
        """Parsing manifest .xml file.

        Returns
        -------
        etree._Element
            XML manifest root
        """
        tree = etree.parse(self._path)
        root = tree.getroot()
        return root

    def parse_manifest_document(
        self,
    ) -> tuple[list[str], list[str], list[str], list[str], PreciseDateTime, tuple[float, float, float, float]]:
        """Parsing SAFE manifest .xml document to gather information to available data, metadata, calibrations and
        product acquisition time.

        Returns
        -------
        list[str]
            list of data relative paths with respect to SAFE product folder
        list[str]
            list of metadata relative paths with respect to SAFE product folder
        list[str]
            list of calibration relative paths with respect to SAFE product folder
        list[str]
            list of noise relative paths with respect to SAFE product folder
        PreciseDateTime
            acquisition start time
        tuple[float, float, float, float]
            product footprint [min lat, max lat, min lon, max lon]
        """
        data_object_section = self._root.find("dataObjectSection")
        data_objects = data_object_section.findall("dataObject")

        # extracting paths to each raster data inside the SAFE product
        data_elements = [d for d in data_objects if dict(d.items())["repID"] == "s1Level1MeasurementSchema"]
        relative_data_paths = self._extract_relative_path_from_nodes(data_elements)

        # extracting paths to each metadata inside the SAFE product
        metadata_elements = [d for d in data_objects if dict(d.items())["repID"] == "s1Level1ProductSchema"]
        relative_metadata_paths = self._extract_relative_path_from_nodes(metadata_elements)

        # extracting paths to each calibration file inside the SAFE product
        calibration_elements = [d for d in data_objects if dict(d.items())["repID"] == "s1Level1CalibrationSchema"]
        relative_calibration_paths = self._extract_relative_path_from_nodes(calibration_elements)

        # extracting paths to each noise file inside the SAFE product
        noise_elements = [d for d in data_objects if dict(d.items())["repID"] == "s1Level1NoiseSchema"]
        relative_noise_paths = self._extract_relative_path_from_nodes(noise_elements)

        # extracting acquisition start time
        metadata_objects = self._root.find("metadataSection").findall("metadataObject")
        acquisition_period = [m for m in metadata_objects if dict(m.items())["ID"] == "acquisitionPeriod"]
        period = [m for m in acquisition_period[0].find("metadataWrap/xmlData") if "acquisitionPeriod" in m.tag][0]
        date = [p.text for p in period if "startTime" in p.tag][0]
        try:
            acq_start_time = PreciseDateTime.from_utc_string(date)
        except ValueError:
            acq_start_time = PreciseDateTime.fromisoformat(date)

        # extracting product footprint
        meas_frame_set_node = [m for m in metadata_objects if dict(m.items())["ID"] == "measurementFrameSet"][0]
        footprint_str = (
            meas_frame_set_node.xpath(
                ".//metadataWrap//xmlData//safe:frameSet//safe:frame//safe:footPrint//gml:coordinates",
                namespaces=self._root.nsmap,
            )[0]
            .text.replace(",", " ")
            .split()
        )
        longitudes = [float(f) for f in footprint_str[1::2]]
        latitudes = [float(f) for f in footprint_str[::2]]
        footprint = (min(latitudes), max(latitudes), min(longitudes), max(longitudes))

        return (
            relative_data_paths,
            relative_metadata_paths,
            relative_calibration_paths,
            relative_noise_paths,
            acq_start_time,
            footprint,
        )


class S1SAFEFolderLayout:
    """SAFE file main directory architecture"""

    def __init__(self, path: Path) -> None:
        """Definition of internal architecture of a SAFE product folder.

        Parameters
        ----------
        path : Path
            path to the SAFE base folder
        """
        self._safe_path = path
        self._manifest_file = path.joinpath("manifest.safe")
        self._annotations_dir = path.joinpath("annotation")
        self._measurements_dir = path.joinpath("measurement")
        self._preview_dir = path.joinpath("preview")
        self._calibrations_dir = self._annotations_dir.joinpath("calibration")

    @property
    def manifest(self) -> Path:
        """Location of manifest file"""
        return self._manifest_file

    @property
    def annotations(self) -> Path:
        """Location of annotation folder"""
        return self._annotations_dir

    @property
    def measurements(self) -> Path:
        """Location of measurement folder"""
        return self._measurements_dir

    @property
    def preview(self) -> Path:
        """Location of preview folder"""
        return self._preview_dir

    @property
    def calibration(self) -> Path:
        """Location of calibration folder in annotations"""
        return self._calibrations_dir


class S1SAFEProduct:
    """Sentinel-1 SAFE product object"""

    def __init__(self, path: str | Path) -> None:
        self._product_path = Path(path)
        self._product_name = self._product_path.name.replace(PRODUCT_FORMAT_EXTENSION, "")
        self._layout = S1SAFEFolderLayout(self._product_path)
        self._manifest = S1Manifest(manifest_path=self._layout.manifest)

        # locating relative data and metadata paths inside SAFE product folder
        (data_rel_paths, metadata_rel_paths, calibration_rel_paths, noise_rel_paths, acquisition_time, footprint) = (
            self._manifest.parse_manifest_document()
        )
        # validating channel data pairs (raster + metadata)
        self._validate_channel_pairs(data_rel_paths, metadata_rel_paths)

        # extracting full path to raster and metadata files
        self._data_paths = self._get_full_file_paths(data_rel_paths)
        self._metadata_paths = self._get_full_file_paths(metadata_rel_paths)
        self._calibration_paths = self._get_full_file_paths(calibration_rel_paths)
        self._noise_paths = self._get_full_file_paths(noise_rel_paths)
        self._channels_number = len(self._data_paths)

        # acquisition time
        self._acq_time = acquisition_time

        # footprint
        self._footprint = footprint

        self._is_wave = False
        if "WV" in self._product_name.split("_")[1]:
            self._is_wave = True

        # computing channel list
        self._channel_list_by_swath_id = [
            _extract_channel_name_from_file(m, is_wave=self._is_wave) for m in metadata_rel_paths
        ]

    @staticmethod
    def _validate_channel_pairs(data_list: list[str], metadata_list: list[str]) -> None:
        """Checking validity of metadata and raster pairs.

        Parameters
        ----------
        data_list : list[str]
            list of raster data files
        metadata_list : list[str]
            list of metadata files
        """
        if not len(data_list) == len(metadata_list):
            raise ChannelDataPairMismatch(
                f"Metadata number != raster data number ({len(metadata_list)} != {len(data_list)})"
            )
        data_names = [d.split("/")[-1].replace(_IMAGE_FORMAT, "") for d in data_list]
        metadata_names = [m.split("/")[-1].replace(_METADATA_EXTENSION, "") for m in metadata_list]
        if not metadata_names == data_names:
            raise ChannelDataPairMismatch("Metadata file names are not the same as the rasters'")

    def _get_full_file_paths(self, rel_paths: list[str]) -> list[Path]:
        """Generating full paths from relative ones by prepending the SAFE product folder path.

        Parameters
        ----------
        rel_paths : list[str]
            list of relative paths inside the SAFE product folder

        Returns
        -------
        list[Path]
            list of full paths
        """
        abs_paths = [self._product_path.joinpath(rel.replace("./", "", 1)) for rel in rel_paths]
        return abs_paths

    @property
    def acquisition_time(self) -> PreciseDateTime:
        """Acquisition start time for this product"""
        return self._acq_time

    @property
    def data_list(self) -> list[Path]:
        """Returning the list of raster data files in SAFE product"""
        return self._data_paths

    @property
    def metadata_list(self) -> list[Path]:
        """Returning the list of metadata files in SAFE product"""
        return self._metadata_paths

    @property
    def calibration_list(self) -> list[Path]:
        """Returning the list of calibration files in SAFE product"""
        return self._calibration_paths

    @property
    def noise_list(self) -> list[Path]:
        """Returning the list of noise files in SAFE product"""
        return self._noise_paths

    @property
    def channels_number(self) -> int:
        """Returning the number of channels in the SAFE product"""
        return self._channels_number

    @property
    def channels_list(self) -> list[str]:
        """Returning the list of channels in terms of SwathID (swath-polarization)"""
        return self._channel_list_by_swath_id

    @property
    def footprint(self) -> tuple[float, float, float, float]:
        """Product footprint as tuple of (min lat, max lat, min lon, max lon)"""
        return self._footprint

    def get_files_from_channel_name(self, channel_name: str) -> tuple[Path, Path, Path, Path]:
        """Get metadata, raster and calibration file paths associated to input channel name.

        Parameters
        ----------
        channel_name : str
            selected channel name

        Returns
        -------
        Path
            metadata file path
        Path
            raster file path
        Path
            calibration file path
        Path
            noise file path
        """

        def _check_name(name: str, file_path: Path, is_wave: bool) -> bool:
            if is_wave:
                channel_name_parts = name.split("-")
                return (
                    True
                    if "-".join(channel_name_parts[:-1]) in file_path.name
                    and channel_name_parts[-1] in file_path.stem.split("-")[-1]
                    else False
                )
            return True if name in file_path.name else False

        metadata = [
            m for m in self.metadata_list if _check_name(name=channel_name, file_path=m, is_wave=self._is_wave)
        ][0]
        raster = [r for r in self.data_list if _check_name(name=channel_name, file_path=r, is_wave=self._is_wave)][0]
        calibration = [
            c for c in self.calibration_list if _check_name(name=channel_name, file_path=c, is_wave=self._is_wave)
        ][0]
        noise = [n for n in self.noise_list if _check_name(name=channel_name, file_path=n, is_wave=self._is_wave)][0]
        return metadata, raster, calibration, noise


def is_s1_safe_product(product: str | Path) -> bool:
    """Check if input path corresponds to a valid S1 Safe product, basic version.

    Conditions to be met for basic validity:
        - path exists
        - path is a directory
        - path ends with SAFE
        - S1SAFEProduct can be instantiated

    Parameters
    ----------
    product : str | Path
        path to the product to be checked

    Returns
    -------
    bool
        True if it is a valid S1 Safe, else False
    """
    product = Path(product)

    if not product.exists() or not product.is_dir():
        return False

    if not product.name.endswith(PRODUCT_FORMAT_EXTENSION):
        return False

    try:
        S1SAFEProduct(product)
    except Exception:
        return False

    return True
