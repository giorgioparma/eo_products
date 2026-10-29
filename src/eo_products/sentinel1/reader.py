# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Sentinel-1 product format reader."""

from __future__ import annotations

from pathlib import Path

import eo_products.sentinel1.safe_prod.utilities as safe_utils
import eo_products.sentinel1.zarr_prod.utilities as zarr_utils
from eo_products.sentinel1.common import InvalidSentinel1Product


def open_product(pf_path: str | Path) -> safe_utils.S1SAFEProduct | zarr_utils.S1ZarrProduct:
    """Open a Sentinel-1 product. Both .SAFE and .zarr formats are supported.

    Parameters
    ----------
    pf_path : str | Path
        Path to the Sentinel-1 product

    Returns
    -------
    S1SAFEProduct | S1ZarrProduct
        S1Product object corresponding to the input SAFE product
    """

    pf_path = Path(pf_path)
    if not pf_path.exists() or not pf_path.is_dir():
        raise InvalidSentinel1Product(f"{pf_path}")

    match pf_path.suffix:
        case safe_utils.PRODUCT_FORMAT_EXTENSION:
            return safe_utils.S1SAFEProduct(path=pf_path)
        case zarr_utils.PRODUCT_FORMAT_EXTENSION:
            return zarr_utils.S1ZarrProduct(path=pf_path)
        case _:
            raise InvalidSentinel1Product(f"{pf_path}")


def is_s1_product(product: str | Path) -> bool:
    """Check if input path corresponds to a valid Sentinel-1 product, basic version.

    Conditions to be met for basic validity:

        - path exists
        - path is a directory
        - path ends with proper format extension
        - S1 Product class can be instantiated

    Parameters
    ----------
    product : str | Path
        path to the product to be checked

    Returns
    -------
    bool
        True if it is a valid Sentinel-1 product, else False
    """
    match Path(product).suffix:
        case safe_utils.PRODUCT_FORMAT_EXTENSION:
            return safe_utils.is_s1_safe_product(product)
        case zarr_utils.PRODUCT_FORMAT_EXTENSION:
            return zarr_utils.is_s1_zarr_product(product)
        case _:
            return False
