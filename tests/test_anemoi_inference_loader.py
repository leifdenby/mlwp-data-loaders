"""Integration tests for the built-in ``anemoi-inference`` loader."""

from __future__ import annotations

from mlwp_data_specs.api import (
    SPACE_TRAIT_ATTR,
    TIME_TRAIT_ATTR,
    UNCERTAINTY_TRAIT_ATTR,
)

from mlwp_data_loaders.api import load_and_validate_dataset
from mlwp_data_loaders.mxalign_api import validate_dataset_with_mxalign

DATASET_PATH = [
    "s3://mlwp-sample-datasets/anemoi-inference/unknown-revision/"
    "anemoi-inference-lam_2020020100.nc",
    "s3://mlwp-sample-datasets/anemoi-inference/unknown-revision/"
    "anemoi-inference-lam_2020020200.nc",
]
ENDPOINT_URL = "https://object-store.os-api.cci2.ecmwf.int"
LOADER = "mlwp_data_loaders.loaders.anemoi.anemoi_inference"


def test_load_dataset_opens_anemoi_inference_from_ewc() -> None:
    """The anemoi-inference loader can open and validate the sample NetCDF files."""
    storage_options: dict[str, object] = {
        "endpoint_url": ENDPOINT_URL,
        "anon": True,
    }

    ds, report_specs = load_and_validate_dataset(  # type: ignore
        DATASET_PATH,
        loader=LOADER,
        storage_options=storage_options,
        chunks=None,
        parallel=False,
        return_validation_report=True,
    )

    report_mxalign = validate_dataset_with_mxalign(
        ds,
        time=ds.attrs.get(TIME_TRAIT_ATTR),
        space=ds.attrs.get(SPACE_TRAIT_ATTR),
        uncertainty=ds.attrs.get(UNCERTAINTY_TRAIT_ATTR),
    )
    if report_mxalign.has_fails():
        report_mxalign.console_print()
    assert not report_mxalign.has_fails()

    if report_specs.has_fails():
        report_specs.console_print()
    assert not report_specs.has_fails()
