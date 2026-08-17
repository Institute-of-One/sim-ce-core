"""Local DICOM / NIfTI time–attenuation extraction. Never downloads."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from sim_ce_core.data.io import write_case
from sim_ce_core.data.types import EnhancementSeries
from sim_ce_core.physio.params import REGION_NAMES


def _acquisition_time_s(ds: object) -> float:
    if hasattr(ds, "TemporalPositionIdentifier"):
        return float(ds.TemporalPositionIdentifier)
    acq = getattr(ds, "AcquisitionTime", None)
    if acq:
        text = str(acq)
        hours = float(text[0:2]) if len(text) >= 2 else 0.0
        mins = float(text[2:4]) if len(text) >= 4 else 0.0
        secs = float(text[4:]) if len(text) > 4 else 0.0
        return hours * 3600.0 + mins * 60.0 + secs
    return float(getattr(ds, "InstanceNumber", 0))


def extract_mean_curve(dicom_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Mean HU per temporal volume from a local DICOM folder.

    Frames are grouped by ``TemporalPositionIdentifier`` (preferred) or
    ``AcquisitionTime``. Enhancement is HU minus the first time point.
    """
    try:
        import pydicom
    except ImportError as exc:  # pragma: no cover
        raise ImportError("pydicom is required to extract DICOM curves") from exc

    paths = sorted(dicom_dir.rglob("*"))
    groups: dict[float, list[np.ndarray]] = defaultdict(list)
    for path in paths:
        if not path.is_file():
            continue
        try:
            ds = pydicom.dcmread(path, force=True)
        except Exception:
            continue
        if not hasattr(ds, "PixelData"):
            continue
        pixels = np.asarray(ds.pixel_array, dtype=np.float64)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        hu = pixels * slope + intercept
        groups[_acquisition_time_s(ds)].append(hu)

    if not groups:
        raise FileNotFoundError(f"no readable DICOM frames in {dicom_dir}")
    times = np.array(sorted(groups), dtype=np.float64)
    times = times - times[0]
    means = np.array(
        [float(np.mean(np.stack(groups[t]))) for t in sorted(groups)],
        dtype=np.float64,
    )
    enhancement = means - means[0]
    return times, enhancement


def series_from_dicom_folder(
    dicom_dir: Path,
    *,
    case_id: str | None = None,
    dataset: str = "ctp_brain",
) -> EnhancementSeries:
    """Build an EnhancementSeries from a local 4D CTP folder.

    Without ROI masks the same mean curve is used for aorta and organ
    (AIF-free fallback). Prefer writing ROI-extracted NPZ for real studies.
    """
    times, enh = extract_mean_curve(dicom_dir)
    curves = np.column_stack([enh, enh, np.zeros_like(enh)])
    meta = {
        "case_id": case_id or dicom_dir.name,
        "dataset": dataset,
        "source": "dicom_extract",
        "dicom_dir": str(dicom_dir),
        "roi": "volume_mean_fallback",
    }
    return EnhancementSeries(
        times_s=times,
        curves_hu=curves,
        region_names=REGION_NAMES,
        aif_hu=enh.copy(),
        metadata=meta,
    )


def extract_and_write(
    dicom_dir: Path,
    out_dir: Path,
    *,
    case_id: str | None = None,
    dataset: str = "ctp_brain",
) -> Path:
    """Extract a local DICOM folder to the NPZ case schema."""
    series = series_from_dicom_folder(dicom_dir, case_id=case_id, dataset=dataset)
    return write_case(out_dir, series, metadata=dict(series.metadata))
