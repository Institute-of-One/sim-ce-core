"""Minimal TCIA HCC-TACE-Seg download + multi-phase HU extraction.

Public collection, CC BY 4.0. Downloads at most ``max_cases`` baseline
patients (PRE + one contrast series). Tests must not import this module
in a way that hits the network.
"""

from __future__ import annotations

import json
import time
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

from sim_ce_core.data.io import write_case
from sim_ce_core.data.mphase import DEFAULT_PHASES_S, liver_protocol
from sim_ce_core.data.types import EnhancementSeries
from sim_ce_core.physio.params import REGION_NAMES

COLLECTION = "HCC-TACE-Seg"
NBIA = "https://services.cancerimagingarchive.net/nbia-api/services/v1"
DOI = "10.7937/TCIA.5FNA-0924"
LICENSE = "CC BY 4.0"
USER_AGENT = "sim_ce_core/0.1 (academic; minimal-subset TCIA download)"


def _get_json(url: str) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=90) as handle:
        return json.loads(handle.read().decode())


def fetch_series_metadata() -> list[dict[str, Any]]:
    return _get_json(f"{NBIA}/getSeries?Collection={COLLECTION}")


def _is_pre(desc: str | None) -> bool:
    text = (desc or "").upper()
    return "PRE" in text and "LIVER" in text


def select_baseline_series(
    metadata: list[dict[str, Any]], *, max_cases: int = 20
) -> list[dict[str, Any]]:
    """One PRE + one contrast CT per patient, first ``max_cases`` usable IDs."""
    by_patient: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for series in metadata:
        by_patient[str(series["PatientID"])].append(series)

    chosen: list[dict[str, Any]] = []
    for pid in sorted(by_patient):
        studies: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for series in by_patient[pid]:
            studies[str(series["StudyInstanceUID"])].append(series)
        baseline = None
        for series_list in studies.values():
            has_pre = any(
                s.get("Modality") == "CT" and _is_pre(s.get("SeriesDescription"))
                for s in series_list
            )
            has_contrast = any(
                s.get("Modality") == "CT" and not _is_pre(s.get("SeriesDescription"))
                for s in series_list
            )
            if has_pre and has_contrast:
                baseline = series_list
                break
        if baseline is None:
            continue
        pre = [
            s
            for s in baseline
            if s.get("Modality") == "CT" and _is_pre(s.get("SeriesDescription"))
        ]
        contrast = [
            s
            for s in baseline
            if s.get("Modality") == "CT" and not _is_pre(s.get("SeriesDescription"))
        ]
        contrast.sort(key=lambda item: int(item.get("FileSize") or 0), reverse=True)
        chosen.append(
            {
                "patient_id": pid,
                "pre": pre[0],
                "contrast": contrast[0],
            }
        )
        if len(chosen) >= max_cases:
            break
    return chosen


def download_series(uid: str, dest_dir: Path) -> Path:
    """Download one series zip and extract DICOM files. Resumes if present."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    marker = dest_dir / ".ok"
    if marker.is_file() and any(dest_dir.rglob("*")):
        return dest_dir
    zip_path = dest_dir / "series.zip"
    url = f"{NBIA}/getImage?SeriesInstanceUID={uid}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=300) as handle, zip_path.open("wb") as out:
        while True:
            chunk = handle.read(1024 * 256)
            if not chunk:
                break
            out.write(chunk)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(dest_dir)
    zip_path.unlink(missing_ok=True)
    marker.write_text("ok", encoding="utf-8")
    return dest_dir


def _read_ct_slices(dicom_dir: Path) -> list[tuple[float, float, np.ndarray]]:
    import pydicom

    frames: list[tuple[float, float, np.ndarray]] = []
    for path in dicom_dir.rglob("*"):
        if not path.is_file() or path.name.startswith("."):
            continue
        try:
            ds = pydicom.dcmread(path, force=True)
        except Exception:
            continue
        if str(getattr(ds, "Modality", "")) != "CT" or not hasattr(ds, "PixelData"):
            continue
        z = (
            float(ds.ImagePositionPatient[2])
            if hasattr(ds, "ImagePositionPatient")
            else float(getattr(ds, "InstanceNumber", 0))
        )
        acq = str(getattr(ds, "AcquisitionTime", "") or getattr(ds, "ContentTime", "0"))
        try:
            t_s = (
                float(acq[0:2]) * 3600.0 + float(acq[2:4]) * 60.0 + float(acq[4:])
                if len(acq) >= 4
                else 0.0
            )
        except ValueError:
            t_s = 0.0
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        hu = np.asarray(ds.pixel_array, dtype=np.float64) * slope + intercept
        frames.append((t_s, z, hu))
    if not frames:
        raise FileNotFoundError(f"no CT frames in {dicom_dir}")
    return frames


def split_phase_volumes(
    frames: list[tuple[float, float, np.ndarray]],
    n_ref: int | None = None,
) -> list[np.ndarray]:
    """Group frames into 3D volumes by acquisition time, else by z-blocks."""
    times = sorted({item[0] for item in frames})
    if len(times) >= 2:
        volumes = []
        for t_s in times:
            group = [item for item in frames if item[0] == t_s]
            group.sort(key=lambda item: item[1])
            volumes.append(np.stack([item[2] for item in group]))
        return volumes
    frames_sorted = sorted(frames, key=lambda item: item[1])
    vol = np.stack([item[2] for item in frames_sorted])
    if n_ref is not None and n_ref >= 8:
        for k in (3, 2, 4):
            expected = k * n_ref
            if abs(vol.shape[0] - expected) <= max(3, int(0.25 * n_ref)):
                usable = vol.shape[0] - (vol.shape[0] % k)
                if usable // k >= 8:
                    return list(np.split(vol[:usable], k))
    return [vol]


def liver_mask(
    volume: np.ndarray, *, hu_lo: float = 20.0, hu_hi: float = 90.0
) -> np.ndarray:
    """Largest HU-window component (hepatic parenchyma heuristic)."""
    binary = (volume >= hu_lo) & (volume <= hu_hi)
    labeled, n_lab = ndimage.label(binary)
    if n_lab == 0:
        return np.ones(volume.shape, dtype=bool)
    counts = ndimage.sum(binary, labeled, index=list(range(1, n_lab + 1)))
    keep = int(np.argmax(counts)) + 1
    return labeled == keep


def mean_liver_hu(volume: np.ndarray, mask: np.ndarray | None = None) -> float:
    used = liver_mask(volume) if mask is None else mask
    return float(volume[used].mean())


def mean_vascular_hu(volume: np.ndarray) -> float:
    """Bright-blood proxy (150-650 HU); unenhanced blood is ~40 HU."""
    vessel = (volume >= 150.0) & (volume <= 650.0)
    if int(vessel.sum()) < 50:
        return mean_liver_hu(volume, mask=liver_mask(volume, hu_lo=10.0, hu_hi=200.0))
    return float(volume[vessel].mean())


def _phase_times(n_contrast: int) -> list[float]:
    if n_contrast <= 0:
        return []
    if n_contrast == 1:
        return [DEFAULT_PHASES_S["ap"]]
    if n_contrast == 2:
        return [DEFAULT_PHASES_S["ap"], DEFAULT_PHASES_S["pvp"]]
    return [DEFAULT_PHASES_S["ap"], DEFAULT_PHASES_S["pvp"], DEFAULT_PHASES_S["dp"]][
        :n_contrast
    ]


def extract_mphase_case(
    pre_dir: Path,
    contrast_dir: Path,
    *,
    patient_id: str,
    extra_meta: dict[str, Any] | None = None,
) -> EnhancementSeries:
    pre_vol = split_phase_volumes(_read_ct_slices(pre_dir))[0]
    contrast_frames = _read_ct_slices(contrast_dir)
    contrast_vols = split_phase_volumes(contrast_frames, n_ref=int(pre_vol.shape[0]))
    nc_mask = liver_mask(pre_vol)
    nc_liver = mean_liver_hu(pre_vol, mask=nc_mask)
    phase_t = _phase_times(len(contrast_vols))
    while len(phase_t) < len(contrast_vols):
        phase_t.append(float(phase_t[-1]) + 30.0)
    times = [0.0, *phase_t[: len(contrast_vols)]]
    organ = [0.0]
    aorta = [0.0]
    for vol in contrast_vols:
        if vol.shape == pre_vol.shape:
            organ.append(mean_liver_hu(vol, mask=nc_mask) - nc_liver)
        else:
            wide = liver_mask(vol, hu_lo=10.0, hu_hi=200.0)
            organ.append(mean_liver_hu(vol, mask=wide) - nc_liver)
        aorta.append(mean_vascular_hu(vol) - 40.0)
    if any(val < -5.0 or val > 160.0 for val in organ[1:]):
        contrast_vols = [np.concatenate(contrast_vols, axis=0)]
        phase_t = _phase_times(1)
        times = [0.0, *phase_t]
        organ = [0.0]
        aorta = [0.0]
        vol = contrast_vols[0]
        wide = liver_mask(vol, hu_lo=10.0, hu_hi=200.0)
        organ.append(mean_liver_hu(vol, mask=wide) - nc_liver)
        aorta.append(mean_vascular_hu(vol) - 40.0)
    times_a = np.asarray(times, dtype=np.float64)
    organ_a = np.asarray(organ, dtype=np.float64)
    aorta_a = np.asarray(aorta, dtype=np.float64)
    recirc = np.zeros_like(organ_a)
    curves = np.column_stack([aorta_a, organ_a, recirc])
    protocol = liver_protocol(70.0)
    meta = {
        "case_id": patient_id,
        "dataset": "mphase_liver",
        "source": "tcia_hcc_tace_seg",
        "collection": COLLECTION,
        "doi": DOI,
        "license": LICENSE,
        "body_weight_kg": 70.0,
        "phases_s": DEFAULT_PHASES_S,
        "injection": protocol.model_dump(),
        "roi": "nc_liver_20_90_propagated_or_wide_10_200; vascular_150_650_minus_40",
        "n_contrast_volumes": len(contrast_vols),
    }
    if extra_meta:
        meta.update(extra_meta)
    return EnhancementSeries(
        times_s=times_a,
        curves_hu=curves,
        region_names=REGION_NAMES,
        aif_hu=aorta_a.copy(),
        metadata=meta,
    )


def download_and_extract(
    *,
    raw_root: Path,
    extract_root: Path,
    max_cases: int = 20,
    sleep_s: float = 0.4,
) -> list[EnhancementSeries]:
    """Download a minimal TCIA subset and write NPZ extracts."""
    metadata = fetch_series_metadata()
    selected = select_baseline_series(metadata, max_cases=max_cases)
    if not selected:
        raise RuntimeError("TCIA metadata returned no usable baseline studies")
    manifest = []
    cohort: list[EnhancementSeries] = []
    dicom_root = raw_root / "_tcia_hcc_tace_seg"
    for item in selected:
        pid = str(item["patient_id"])
        pre_uid = str(item["pre"]["SeriesInstanceUID"])
        con_uid = str(item["contrast"]["SeriesInstanceUID"])
        pre_dir = dicom_root / pid / "pre"
        con_dir = dicom_root / pid / "contrast"
        print(f"downloading {pid} PRE {pre_uid[-12:]} ...", flush=True)
        download_series(pre_uid, pre_dir)
        time.sleep(sleep_s)
        print(f"downloading {pid} CONTRAST {con_uid[-12:]} ...", flush=True)
        download_series(con_uid, con_dir)
        time.sleep(sleep_s)
        series = extract_mphase_case(
            pre_dir,
            con_dir,
            patient_id=pid,
            extra_meta={
                "pre_series_uid": pre_uid,
                "contrast_series_uid": con_uid,
                "pre_description": item["pre"].get("SeriesDescription"),
                "contrast_description": item["contrast"].get("SeriesDescription"),
            },
        )
        write_case(extract_root / pid, series, metadata=dict(series.metadata))
        cohort.append(series)
        manifest.append(
            {
                "patient_id": pid,
                "pre_series_uid": pre_uid,
                "contrast_series_uid": con_uid,
                "times_s": series.times_s.tolist(),
                "organ_hu": series.region("organ").tolist(),
            }
        )
        print(
            f"  extracted {pid}: t={series.times_s.tolist()} "
            f"organ={np.round(series.region('organ'), 1).tolist()}",
            flush=True,
        )
    extract_root.mkdir(parents=True, exist_ok=True)
    (extract_root / "manifest.json").write_text(
        json.dumps(
            {
                "collection": COLLECTION,
                "doi": DOI,
                "license": LICENSE,
                "n_cases": len(cohort),
                "cases": manifest,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return cohort


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Download a minimal TCIA HCC-TACE-Seg subset."
    )
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--extract-root", type=Path, default=Path("data/raw/mphase_liver")
    )
    args = parser.parse_args(argv)
    n_cases = max(1, min(int(args.n), 30))
    cohort = download_and_extract(
        raw_root=args.raw_root, extract_root=args.extract_root, max_cases=n_cases
    )
    print(f"done: {len(cohort)} cases -> {args.extract_root}")


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
