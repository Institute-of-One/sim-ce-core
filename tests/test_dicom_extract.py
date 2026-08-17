"""Tiny local DICOM extract. Writes fixtures in tmp; no network."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sim_ce_core.data.dicom_extract import extract_mean_curve, series_from_dicom_folder


def _write_ct_frame(path: Path, value: int, temporal: int, instance: int) -> None:
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = CTImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.SOPClassUID = CTImageStorage
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.Modality = "CT"
    ds.Rows = 4
    ds.Columns = 4
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.RescaleSlope = 1.0
    ds.RescaleIntercept = 0.0
    ds.TemporalPositionIdentifier = temporal
    ds.InstanceNumber = instance
    ds.PixelData = np.full((4, 4), value, dtype=np.uint16).tobytes()
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.save_as(path, write_like_original=False)


def test_extract_mean_curve_from_two_frames(tmp_path: Path) -> None:
    _write_ct_frame(tmp_path / "t0.dcm", 40, temporal=1, instance=1)
    _write_ct_frame(tmp_path / "t1.dcm", 90, temporal=2, instance=2)
    times, enh = extract_mean_curve(tmp_path)
    np.testing.assert_allclose(times, [0.0, 1.0])
    np.testing.assert_allclose(enh, [0.0, 50.0])
    series = series_from_dicom_folder(tmp_path, case_id="demo")
    assert series.metadata["source"] == "dicom_extract"
    assert series.curves_hu.shape[0] == 2
