"""Pruebas de lectura y escritura de archivos con datos sintéticos."""
import nibabel as nib
import numpy as np

from logic.helpers import identify_high_correlations
from logic.lut import get_lut
from logic.results import read_features_csv, write_features_csv
from logic.subjects import find_subject_files, scan_directory
from logic.volumes import mask_labels, to_nifti


def test_features_csv_round_trip(tmp_path):
    features = {53: {"original_shape_Sphericity": 0.7, "original_shape_VoxelVolume": 3200.0},
                17: {"original_shape_Sphericity": 0.6, "original_shape_VoxelVolume": 3000.5}}
    path = write_features_csv(tmp_path / "sub" / "paciente.csv", "paciente", features)
    patient_id, loaded = read_features_csv(path)
    assert patient_id == "paciente"
    assert loaded == features
    # Mismo formato que los datos de entrenamiento: PatientID, LabelID y las características
    assert path.read_text().splitlines()[0].startswith("PatientID,LabelID,original_shape_")


def test_subject_detection(tmp_path):
    good = tmp_path / "sujeto1" / "mri"
    good.mkdir(parents=True)
    for name in ("brain.mgz", "aparc+aseg.mgz"):
        (good / name).touch()
    (tmp_path / "sujeto2").mkdir()
    (tmp_path / "sujeto2" / "brain.nii.gz").touch()          # sin máscara: no es válido
    subject = find_subject_files(tmp_path / "sujeto1")
    assert subject.id == "sujeto1" and subject.mask.name == "aparc+aseg.mgz"
    assert [s.id for s in scan_directory(tmp_path)] == ["sujeto1"]


def test_mgz_conversion_preserves_voxels_and_geometry(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    affine = np.array([[-1, 0, 0, 128.5], [0, 0, 1, -110.2], [0, -1, 0, 127.3], [0, 0, 0, 1]], float)
    data = np.zeros((20, 20, 20), np.int32)
    data[5:10, 5:10, 5:10] = 17
    data[12:15, 12:15, 12:15] = 1006
    source = tmp_path / "aparc+aseg.mgz"
    nib.save(nib.MGHImage(data, affine), source)

    converted = to_nifti(source)
    assert converted.name.endswith(".nii.gz") and converted.parent == tmp_path / "cache" / "neuroradiomics"
    image = nib.load(converted)
    assert np.array_equal(np.asanyarray(image.dataobj), data)
    assert np.allclose(image.affine, affine)
    assert to_nifti(source) == converted                     # la segunda vez usa la caché
    assert mask_labels(source) == [17, 1006]


def test_lut_colors_and_names():
    lut = get_lut()
    assert lut.name(17) == "Left-Hippocampus"
    assert tuple(lut.color_table()[17]) == (lut.entries[17].r, lut.entries[17].g, lut.entries[17].b)
    assert lut.name(-5) == "Región -5"


def test_high_correlations():
    regions = [{"a": float(i), "b": 2.0 * i + 1, "c": (-1) ** i, "d": 3.0} for i in range(6)]
    pairs = identify_high_correlations(regions)
    assert [(v1, v2) for v1, v2, _ in pairs] == [("a", "b")]
