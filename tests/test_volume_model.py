"""Orientación de los planos del visualizador con un volumen sintético en LIA (como FreeSurfer)."""
import nibabel as nib
import numpy as np
import pytest

from logic.volume_model import VolumeModel

N = 21
LEFT, ANTERIOR, SUPERIOR = 1, 2, 3          # etiquetas de los marcadores


def synthetic_lia(tmp_path):
    """Máscara LIA (eje 0 hacia la izquierda, eje 1 hacia abajo, eje 2 hacia adelante) con tres marcadores."""
    mask = np.zeros((N, N, N), np.int32)
    mask[17:20, 9:12, 9:12] = LEFT          # izquierda del paciente
    mask[9:12, 9:12, 17:20] = ANTERIOR      # adelante
    mask[9:12, 1:4, 9:12] = SUPERIOR        # arriba (índice bajo del eje inferior)
    affine = np.array([[-1, 0, 0, 10], [0, 0, 1, -10], [0, -1, 0, 10], [0, 0, 0, 1]], float)
    assert nib.aff2axcodes(affine) == ("L", "I", "A")
    image = nib.Nifti1Image(mask.astype(np.float32) * 50 + 10, affine)
    nib.save(image, tmp_path / "lia.nii.gz")
    nib.save(nib.Nifti1Image(mask, affine), tmp_path / "lia_mask.nii.gz")
    # El mismo volumen guardado en RAS
    nib.save(nib.as_closest_canonical(image), tmp_path / "ras.nii.gz")
    nib.save(nib.as_closest_canonical(nib.Nifti1Image(mask, affine)), tmp_path / "ras_mask.nii.gz")
    return tmp_path


def centroid(model, plane, label):
    """Centro (fila, columna) del marcador en el corte que se muestra en pantalla."""
    model.coords = {"sagittal": N // 2, "coronal": N // 2, "axial": N // 2}
    rows, cols = np.nonzero(model._oriented_slice(model.mask, plane) == label)
    return rows.mean(), cols.mean()


@pytest.fixture
def model(tmp_path):
    folder = synthetic_lia(tmp_path)
    m = VolumeModel()
    m.load_volume(str(folder / "lia.nii.gz"))
    m.load_mask(str(folder / "lia_mask.nii.gz"))
    return m


def test_standard_radiological_orientation(model):
    middle = N // 2
    # Sagital: anterior a la izquierda, superior arriba
    assert centroid(model, "sagittal", ANTERIOR)[1] < middle
    assert centroid(model, "sagittal", SUPERIOR)[0] < middle
    # Axial: anterior arriba, izquierda del paciente a la derecha de la pantalla
    assert centroid(model, "axial", ANTERIOR)[0] < middle
    assert centroid(model, "axial", LEFT)[1] > middle
    # Coronal: superior arriba, izquierda del paciente a la derecha de la pantalla
    assert centroid(model, "coronal", SUPERIOR)[0] < middle
    assert centroid(model, "coronal", LEFT)[1] > middle


def test_other_orientations_are_displayed_the_same(model, tmp_path):
    ras = VolumeModel()
    ras.load_volume(str(tmp_path / "ras.nii.gz"))
    ras.load_mask(str(tmp_path / "ras_mask.nii.gz"))
    assert np.array_equal(ras.data, model.data)
    assert np.array_equal(ras.mask, model.mask)


@pytest.mark.parametrize("plane", ["axial", "coronal", "sagittal"])
def test_click_and_crosshair_mappings_are_inverse(model, plane):
    """Un clic en la pantalla y la cruz dibujada deben señalar el mismo vóxel (con los flips de view_config)."""
    cfg = model.view_config[plane]
    x_size = model.shape[model.axis_map[cfg["x_axis"]]]
    y_size = model.shape[model.axis_map[cfg["y_axis"]]]
    for col, row in [(0, 0), (3, 15), (x_size - 1, y_size - 1)]:
        # Clic -> vóxel (misma fórmula que VolumeView.on_plane_interaction)
        col_val = x_size - 1 - col if cfg["flip_x"] else col
        row_val = y_size - 1 - row if cfg["flip_y"] else row
        model.set_coord(cfg["x_axis"], col_val)
        model.set_coord(cfg["y_axis"], row_val)
        # Vóxel -> posición de la cruz (misma fórmula que VolumeView._refresh_plane_crosshairs)
        c = model.coords[cfg["x_axis"]]
        r = model.coords[cfg["y_axis"]]
        assert (x_size - 1 - c if cfg["flip_x"] else c, y_size - 1 - r if cfg["flip_y"] else r) == (col, row)
        # Y el vóxel señalado es el que se ve en esa posición del corte
        shown = model._oriented_slice(model.mask, plane)[row, col]
        assert shown == model.label_at_cursor()
