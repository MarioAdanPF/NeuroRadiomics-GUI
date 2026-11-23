"""Extracción de punta a punta con el backend real (se omite si no está instalado)."""
import asyncio

import nibabel as nib
import numpy as np
import pytest

from logic.extraction import ExtractionService

service = ExtractionService()
pytestmark = pytest.mark.skipif(not service.available, reason="Backend de PyRadiomics no instalado (.venv-backend)")


def synthetic_subject(folder):
    affine = np.diag([1.0, 1.0, 1.0, 1.0])
    z, y, x = np.mgrid[:40, :40, :40]
    mask = np.zeros((40, 40, 40), np.int32)
    mask[((z - 12) / 6) ** 2 + ((y - 20) / 8) ** 2 + ((x - 20) / 5) ** 2 <= 1] = 17
    mask[((z - 28) / 5) ** 2 + ((y - 20) / 5) ** 2 + ((x - 20) / 5) ** 2 <= 1] = 53
    image = (mask > 0) * 80.0 + np.random.default_rng(0).normal(30, 5, mask.shape)
    nib.save(nib.MGHImage(image.astype(np.float32), affine), folder / "brain.mgz")
    nib.save(nib.MGHImage(mask, affine), folder / "aparc+aseg.mgz")
    return folder / "brain.mgz", folder / "aparc+aseg.mgz"


@pytest.mark.parametrize("workers", [1, 2])
def test_extraction_reports_progress_and_shape_features(tmp_path, monkeypatch, workers):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    image, mask = synthetic_subject(tmp_path)
    progress = []
    result = asyncio.run(service.extract(image, mask, labels=[17, 53, 99], shape_only=True, workers=workers,
                                         on_progress=progress.append))
    assert set(result.features) == {17, 53}
    assert set(result.errors) == {99}
    assert all(len(f) == 14 and all(k.startswith("original_shape_") for k in f) for f in result.features.values())
    assert progress[-1].done == progress[-1].total == 3
    # Una esfera de radio 5 tiene esfericidad cercana a 1
    assert result.features[53]["original_shape_Sphericity"] > 0.85


def test_parallel_and_sequential_give_identical_values(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    image, mask = synthetic_subject(tmp_path)
    one = asyncio.run(service.extract(image, mask, shape_only=False, workers=1))
    two = asyncio.run(service.extract(image, mask, shape_only=False, workers=2))
    assert one.features == two.features
    assert len(one.features[17]) > 14          # también las familias de intensidad y textura
