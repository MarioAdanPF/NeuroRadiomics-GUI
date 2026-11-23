"""Pruebas del predictor con un modelo y un paciente sintéticos (sin datos reales)."""
import json
import math

import pytest

from logic.paths import MODELS_DIR
from logic.predictor import ModelFormatError, PortableModel, PredictionInputError, load_models

REGIONS = [17, 53, 1006]


def synthetic_model(threshold=0.5):
    return {
        "formato": "neuroradiomics-modelo", "version": 1, "comparativa": "1Y_vs_Ctrl", "variante": "completo",
        "descripcion": "modelo de prueba", "umbral": threshold, "regiones": REGIONS,
        "metricas_tamano": ["shape_MeshVolume", "shape_VoxelVolume", "shape_SurfaceArea"],
        "covariables": [{"nombre": "Edad", "media": 70.0}, {"nombre": "Sexo_F", "media": 0.5}],
        "intercepto": -0.2,
        "variables": [
            {"columna": "Label_17_original_shape_VoxelVolume", "label": 17, "region": "Hipocampo izq.",
             "descriptor": "Volumen (vóxeles)", "ajuste": {"Edad": -0.001, "Sexo_F": 0.002},
             "centro": 0.3, "escala": 0.05, "peso": -1.5},
            {"columna": "Label_53_original_shape_Sphericity", "label": 53, "region": "Hipocampo der.",
             "descriptor": "Esfericidad", "ajuste": {"Edad": 0.0, "Sexo_F": 0.0},
             "centro": 0.7, "escala": 0.1, "peso": 0.8},
        ],
        "rendimiento": {"AUC": 0.86},
    }


def synthetic_features():
    return {
        17: {"original_shape_VoxelVolume": 3000.0, "original_shape_MeshVolume": 2900.0,
             "original_shape_SurfaceArea": 1500.0, "original_shape_Sphericity": 0.65},
        53: {"original_shape_VoxelVolume": 3200.0, "original_shape_MeshVolume": 3100.0,
             "original_shape_SurfaceArea": 1600.0, "original_shape_Sphericity": 0.75},
        1006: {"original_shape_VoxelVolume": 3800.0, "original_shape_MeshVolume": 3700.0,
               "original_shape_SurfaceArea": 2000.0, "original_shape_Sphericity": 0.5},
        # Región que no está en el modelo: no debe cambiar el total de la normalización
        99: {"original_shape_VoxelVolume": 1e9},
    }


def expected_logit(age, sex):
    sex_f = 1.0 if sex == "F" else 0.0
    volume = 3000.0 / (3000.0 + 3200.0 + 3800.0)
    volume -= -0.001 * (age - 70.0) + 0.002 * (sex_f - 0.5)
    z1 = (volume - 0.3) / 0.05
    z2 = (0.75 - 0.7) / 0.1
    return -0.2 - 1.5 * z1 + 0.8 * z2


@pytest.mark.parametrize("age,sex", [(70, "F"), (82.5, "M"), (55, "F")])
def test_prediction_matches_manual_computation(age, sex):
    prediction = PortableModel(synthetic_model()).predict(synthetic_features(), age, sex)
    assert prediction.logit == pytest.approx(expected_logit(age, sex), abs=1e-12)
    assert prediction.probability == pytest.approx(1 / (1 + math.exp(-expected_logit(age, sex))), abs=1e-12)


def test_contributions_add_up_to_logit_and_are_sorted():
    prediction = PortableModel(synthetic_model()).predict(synthetic_features(), 75, "M")
    total = -0.2 + sum(c.contribution for c in prediction.contributions)
    assert total == pytest.approx(prediction.logit)
    sizes = [abs(c.contribution) for c in prediction.contributions]
    assert sizes == sorted(sizes, reverse=True)


def test_threshold_decides_class():
    features = synthetic_features()
    probability = PortableModel(synthetic_model()).predict(features, 70, "F").probability
    assert PortableModel(synthetic_model(threshold=probability - 1e-6)).predict(features, 70, "F").is_positive
    assert not PortableModel(synthetic_model(threshold=probability + 1e-6)).predict(features, 70, "F").is_positive


def test_missing_region_is_reported():
    features = synthetic_features()
    del features[1006]
    with pytest.raises(PredictionInputError, match="Faltan 1 regiones"):
        PortableModel(synthetic_model()).predict(features, 70, "F")


def test_missing_feature_and_invalid_values_are_reported():
    features = synthetic_features()
    del features[53]["original_shape_Sphericity"]
    with pytest.raises(PredictionInputError, match="le falta"):
        PortableModel(synthetic_model()).predict(features, 70, "F")
    features = synthetic_features()
    features[17]["original_shape_VoxelVolume"] = float("nan")
    with pytest.raises(PredictionInputError, match="faltante o infinito"):
        PortableModel(synthetic_model()).predict(features, 70, "F")


def test_sex_must_be_f_or_m():
    with pytest.raises(PredictionInputError):
        PortableModel(synthetic_model()).predict(synthetic_features(), 70, "X")


def test_extreme_logit_does_not_overflow():
    spec = synthetic_model()
    spec["intercepto"] = -5000
    assert PortableModel(spec).predict(synthetic_features(), 70, "F").probability == 0.0


def test_invalid_files_are_rejected(tmp_path):
    (tmp_path / "otro.json").write_text(json.dumps({"formato": "otro"}))
    (tmp_path / "roto.json").write_text("{")
    (tmp_path / "bueno.json").write_text(json.dumps(synthetic_model()))
    models, errors = load_models(tmp_path)
    assert [m.key for m in models] == ["1Y_vs_Ctrl_completo"]
    assert len(errors) == 2
    with pytest.raises(ModelFormatError):
        PortableModel({"formato": "neuroradiomics-modelo", "version": 99})


def test_bundled_models_load():
    models, errors = load_models(MODELS_DIR)
    assert models and not errors
    for model in models:
        assert len(model.regions) > 0 and len(model.variables) > 0
        assert 0 < model.threshold < 1
        assert model.variable_labels <= set(model.regions)
