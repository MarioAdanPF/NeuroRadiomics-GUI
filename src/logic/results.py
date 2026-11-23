"""
Lectura y escritura de resultados.

El CSV de características tiene una fila por región: PatientID, LabelID y las características
de PyRadiomics. Es el mismo formato con el que se entrenaron los modelos, así que un CSV
guardado aquí se puede volver a cargar para predecir sin repetir la extracción.
"""
import csv
import json
from datetime import datetime
from pathlib import Path

from logic.extraction import Features


def write_features_csv(path, patient_id: str, features: Features) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Unión de las columnas de todas las regiones, en el orden en que aparecen
    columns = list(dict.fromkeys(key for values in features.values() for key in values))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["PatientID", "LabelID", *columns])
        writer.writeheader()
        for label in sorted(features):
            writer.writerow({"PatientID": patient_id, "LabelID": label, **features[label]})
    return path


def read_features_csv(path) -> tuple[str, Features]:
    """Lee un CSV de características. Devuelve el PatientID y {LabelID: {característica: valor}}."""
    features, patient_id = {}, Path(path).stem
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "LabelID" not in reader.fieldnames:
            raise ValueError("El CSV no tiene la columna LabelID.")
        for row in reader:
            patient_id = row.pop("PatientID", None) or patient_id
            label = int(float(row.pop("LabelID")))
            row.pop("Target", None)
            values = {}
            for key, text in row.items():
                try:
                    values[key] = float(text)
                except (TypeError, ValueError):
                    continue        # columnas de texto (diagnósticos de PyRadiomics)
            features[label] = values
    return patient_id, features


def write_prediction_report(path, patient_id, age, sex, model, prediction, image=None, mask=None) -> Path:
    """Reporte JSON con los datos de entrada, el modelo y el resultado de la predicción."""
    report = {
        "generado": datetime.now().isoformat(timespec="seconds"),
        "paciente": {"id": patient_id, "edad": age, "sexo": sex,
                     "imagen": str(image) if image else None, "mascara": str(mask) if mask else None},
        "modelo": {"archivo": model.path.name if model.path else None, "comparativa": model.comparison,
                   "variante": model.variant, "descripcion": model.description, "umbral": model.threshold,
                   "rendimiento_validacion": model.performance},
        "resultado": {"probabilidad": prediction.probability, "logit": prediction.logit,
                      "clasificacion": "conversor" if prediction.is_positive else "no conversor"},
        "contribuciones": [{"region": c.region, "descriptor": c.descriptor, "columna": c.column,
                            "valor": c.value, "valor_escalado": c.z, "peso": c.weight, "aporte_logit": c.contribution}
                           for c in prediction.contributions],
        "aviso": "Herramienta de investigación. No es un dispositivo médico ni sustituye el diagnóstico clínico.",
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
