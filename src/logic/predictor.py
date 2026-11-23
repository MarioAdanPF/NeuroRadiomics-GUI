"""
Modelos de predicción en formato portable (JSON "neuroradiomics-modelo", versión 1).

Cada modelo es una regresión logística sobre descriptores de forma. Para una imagen:

1. Se toman las regiones del modelo y cada volumen o área se divide entre el total de esa
   métrica en todas esas regiones (normalización por volumen total segmentado).
2. A cada variable se le resta el efecto de la edad y el sexo, se le resta su mediana y se
   divide entre su rango intercuartílico.
3. La suma de cada variable por su peso, más el intercepto, es el logit; la función
   logística lo convierte en probabilidad.

El archivo no contiene datos de pacientes, solo los parámetros de cada variable.
"""
import json
import math
from dataclasses import dataclass
from pathlib import Path

FORMAT = "neuroradiomics-modelo"
SUPPORTED_VERSIONS = {1}
# Valor escalado a partir del cual una variable se considera fuera del rango del entrenamiento
ATYPICAL_Z = 4.0


class ModelFormatError(ValueError):
    pass


class PredictionInputError(ValueError):
    pass


@dataclass
class Contribution:
    column: str
    label: int
    region: str
    descriptor: str
    value: float          # valor de la variable después de normalizar
    z: float              # valor ajustado y escalado (en rangos intercuartílicos)
    weight: float
    contribution: float   # aporte al logit: peso * z


@dataclass
class Prediction:
    probability: float
    threshold: float
    logit: float
    contributions: list[Contribution]

    @property
    def is_positive(self) -> bool:
        return self.probability >= self.threshold

    @property
    def atypical(self) -> list[Contribution]:
        return [c for c in self.contributions if abs(c.z) > ATYPICAL_Z]


class PortableModel:
    def __init__(self, spec: dict, path: Path | None = None):
        if spec.get("formato") != FORMAT:
            raise ModelFormatError("El archivo no es un modelo de NeuroRadiomics.")
        if spec.get("version") not in SUPPORTED_VERSIONS:
            raise ModelFormatError(f"Versión de modelo no soportada: {spec.get('version')}.")
        self.spec = spec
        self.path = path

    @classmethod
    def load(cls, path) -> "PortableModel":
        path = Path(path)
        try:
            spec = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ModelFormatError(f"No se pudo leer el modelo: {error}") from error
        return cls(spec, path)

    # --- Descripción ---------------------------------------------------------------------

    @property
    def comparison(self) -> str:
        return self.spec["comparativa"]

    @property
    def variant(self) -> str:
        return self.spec["variante"]

    @property
    def threshold(self) -> float:
        return float(self.spec["umbral"])

    @property
    def regions(self) -> list[int]:
        return [int(label) for label in self.spec["regiones"]]

    @property
    def variables(self) -> list[dict]:
        return self.spec["variables"]

    @property
    def variable_labels(self) -> set[int]:
        """Regiones que usa alguna variable del modelo."""
        return {int(v["label"]) for v in self.variables}

    @property
    def performance(self) -> dict:
        return self.spec.get("rendimiento") or {}

    @property
    def description(self) -> str:
        return self.spec.get("descripcion", "")

    @property
    def display_name(self) -> str:
        return f"{self.comparison.replace('_', ' ')} · {self.variant.capitalize()} ({len(self.variables)} variables)"

    @property
    def key(self) -> str:
        return f"{self.comparison}_{self.variant}"

    # --- Predicción ----------------------------------------------------------------------

    def predict(self, features, age: float, sex: str) -> Prediction:
        """
        features: {LabelID: {"original_shape_...": valor}}, como lo entrega el extractor
        age:      edad en años al momento de la imagen
        sex:      "F" o "M"
        """
        if sex not in ("F", "M"):
            raise PredictionInputError("Indique el sexo (F o M).")
        covariates = {"Edad": float(age), "Sexo_F": 1.0 if sex == "F" else 0.0}
        missing = [label for label in self.regions if label not in features]
        if missing:
            raise PredictionInputError(f"Faltan {len(missing)} regiones que necesita el modelo: {missing[:5]}.")

        # Total de cada métrica de tamaño en las regiones del modelo
        totals = {}
        for metric in self.spec["metricas_tamano"]:
            key = f"original_{metric}"
            totals[metric] = sum(self._value(features, label, key) for label in self.regions)

        logit = float(self.spec["intercepto"])
        contributions = []
        for variable in self.variables:
            label = int(variable["label"])
            key = variable["columna"].split("_", 2)[2]          # "original_shape_..."
            value = self._value(features, label, key)
            metric = key.removeprefix("original_")
            if metric in totals:
                value /= totals[metric]
            adjusted = value - sum(variable["ajuste"][c["nombre"]] * (covariates[c["nombre"]] - c["media"])
                                   for c in self.spec["covariables"])
            z = (adjusted - variable["centro"]) / variable["escala"]
            contribution = variable["peso"] * z
            logit += contribution
            contributions.append(Contribution(variable["columna"], label, variable["region"], variable["descriptor"],
                                              value, z, variable["peso"], contribution))

        contributions.sort(key=lambda c: abs(c.contribution), reverse=True)
        return Prediction(_sigmoid(logit), self.threshold, logit, contributions)

    @staticmethod
    def _value(features, label, key) -> float:
        try:
            value = float(features[label][key])
        except KeyError:
            raise PredictionInputError(f"A la región {label} le falta la característica {key}.") from None
        except (TypeError, ValueError):
            raise PredictionInputError(f"Valor no numérico en la región {label}, {key}.") from None
        if not math.isfinite(value):
            raise PredictionInputError(f"Valor faltante o infinito en la región {label}, {key}.")
        return value


def _sigmoid(x: float) -> float:
    # Forma estable para logits muy grandes en valor absoluto
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def load_models(directory) -> tuple[list[PortableModel], list[str]]:
    """Carga todos los modelos .json de una carpeta. Devuelve los modelos y los errores."""
    models, errors = [], []
    for path in sorted(Path(directory).glob("*.json")):
        try:
            models.append(PortableModel.load(path))
        except ModelFormatError as error:
            errors.append(f"{path.name}: {error}")
    # Primero por comparativa y después la variante completa antes que la compacta
    models.sort(key=lambda m: (m.comparison, m.variant != "completo"))
    return models, errors
