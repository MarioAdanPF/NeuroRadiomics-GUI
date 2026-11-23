"""
Comunicación con el backend de extracción (PyRadiomics en Python 3.9).

Por cada sujeto se lanza un solo proceso que extrae todas las regiones y reporta el avance
con una línea JSON por región (ver legacy_service/extractor.py).
"""
import asyncio
import json
import os
from dataclasses import dataclass, field
from typing import Callable

from logic.paths import EXTRACTOR_SCRIPT, find_backend_python
from logic.volumes import to_nifti

# {LabelID: {nombre de la característica: valor}}
Features = dict[int, dict[str, float]]


# Procesos por sujeto: las dos regiones de sustancia blanca tardan casi todo el tiempo,
# así que más de cuatro procesos apenas acelera
DEFAULT_WORKERS = max(1, min(4, os.cpu_count() or 1))


class BackendNotFoundError(RuntimeError):
    pass


class ExtractionError(RuntimeError):
    pass


@dataclass
class ExtractionResult:
    features: Features = field(default_factory=dict)
    errors: dict[int, str] = field(default_factory=dict)


@dataclass
class Progress:
    done: int
    total: int
    label: int | None = None


class ExtractionService:
    def __init__(self, backend_python=None):
        self.backend_python = backend_python or find_backend_python()

    @property
    def available(self) -> bool:
        return self.backend_python is not None

    async def extract(self, image, mask, labels=None, shape_only=True, workers=DEFAULT_WORKERS,
                      on_progress: Callable[[Progress], None] | None = None) -> ExtractionResult:
        """
        Extrae las características de las regiones indicadas (todas si labels es None),
        repartiéndolas entre `workers` procesos. Si la tarea se cancela, el backend se termina.
        """
        if not self.available:
            raise BackendNotFoundError("No se encontró el Python del backend (.venv-backend o PY_LEGACY).")

        # La conversión de .mgz lee el volumen completo: se hace fuera del hilo de la interfaz
        image_nii, mask_nii = await asyncio.gather(asyncio.to_thread(to_nifti, image), asyncio.to_thread(to_nifti, mask))

        command = [str(self.backend_python), str(EXTRACTOR_SCRIPT), str(image_nii), str(mask_nii)]
        if labels is not None:
            command += ["--labels", ",".join(str(label) for label in labels)]
        if shape_only:
            command.append("--shape-only")
        command += ["--workers", str(workers)]

        process = await asyncio.create_subprocess_exec(
            *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, limit=2**22)
        # stderr se lee en paralelo para que el proceso no se bloquee si escribe mucho
        stderr_task = asyncio.create_task(process.stderr.read())
        result = ExtractionResult()
        try:
            async for line in process.stdout:
                event = json.loads(line)
                kind = event["event"]
                if kind == "start" and on_progress:
                    on_progress(Progress(0, event["total"]))
                elif kind == "region":
                    result.features[event["label"]] = event["features"]
                elif kind == "region_error":
                    result.errors[event["label"]] = event["message"]
                elif kind == "error":
                    raise ExtractionError(event["message"])
                if kind in ("region", "region_error") and on_progress:
                    on_progress(Progress(event["index"], event["total"], event["label"]))
            await process.wait()
        except BaseException:
            # Cancelación o error: no dejar el proceso del backend vivo
            if process.returncode is None:
                # terminate deja que el Pool cierre sus procesos hijos; kill si no responde
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=3)
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
            raise
        finally:
            stderr = (await stderr_task).decode(errors="replace").strip()

        if process.returncode != 0:
            raise ExtractionError(stderr.splitlines()[-1] if stderr else f"El backend terminó con código {process.returncode}")
        return result
