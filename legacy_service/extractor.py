"""
Extracción de características con PyRadiomics (se ejecuta con el Python 3.9 del backend).

Procesa todas las regiones pedidas en un solo proceso: la imagen y la máscara se leen una
vez y cada región se recorta a su caja delimitadora antes de extraer. El recorte no cambia
ningún valor (las características solo dependen de los vóxeles de la región y del tamaño
del vóxel) y hace la extracción unas diez veces más rápida.

Uso:
    python extractor.py IMAGEN MASCARA [--labels 17,53] [--shape-only] [--workers 4]

Sin --labels se procesan todas las regiones de la máscara. Con --workers las regiones se
reparten entre varios procesos. La salida es una línea JSON por
evento, para que la interfaz pueda mostrar el progreso:
    {"event": "start", "labels": [...], "total": N}
    {"event": "region", "label": 17, "index": 1, "total": N, "features": {...}}   (en orden de término)
    {"event": "region_error", "label": 17, "index": 1, "total": N, "message": "..."}
    {"event": "done"}
    {"event": "error", "message": "..."}      (error que detiene todo el proceso)
"""
import argparse
import json
import logging
import multiprocessing
import signal
import sys

import numpy as np
import SimpleITK as sitk
from radiomics import featureextractor

# Margen en vóxeles alrededor de la caja de cada región
CROP_MARGIN = 3


def emit(event, **data):
    """Escribe un evento como una línea JSON y la envía de inmediato."""
    data["event"] = event
    sys.stdout.write(json.dumps(data) + "\n")
    sys.stdout.flush()


def build_extractor(shape_only):
    # Misma configuración con la que se extrajeron los datos de entrenamiento
    settings = {"binWidth": 25, "resampledPixelSpacing": None, "interpolator": sitk.sitkBSpline}
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    if shape_only:
        extractor.disableAllFeatures()
        extractor.enableFeatureClassByName("shape")
    return extractor


def crop_to_label(image, mask, mask_array, label):
    """Recorta imagen y máscara a la caja de la región, con un margen de CROP_MARGIN vóxeles."""
    voxels = np.argwhere(mask_array == label)            # índices (z, y, x)
    low = np.maximum(voxels.min(axis=0) - CROP_MARGIN, 0)
    high = np.minimum(voxels.max(axis=0) + CROP_MARGIN + 1, mask_array.shape)
    # SimpleITK indexa en orden (x, y, z)
    start = [int(v) for v in low[::-1]]
    size = [int(v) for v in (high - low)[::-1]]
    return sitk.RegionOfInterest(image, size, start), sitk.RegionOfInterest(mask, size, start)


# Estado de cada proceso de trabajo: la imagen y la máscara se leen una sola vez por proceso
_worker = {}


def init_worker(image_path, mask_path, shape_only):
    logging.getLogger("radiomics").setLevel(logging.ERROR)
    _worker["image"] = sitk.ReadImage(image_path)
    _worker["mask"] = sitk.ReadImage(mask_path)
    _worker["mask_array"] = sitk.GetArrayFromImage(_worker["mask"])
    _worker["extractor"] = build_extractor(shape_only)


def extract_label(label):
    """Extrae una región. Devuelve (label, características, mensaje de error)."""
    try:
        image_roi, mask_roi = crop_to_label(_worker["image"], _worker["mask"], _worker["mask_array"], label)
        result = _worker["extractor"].execute(image_roi, mask_roi, label=label)
        # Solo los valores numéricos; los metadatos de diagnóstico no se envían
        features = {key: float(value) for key, value in result.items() if not key.startswith("diagnostics")}
        return label, features, None
    except Exception as error:
        return label, None, str(error)


def run(image_path, mask_path, labels, shape_only, workers):
    mask_array = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
    # Número de vóxeles de cada región presente en la máscara
    values, counts = np.unique(mask_array, return_counts=True)
    sizes = {int(v): int(c) for v, c in zip(values, counts) if v != 0}
    del mask_array

    if labels is None:
        labels = sorted(sizes)
    emit("start", labels=labels, total=len(labels))

    index = 0
    for label in labels:
        if label not in sizes:
            index += 1
            emit("region_error", label=label, index=index, total=len(labels), message="La región no está en la máscara")
    # Las regiones grandes (sustancia blanca) tardan mucho más: se empiezan primero para repartir mejor el trabajo
    pending = sorted((l for l in labels if l in sizes), key=lambda l: sizes[l], reverse=True)

    def report(label, features, error):
        nonlocal index
        index += 1
        if error is None:
            emit("region", label=label, index=index, total=len(labels), features=features)
        else:
            emit("region_error", label=label, index=index, total=len(labels), message=error)

    workers = max(1, min(workers, len(pending)))
    if workers == 1:
        init_worker(image_path, mask_path, shape_only)
        for label in pending:
            report(*extract_label(label))
    else:
        with multiprocessing.Pool(workers, initializer=init_worker, initargs=(image_path, mask_path, shape_only)) as pool:
            for result in pool.imap_unordered(extract_label, pending):
                report(*result)
    emit("done")


def parse_labels(text):
    if not text:
        return None
    return [int(part) for part in text.split(",") if part.strip()]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image")
    parser.add_argument("mask")
    parser.add_argument("--labels", default=None, help="LabelID separados por comas (por defecto, todos)")
    parser.add_argument("--shape-only", action="store_true", help="extraer solo los descriptores de forma")
    parser.add_argument("--workers", type=int, default=1, help="procesos en paralelo (por defecto, 1)")
    args = parser.parse_args()

    # PyRadiomics escribe advertencias en el log; se silencian para no mezclarlas con el JSON
    logging.getLogger("radiomics").setLevel(logging.ERROR)
    # Al cancelar, la interfaz envía SIGTERM: sys.exit cierra el Pool y termina sus procesos
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(1))
    try:
        run(args.image, args.mask, parse_labels(args.labels), args.shape_only, args.workers)
    except Exception as error:
        emit("error", message=str(error))
        sys.exit(1)
