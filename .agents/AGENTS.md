# Contexto del Proyecto: NeuroRadiomics (GUI de extracción radiómica y predicción)

## 1. Qué es este software
NeuroRadiomics es una aplicación de escritorio que procesa resonancias magnéticas cerebrales segmentadas con FreeSurfer (`brain` + `aparc+aseg`): extrae características radiómicas con `PyRadiomics`, las muestra, y aplica modelos de predicción en formato JSON portable. Versión 1.0.0.

## 2. Arquitectura dual (Python 3.14 / Python 3.9)
- **Interfaz (`src/`):** Python 3.14 con **Flet 1.0.5** (`requirements-ui.txt`, entorno `.venv/`).
- **Backend (`legacy_service/extractor.py`):** Python 3.9, porque PyRadiomics 3.1.0 solo publica paquetes hasta 3.9 (`requirements-backend.txt`, entorno `.venv-backend/`). `logic/paths.find_backend_python()` lo busca en `NEURORADIOMICS_BACKEND_PYTHON`, `PY_LEGACY` o `.venv-backend/`.
- **Protocolo:** la interfaz lanza un proceso del backend por sujeto (`python extractor.py IMAGEN MASCARA [--labels] [--shape-only] [--workers N]`). El backend emite una línea JSON por evento (`start`, `region`, `region_error`, `done`, `error`). Cancelar = SIGTERM; el backend lo convierte en `sys.exit` para que el Pool cierre sus procesos hijos.
- **Invariantes verificados (no romper):** recortar cada región a su caja delimitadora da exactamente los mismos valores que la imagen completa; `--shape-only` da los mismos 14 descriptores de forma que la extracción completa; la conversión `.mgz` → `.nii.gz` con nibabel (`logic/volumes.to_nifti`) da la misma geometría y vóxeles que `mri_convert`; la extracción en paralelo y la secuencial dan valores idénticos. La configuración del extractor (`binWidth` 25, sin remuestreo) es la de los datos de entrenamiento: no cambiarla.

## 3. Estructura
- `src/main.py`: navegación (Predicción, Extracción, Visualizador, Regiones), tema y `AppContext` (en `ui/common.py`) con servicios compartidos y acciones entre vistas (`open_in_viewer`, `show_regions`, `notify`, `show_error`).
- `src/logic/`: `extraction.py` (servicio asíncrono del backend), `predictor.py` (modelos JSON), `volumes.py`, `subjects.py`, `results.py` (CSV y reporte), `volume_model.py` (cortes con la segmentación superpuesta), `lut.py` (`FreeSurferColorLUT.csv`), `helpers.py` (correlaciones), `paths.py`.
- `src/ui/`: `prediction_view.py`, `extraction_view.py`, `volume_view.py` + `plane_view.py`, `info_view.py`, `correlations_dialog.py`, `common.py`.
- `models/`: modelos `modelo_<comparativa>_<variante>.json`. `assets/logos/`: logos de la UAZ y la UDEM al pie de la barra lateral (el de la UDEM se tiñe con el color del tema). `tests/`: pytest con datos sintéticos. `docs/img/`: capturas del README.

## 4. Contrato del modelo (JSON `neuroradiomics-modelo`, versión 1)
Campos: `comparativa`, `variante`, `descripcion`, `umbral`, `regiones` (LabelID requeridos; también definen los totales de la normalización), `metricas_tamano`, `covariables` (`nombre`, `media`), `intercepto`, `variables` (`columna`, `label`, `region`, `descriptor`, `ajuste` {covariable: coef}, `centro`, `escala`, `peso`) y `rendimiento` (opcional). Predicción: normalizar volúmenes y áreas por su total en `regiones` → restar el ajuste por covariables → `(valor - centro) / escala` → `logit = intercepto + Σ peso·z`. Los archivos solo contienen parámetros, nunca datos de pacientes. Si cambia el formato, subir `version` y mantener la lectura de la anterior.

## 5. Directrices de trabajo
- Usar la API de **Flet 1.0** (`ft.Button`/`FilledButton`, `ft.run`, `Dropdown.on_select`, `page.show_dialog`/`pop_dialog`, `ft.Image(src=bytes)`). Los servicios como `FilePicker` se registran solos al construirse dentro de la página.
- Las vistas heredan de `ft.Container`: **no nombrar atributos como campos de `Container`** (`image`, `opacity`, `content`, `data`, `key`, `width`…), porque Flet los serializa y falla al validar.
- En un `Row`/`Column` sin ancho acotado no usar hijos con `expand` (un `Slider` con `expand` dentro de un `Column` centrado rompe el layout: Flutter dibuja un bloque gris). Lo mismo en filas con `wrap=True`.
- Nada bloqueante en los manejadores: lecturas de volúmenes con `asyncio.to_thread`, tareas largas con `page.run_task` y cancelables.
- Al modificar `volume_view.py`/`plane_view.py`/`volume_model.py`, no alterar `axis_map` ni `view_config` (rotación y flips) salvo petición expresa. Los volúmenes se reorientan a LIA al cargarse (`VolumeModel._read_lia`) y los planos siguen la convención radiológica: izquierda del paciente a la derecha de la pantalla, anterior arriba en el axial, superior arriba en el coronal y anterior a la izquierda en el sagital (`flip_x=True`, pedido por el usuario el 2026-11-23). `tests/test_volume_model.py` lo verifica, junto con que clic y cruz señalen el mismo vóxel.
- Las etiquetas de FreeSurfer se mapean con `logic/lut.py`.
- Antes de entregar cambios de interfaz, ejecutar `pytest` y revisar la app con capturas (`page.enable_screenshots = True` y `await page.take_screenshot()`), usando el sujeto público `bert` de FreeSurfer, nunca imágenes de pacientes.
- Este repositorio consume los modelos como archivos JSON y no depende del código de entrenamiento; la metodología de ML vive en su propio repositorio.
