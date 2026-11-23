"""
Vista de predicción: flujo guiado para estimar la probabilidad de conversión de DCL a EA.

1. Sujeto: carpeta de FreeSurfer (brain + aparc+aseg), archivos sueltos o un CSV ya extraído.
2. Datos clínicos: edad y sexo al momento de la imagen (los modelos están ajustados por ellos).
3. Modelo: cualquiera de los modelos de la carpeta models/ o uno cargado desde un archivo.

Las características se extraen una sola vez; al cambiar de modelo, de edad o de sexo, la
predicción se recalcula al instante.
"""
import asyncio
import time
from pathlib import Path

import flet as ft

from logic.extraction import ExtractionError, Progress
from logic.predictor import ModelFormatError, PortableModel, Prediction, PredictionInputError
from logic.results import read_features_csv, write_features_csv, write_prediction_report
from logic.subjects import find_subject_files
from logic.volumes import VOLUME_EXTENSIONS
from ui.common import (ACCENT, MUTED, NEGATIVE, POSITIVE, AppContext, disclaimer, file_chip, section,
                       status_row, view_header)

TOP_CONTRIBUTIONS = 10
METRIC_NAMES = {"AUC": "AUC", "Exactitud_Balanceada": "Exactitud balanceada",
                "Sensibilidad": "Sensibilidad", "Especificidad": "Especificidad"}


class PredictionView(ft.Container):
    def __init__(self, ctx: AppContext):
        super().__init__()
        self.ctx = ctx
        self.expand = True
        self.padding = ft.Padding.only(left=24, right=24, top=20, bottom=8)

        # Estado
        self.image_path: Path | None = None
        self.mask_path: Path | None = None
        self.csv: Path | None = None
        self.features = None           # {LabelID: {característica: valor}}
        self.prediction: Prediction | None = None
        self.task: asyncio.Task | None = None
        self.show_all = False
        self.extra_models: list[PortableModel] = []

        self.picker = ft.FilePicker()

        # --- 1. Sujeto ---
        self.subject_status = ft.Column(spacing=6)
        self.patient_id = ft.TextField(label="Identificador del paciente", dense=True, on_change=self._on_input_change)
        other_sources = ft.PopupMenuButton(
            content=ft.Row([ft.Icon(ft.Icons.MORE_HORIZ, size=18), ft.Text("Otras fuentes", size=13)], spacing=6, tight=True),
            tooltip="Elegir los archivos por separado o un CSV ya extraído",
            items=[ft.PopupMenuItem("Elegir imagen (brain)", icon=ft.Icons.IMAGE_OUTLINED, on_click=self._pick_image),
                   ft.PopupMenuItem("Elegir máscara (aparc+aseg)", icon=ft.Icons.LAYERS_OUTLINED, on_click=self._pick_mask),
                   ft.PopupMenuItem("Cargar CSV de características", icon=ft.Icons.TABLE_CHART_OUTLINED, on_click=self._pick_csv)])
        subject_card = section("Sujeto", None, [
            ft.Text("Carpeta de un sujeto procesado con FreeSurfer. Se buscan brain y aparc+aseg en ella o en su subcarpeta mri.",
                    size=12.5, color=MUTED),
            ft.Row([ft.FilledTonalButton("Carpeta del sujeto", icon=ft.Icons.FOLDER_OPEN, on_click=self._pick_folder),
                    other_sources], wrap=True),
            self.subject_status, self.patient_id], step=1)

        # --- 2. Datos clínicos ---
        self.age = ft.TextField(label="Edad", suffix="años", width=120, dense=True,
                                input_filter=ft.InputFilter(r"[0-9.,]"), on_change=self._on_input_change)
        self.sex = ft.SegmentedButton(
            segments=[ft.Segment("F", label="Femenino"), ft.Segment("M", label="Masculino")],
            selected=[], allow_empty_selection=True, show_selected_icon=False, on_change=self._on_input_change)
        clinical_card = section("Datos clínicos", None, [
            ft.Text("Al momento de la imagen. Los modelos están ajustados por edad y sexo.", size=12.5, color=MUTED),
            ft.Row([self.age, self.sex], spacing=16, vertical_alignment=ft.CrossAxisAlignment.CENTER, wrap=True)], step=2)

        # --- 3. Modelo ---
        self.model_dropdown = ft.Dropdown(label="Modelo", dense=True, expand=True, on_select=self._on_model_change)
        self.model_info = ft.Text(size=12.5, color=MUTED)
        model_card = section("Modelo", None, [
            ft.Row([self.model_dropdown,
                    ft.IconButton(ft.Icons.FILE_UPLOAD_OUTLINED, tooltip="Cargar un modelo desde un archivo JSON",
                                  on_click=self._pick_model)]),
            self.model_info], step=3)

        # --- Acción ---
        self.run_button = ft.FilledButton("Extraer y predecir", icon=ft.Icons.PLAY_ARROW, on_click=self._on_run, height=44)
        self.cancel_button = ft.OutlinedButton("Cancelar", icon=ft.Icons.STOP, on_click=self._on_cancel, visible=False, height=44)
        self.progress = ft.ProgressBar(value=0, visible=False, border_radius=4)
        self.status = ft.Text("", size=12.5, color=MUTED)
        self.live_hint = status_row(True, "La predicción se recalcula al cambiar el modelo, la edad o el sexo.")

        left = ft.Column([subject_card, clinical_card, model_card,
                          ft.Row([self.run_button, self.cancel_button]), self.progress, self.status, self.live_hint],
                         spacing=14, scroll=ft.ScrollMode.AUTO, width=430)

        # --- Resultados ---
        self.result_area = ft.Column(spacing=14, scroll=ft.ScrollMode.AUTO, expand=True)
        right = ft.Container(self.result_area, expand=True)

        self.content = ft.Column([
            view_header("Predicción", "Probabilidad de conversión de DCL a enfermedad de Alzheimer a partir de la morfología cerebral"),
            ft.Row([left, ft.VerticalDivider(width=1), right], expand=True, spacing=20,
                   vertical_alignment=ft.CrossAxisAlignment.START),
            disclaimer()], expand=True, spacing=14)

        self.reload_models()
        self._render_subject()
        self._render_placeholder()
        self._refresh_controls()

    # --- Modelos -------------------------------------------------------------------------

    @property
    def models(self) -> list[PortableModel]:
        return self.ctx.models + self.extra_models

    @property
    def model(self) -> PortableModel | None:
        return next((m for m in self.models if m.key == self.model_dropdown.value), None)

    def reload_models(self):
        self.model_dropdown.options = [ft.DropdownOption(key=m.key, text=m.display_name) for m in self.models]
        if self.model is None and self.models:
            self.model_dropdown.value = self.models[0].key
        self._render_model_info()

    def _render_model_info(self):
        model = self.model
        if model is None:
            self.model_info.value = "No hay modelos en la carpeta models/. Cargue uno con el botón de la derecha."
            return
        lines = [model.description or model.display_name, f"Umbral de decisión: {model.threshold:.2f}"]
        if model.performance:
            metrics = ", ".join(f"{METRIC_NAMES.get(k, k)} {v:.3f}" for k, v in model.performance.items())
            lines.append(f"Rendimiento en validación cruzada: {metrics}")
        self.model_info.value = "\n".join(lines)

    async def _pick_model(self, e):
        files = await self.picker.pick_files(dialog_title="Cargar modelo", allowed_extensions=["json"])
        if not files:
            return
        try:
            model = PortableModel.load(files[0].path)
        except ModelFormatError as error:
            self.ctx.show_error("Modelo no válido", str(error))
            return
        # Un modelo con la misma clave reemplaza al anterior
        self.extra_models = [m for m in self.extra_models if m.key != model.key] + [model]
        self.ctx.models = [m for m in self.ctx.models if m.key != model.key]
        self.reload_models()
        self.model_dropdown.value = model.key
        self._on_model_change(None)

    def _on_model_change(self, e):
        self._render_model_info()
        self._update_prediction()

    # --- Sujeto --------------------------------------------------------------------------

    async def _pick_folder(self, e):
        path = await self.picker.get_directory_path(dialog_title="Carpeta del sujeto")
        if not path:
            return
        subject = find_subject_files(path)
        if subject is None:
            self.ctx.show_error("No se encontró el sujeto",
                                f"En {path} (ni en su subcarpeta mri) no están brain y aparc+aseg en formato .mgz, .nii o .nii.gz.")
            return
        self._set_subject(subject.image, subject.mask, None, subject.id)

    async def _pick_image(self, e):
        files = await self.picker.pick_files(dialog_title="Imagen (brain)", allowed_extensions=["mgz", "nii", "gz"])
        if files:
            self._set_subject(Path(files[0].path), self.mask_path, None, self.patient_id.value or Path(files[0].path).parent.name)

    async def _pick_mask(self, e):
        files = await self.picker.pick_files(dialog_title="Máscara (aparc+aseg)", allowed_extensions=["mgz", "nii", "gz"])
        if files:
            self._set_subject(self.image_path, Path(files[0].path), None, self.patient_id.value or Path(files[0].path).parent.name)

    async def _pick_csv(self, e):
        files = await self.picker.pick_files(dialog_title="CSV de características", allowed_extensions=["csv"])
        if not files:
            return
        try:
            patient_id, features = read_features_csv(files[0].path)
        except (OSError, ValueError) as error:
            self.ctx.show_error("No se pudo leer el CSV", str(error))
            return
        self._set_subject(None, None, Path(files[0].path), patient_id)
        self.features = features
        self.status.value = f"CSV cargado: {len(features)} regiones."
        self._update_prediction()
        self._refresh_controls()

    def _set_subject(self, image, mask, csv, patient_id):
        for path in (image, mask):
            if path is not None and not str(path).lower().endswith(VOLUME_EXTENSIONS):
                self.ctx.show_error("Formato no soportado", f"{Path(path).name} no es .mgz, .nii ni .nii.gz.")
                return
        self.image_path, self.mask_path, self.csv = image, mask, csv
        self.patient_id.value = patient_id or ""
        # Las características anteriores ya no corresponden a este sujeto
        self.features, self.prediction = None, None
        self.status.value = ""
        self._render_subject()
        self._render_placeholder()
        self._refresh_controls()

    def _render_subject(self):
        rows = []
        if self.csv:
            rows.append(status_row(True, f"CSV: {self.csv.name}"))
        else:
            rows.append(ft.Row([ft.Text("Imagen", size=13, width=70, color=MUTED), file_chip(self.image_path)]))
            rows.append(ft.Row([ft.Text("Máscara", size=13, width=70, color=MUTED), file_chip(self.mask_path)]))
        self.subject_status.controls = rows

    # --- Entradas ------------------------------------------------------------------------

    def _age_value(self) -> float | None:
        try:
            age = float(self.age.value.replace(",", "."))
        except ValueError:
            return None
        return age if 18 <= age <= 110 else None

    def _sex_value(self) -> str | None:
        return self.sex.selected[0] if self.sex.selected else None

    def _on_input_change(self, e):
        self._refresh_controls()
        self._update_prediction()

    def _refresh_controls(self):
        running = self.task is not None and not self.task.done()
        has_source = bool(self.csv) or bool(self.image_path and self.mask_path)
        ready = has_source and self._age_value() is not None and self._sex_value() and self.model is not None
        # Con las características ya extraídas la predicción se recalcula sola: el botón sobra
        self.run_button.visible = not running and self.features is None
        self.run_button.disabled = not ready
        self.live_hint.visible = self.features is not None
        self.cancel_button.visible = running
        self.progress.visible = running
        self.age.error = None if not self.age.value or self._age_value() is not None else "Edad no válida"
        if not ready and not running:
            missing = [text for ok, text in [(has_source, "el sujeto"), (self._age_value() is not None, "la edad"),
                                             (self._sex_value(), "el sexo"), (self.model is not None, "un modelo")] if not ok]
            self.run_button.tooltip = "Falta " + ", ".join(missing)
        else:
            self.run_button.tooltip = None

    # --- Extracción ----------------------------------------------------------------------

    def _on_run(self, e):
        if not self.ctx.extraction.available:
            self.ctx.show_error("Backend no disponible",
                                "No se encontró el Python 3.9 con PyRadiomics. Cree .venv-backend como indica el README "
                                "o defina la variable de entorno NEURORADIOMICS_BACKEND_PYTHON.")
            return
        self.task = self.page.run_task(self._extract)
        self._refresh_controls()

    def _on_cancel(self, e):
        if self.task:
            self.task.cancel()

    async def _extract(self):
        # Solo hacen falta las regiones de los modelos y los descriptores de forma
        labels = sorted({label for model in self.models for label in model.regions})
        start = time.monotonic()
        self.status.value = "Preparando la imagen..."
        self.progress.value = None
        self.update()

        def on_progress(p: Progress):
            self.progress.value = p.done / p.total if p.total else None
            name = self.ctx.lut.name(p.label) if p.label is not None else ""
            self.status.value = f"Extrayendo región {p.done} de {p.total} {name}"
            self.update()

        try:
            result = await self.ctx.extraction.extract(self.image_path, self.mask_path, labels=labels, shape_only=True, on_progress=on_progress)
        except asyncio.CancelledError:
            self.status.value = "Extracción cancelada."
            self.task = None
            self._refresh_controls()
            self.update()
            return
        except (ExtractionError, OSError, ValueError) as error:
            self.status.value = "La extracción falló."
            self.task = None
            self._refresh_controls()
            self.update()
            self.ctx.show_error("Error en la extracción", str(error))
            return

        self.task = None
        self.features = result.features
        message = f"{len(result.features)} regiones extraídas en {time.monotonic() - start:.0f} s."
        if result.errors:
            message += f" {len(result.errors)} regiones con error o ausentes en la máscara."
        self.status.value = message
        self._update_prediction()
        self._refresh_controls()
        self.update()

    # --- Predicción ----------------------------------------------------------------------

    def _update_prediction(self):
        """Recalcula la predicción si ya hay características. Es inmediato: no repite la extracción."""
        if self.features is None:
            return
        model, age, sex = self.model, self._age_value(), self._sex_value()
        if model is None or age is None or sex is None:
            self.prediction = None
            self._render_placeholder("Complete la edad, el sexo y el modelo para ver la predicción.")
            return
        try:
            self.prediction = model.predict(self.features, age, sex)
        except PredictionInputError as error:
            self.prediction = None
            self._render_message(ft.Icons.ERROR_OUTLINE, "No se puede aplicar este modelo", str(error), ft.Colors.ERROR)
            return
        self._render_result(model, self.prediction)
        self._refresh_controls()

    def _render_placeholder(self, text=None):
        self._render_message(ft.Icons.INSIGHTS, "Los resultados aparecerán aquí",
                             text or "Elija un sujeto, capture su edad y sexo y presione Extraer y predecir. "
                                     "La extracción de las regiones que usan los modelos tarda alrededor de un minuto.", MUTED)

    def _render_message(self, icon, title, text, color):
        self.result_area.controls = [ft.Container(
            ft.Column([ft.Icon(icon, size=48, color=color), ft.Text(title, size=17, weight=ft.FontWeight.W_600),
                       ft.Text(text, size=13, color=MUTED, text_align=ft.TextAlign.CENTER)],
                      horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=8),
            padding=40, alignment=ft.Alignment.CENTER)]

    def _render_result(self, model: PortableModel, prediction: Prediction):
        positive = prediction.is_positive
        color = POSITIVE if positive else NEGATIVE
        verdict = "Patrón compatible con conversión a EA" if positive else "Patrón compatible con DCL estable"

        summary = section("Resultado", ft.Icons.INSIGHTS, [
            ft.Row([
                ft.Text(f"{prediction.probability:.0%}", size=44, weight=ft.FontWeight.BOLD, color=color),
                ft.Column([ft.Text("Probabilidad de conversión", size=13, color=MUTED),
                           ft.Container(ft.Text(verdict, size=13, weight=ft.FontWeight.W_600, color=ft.Colors.BLACK),
                                        bgcolor=color, border_radius=12, padding=ft.Padding.symmetric(horizontal=10, vertical=4))],
                          spacing=6)], spacing=18, vertical_alignment=ft.CrossAxisAlignment.CENTER),
            self._probability_bar(prediction.probability, prediction.threshold, color),
            ft.Text(f"{model.display_name}. Se clasifica como conversión cuando la probabilidad es de al menos "
                    f"{prediction.threshold:.0%}. Los modelos detectan conversión en un horizonte de 1 a 2 años; "
                    "no distinguen si ocurrirá a los 12 o a los 24 meses.", size=12.5, color=MUTED)])

        blocks = [summary]
        if prediction.atypical:
            names = ", ".join(f"{c.region} ({c.descriptor})" for c in prediction.atypical[:5])
            blocks.append(ft.Container(
                ft.Row([ft.Icon(ft.Icons.WARNING_AMBER, color=ft.Colors.AMBER_400),
                        ft.Text(f"{len(prediction.atypical)} variables están muy lejos de los valores del entrenamiento: "
                                f"{names}. Revise la segmentación antes de interpretar el resultado.", size=12.5, expand=True)],
                       vertical_alignment=ft.CrossAxisAlignment.START),
                padding=12, border_radius=10, bgcolor=ft.Colors.with_opacity(0.12, ft.Colors.AMBER)))

        blocks.append(self._contributions_section(prediction))
        blocks.append(self._other_models_section(model))
        blocks.append(ft.Row([
            ft.OutlinedButton("Ver en el visualizador", icon=ft.Icons.VIEW_IN_AR, on_click=self._open_viewer,
                              disabled=not (self.image_path and self.mask_path)),
            ft.OutlinedButton("Guardar reporte", icon=ft.Icons.SAVE_OUTLINED, on_click=self._save_report),
            ft.OutlinedButton("Guardar CSV", icon=ft.Icons.TABLE_CHART_OUTLINED, on_click=self._save_csv,
                              disabled=self.csv is not None)], wrap=True))
        self.result_area.controls = blocks

    @staticmethod
    def _probability_bar(probability, threshold, color):
        # Barras proporcionales con expand: no dependen del ancho de la ventana
        scale = 1000
        filled = max(1, int(probability * scale))
        bar = ft.Container(ft.Row([ft.Container(bgcolor=color, expand=filled, height=12, border_radius=6),
                                   ft.Container(expand=max(1, scale - filled))], spacing=0),
                           bgcolor=ft.Colors.with_opacity(0.12, ft.Colors.ON_SURFACE), border_radius=6, height=12)
        marker_at = max(1, int(threshold * scale))
        marker = ft.Row([ft.Container(expand=marker_at),
                         ft.Column([ft.Container(width=2, height=10, bgcolor=ft.Colors.ON_SURFACE),
                                    ft.Text(f"umbral {threshold:.2f}", size=11, color=MUTED)],
                                   spacing=2, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                         ft.Container(expand=max(1, scale - marker_at))], spacing=0)
        return ft.Column([bar, marker], spacing=0)

    def _contributions_section(self, prediction: Prediction):
        items = prediction.contributions if self.show_all else prediction.contributions[:TOP_CONTRIBUTIONS]
        largest = max((abs(c.contribution) for c in prediction.contributions), default=1) or 1
        scale = 1000
        rows = []
        for c in items:
            size = max(1, int(abs(c.contribution) / largest * scale))
            bar = ft.Container(bgcolor=POSITIVE if c.contribution > 0 else NEGATIVE, expand=size, height=14, border_radius=3)
            space = ft.Container(expand=scale - size + 1)
            # Barra divergente: hacia la izquierda disminuye, hacia la derecha aumenta
            left = ft.Row([space, bar] if c.contribution < 0 else [ft.Container(expand=1)], spacing=0, expand=True)
            right = ft.Row([bar, space] if c.contribution > 0 else [ft.Container(expand=1)], spacing=0, expand=True)
            rows.append(ft.Row([
                ft.Column([ft.Text(c.region, size=13, no_wrap=True, overflow=ft.TextOverflow.ELLIPSIS),
                           ft.Text(c.descriptor, size=11.5, color=MUTED, no_wrap=True, overflow=ft.TextOverflow.ELLIPSIS)],
                          spacing=0, width=230),
                ft.Row([left, ft.Container(width=1, height=22, bgcolor=ft.Colors.with_opacity(0.4, ft.Colors.ON_SURFACE)), right],
                       spacing=0, expand=True),
                ft.Text(f"{c.contribution:+.2f}", size=12, width=48, text_align=ft.TextAlign.RIGHT)],
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                tooltip=f"Valor escalado {c.z:+.2f} × peso {c.weight:+.3f}"))

        legend = ft.Row([ft.Container(width=12, height=12, bgcolor=NEGATIVE, border_radius=2),
                         ft.Text("disminuye la probabilidad", size=12, color=MUTED),
                         ft.Container(width=12),
                         ft.Container(width=12, height=12, bgcolor=POSITIVE, border_radius=2),
                         ft.Text("aumenta la probabilidad", size=12, color=MUTED)], spacing=6)
        controls = [ft.Text("Cuánto aportó cada variable al resultado (en la escala del logit). Los volúmenes y áreas son "
                            "proporciones del total segmentado, ajustadas por edad y sexo.", size=12.5, color=MUTED),
                    legend, *rows]
        total = len(prediction.contributions)
        if total > TOP_CONTRIBUTIONS:
            controls.append(ft.TextButton(f"Ver las {total} variables" if not self.show_all else "Ver solo las principales",
                                          on_click=self._toggle_all))
        title = "Variables que más influyeron" if total > TOP_CONTRIBUTIONS else "Variables del modelo"
        return section(title, ft.Icons.BAR_CHART, controls)

    def _toggle_all(self, e):
        self.show_all = not self.show_all
        self._update_prediction()

    def _other_models_section(self, current: PortableModel):
        """Resultado del mismo paciente con los demás modelos: el cálculo es inmediato."""
        rows = []
        for model in self.models:
            try:
                p = model.predict(self.features, self._age_value(), self._sex_value())
            except PredictionInputError:
                continue
            rows.append(ft.DataRow(
                selected=model.key == current.key,
                on_select_change=lambda e, key=model.key: self._select_model(key),
                cells=[ft.DataCell(ft.Text(model.display_name, size=13)),
                       ft.DataCell(ft.Text(f"{p.probability:.0%}", size=13)),
                       ft.DataCell(ft.Text(f"{p.threshold:.2f}", size=13)),
                       ft.DataCell(ft.Text("Conversión" if p.is_positive else "DCL estable", size=13,
                                           color=POSITIVE if p.is_positive else NEGATIVE))]))
        table = ft.DataTable(columns=[ft.DataColumn(ft.Text("Modelo")), ft.DataColumn(ft.Text("Probabilidad"), numeric=True),
                                      ft.DataColumn(ft.Text("Umbral"), numeric=True), ft.DataColumn(ft.Text("Clasificación"))],
                             rows=rows, heading_row_height=36, data_row_min_height=36, data_row_max_height=40,
                             column_spacing=24, show_checkbox_column=False)
        return section("Con todos los modelos", ft.Icons.TABLE_ROWS_OUTLINED, [table])

    def _select_model(self, key):
        self.model_dropdown.value = key
        self._on_model_change(None)
        self.update()

    # --- Acciones ------------------------------------------------------------------------

    def _open_viewer(self, e):
        model = self.model
        self.ctx.open_in_viewer(self.image_path, self.mask_path, model.variable_labels if model else None)

    def _safe_id(self):
        return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in (self.patient_id.value or "paciente"))

    async def _save_report(self, e):
        path = await self.picker.save_file(dialog_title="Guardar reporte", file_name=f"{self._safe_id()}_prediccion.json",
                                           initial_directory=str(self.ctx.output_dir), allowed_extensions=["json"])
        if not path:
            return
        write_prediction_report(path, self.patient_id.value, self._age_value(), self._sex_value(), self.model,
                                self.prediction, self.image_path, self.mask_path)
        self.ctx.notify(f"Reporte guardado en {path}")

    async def _save_csv(self, e):
        path = await self.picker.save_file(dialog_title="Guardar características", file_name=f"{self._safe_id()}.csv",
                                           initial_directory=str(self.ctx.output_dir), allowed_extensions=["csv"])
        if not path:
            return
        write_features_csv(path, self.patient_id.value or self._safe_id(), self.features)
        self.ctx.notify(f"Características guardadas en {path}")
