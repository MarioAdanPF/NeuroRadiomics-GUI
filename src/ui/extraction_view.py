"""
Vista de extracción: características de PyRadiomics de un sujeto o de una carpeta con
muchos sujetos (modo por lotes). Cada sujeto se guarda como <ID>.csv, con una fila por región:
el mismo formato con el que se entrenaron los modelos.
"""
import asyncio
import os
import time
from pathlib import Path

import flet as ft

from logic.extraction import ExtractionError
from logic.helpers import identify_high_correlations
from logic.results import write_features_csv
from logic.subjects import SubjectFiles, find_subject_files, scan_directory
from logic.volumes import mask_labels
from ui.common import MUTED, AppContext, file_chip, section, status_row, view_header
from ui.correlations_dialog import CorrelationsDialog


class ExtractionView(ft.Container):
    def __init__(self, ctx: AppContext):
        super().__init__()
        self.ctx = ctx
        self.expand = True
        self.padding = ft.Padding.only(left=24, right=24, top=20, bottom=16)

        self.image_path: Path | None = None
        self.mask_path: Path | None = None
        self.subject_id = ""
        self.available_labels: list[int] | None = None
        self.batch_root: Path | None = None
        self.batch_subjects: list[SubjectFiles] = []
        self.output_dir = ctx.output_dir
        self.features = None
        self.vars_to_remove: set[str] = set()
        self.task = None

        self.picker = ft.FilePicker()

        # --- Modo ---
        self.mode = ft.SegmentedButton(
            segments=[ft.Segment("single", label="Un sujeto", icon=ft.Icons.PERSON_OUTLINE),
                      ft.Segment("batch", label="Lote", icon=ft.Icons.GROUPS_OUTLINED)],
            selected=["single"], show_selected_icon=False, on_change=self._on_mode_change)

        # --- Entrada: un sujeto ---
        self.single_status = ft.Column(spacing=6)
        self.labels_button = ft.TextButton("Ver regiones de la máscara", icon=ft.Icons.INFO_OUTLINE, visible=False,
                                           on_click=lambda e: self.ctx.show_regions(self.available_labels))
        self.single_input = ft.Column([
            ft.Row([ft.FilledTonalButton("Carpeta del sujeto", icon=ft.Icons.FOLDER_OPEN, on_click=self._pick_folder),
                    ft.OutlinedButton("Imagen", icon=ft.Icons.IMAGE_OUTLINED, on_click=self._pick_image),
                    ft.OutlinedButton("Máscara", icon=ft.Icons.LAYERS_OUTLINED, on_click=self._pick_mask)], wrap=True),
            self.single_status, self.labels_button], spacing=10)

        # --- Entrada: lote ---
        self.batch_status = ft.Column(spacing=6)
        self.skip_existing = ft.Switch(label="Omitir sujetos que ya tienen su CSV", value=True)
        cores = os.cpu_count() or 2
        self.workers = ft.Slider(min=1, max=max(2, min(8, cores)), divisions=max(1, min(8, cores) - 1),
                                 value=max(1, min(4, cores // 2)), label="{value}", expand=True)
        self.batch_input = ft.Column([
            ft.Text("Una carpeta con una subcarpeta por sujeto; en cada una se buscan brain y aparc+aseg (o en su carpeta mri).",
                    size=12.5, color=MUTED),
            ft.FilledTonalButton("Carpeta con sujetos", icon=ft.Icons.FOLDER_COPY_OUTLINED, on_click=self._pick_batch_root),
            self.batch_status, self.skip_existing,
            ft.Row([ft.Text("Procesos en paralelo", size=13), self.workers])], spacing=10, visible=False)

        # --- Regiones ---
        self.custom_labels = ft.TextField(hint_text="Ej.: 17, 53, 1006", dense=True, disabled=True,
                                          input_filter=ft.InputFilter(r"[0-9, ]"))
        n_model_regions = len({label for m in ctx.models for label in m.regions})
        self.region_mode = ft.RadioGroup(
            value="all", on_change=self._on_region_mode_change,
            content=ft.Column([
                ft.Radio(value="all", label="Todas las regiones de la máscara"),
                ft.Radio(value="model", label=f"Las que usan los modelos ({n_model_regions})", disabled=n_model_regions == 0),
                ft.Radio(value="custom", label="Lista personalizada")], spacing=0))

        # --- Características ---
        self.shape_only = ft.Switch(label="Solo descriptores de forma", value=True)

        # --- Salida ---
        self.output_text = file_chip(self.output_dir)

        self.run_button = ft.FilledButton("Iniciar extracción", icon=ft.Icons.PLAY_ARROW, on_click=self._on_run, height=44)
        self.cancel_button = ft.OutlinedButton("Cancelar", icon=ft.Icons.STOP, on_click=self._on_cancel, visible=False, height=44)
        self.progress = ft.ProgressBar(value=0, visible=False, border_radius=4)
        self.status = ft.Text("", size=12.5, color=MUTED)

        left = ft.Column([
            section("Entrada", ft.Icons.INPUT, [self.mode, self.single_input, self.batch_input]),
            section("Regiones", ft.Icons.GRID_VIEW, [self.region_mode, self.custom_labels]),
            section("Características", ft.Icons.TUNE, [
                self.shape_only,
                ft.Text("Los modelos solo usan la forma. Con todas las familias (intensidad y textura) la extracción "
                        "tarda más y el CSV ocupa unas ocho veces más.", size=12.5, color=MUTED)]),
            section("Carpeta de salida", ft.Icons.SAVE_OUTLINED, [
                ft.Row([ft.Icon(ft.Icons.FOLDER_OUTLINED, size=16, color=MUTED), ft.Container(self.output_text, expand=True),
                        ft.TextButton("Cambiar", on_click=self._pick_output)]),
                ft.Text("Se guarda un CSV por sujeto con el nombre <ID>.csv.", size=12.5, color=MUTED)]),
            ft.Row([self.run_button, self.cancel_button]), self.progress, self.status,
        ], spacing=14, scroll=ft.ScrollMode.AUTO, width=430)

        self.result_area = ft.Column(spacing=12, expand=True, scroll=ft.ScrollMode.AUTO)
        self.content = ft.Column([
            view_header("Extracción", "Características radiómicas con PyRadiomics de un sujeto o de una carpeta de sujetos"),
            ft.Row([left, ft.VerticalDivider(width=1), ft.Container(self.result_area, expand=True)],
                   expand=True, spacing=20, vertical_alignment=ft.CrossAxisAlignment.START)], expand=True, spacing=14)

        self._render_inputs()
        self._render_empty()

    # --- Entrada -------------------------------------------------------------------------

    @property
    def is_batch(self) -> bool:
        return self.mode.selected == ["batch"]

    def _on_mode_change(self, e):
        self.single_input.visible = not self.is_batch
        self.batch_input.visible = self.is_batch
        self._render_empty()
        self._refresh_controls()

    def _on_region_mode_change(self, e):
        self.custom_labels.disabled = self.region_mode.value != "custom"
        self._refresh_controls()

    async def _pick_folder(self, e):
        path = await self.picker.get_directory_path(dialog_title="Carpeta del sujeto")
        if not path:
            return
        subject = find_subject_files(path)
        if subject is None:
            self.ctx.show_error("No se encontró el sujeto", f"En {path} no están brain y aparc+aseg (.mgz, .nii o .nii.gz).")
            return
        self.image_path, self.mask_path, self.subject_id = subject.image, subject.mask, subject.id
        await self._load_labels()

    async def _pick_image(self, e):
        files = await self.picker.pick_files(dialog_title="Imagen", allowed_extensions=["mgz", "nii", "gz"])
        if files:
            self.image_path = Path(files[0].path)
            self.subject_id = self.subject_id or self.image_path.parent.name
            self._render_inputs()

    async def _pick_mask(self, e):
        files = await self.picker.pick_files(dialog_title="Máscara", allowed_extensions=["mgz", "nii", "gz"])
        if files:
            self.mask_path = Path(files[0].path)
            self.subject_id = self.subject_id or self.mask_path.parent.name
            await self._load_labels()

    async def _load_labels(self):
        self.available_labels = None
        self._render_inputs()
        self.update()
        try:
            self.available_labels = await asyncio.to_thread(mask_labels, self.mask_path)
        except Exception as error:
            self.ctx.show_error("No se pudo leer la máscara", str(error))
        self._render_inputs()
        self.update()

    async def _pick_batch_root(self, e):
        path = await self.picker.get_directory_path(dialog_title="Carpeta con sujetos")
        if not path:
            return
        self.batch_root = Path(path)
        self.batch_subjects = await asyncio.to_thread(scan_directory, self.batch_root)
        self._render_inputs()
        self.update()

    async def _pick_output(self, e):
        path = await self.picker.get_directory_path(dialog_title="Carpeta de salida", initial_directory=str(self.output_dir))
        if path:
            self.output_dir = Path(path)
            self.output_text.value = self.output_dir.name
            self.output_text.tooltip = str(self.output_dir)
            self.update()

    def _render_inputs(self):
        rows = [ft.Row([ft.Text("Imagen", size=13, width=70, color=MUTED), file_chip(self.image_path)]),
                ft.Row([ft.Text("Máscara", size=13, width=70, color=MUTED), file_chip(self.mask_path)])]
        if self.mask_path is not None:
            if self.available_labels is None:
                rows.append(status_row(None, "Leyendo las regiones de la máscara..."))
            else:
                rows.append(status_row(True, f"{len(self.available_labels)} regiones en la máscara"))
        self.single_status.controls = rows
        self.labels_button.visible = bool(self.available_labels)

        if self.batch_root is None:
            self.batch_status.controls = []
        else:
            found = len(self.batch_subjects)
            self.batch_status.controls = [
                ft.Row([ft.Text("Carpeta", size=13, width=70, color=MUTED), file_chip(self.batch_root)]),
                status_row(found > 0, f"{found} sujetos con brain y aparc+aseg")]
        self._refresh_controls()

    def _labels_to_process(self, available=None) -> list[int] | None:
        """Regiones a extraer según la opción elegida; None significa todas las de la máscara."""
        if self.region_mode.value == "model":
            return sorted({label for m in self.ctx.models for label in m.regions})
        if self.region_mode.value == "custom":
            return sorted({int(part) for part in self.custom_labels.value.replace(" ", ",").split(",") if part.strip().isdigit()})
        return None

    def _refresh_controls(self):
        running = self.task is not None and not self.task.done()
        if self.is_batch:
            ready = bool(self.batch_subjects)
        else:
            ready = self.image_path is not None and self.mask_path is not None
        if self.region_mode.value == "custom" and not self._labels_to_process():
            ready = False
        self.run_button.disabled = not ready
        self.run_button.visible = not running
        self.cancel_button.visible = running
        self.progress.visible = running

    # --- Ejecución -----------------------------------------------------------------------

    def _on_run(self, e):
        if not self.ctx.extraction.available:
            self.ctx.show_error("Backend no disponible",
                                "No se encontró el Python 3.9 con PyRadiomics. Cree .venv-backend como indica el README "
                                "o defina la variable de entorno NEURORADIOMICS_BACKEND_PYTHON.")
            return
        self.task = self.page.run_task(self._run_batch if self.is_batch else self._run_single)
        self._refresh_controls()
        self.update()

    def _on_cancel(self, e):
        if self.task:
            self.task.cancel()

    def _finish(self, message):
        self.task = None
        self.status.value = message
        self._refresh_controls()
        self.update()

    async def _run_single(self):
        start = time.monotonic()
        self.progress.value = None
        self.status.value = "Preparando la imagen..."
        self.result_area.controls = []
        self.update()

        def on_progress(p):
            self.progress.value = p.done / p.total if p.total else None
            self.status.value = f"Región {p.done} de {p.total}" + (f": {self.ctx.lut.name(p.label)}" if p.label else "")
            self.update()

        try:
            result = await self.ctx.extraction.extract(self.image_path, self.mask_path, labels=self._labels_to_process(),
                                                       shape_only=self.shape_only.value, on_progress=on_progress)
        except asyncio.CancelledError:
            self._finish("Extracción cancelada.")
            return
        except (ExtractionError, OSError, ValueError) as error:
            self._finish("La extracción falló.")
            self.ctx.show_error("Error en la extracción", str(error))
            return

        self.features = result.features
        self.vars_to_remove = set()
        message = f"{len(result.features)} regiones en {time.monotonic() - start:.0f} s."
        if result.features:
            path = write_features_csv(self.output_dir / f"{self.subject_id or 'sujeto'}.csv", self.subject_id, result.features)
            message += f" Guardado en {path}."
        if result.errors:
            message += f" {len(result.errors)} regiones con error o ausentes."
        self._render_single_result(result.errors)
        self._finish(message)

    async def _run_batch(self):
        subjects = self.batch_subjects
        if self.skip_existing.value:
            subjects = [s for s in subjects if not (self.output_dir / f"{s.id}.csv").exists()]
        total, done, failed = len(subjects), 0, 0
        log = ft.ListView(spacing=2, expand=True, auto_scroll=True)
        self.result_area.controls = [section("Registro", ft.Icons.LIST_ALT, [ft.Container(log, height=520)])]
        skipped = len(self.batch_subjects) - total
        if skipped:
            log.controls.append(status_row(None, f"{skipped} sujetos omitidos porque ya tienen su CSV."))
        self.progress.value = 0
        self.status.value = f"0 de {total} sujetos"
        self.update()

        semaphore = asyncio.Semaphore(int(self.workers.value))
        labels = self._labels_to_process()

        async def process(subject: SubjectFiles):
            nonlocal done, failed
            async with semaphore:
                start = time.monotonic()
                try:
                    # El paralelismo del lote es entre sujetos: un proceso por sujeto
                    result = await self.ctx.extraction.extract(subject.image, subject.mask, labels=labels,
                                                               shape_only=self.shape_only.value, workers=1)
                    write_features_csv(self.output_dir / f"{subject.id}.csv", subject.id, result.features)
                    text = f"{subject.id}: {len(result.features)} regiones en {time.monotonic() - start:.0f} s"
                    if result.errors:
                        text += f" ({len(result.errors)} con error)"
                    log.controls.append(status_row(True, text))
                except (ExtractionError, OSError, ValueError) as error:
                    failed += 1
                    log.controls.append(status_row(False, f"{subject.id}: {error}"))
                done += 1
                self.progress.value = done / total
                self.status.value = f"{done} de {total} sujetos"
                self.update()

        start = time.monotonic()
        try:
            await asyncio.gather(*(process(s) for s in subjects))
        except asyncio.CancelledError:
            self._finish(f"Cancelado: {done} de {total} sujetos procesados.")
            return
        minutes = (time.monotonic() - start) / 60
        self._finish(f"Lote terminado: {done - failed} de {total} sujetos en {minutes:.1f} min. CSV en {self.output_dir}.")

    # --- Resultados ----------------------------------------------------------------------

    def _render_empty(self):
        text = ("Elija la carpeta con los sujetos y presione Iniciar extracción. Cada sujeto se procesa en su propio "
                "proceso; con varios procesos en paralelo el lote termina antes.") if self.is_batch else \
               ("Elija la carpeta del sujeto o su imagen y su máscara, y presione Iniciar extracción. "
                "Las características de cada región aparecerán aquí.")
        self.result_area.controls = [ft.Container(
            ft.Column([ft.Icon(ft.Icons.ANALYTICS_OUTLINED, size=48, color=MUTED),
                       ft.Text("Sin resultados todavía", size=17, weight=ft.FontWeight.W_600),
                       ft.Text(text, size=13, color=MUTED, text_align=ft.TextAlign.CENTER)],
                      horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=8),
            padding=40, alignment=ft.Alignment.CENTER)]

    def _render_single_result(self, errors):
        if not self.features:
            self.result_area.controls = [status_row(False, "No se extrajo ninguna región.")]
            return
        labels = sorted(self.features)
        self.region_selector = ft.Dropdown(
            label="Región", dense=True, width=360, value=str(labels[0]), enable_filter=True, editable=True,
            options=[ft.DropdownOption(key=str(l), text=f"{l} · {self.ctx.lut.name(l)}") for l in labels],
            on_select=lambda e: self._render_feature_table())
        self.feature_table = ft.Column()
        actions = ft.Row([
            ft.OutlinedButton("Identificar correlaciones", icon=ft.Icons.HUB_OUTLINED, on_click=self._on_correlations,
                              disabled=len(labels) < 2),
            ft.OutlinedButton("Ver en el visualizador", icon=ft.Icons.VIEW_IN_AR,
                              on_click=lambda e: self.ctx.open_in_viewer(self.image_path, self.mask_path, set(labels)))], wrap=True)
        controls = [ft.Row([self.region_selector, actions], wrap=True, spacing=16)]
        if errors:
            names = ", ".join(str(l) for l in sorted(errors)[:10])
            controls.append(status_row(False, f"Regiones con error o ausentes: {names}{'...' if len(errors) > 10 else ''}"))
        self.result_area.controls = [section(f"Resultados de {self.subject_id or 'sujeto'}", ft.Icons.TABLE_CHART_OUTLINED,
                                             [*controls, self.feature_table])]
        self._render_feature_table()

    def _render_feature_table(self):
        label = int(self.region_selector.value)
        values = self.features.get(label, {})
        rows = [ft.DataRow(cells=[ft.DataCell(ft.Text(name.removeprefix("original_"), size=13, selectable=True)),
                                  ft.DataCell(ft.Text(f"{value:.6g}", size=13, selectable=True))])
                for name, value in values.items() if name not in self.vars_to_remove]
        hidden = len(values) - len(rows)
        self.feature_table.controls = [
            ft.Text(f"{len(rows)} características" + (f" ({hidden} ocultas por redundantes)" if hidden else ""),
                    size=12.5, color=MUTED),
            ft.DataTable(columns=[ft.DataColumn(ft.Text("Característica")), ft.DataColumn(ft.Text("Valor"), numeric=True)],
                         rows=rows, heading_row_height=36, data_row_min_height=32, data_row_max_height=34)]
        if self.page:
            self.update()

    def _on_correlations(self, e):
        pairs = identify_high_correlations([self.features[l] for l in sorted(self.features)])

        def on_remove(new_vars):
            self.vars_to_remove = set(new_vars)
            self._render_feature_table()

        self.page.show_dialog(CorrelationsDialog(self.page, pairs, self.vars_to_remove, on_remove))
