import asyncio
from pathlib import Path

import flet as ft

from logic.volume_model import VolumeModel
from ui.common import MUTED, AppContext, file_chip, view_header
from ui.plane_view import PlaneView

# Espacio que ocupan la barra lateral, los márgenes y los controles alrededor de los planos
SIDEBAR_WIDTH = 104
CHROME_WIDTH = 48 + 2 * 12 + 3 * 16
CHROME_HEIGHT = 330


class VolumeView(ft.Container):
    """
    Controlador principal. Une el Modelo (datos) con las Vistas (Componentes UI).
    """
    def __init__(self, ctx: AppContext):
        super().__init__()
        self.ctx = ctx
        self.expand = True
        self.padding = ft.Padding.only(left=24, right=24, top=20, bottom=16)

        # ── Instancias ──────────────────────────────────────────────
        self.model = VolumeModel()
        self.focused_plane = "axial"
        self.image_path: Path | None = None
        self.mask_path: Path | None = None
        self.model_labels: set[int] | None = None
        self.picker = ft.FilePicker()

        # ── Instanciar Vistas (Componentes) ─────────────────────────
        self.views = {
            "axial": PlaneView("axial", "Axial", ft.Colors.CYAN_700, self.on_plane_interaction, self.on_slice_change),
            "coronal": PlaneView("coronal", "Coronal", ft.Colors.TEAL_700, self.on_plane_interaction, self.on_slice_change),
            "sagittal": PlaneView("sagittal", "Sagital", ft.Colors.INDIGO_700, self.on_plane_interaction, self.on_slice_change),
        }

        # ── Interfaz Global ─────────────────────────────────────────
        self.image_label = file_chip(None, "Ningún volumen cargado")
        self.mask_label = file_chip(None, "Sin máscara")
        self.info_text = ft.Text("", size=12.5, color=MUTED)
        self.loading = ft.ProgressRing(visible=False, width=18, height=18, stroke_width=2)

        self.show_mask = ft.Switch(label="Segmentación", value=True, on_change=self._on_overlay_change, disabled=True)
        self.opacity_slider = ft.Slider(min=0, max=1, value=self.model.mask_opacity, width=160, disabled=True,
                                 on_change=self._on_overlay_change, tooltip="Opacidad de la segmentación")
        self.only_model = ft.Switch(label="Solo regiones del modelo", value=False, visible=False,
                                    on_change=self._on_overlay_change)
        self.region_swatch = ft.Container(width=14, height=14, border_radius=3, visible=False)
        self.region_text = ft.Text("", size=13)

        toolbar = ft.Row([
            ft.FilledTonalButton("Volumen", icon=ft.Icons.FOLDER_OPEN, on_click=self._pick_volume,
                                 tooltip="Imagen de resonancia (.mgz, .nii, .nii.gz)"),
            ft.Column([self.image_label, self.info_text], spacing=0, width=260),
            ft.OutlinedButton("Máscara", icon=ft.Icons.LAYERS_OUTLINED, on_click=self._pick_mask,
                              tooltip="Segmentación de FreeSurfer (aparc+aseg) con las mismas dimensiones"),
            ft.Container(self.mask_label, width=180),
            self.loading,
        ], spacing=12, vertical_alignment=ft.CrossAxisAlignment.CENTER, wrap=True)

        overlay = ft.Row([self.show_mask, ft.Icon(ft.Icons.OPACITY, size=16, color=MUTED), self.opacity_slider, self.only_model,
                          ft.Container(expand=True),
                          ft.Row([self.region_swatch, self.region_text], spacing=8)],
                         vertical_alignment=ft.CrossAxisAlignment.CENTER)

        self.planes_row = ft.Row([self.views["axial"], self.views["coronal"], self.views["sagittal"]],
                                 alignment=ft.MainAxisAlignment.CENTER, spacing=12)
        help_text = ft.Text("Clic o arrastre para mover la cruz · rueda del ratón o barra para cambiar de corte · "
                            "flechas para moverse en el plano activo · RePág/AvPág para cambiar de corte",
                            size=11.5, color=MUTED)

        self.content = ft.Column([
            view_header("Visualizador", "Planos axial, coronal y sagital con la segmentación de FreeSurfer superpuesta"),
            toolbar, overlay, self.planes_row, help_text,
        ], expand=True, scroll=ft.ScrollMode.AUTO, spacing=12)

    # ── Callbacks del Controlador ───────────────────────────────
    def on_plane_interaction(self, plane_clicked: str, col: int, row: int):
        if self.model.data is None: return
        self.focused_plane = plane_clicked
        model = self.model

        # Guardar estado anterior para saber si cambiamos de slice
        old_coords = model.coords.copy()

        # Obtenemos la configuración dinámica del plano para asegurar
        # que la navegación SIEMPRE concuerde con la rotación y el flip visual.
        cfg = model.view_config[plane_clicked]
        x_axis = cfg["x_axis"]
        y_axis = cfg["y_axis"]

        # Calcular coordenadas para la matriz real teniendo en cuenta los flips
        # Eje X de la imagen (movimiento horizontal del mouse) -> columnas
        col_val = model.shape[model.axis_map[x_axis]] - 1 - col if cfg["flip_x"] else col
        # Eje Y de la imagen (movimiento vertical del mouse) -> filas
        row_val = model.shape[model.axis_map[y_axis]] - 1 - row if cfg["flip_y"] else row

        model.set_coord(x_axis, col_val)
        model.set_coord(y_axis, row_val)

        # Actualizar UI inteligentemente:
        # Solo re-renderizamos las imagenes si el usuario cambió de corte
        for p_name, view in self.views.items():
            if model.coords[p_name] != old_coords[p_name]:
                self._refresh_plane_image(p_name)

            # Las cruces sí se actualizan siempre para que el cursor fluya
            self._refresh_plane_crosshairs(p_name)

            # Actualizar bordes
            view.set_focus(p_name == self.focused_plane)
        self._refresh_region_label()

    def on_slice_change(self, plane: str, value: int):
        """Cambio de corte desde la barra de desplazamiento o la rueda del ratón."""
        if self.model.data is None: return
        old = self.model.coords[plane]
        self.model.set_coord(plane, value)
        if self.model.coords[plane] == old: return
        self.focused_plane = plane
        self._refresh_plane_image(plane)
        # El corte de este plano mueve la cruz en los otros dos
        for p_name, view in self.views.items():
            self._refresh_plane_crosshairs(p_name)
            view.set_focus(p_name == plane)
        self._refresh_region_label()

    # ── Funciones de Actualización de Vistas ────────────────────
    def _refresh_plane_image(self, plane: str):
        png, w, h = self.model.get_slice_png(plane)
        if png:
            current = self.model.coords[plane]
            maximum = self.model.shape[self.model.axis_map[plane]]
            self.views[plane].update_image(png, w, h, current, maximum)

    def _refresh_plane_crosshairs(self, plane: str):
        model = self.model
        if model.data is None: return

        # Calcular de forma perfectamente coordinada dónde está la cruz
        # apoyándonos en la configuración orientativa dictada en el modelo.
        cfg = model.view_config[plane]
        x_axis = cfg["x_axis"]
        y_axis = cfg["y_axis"]

        # Coordenadas crudas desde el modelo
        col_val = model.coords[x_axis]
        row_val = model.coords[y_axis]

        # Invertir para adaptar a coordenada de vista de acuerdo a los flips!
        col = model.shape[model.axis_map[x_axis]] - 1 - col_val if cfg["flip_x"] else col_val
        row = model.shape[model.axis_map[y_axis]] - 1 - row_val if cfg["flip_y"] else row_val

        self.views[plane].update_crosshairs(col, row)

    def _refresh_all(self):
        for p in self.views.keys():
            self._refresh_plane_image(p)
            self._refresh_plane_crosshairs(p)
            self.views[p].set_focus(p == self.focused_plane)
        self._refresh_region_label()

    def _refresh_region_label(self):
        label = self.model.label_at_cursor()
        if label is None:
            self.region_text.value = ""
            self.region_swatch.visible = False
        elif label == 0:
            self.region_text.value = "Fondo"
            self.region_swatch.visible = False
        else:
            entry = self.ctx.lut.entries.get(label)
            self.region_text.value = f"{label} · {self.ctx.lut.name(label)}"
            self.region_swatch.visible = entry is not None
            if entry:
                self.region_swatch.bgcolor = entry.hex_color
        self.region_text.update()
        self.region_swatch.update()

    # ── Navegación con Teclado ──────────────────────────────────
    def on_keyboard(self, e: ft.KeyboardEvent):
        model = self.model
        if model.data is None or self.focused_plane is None: return

        if e.key in ("Page Up", "PageUp", "Page Down", "PageDown"):
            step = 1 if e.key in ("Page Up", "PageUp") else -1
            self.on_slice_change(self.focused_plane, model.coords[self.focused_plane] + step)
            return

        # Obtenemos la cruz actual en la vista focado usando la configuración dinámica
        cfg = model.view_config[self.focused_plane]
        x_axis = cfg["x_axis"]
        y_axis = cfg["y_axis"]

        col_val = model.coords[x_axis]
        row_val = model.coords[y_axis]

        col = model.shape[model.axis_map[x_axis]] - 1 - col_val if cfg["flip_x"] else col_val
        row = model.shape[model.axis_map[y_axis]] - 1 - row_val if cfg["flip_y"] else row_val

        # Modificamos según la flecha
        if e.key in ["Arrow Up", "ArrowUp", "Up"]: row -= 1
        elif e.key in["Arrow Down", "ArrowDown", "Down"]: row += 1
        elif e.key in["Arrow Left", "ArrowLeft", "Left"]: col -= 1
        elif e.key in["Arrow Right", "ArrowRight", "Right"]: col += 1
        else: return

        # Simulamos que el usuario hizo click en esa nueva posición
        self.on_plane_interaction(self.focused_plane, col, row)

    # ── Tamaño ──────────────────────────────────────────────────
    def on_resize(self):
        page = self.page
        if not page or not page.width or not page.height:
            return
        by_width = (page.width - SIDEBAR_WIDTH - CHROME_WIDTH) / 3
        by_height = page.height - CHROME_HEIGHT
        size = int(max(200, min(by_width, by_height, 640)))
        if size != self.views["axial"].size:
            for view in self.views.values():
                view.set_size(size)
            self.update()

    def did_mount(self):
        self.on_resize()

    # ── Carga de archivos ───────────────────────────────────────
    async def _pick_volume(self, e):
        files = await self.picker.pick_files(dialog_title="Abrir volumen", allowed_extensions=["mgz", "gz", "nii"])
        if files:
            await self.load(Path(files[0].path), None, None)

    async def _pick_mask(self, e):
        if self.model.data is None:
            self.ctx.notify("Cargue primero el volumen.")
            return
        files = await self.picker.pick_files(dialog_title="Abrir máscara", allowed_extensions=["mgz", "gz", "nii"])
        if files:
            await self.load(None, Path(files[0].path), self.model_labels)

    def open(self, image, mask, highlight=None):
        """Abre un volumen y su máscara desde otra vista (se llama al navegar al visualizador)."""
        self.page.run_task(self.load, Path(image) if image else None, Path(mask) if mask else None, highlight)

    async def load(self, image: Path | None, mask: Path | None, highlight: set[int] | None):
        self.loading.visible = True
        self.update()
        try:
            if image is not None:
                zooms = await asyncio.to_thread(self.model.load_volume, str(image))
                self.image_path = image
                self.image_label.value, self.image_label.color, self.image_label.tooltip = image.name, None, str(image)
                self.info_text.value = f"{self.model.shape[0]}×{self.model.shape[1]}×{self.model.shape[2]} · vóxel " + \
                                       "×".join(f"{v:.2f}" for v in zooms) + " mm"
                self.mask_path = None
                self.mask_label.value, self.mask_label.color, self.mask_label.tooltip = "Sin máscara", MUTED, None
            if mask is not None:
                await asyncio.to_thread(self.model.load_mask, str(mask))
                self.mask_path = mask
                self.mask_label.value, self.mask_label.color, self.mask_label.tooltip = mask.name, None, str(mask)
            self.model_labels = highlight
        except Exception as ex:
            self.ctx.show_error("No se pudo abrir el archivo", str(ex))
        finally:
            self.loading.visible = False

        has_mask = self.model.mask is not None
        self.show_mask.disabled = self.opacity_slider.disabled = not has_mask
        self.only_model.visible = has_mask and bool(self.model_labels)
        self.only_model.value = self.only_model.visible
        self._apply_overlay_settings()
        self._refresh_all()
        self.update()

    # ── Superposición ───────────────────────────────────────────
    def _apply_overlay_settings(self):
        self.model.show_mask = self.show_mask.value
        self.model.mask_opacity = self.opacity_slider.value
        self.model.highlight_labels = self.model_labels if self.only_model.visible and self.only_model.value else None

    def _on_overlay_change(self, e):
        self._apply_overlay_settings()
        for plane in self.views:
            self._refresh_plane_image(plane)
