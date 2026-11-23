import flet as ft

from logic.extraction import ExtractionService
from logic.lut import get_lut
from logic.paths import LOGOS_DIR, MODELS_DIR
from logic.predictor import load_models
from ui.common import ACCENT, AppContext
from ui.extraction_view import ExtractionView
from ui.info_view import InfoView
from ui.prediction_view import PredictionView
from ui.volume_view import VolumeView

APP_VERSION = "1.0.0"


async def main(page: ft.Page):
    page.title = "NeuroRadiomics"
    page.theme_mode = ft.ThemeMode.DARK
    page.theme = ft.Theme(color_scheme_seed=ft.Colors.CYAN)
    page.dark_theme = ft.Theme(color_scheme_seed=ft.Colors.CYAN)
    page.window.width = 1360
    page.window.height = 860
    page.window.min_width = 1100
    page.window.min_height = 680
    page.padding = 0
    page.spacing = 0
    sidebar_width = 104

    # ── Servicios compartidos ───────────────────────────────────
    models, model_errors = load_models(MODELS_DIR)
    ctx = AppContext(page=page, lut=get_lut(), extraction=ExtractionService(), models=models)

    # ── Construir las vistas ────────────────────────────────────
    prediction_view = PredictionView(ctx)
    extraction_view = ExtractionView(ctx)
    volume_view = VolumeView(ctx)
    info_view = InfoView(ctx.lut)
    views = [prediction_view, extraction_view, volume_view, info_view]

    # Contenedor que aloja la vista activa
    content_area = ft.Container(content=prediction_view, expand=True)

    def navigate(index: int):
        nav_rail.selected_index = index
        content_area.content = views[index]
        page.update()
        if views[index] is volume_view:
            volume_view.on_resize()

    def show_regions(labels=None):
        if labels:
            info_view.set_available_labels(labels)
        else:
            info_view.set_force_all()
        navigate(3)

    def open_in_viewer(image, mask, highlight=None):
        navigate(2)
        volume_view.open(image, mask, highlight)

    ctx.show_regions = show_regions
    ctx.open_in_viewer = open_in_viewer

    # ── Navegación lateral ──────────────────────────────────────
    def on_nav_change(e):
        index = e.control.selected_index
        if views[index] is info_view:
            info_view.set_force_all()
        navigate(index)

    def toggle_theme(e):
        dark = page.theme_mode == ft.ThemeMode.DARK
        page.theme_mode = ft.ThemeMode.LIGHT if dark else ft.ThemeMode.DARK
        theme_button.icon = ft.Icons.DARK_MODE_OUTLINED if dark else ft.Icons.LIGHT_MODE_OUTLINED
        page.update()

    theme_button = ft.IconButton(ft.Icons.LIGHT_MODE_OUTLINED, tooltip="Cambiar tema claro/oscuro", on_click=toggle_theme)

    nav_rail = ft.NavigationRail(
        selected_index=0,
        label_type=ft.NavigationRailLabelType.ALL,
        min_width=sidebar_width,
        group_alignment=-0.9,
        on_change=on_nav_change,
        bgcolor=ft.Colors.TRANSPARENT,
        leading=ft.Container(
            ft.Column([ft.Icon(ft.Icons.PSYCHOLOGY, size=34, color=ACCENT),
                       ft.Text("NeuroRadiomics", size=11, weight=ft.FontWeight.BOLD, color=ACCENT)],
                      horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=2),
            padding=ft.Padding.only(top=12, bottom=16)),
        trailing=ft.Column([theme_button, ft.Text(f"v{APP_VERSION}", size=10, color=ft.Colors.ON_SURFACE_VARIANT)],
                           horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=4),
        destinations=[
            ft.NavigationRailDestination(icon=ft.Icons.INSIGHTS_OUTLINED, selected_icon=ft.Icons.INSIGHTS, label="Predicción"),
            ft.NavigationRailDestination(icon=ft.Icons.ANALYTICS_OUTLINED, selected_icon=ft.Icons.ANALYTICS, label="Extracción"),
            ft.NavigationRailDestination(icon=ft.Icons.VIEW_IN_AR_OUTLINED, selected_icon=ft.Icons.VIEW_IN_AR, label="Visualizador"),
            ft.NavigationRailDestination(icon=ft.Icons.MENU_BOOK_OUTLINED, selected_icon=ft.Icons.MENU_BOOK, label="Regiones"),
        ],
    )

    # ── Logos institucionales al pie de la barra lateral ────────
    logos = ft.Container(
        ft.Column([
            ft.Image(src=(LOGOS_DIR / "uaz.png").read_bytes(), width=78, tooltip="Universidad Autónoma de Zacatecas"),
            # El logo de la UDEM es de un solo color sobre fondo transparente: se tiñe con el color
            # del texto del tema para que se vea tanto en el tema oscuro como en el claro
            ft.Image(src=(LOGOS_DIR / "udem.png").read_bytes(), width=72, tooltip="Universidad de Monterrey",
                     color=ft.Colors.ON_SURFACE, color_blend_mode=ft.BlendMode.SRC_IN),
        ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=12),
        padding=ft.Padding.only(bottom=16, top=8))

    sidebar = ft.Container(
        content=ft.Column([ft.Container(nav_rail, expand=True), logos], spacing=0,
                          horizontal_alignment=ft.CrossAxisAlignment.CENTER),
        width=sidebar_width,
        bgcolor=ft.Colors.with_opacity(0.04, ft.Colors.ON_SURFACE),
        border=ft.Border(right=ft.BorderSide(1, ft.Colors.with_opacity(0.1, ft.Colors.ON_SURFACE))),
    )

    # ── Eventos globales: se envían a la vista activa ───────────
    def on_keyboard(e: ft.KeyboardEvent):
        if content_area.content is volume_view:
            volume_view.on_keyboard(e)

    def on_resize(e):
        if content_area.content is volume_view:
            volume_view.on_resize()

    page.on_keyboard_event = on_keyboard
    page.on_resize = on_resize

    # ── Layout principal ────────────────────────────────────────
    page.add(ft.Row([sidebar, content_area], expand=True, spacing=0))

    # ── Avisos de configuración ─────────────────────────────────
    if not ctx.extraction.available:
        ctx.show_error("Backend de extracción no encontrado",
                       "No se encontró el Python 3.9 con PyRadiomics, así que no se podrán extraer características "
                       "(sí se pueden cargar CSV ya extraídos y usar el visualizador).\n\n"
                       "Cree el entorno .venv-backend en la carpeta del repositorio como indica el README, o defina la "
                       "variable de entorno NEURORADIOMICS_BACKEND_PYTHON con la ruta de ese intérprete.")
    if model_errors:
        ctx.notify("Algunos modelos no se pudieron cargar: " + "; ".join(model_errors), error=True)


if __name__ == "__main__":
    ft.run(main)
