"""Estilo, componentes reutilizables y contexto compartido entre las vistas."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import flet as ft

from logic.extraction import ExtractionService
from logic.lut import FreeSurferLUT
from logic.paths import DEFAULT_OUTPUT_DIR
from logic.predictor import PortableModel

ACCENT = ft.Colors.CYAN_400
POSITIVE = ft.Colors.DEEP_ORANGE_400   # aumenta la probabilidad de conversión
NEGATIVE = ft.Colors.LIGHT_BLUE_400    # la disminuye
MUTED = ft.Colors.ON_SURFACE_VARIANT
DISCLAIMER = ("Herramienta de investigación. No es un dispositivo médico ni sustituye el diagnóstico clínico. "
              "Los modelos se entrenaron con imágenes de ADNI de pacientes con deterioro cognitivo leve (DCL) "
              "segmentadas con FreeSurfer; sus resultados solo tienen sentido para ese tipo de pacientes.")


@dataclass
class AppContext:
    """Servicios y acciones que comparten las vistas."""
    page: ft.Page
    lut: FreeSurferLUT
    extraction: ExtractionService
    models: list[PortableModel] = field(default_factory=list)
    output_dir: Path = DEFAULT_OUTPUT_DIR
    open_in_viewer: Callable[..., None] = lambda *args, **kwargs: None
    show_regions: Callable[[list[int] | None], None] = lambda labels: None

    def notify(self, message: str, error: bool = False):
        self.page.show_dialog(ft.SnackBar(
            ft.Text(message, color=ft.Colors.ON_ERROR_CONTAINER if error else None),
            bgcolor=ft.Colors.ERROR_CONTAINER if error else None,
            show_close_icon=True, duration=6000 if error else 3500))

    def show_error(self, title: str, details: str):
        def close(e):
            self.page.pop_dialog()
        self.page.show_dialog(ft.AlertDialog(
            icon=ft.Icon(ft.Icons.ERROR_OUTLINE, color=ft.Colors.ERROR),
            title=ft.Text(title),
            content=ft.Container(ft.Text(details, selectable=True, size=13), width=520),
            actions=[ft.TextButton("Cerrar", on_click=close)]))


def view_header(title: str, subtitle: str, trailing: ft.Control | None = None) -> ft.Control:
    text = ft.Column([ft.Text(title, size=24, weight=ft.FontWeight.BOLD),
                      ft.Text(subtitle, size=13, color=MUTED)], spacing=2, expand=True)
    return ft.Row([text, trailing] if trailing else [text], vertical_alignment=ft.CrossAxisAlignment.START)


def section(title: str, icon, controls: list[ft.Control], step: int | None = None, expand=False) -> ft.Container:
    """Tarjeta con título e ícono, o con el número del paso en un flujo guiado."""
    if step is not None:
        badge = ft.Container(ft.Text(str(step), size=12, weight=ft.FontWeight.BOLD, color=ft.Colors.ON_PRIMARY),
                             width=22, height=22, border_radius=11, bgcolor=ACCENT, alignment=ft.Alignment.CENTER)
    else:
        badge = ft.Icon(icon, size=18, color=ACCENT)
    header = ft.Row([badge, ft.Text(title, size=15, weight=ft.FontWeight.W_600)], spacing=10)
    return ft.Container(
        content=ft.Column([header, *controls], spacing=12),
        padding=16, border_radius=12, expand=expand,
        bgcolor=ft.Colors.with_opacity(0.04, ft.Colors.ON_SURFACE),
        border=ft.Border.all(1, ft.Colors.with_opacity(0.08, ft.Colors.ON_SURFACE)))


def file_chip(path, placeholder="Sin seleccionar") -> ft.Text:
    """Texto con el nombre de un archivo y su ruta completa como tooltip."""
    return ft.Text(Path(path).name if path else placeholder, size=13, no_wrap=True, overflow=ft.TextOverflow.ELLIPSIS,
                   color=None if path else MUTED, tooltip=str(path) if path else None)


def status_row(ok: bool | None, text: str) -> ft.Row:
    icon = {True: (ft.Icons.CHECK_CIRCLE, ft.Colors.GREEN_400), False: (ft.Icons.ERROR_OUTLINE, ft.Colors.ERROR),
            None: (ft.Icons.RADIO_BUTTON_UNCHECKED, MUTED)}[ok]
    return ft.Row([ft.Icon(icon[0], size=16, color=icon[1]), ft.Text(text, size=13, expand=True)], spacing=8)


def disclaimer() -> ft.Container:
    return ft.Container(
        ft.Row([ft.Icon(ft.Icons.INFO_OUTLINE, size=16, color=MUTED),
                ft.Text(DISCLAIMER, size=11.5, color=MUTED, expand=True)], spacing=8,
               vertical_alignment=ft.CrossAxisAlignment.START),
        padding=ft.Padding.symmetric(horizontal=4, vertical=8))
