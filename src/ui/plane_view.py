import io

import flet as ft
from PIL import Image


def _blank_png(size=8) -> bytes:
    """Imagen negra para el lienzo antes de cargar un volumen."""
    buf = io.BytesIO()
    Image.new("L", (size, size), 0).save(buf, format="PNG")
    return buf.getvalue()


class PlaneView(ft.Container):
    def __init__(self, plane_name: str, title: str, color: str, on_crosshair_move, on_slice_change, size: int = 300):
        super().__init__()
        self.plane_name = plane_name
        self.on_crosshair_move = on_crosshair_move
        self.on_slice_change = on_slice_change
        self.border_color = color
        self.size = size

        # UI Elements
        self.title = ft.Text(title, size=12, color=color, weight=ft.FontWeight.BOLD)
        self.info_text = ft.Text("", size=11, color=ft.Colors.ON_SURFACE_VARIANT)

        self.img = ft.Image(src=_blank_png(), width=size, height=size, fit=ft.BoxFit.CONTAIN, gapless_playback=True,
                            error_content=ft.Container(bgcolor=ft.Colors.BLACK))
        # Líneas nativas de Flet
        self.vline = ft.Container(width=1, height=size, bgcolor=ft.Colors.RED_ACCENT_400, visible=False)
        self.hline = ft.Container(width=size, height=1, bgcolor=ft.Colors.RED_ACCENT_400, visible=False)

        self.stack = ft.Stack(controls=[self.img, self.vline, self.hline], width=size, height=size)

        gesture = ft.GestureDetector(
            content=self.stack,
            on_tap_down=self._handle_mouse,
            on_pan_update=self._handle_mouse,
            on_scroll=self._handle_scroll,
            drag_interval=30,
            mouse_cursor=ft.MouseCursor.PRECISE
        )

        # Barra de desplazamiento para recorrer los cortes de este plano
        self.slider = ft.Slider(min=0, max=1, value=0, disabled=True, on_change=self._handle_slider, width=size)

        # Configuración del contenedor principal (self)
        self.content = ft.Column([self.title, gesture, self.slider, self.info_text],
                                 horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=4)
        self.border_radius = 8
        self.border = ft.Border.all(1, color)
        self.padding = ft.Padding.only(top=6, bottom=8, left=6, right=6)

        # Dimensiones de la imagen cargada para cálculos
        self._img_w = 1
        self._img_h = 1
        self._crosshair = None

    def set_size(self, size: int):
        """Cambia el tamaño del lienzo (al redimensionar la ventana)."""
        self.size = size
        self.img.width = self.img.height = size
        self.stack.width = self.stack.height = size
        self.slider.width = size
        if self._crosshair:
            self.update_crosshairs(*self._crosshair)

    def update_image(self, png: bytes, img_w: int, img_h: int, current_slice: int, max_slice: int):
        """Actualiza la imagen de fondo solo cuando cambiamos de corte."""
        if png:
            self.img.src = png
            self._img_w = img_w
            self._img_h = img_h
            self.info_text.value = f"Corte {current_slice + 1} / {max_slice}"
            self.slider.disabled = False
            self.slider.max = max(1, max_slice - 1)
            self.slider.divisions = max(1, max_slice - 1)
            self.slider.value = current_slice
            self.update()

    def _geometry(self):
        """Tamaño dibujado de la imagen dentro del lienzo cuadrado y sus márgenes."""
        AR = self._img_w / self._img_h
        if AR > 1:
            draw_w, draw_h = self.size, self.size / AR
            pad_x, pad_y = 0, (self.size - draw_h) / 2
        else:
            draw_w, draw_h = self.size * AR, self.size
            pad_x, pad_y = (self.size - draw_w) / 2, 0
        return draw_w, draw_h, pad_x, pad_y

    def update_crosshairs(self, col: int, row: int):
        """Mueve las líneas rojas usando coordenadas visuales. Es inmediato."""
        if self._img_w <= 1 or self._img_h <= 1: return
        self._crosshair = (col, row)

        # Calcular aspect ratio para saber dónde están los bordes negros
        draw_w, draw_h, pad_x, pad_y = self._geometry()

        # Posiciones en pantalla
        px = pad_x + (col / self._img_w) * draw_w
        py = pad_y + (row / self._img_h) * draw_h

        self.vline.left = px
        self.vline.top = pad_y
        self.vline.height = draw_h
        self.vline.visible = True

        self.hline.top = py
        self.hline.left = pad_x
        self.hline.width = draw_w
        self.hline.visible = True

        self.update()

    def set_focus(self, is_focused: bool):
        self.border = ft.Border.all(2 if is_focused else 1, ft.Colors.AMBER if is_focused else self.border_color)
        self.update()

    def _handle_mouse(self, e):
        """Detecta el click/arrastre y lo emite al controlador principal."""
        lx = e.local_position.x if hasattr(e, "local_position") else (e.local_x if hasattr(e, "local_x") else None)
        ly = e.local_position.y if hasattr(e, "local_position") else (e.local_y if hasattr(e, "local_y") else None)
        if lx is None or ly is None: return

        draw_w, draw_h, pad_x, pad_y = self._geometry()

        img_x, img_y = lx - pad_x, ly - pad_y
        if img_x < 0 or img_x > draw_w or img_y < 0 or img_y > draw_h: return

        # Mapeamos a índice de columna y fila
        col = min(int((img_x / draw_w) * self._img_w), self._img_w - 1)
        row = min(int((img_y / draw_h) * self._img_h), self._img_h - 1)

        # Disparamos el evento al Controlador
        self.on_crosshair_move(self.plane_name, col, row)

    def _handle_scroll(self, e):
        """La rueda del ratón avanza o retrocede un corte."""
        dy = e.scroll_delta.y if getattr(e, "scroll_delta", None) is not None else 0
        if dy:
            self.on_slice_change(self.plane_name, int(self.slider.value) + (1 if dy > 0 else -1))

    def _handle_slider(self, e):
        self.on_slice_change(self.plane_name, int(e.control.value))
