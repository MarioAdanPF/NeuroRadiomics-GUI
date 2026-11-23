import flet as ft

from logic.lut import FreeSurferLUT
from ui.common import MUTED, view_header


class InfoView(ft.Container):
    def __init__(self, lut: FreeSurferLUT):
        super().__init__()
        self.all_labels = list(lut.entries.values())
        self.filtered_ids = None
        self.show_all_force = True
        self.page_size = 100
        self.current_page = 1
        self.filtered_labels_cache = self.all_labels.copy()

        self.search_bar = ft.TextField(
            hint_text="Buscar región por ID o nombre...",
            prefix_icon=ft.Icons.SEARCH,
            on_change=self.filter_table,
            width=320,
            dense=True,
        )

        self.table = ft.DataTable(
            columns=[
                ft.DataColumn(ft.Text("ID"), numeric=True),
                ft.DataColumn(ft.Text("Color")),
                ft.DataColumn(ft.Text("Nombre de la región")),
            ],
            rows=[],
            heading_row_height=36,
            data_row_min_height=32,
            data_row_max_height=34,
        )

        self.list_view = ft.ListView(
            controls=[ft.Row([self.table], scroll=ft.ScrollMode.AUTO)],
            expand=True
        )

        self.subtitle_text = ft.Text(f"{len(self.all_labels)} etiquetas en la tabla", size=13, color=MUTED)
        self.show_all_button = ft.TextButton("Ver toda la tabla", visible=False, on_click=lambda e: self.set_force_all())

        # --- Controles de Paginación ---
        self.btn_prev = ft.IconButton(
            icon=ft.Icons.ARROW_BACK_IOS,
            on_click=self.prev_page,
            tooltip="Anterior"
        )
        self.btn_next = ft.IconButton(
            icon=ft.Icons.ARROW_FORWARD_IOS,
            on_click=self.next_page,
            tooltip="Siguiente"
        )
        self.page_indicator = ft.Text("Página 1 / 1", weight="bold")

        self.page_jump_input = ft.TextField(
            hint_text="Ir a pág...",
            width=100,
            height=35,
            content_padding=5,
            on_submit=self.jump_page,
            dense=True,
        )

        pagination_row = ft.Row([
            self.btn_prev,
            self.page_indicator,
            self.btn_next,
            ft.VerticalDivider(width=1),
            self.page_jump_input
        ], alignment=ft.MainAxisAlignment.CENTER)

        self.content = ft.Column([
            view_header("Regiones", "Etiquetas de FreeSurfer (FreeSurferColorLUT): número, color y nombre de cada estructura",
                        trailing=self.search_bar),
            ft.Row([self.subtitle_text, self.show_all_button]),
            pagination_row,
            self.list_view
        ], expand=True)
        self.padding = ft.Padding.only(left=24, right=24, top=20, bottom=16)
        self.expand = True

        self.apply_filters() # Aplica el filtro inicial y dibuja

    def set_available_labels(self, labels: list[int]):
        self.filtered_ids = set(labels)
        self.show_all_force = False
        self.subtitle_text.value = f"Mostrando las {len(labels)} regiones de la máscara"
        self.show_all_button.visible = True
        self.apply_filters()

    def set_force_all(self):
        self.show_all_force = True
        self.subtitle_text.value = f"{len(self.all_labels)} etiquetas en la tabla"
        self.show_all_button.visible = False
        self.apply_filters()

    def filter_table(self, e):
        self.apply_filters()

    def apply_filters(self):
        query = self.search_bar.value.lower() if self.search_bar.value else ""
        self.filtered_labels_cache = []

        for lbl in self.all_labels:
            if self.filtered_ids is not None and not self.show_all_force:
                if lbl.id not in self.filtered_ids:
                    continue

            if query and query not in lbl.name.lower() and query != str(lbl.id):
                continue

            self.filtered_labels_cache.append(lbl)

        self.current_page = 1
        self.render_page()

    def prev_page(self, e=None):
        if self.current_page > 1:
            self.current_page -= 1
            self.render_page()

    def next_page(self, e=None):
        max_page = max(1, (len(self.filtered_labels_cache) + self.page_size - 1) // self.page_size)
        if self.current_page < max_page:
            self.current_page += 1
            self.render_page()

    def jump_page(self, e):
        try:
            target = int(e.control.value)
            max_page = max(1, (len(self.filtered_labels_cache) + self.page_size - 1) // self.page_size)
            if 1 <= target <= max_page:
                self.current_page = target
                self.render_page()
            e.control.value = "" # Clear after jump
        except ValueError:
            pass

    def render_page(self):
        self.table.rows.clear()

        total_items = len(self.filtered_labels_cache)
        max_page = max(1, (total_items + self.page_size - 1) // self.page_size)

        self.page_indicator.value = f"Página {self.current_page} / {max_page} ({total_items} totales)"
        self.btn_prev.disabled = self.current_page <= 1
        self.btn_next.disabled = self.current_page >= max_page

        start_idx = (self.current_page - 1) * self.page_size
        end_idx = start_idx + self.page_size

        page_labels = self.filtered_labels_cache[start_idx:end_idx]

        for lbl in page_labels:
            self.table.rows.append(
                ft.DataRow(
                    cells=[
                        ft.DataCell(ft.Text(str(lbl.id))),
                        ft.DataCell(ft.Container(width=20, height=20, bgcolor=lbl.hex_color, border_radius=4,
                                                 border=ft.Border.all(1, ft.Colors.with_opacity(0.3, ft.Colors.ON_SURFACE)))),
                        ft.DataCell(ft.Text(lbl.name, selectable=True)),
                    ]
                )
            )
        try:
            self.update()
        except RuntimeError:
            pass
