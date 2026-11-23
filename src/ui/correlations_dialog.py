import flet as ft
from logic.helpers import remove_redundant_variables

class CorrelationsDialog(ft.AlertDialog):
    def __init__(self, page: ft.Page, pairs, vars_to_remove_state, on_remove_callback):
        super().__init__()
        self.parent_page = page
        self.pairs = pairs
        self.vars_to_remove_state = vars_to_remove_state
        self.on_remove_callback = on_remove_callback

        self.modal = True
        self.title = ft.Text(f"Correlaciones altas entre variables ({len(pairs)} pares)")

        self.content = ft.Column(scroll=ft.ScrollMode.AUTO, width=600, height=400)
        self.content.controls.append(ft.Text(
            "Correlación absoluta mayor a 0.9 entre las regiones analizadas. Ocultar las redundantes solo cambia "
            "la tabla de resultados: el CSV guardado conserva todas las variables.", size=12.5,
            color=ft.Colors.ON_SURFACE_VARIANT))

        if not pairs:
            self.content.controls.append(ft.Text("No se encontraron correlaciones altas (>0.9) entre las variables."))
        else:
            for v1, v2, val in pairs:
                self.content.controls.append(
                    ft.Text(
                        spans=[
                            ft.TextSpan(f"{v1}", ft.TextStyle(color=ft.Colors.BLUE_400)),
                            ft.TextSpan(" × "),
                            ft.TextSpan(f"{v2}", ft.TextStyle(color=ft.Colors.RED_400)),
                            ft.TextSpan(f": {val:.4f}")
                        ],
                        size=13
                    )
                )

        self.actions = [
            ft.TextButton("Cerrar", on_click=self.close_dialog),
            ft.FilledButton("Ocultar variables redundantes", on_click=self.remove_and_close, disabled=not pairs),
        ]

    def close_dialog(self, e):
        self.parent_page.pop_dialog()

    def remove_and_close(self, e):
        new_vars_to_remove = remove_redundant_variables(self.pairs, self.vars_to_remove_state)
        # Notify the parent to update state
        self.on_remove_callback(new_vars_to_remove)
        self.close_dialog(e)
