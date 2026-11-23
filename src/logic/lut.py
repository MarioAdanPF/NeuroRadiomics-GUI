"""Tabla de etiquetas de FreeSurfer (FreeSurferColorLUT.csv): nombre y color de cada región."""
import csv
from dataclasses import dataclass
from functools import cache

import numpy as np

from logic.paths import LUT_PATH


@dataclass(frozen=True)
class LutEntry:
    id: int
    name: str
    r: int
    g: int
    b: int
    a: int

    @property
    def hex_color(self) -> str:
        return f"#{self.r:02x}{self.g:02x}{self.b:02x}"


class FreeSurferLUT:
    def __init__(self, path=LUT_PATH):
        self.entries: dict[int, LutEntry] = {}
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                entry = LutEntry(int(row["No."]), row["Label Name"], int(row["R"]), int(row["G"]), int(row["B"]), int(row["A"]))
                self.entries[entry.id] = entry
        self._color_table = None

    def name(self, label: int) -> str:
        entry = self.entries.get(label)
        return entry.name if entry else f"Región {label}"

    def color_table(self) -> np.ndarray:
        """Arreglo (id máximo + 1, 3) con el color RGB de cada etiqueta, para pintar máscaras."""
        if self._color_table is None:
            table = np.zeros((max(self.entries) + 1, 3), dtype=np.uint8)
            for entry in self.entries.values():
                table[entry.id] = (entry.r, entry.g, entry.b)
            self._color_table = table
        return self._color_table


@cache
def get_lut() -> FreeSurferLUT:
    """Instancia compartida de la tabla (se lee una sola vez)."""
    return FreeSurferLUT()
