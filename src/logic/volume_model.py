import io

import nibabel as nib
import numpy as np
from PIL import Image

from logic.lut import get_lut


class VolumeModel:
    def __init__(self):
        self.data = None
        self.mask = None
        self.shape = (0, 0, 0)
        self.zooms = (1.0, 1.0, 1.0)

        # Superposición de la máscara de segmentación
        self.show_mask = True
        self.mask_opacity = 0.45
        self.highlight_labels: set[int] | None = None   # None: todas las regiones

        # Coordenadas actuales de navegación
        self.coords = {"sagittal": 0, "coronal": 0, "axial": 0}

        # NIfTI estándar suele ser: 0=Sag, 1=Cor, 2=Ax
        self.axis_map = {
            "sagittal": 0,
            "axial": 1,
            "coronal": 2
        }

        # --- SISTEMA DE NAVEGACIÓN Y ORIENTACIÓN VISUAL ---
        # Permite rotar y hacer efecto espejo (flip) de posición en las imágenes para que se
        # muestren correctamente. Al mismo tiempo asegura que la navegación (cruces rojas)
        # siga estando 100% coordinada según estas reglas de manera automática.
        #
        # x_axis: Eje anatómico que cambia al mover el mouse de Izq a Der (Columnas).
        # y_axis: Eje anatómico que cambia al mover el mouse de Arriba a Abajo (Filas).
        # flip_x: True si quieres cambiar la posición/invertir la imagen Horizontalmente.
        # flip_y: True si quieres cambiar la posición/invertir la imagen Verticalmente.
        self.view_config = {
            "axial": {
                "x_axis": "sagittal",
                "y_axis": "coronal",
                "flip_x": False,
                "flip_y": True
            },
            "coronal": {
                "x_axis": "sagittal",
                "y_axis": "axial",
                "flip_x": False,
                "flip_y": False
            },
            "sagittal": {
                # Puedes modificar esto si el plano sagital se ve rotado o invertido:
                # - Si quieres rotarlo, intercambia los valores de 'x_axis' y 'y_axis'.
                # - Si el movimiento horizontal parece invertido, cambia 'flip_x'.
                # - Si el movimiento vertical parece invertido, cambia 'flip_y'.
                # flip_x=True: la cara del paciente mira hacia la izquierda (anterior a la izquierda),
                # como en la convención radiológica estándar.
                "x_axis": "coronal",
                "y_axis": "axial",
                "flip_x": True,
                "flip_y": False
            }
        }

    @staticmethod
    def _read_lia(path: str):
        """
        Lee un volumen y lo reorienta a LIA, la orientación de los volúmenes conformados de
        FreeSurfer, que es la que suponen axis_map y view_config: eje 0 hacia la izquierda del
        paciente, eje 1 hacia abajo (inferior) y eje 2 hacia adelante (anterior). Con esto los
        planos se ven en la orientación estándar aunque el archivo esté guardado en otra (p. ej. RAS).
        Devuelve el arreglo y el tamaño del vóxel en ese mismo orden.
        """
        image = nib.load(path)
        transform = nib.orientations.ornt_transform(nib.io_orientation(image.affine),
                                                    nib.orientations.axcodes2ornt(("L", "I", "A")))
        data = nib.orientations.apply_orientation(np.asanyarray(image.dataobj), transform)
        zooms = np.asarray(image.header.get_zooms()[:3], dtype=float)
        # transform[i, 0] es el eje de salida al que va el eje de entrada i
        reordered = np.empty(3)
        reordered[transform[:, 0].astype(int)] = zooms
        return data, tuple(float(z) for z in reordered)

    def load_volume(self, path: str):
        raw, zooms = self._read_lia(path)
        arr = np.asarray(raw, dtype=np.float32)

        # Normalización rápida y segura
        vmin, vmax = np.percentile(arr[arr > 0], [2, 98]) if np.any(arr > 0) else (0, 1)
        # Se guarda en 8 bits: ocupa la cuarta parte de memoria y los cortes salen directo para PNG
        self.data = (np.clip((arr - vmin) / (vmax - vmin + 1e-8), 0, 1) * 255).astype(np.uint8)
        self.shape = self.data.shape
        self.zooms = zooms
        # La máscara del volumen anterior ya no corresponde
        self.mask = None

        # Posicionar en el centro
        self.coords["sagittal"] = self.shape[self.axis_map["sagittal"]] // 2
        self.coords["coronal"] = self.shape[self.axis_map["coronal"]] // 2
        self.coords["axial"] = self.shape[self.axis_map["axial"]] // 2

        return self.zooms

    def load_mask(self, path: str) -> list[int]:
        """Carga la máscara de segmentación. Debe tener las mismas dimensiones que el volumen."""
        # Misma reorientación que el volumen, para que cada vóxel de la máscara caiga en su lugar
        mask = np.asarray(self._read_lia(path)[0]).astype(np.int32)
        if self.data is not None and mask.shape != self.shape:
            raise ValueError(f"La máscara mide {mask.shape} y el volumen {self.shape}.")
        self.mask = mask
        return [int(v) for v in np.unique(mask) if v != 0]

    def set_coord(self, plane: str, value: int):
        """Asegura que las coordenadas no salgan de los límites de la matriz."""
        if self.data is None: return
        max_val = self.shape[self.axis_map[plane]] - 1
        self.coords[plane] = max(0, min(value, max_val))

    def label_at_cursor(self) -> int | None:
        """Etiqueta de la máscara en la posición de la cruz."""
        if self.mask is None:
            return None
        index = [0, 0, 0]
        for plane, axis in self.axis_map.items():
            index[axis] = self.coords[plane]
        return int(self.mask[tuple(index)])

    def _oriented_slice(self, volume, plane: str):
        """
        Extrae el corte 2D de la matriz 3D y lo alinea según la configuración
        en view_config de manera que la rotación e inversión aplicadas coincidan
        perfectamente con la navegación en la vista.
        """
        axis_idx = self.axis_map[plane]
        raw_slice = np.take(volume, self.coords[plane], axis=axis_idx)

        # Obtener los nombres de los 2 ejes que forman la imagen raw (en su orden original de la matriz)
        raw_axes = [k for k, v in sorted(self.axis_map.items(), key=lambda item: item[1]) if v != axis_idx]

        cfg = self.view_config[plane]
        x_axis_name = cfg["x_axis"]
        y_axis_name = cfg["y_axis"]

        # Alinear el corte 2D (Rotaciones implícitas)
        if raw_axes[0] == y_axis_name and raw_axes[1] == x_axis_name:
            vis_slice = raw_slice
        elif raw_axes[0] == x_axis_name and raw_axes[1] == y_axis_name:
            vis_slice = raw_slice.T
        else:
            vis_slice = raw_slice # Fallback por seguridad

        # Posicionamiento: Espejo Horizontal o Vertical
        if cfg["flip_y"]:
            vis_slice = vis_slice[::-1, :]
        if cfg["flip_x"]:
            vis_slice = vis_slice[:, ::-1]
        return vis_slice

    def get_slice_png(self, plane: str):
        """
        Corte del plano como PNG (bytes), con la máscara superpuesta si está activa.
        Devuelve (png, ancho, alto).
        """
        if self.data is None: return None, 0, 0

        gray = self._oriented_slice(self.data, plane)
        if self.mask is not None and self.show_mask and self.mask_opacity > 0:
            labels = self._oriented_slice(self.mask, plane)
            table = get_lut().color_table()
            visible = labels > 0
            if self.highlight_labels is not None:
                visible &= np.isin(labels, list(self.highlight_labels))
            rgb = np.repeat(gray[:, :, None], 3, axis=2).astype(np.float32)
            colors = table[np.clip(labels, 0, len(table) - 1)].astype(np.float32)
            alpha = self.mask_opacity
            rgb[visible] = (1 - alpha) * rgb[visible] + alpha * colors[visible]
            img = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
        else:
            img = Image.fromarray(np.ascontiguousarray(gray), mode="L")

        # Compresión baja: el corte es pequeño y así se codifica mucho más rápido
        buf = io.BytesIO()
        img.save(buf, format="PNG", compress_level=1)
        return buf.getvalue(), gray.shape[1], gray.shape[0]
