"""
Lectura de volúmenes (.mgz, .nii, .nii.gz) y conversión a NIfTI para el backend.

SimpleITK no lee .mgz, así que antes de extraer se convierten a .nii.gz con nibabel. La
conversión conserva los vóxeles, su tipo y la geometría (se verificó que da exactamente lo
mismo que mri_convert de FreeSurfer), así que ya no hace falta tener FreeSurfer instalado.
Los archivos convertidos se guardan en la caché, no junto a los datos del usuario.
"""
import hashlib
from pathlib import Path

import nibabel as nib
import numpy as np

from logic.paths import cache_dir

VOLUME_EXTENSIONS = (".nii.gz", ".nii", ".mgz")


def is_volume(path) -> bool:
    return str(path).lower().endswith(VOLUME_EXTENSIONS)


def to_nifti(path) -> Path:
    """Devuelve una ruta .nii/.nii.gz del volumen, convirtiéndolo si es .mgz."""
    path = Path(path)
    if not path.name.lower().endswith(".mgz"):
        return path

    # El nombre en caché depende de la ruta, la fecha y el tamaño: si el archivo cambia, se reconvierte
    stat = path.stat()
    key = hashlib.sha1(f"{path.resolve()}|{stat.st_mtime_ns}|{stat.st_size}".encode()).hexdigest()[:16]
    output = cache_dir() / f"{path.name[:-4]}_{key}.nii.gz"
    if not output.exists():
        image = nib.load(path)
        # Se escribe con otro nombre y se renombra al final, para no dejar archivos a medias
        temporary = output.with_name(output.name.replace(".nii.gz", ".tmp.nii.gz"))
        nib.save(nib.Nifti1Image(np.asanyarray(image.dataobj), image.affine), temporary)
        temporary.replace(output)
    return output


def mask_labels(path) -> list[int]:
    """Etiquetas (LabelID) presentes en una máscara de segmentación, sin el fondo (0)."""
    data = np.asanyarray(nib.load(path).dataobj)
    return [int(v) for v in np.unique(data) if v != 0]
