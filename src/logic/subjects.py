"""Búsqueda de la imagen y la máscara de un sujeto procesado con FreeSurfer."""
from dataclasses import dataclass
from pathlib import Path

# Nombres de archivo de FreeSurfer: imagen sin cráneo y segmentación aparc+aseg
IMAGE_NAME = "brain"
MASK_NAME = "aparc+aseg"
EXTENSIONS = (".nii.gz", ".nii", ".mgz")


@dataclass
class SubjectFiles:
    id: str
    image: Path
    mask: Path


def _find(folder: Path, name: str) -> Path | None:
    # Prioridad a NIfTI, que no necesita conversión
    for extension in EXTENSIONS:
        candidate = folder / f"{name}{extension}"
        if candidate.is_file():
            return candidate
    return None


def find_subject_files(folder) -> SubjectFiles | None:
    """Busca brain y aparc+aseg en la carpeta del sujeto o en su subcarpeta mri/."""
    folder = Path(folder)
    for candidate in (folder, folder / "mri"):
        if candidate.is_dir():
            image, mask = _find(candidate, IMAGE_NAME), _find(candidate, MASK_NAME)
            if image and mask:
                return SubjectFiles(folder.name, image, mask)
    return None


def scan_directory(root) -> list[SubjectFiles]:
    """Sujetos válidos dentro de una carpeta (una subcarpeta por sujeto)."""
    root = Path(root)
    subjects = [find_subject_files(d) for d in sorted(root.iterdir()) if d.is_dir()]
    return [s for s in subjects if s is not None]
