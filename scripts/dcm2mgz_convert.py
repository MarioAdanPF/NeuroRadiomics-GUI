"""
Convierte series DICOM a .mgz con mri_convert de FreeSurfer, antes de segmentarlas con recon-all.

Uso:
    python scripts/dcm2mgz_convert.py CARPETA_DICOM CARPETA_SALIDA

CARPETA_DICOM tiene una subcarpeta por paciente (p. ej. sujeto_001/...). De cada una se toma el
primer archivo .dcm: mri_convert lee a partir de él toda la serie de esa carpeta.
"""
import argparse
import os
import shutil
import subprocess


def first_dicom(folder):
    for root, _, files in os.walk(folder):
        for name in sorted(files):
            if name.lower().endswith(".dcm"):
                return os.path.join(root, name)
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dicom_dir", help="carpeta con una subcarpeta DICOM por paciente")
    parser.add_argument("output_dir", help="carpeta donde se guardan los .mgz")
    args = parser.parse_args()

    if shutil.which("mri_convert") is None:
        parser.error("No se encontró mri_convert. Cargue el entorno de FreeSurfer (SetUpFreeSurfer.sh).")
    os.makedirs(args.output_dir, exist_ok=True)

    for patient_id in sorted(os.listdir(args.dicom_dir)):
        patient_path = os.path.join(args.dicom_dir, patient_id)
        if not os.path.isdir(patient_path):
            continue
        print(f"--- Procesando paciente: {patient_id} ---")

        dicom = first_dicom(patient_path)
        if dicom is None:
            print(f"No se encontraron archivos .dcm para el paciente {patient_id}\n")
            continue

        output_file = os.path.join(args.output_dir, f"{patient_id}.mgz")
        try:
            # Lista de argumentos (sin shell): funciona con rutas que tienen espacios
            subprocess.run(["mri_convert", dicom, output_file], check=True)
            print(f"Éxito: {patient_id}.mgz generado.\n")
        except subprocess.CalledProcessError as e:
            print(f"Error al procesar {patient_id}: {e}\n")

    print("Proceso finalizado.")


if __name__ == "__main__":
    main()
