import os
import subprocess
import sys
import logging
from pathlib import Path

# Configurar logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def install_packages(packages):
    """Instala una lista de paquetes usando pip."""
    for package in packages:
        try:
            logging.info(f"Instalando {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        except subprocess.CalledProcessError as e:
            logging.error(f"Error al instalar {package}: {e}")

def create_shortcut(shortcut_path, target_command):
    """Crea un script ejecutable en el escritorio para macOS."""
    desktop = Path.home() / "Desktop"
    shortcut = desktop / shortcut_path

    # Crear el archivo de script
    with open(shortcut.with_suffix(".command"), "w") as command_file:
        command_file.write(f"#!/bin/bash\n")
        command_file.write(f"cd \"{os.getcwd()}\"/ttgenerator\n")
        command_file.write(f"{target_command}\n")
        command_file.write(f"exit\n")

    # Hacer el archivo ejecutable
    os.chmod(shortcut.with_suffix(".command"), 0o755)
    logging.info(f"Script creado en: {shortcut.with_suffix('.command')}\nHaz doble clic para ejecutar.")

def main():
    packages = ["ezdxf", "torch", "flet", "tkinter", "numpy", "networkx", "pandas", "sys", "json", "os", "subprocess", "logging", "typing", "math", "re"]
    shortcut_name = "Generador"
    target_command = "flet run"

    logging.info("Iniciando instalación de paquetes...")
    install_packages(packages)

    logging.info("Creando script ejecutable en el escritorio...")
    create_shortcut(shortcut_name, target_command)

    logging.info("Proceso completado. Puedes usar el script en tu escritorio para ejecutar la aplicación.")

if __name__ == "__main__":
    main()
