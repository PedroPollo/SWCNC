import flet as ft
import os
import subprocess
import logging
from cargar_modelo import obtener_codigo

def download_code(name, content, page):
    download_rout = os.path.join(os.path.expanduser("~"), "Downloads")
    
    name += ".txt"
    
    file_route = os.path.join(download_rout, name)
    
    try:
        # Verificar si el archivo ya existe
        if os.path.exists(file_route):
            mensaje_error = f"El archivo {name} ya existe en {download_rout}."
            # Mostrar SnackBar con el mensaje de error
            snack_bar = ft.SnackBar(ft.Text(mensaje_error))
            snack_bar.open = True
            page.overlay.append(snack_bar)
            return mensaje_error

        with open(file_route, "w") as file:
            file.write(content)
        
        mensaje = f"Archivo {name} guardado en {file_route}."
        logging.info(mensaje)
        
        # Mostrar SnackBar
        snack_bar=ft.SnackBar(ft.Text(mensaje))
        snack_bar.open = True
        page.overlay.append(snack_bar)
        
        return mensaje
    
    except Exception as e:
        mensaje_error = f"Error al guardar el archivo: {str(e)}"
        logging.error(mensaje_error)
        
        # Mostrar SnackBar en caso de error
        snack_bar=ft.SnackBar(ft.Text(mensaje))
        snack_bar.open = True
        page.overlay.append(snack_bar)
        
        return mensaje_error
    
def generar_codigo(input):
    codigo = obtener_codigo(input)
    return codigo

def simular_codigo_g(name, content, height, width):
    try:
        # Suponiendo que simulador.py está en el mismo directorio
        subprocess.run(['python', 'simulador.py', name, content, str(width), str(height)])
    except Exception as e:
        logging.error(f"Error en el simulador: {e}")

def open_simulator(gcode):
    ft.launch_url("https://nraynaud.github.io/webgcode/")