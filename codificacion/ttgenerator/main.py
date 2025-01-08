from typing import Dict
import torch
import flet as ft
from flet import (
    FilePicker,
    FilePickerResultEvent,
)
import logging as lg
import json
import os
from simulador import *
from extractor import *
from generador import *
import subprocess
from figure_determinator import *

def get_coordinates(path):
    try:
        result = subprocess.run(['python', 'extractor.py', path], capture_output=True, text=True)
        
        #Verificamos errores en la ejecucion
        if result.returncode != 0:
            print(f"Error al ejecutar extractor.py: {result.stderr}")
            return None
        
        coordinates = json.loads(result.stdout)
        return coordinates
    
    except Exception as e:
        print(f"Error:{e}")
        return None

# Archivo donde se guardarán los datos de las máquinas
FILE_PATH = "maquinas.json"

# Función para cargar datos desde el archivo JSON
def cargar_datos():
    if os.path.exists(FILE_PATH):
        with open(FILE_PATH, "r") as file:
            return json.load(file)
        lg.info("Se cargo el archivo JSon correctamente")
    return {}

# Función para guardar los datos en el archivo JSON
def guardar_datos(maquinas):
    with open(FILE_PATH, "w") as file:
        json.dump(maquinas, file, indent=4)
        lg.info("Se guardaron los datos en el archivo JSon correctamente")

def main(page: ft.Page):
    page.window.width = 1000
    page.window.height = 600
    page.bgcolor = '#DDDDDD'
    page.theme_mode = ft.ThemeMode.LIGHT

    # Cargar los datos de las máquinas
    maquinas = cargar_datos()

    maquina_actual = ft.Ref[ft.Dropdown]()
    nueva_maquina_nombre = ft.Ref[ft.TextField]()
    velocidad = ft.Ref[ft.TextField]()
    amperaje = ft.Ref[ft.TextField]()
    diametro = ft.Ref[ft.TextField]()
    direccion = ft.Ref[ft.Dropdown]()
    loaded_file = False
    coordinates = None
    codigo = None
    
    def checker_fields():
        if (
            maquina_actual.current.value
            and velocidad.current.value
            and amperaje.current.value
            and diametro.current.value
            and direccion.current.value
            and altura_ref.current.value
            and ancho_ref.current.value
            and loaded_file  # Verificar que el archivo haya sido cargado
        ):
            button_generador_ref.current.disabled = False
            button_generador_ref.current.bgcolor = "#4CAF50"
            button_generador_ref.current.color = ft.colors.WHITE
            button_generador_ref.current.update()
        else:
            button_generador_ref.current.disabled = True
            button_generador_ref.current.update()

    def seleccionar_maquina(e):
        seleccion = maquina_actual.current.value
        if seleccion in maquinas:
            velocidad.current.value = maquinas[seleccion]["velocidad"]
            amperaje.current.value = maquinas[seleccion]["amperaje"]
            diametro.current.value = maquinas[seleccion]["diámetro"]
            direccion.current.value = maquinas[seleccion]["dirección"]
            lg.info(f"Se selecciono la maquina: {seleccion} correctamente")
            checker_fields()
            button_generador_ref.current.update()
        page.update()

    def guardar_maquina(e):
        seleccion = maquina_actual.current.value
        # Verificar que ningún campo esté vacío
        if not seleccion or not velocidad.current.value or not amperaje.current.value or not diametro.current.value or not direccion.current.value:
            page.snack_bar = ft.SnackBar(ft.Text("Ningún campo puede estar vacío"))
            page.snack_bar.open = True
            page.update()
            return
            
        if seleccion in maquinas:
            maquinas[seleccion] = {
                "velocidad": velocidad.current.value,
                "amperaje": amperaje.current.value,
                "diámetro": diametro.current.value,
                "dirección": direccion.current.value,
            }
            guardar_datos(maquinas)
            page.snack_bar = ft.SnackBar(ft.Text(f"Datos de {seleccion} actualizados"))
            page.snack_bar.open = True
            page.update()

    def agregar_maquina(e):
        nuevo_nombre = nueva_maquina_nombre.current.value
        if not nuevo_nombre or not velocidad.current.value or not amperaje.current.value or not diametro.current.value or not direccion.current.value:
            page.snack_bar = ft.SnackBar(ft.Text("Ningún campo puede estar vacío"))
            page.snack_bar.open = True
            page.update()
            return
            
        if nuevo_nombre not in maquinas:
            maquinas[nuevo_nombre] = {
                "velocidad": velocidad.current.value,
                "amperaje": amperaje.current.value,
                "diámetro": diametro.current.value,
                "dirección": direccion.current.value,
            }
            maquina_actual.current.options.append(ft.dropdown.Option(nuevo_nombre))
            maquina_actual.current.value = nuevo_nombre
            guardar_datos(maquinas)
            page.snack_bar = ft.SnackBar(ft.Text(f"Nueva máquina '{nuevo_nombre}' agregada"))
            page.snack_bar.open = True
            nueva_maquina_nombre.current.value = ""
            page.update()
        else:
            page.snack_bar = ft.SnackBar(ft.Text("Nombre de la máquina inválido o ya existe"))
            page.snack_bar.open = True
            page.update()
        

    def on_dialog_result(e: FilePickerResultEvent):
        nonlocal loaded_file, coordinates
        if file_picker.result != None and file_picker.result.files != None:
            for f in file_picker.result.files:
                loaded_file = True
                file_name_ref.current.value = f"Archivo: {f.name}"
                file_icon_ref.current.name = ft.icons.DONE_ROUNDED
                file_icon_ref.current.color = ft.colors.GREEN
                getcoordinates = get_coordinates(f.path)
                if getcoordinates['lines'] != None:
                    coordinates = classify_lines(getcoordinates)
                    coordinates = new_data_estructure(getcoordinates, coordinates)
                else :
                    coordinates = new_data(get_coordinates)
                print(coordinates)

                file_name_ref.current.update()
                file_icon_ref.current.update()
                lg.info("Se cargo archivo correctamente")
        checker_fields()
        button_generador_ref.current.update()
    
    file_name_ref = ft.Ref[ft.Text]()
    file_icon_ref = ft.Ref[ft.Icon]()
    button_generador_ref = ft.Ref[ft.ElevatedButton]()
    file_picker = FilePicker(on_result=on_dialog_result)
    altura_ref = ft.Ref[ft.TextField]()
    ancho_ref = ft.Ref[ft.TextField]()

    # Llamar a checker_fields cuando se modifiquen altura o ancho
    def on_value_change(e):
        checker_fields()
        button_generador_ref.current.update()

    # Crear la estructura de la pantalla principal (Home Page)
    def home_page():
        page.appbar = None
        
        page.overlay.append(file_picker)
        page.update()
        
        return ft.Row(
            controls=[
                # Primera columna: Selección de máquina y parámetros
                ft.Container(
                    padding=10,
                    width=300,
                    content=ft.Column(
                        controls=[
                            ft.Text("Parámetros de la Máquina"),
                            ft.Divider(color=ft.colors.BLACK),
                            ft.Text("Seleccionar Máquina"),
                            ft.Dropdown(
                                ref=maquina_actual,
                                options=[ft.dropdown.Option(key, key) for key in maquinas.keys()],
                                on_change=seleccionar_maquina,
                                bgcolor = ft.colors.WHITE
                            ),
                            ft.Divider(color=ft.colors.BLACK),
                            ft.TextField(
                                ref=velocidad,
                                label="Velocidad de avance (MM/MIN)",
                                bgcolor = ft.colors.WHITE,
                                label_style=ft.TextStyle(color=ft.colors.BLACK),
                                border_color=ft.colors.BLACK,
                                focused_border_color=ft.colors.BLACK,
                                on_change = on_value_change
                            ),
                            ft.TextField(
                                ref=amperaje,
                                label="Amperaje",
                                bgcolor = ft.colors.WHITE,
                                label_style=ft.TextStyle(color=ft.colors.BLACK),
                                border_color=ft.colors.BLACK,
                                focused_border_color=ft.colors.BLACK,
                                on_change = on_value_change
                            ),
                        ],
                    ),
                ),
                # Segunda columna: Más parámetros
                ft.Container(
                    padding=10,
                    width=300,
                    content=ft.Column(
                        controls=[
                            ft.Text("Más Parámetros"),
                            ft.Divider(color=ft.colors.BLACK),
                            ft.TextField(
                                ref=diametro,
                                label="Diámetro (MM)",
                                bgcolor = ft.colors.WHITE,
                                label_style=ft.TextStyle(color=ft.colors.BLACK),
                                border_color=ft.colors.BLACK,
                                focused_border_color=ft.colors.BLACK,
                                on_change = on_value_change
                            ),
                            ft.Dropdown(
                                ref=direccion,
                                options=[
                                    ft.dropdown.Option("Dentro"),
                                    ft.dropdown.Option("Centro"),
                                    ft.dropdown.Option("Fuera"),
                                ],
                                label="Dirección",
                                bgcolor = ft.colors.WHITE,
                                label_style=ft.TextStyle(color=ft.colors.BLACK),
                                border_color=ft.colors.BLACK,
                                focused_border_color=ft.colors.BLACK,
                            ),
                            ft.Divider(color=ft.colors.BLACK),
                            ft.ElevatedButton(
                                text="Guardar Máquina",
                                bgcolor = ft.colors.BLACK,
                                color = ft.colors.WHITE,
                                on_click=guardar_maquina,
                            ),
                            ft.Divider(color=ft.colors.BLACK),
                            ft.Text("Dimensiones del lienzo"),
                            ft.TextField(
                                ref = altura_ref,
                                label="Altura",
                                bgcolor = ft.colors.WHITE,
                                label_style=ft.TextStyle(color=ft.colors.BLACK),
                                border_color=ft.colors.BLACK,
                                focused_border_color=ft.colors.BLACK,
                                on_change = on_value_change
                            ),
                            ft.TextField(
                                ref = ancho_ref,
                                label="Ancho",
                                bgcolor = ft.colors.WHITE,
                                label_style=ft.TextStyle(color=ft.colors.BLACK),
                                border_color=ft.colors.BLACK,
                                focused_border_color=ft.colors.BLACK,
                                on_change = on_value_change
                            )
                        ],
                    ),
                ),
                # Tercera columna: Agregar nueva máquina y Simulación
                ft.Container(
                    padding=10,
                    width=300,
                    content=ft.Column(
                        controls=[
                            ft.Text("Agregar Nueva Máquina"),
                            ft.Divider(color=ft.colors.BLACK),
                            ft.TextField(
                                ref=nueva_maquina_nombre,
                                label="Nombre de la Máquina",
                                bgcolor = ft.colors.WHITE,
                                label_style=ft.TextStyle(color=ft.colors.BLACK),
                                border_color=ft.colors.BLACK,
                                focused_border_color=ft.colors.BLACK,
                            ),
                            ft.ElevatedButton(
                                text="Agregar Nueva Máquina",
                                bgcolor = ft.colors.BLACK,
                                color = ft.colors.WHITE,
                                on_click=agregar_maquina,
                            ),
                            ft.Divider(color=ft.colors.BLACK),
                            ft.ElevatedButton(
                                text="Generar código G",
                                on_click=on_generate_click,
                                ref=button_generador_ref,
                                disabled=True,
                            ),
                            ft.Divider(color=ft.colors.BLACK),
                            ft.ElevatedButton(
                                text = "Cargar archivo",
                                bgcolor = ft.colors.BLACK,
                                color = ft.colors.WHITE,
                                icon=ft.icons.UPLOAD_FILE,
                                on_click = lambda _: file_picker.pick_files(
                                    allow_multiple=False,
                                    file_type=ft.FilePickerFileType.CUSTOM,
                                    allowed_extensions=["dxf"]
                                    ),
                            ),
                            ft.Divider(color=ft.colors.BLACK),
                            ft.Row([ft.Icon(ref=file_icon_ref, name = ft.icons.NOT_INTERESTED, color= ft.colors.RED), ft.Text(ref=file_name_ref, value = "Archivo: No se ha cargado ningun archivo")])
                        ],
                    ),
                ),
            ],
        )

    def generador_page(name,data):
        page.appbar=ft.AppBar(
            title=ft.Text("Generador de código G"),
            center_title=True,
            color = ft.colors.BLACK,
            bgcolor=ft.colors.WHITE,
            actions=[
                ft.IconButton(
                    icon=ft.icons.ARROW_BACK,
                    tooltip="Regresar a la pagina principal",
                    on_click=lambda e: page.go("/")
                )
            ]
        )
        return ft.Container(
            expand=True,
            alignment=ft.alignment.center,
            content = ft.Row(
                alignment=ft.MainAxisAlignment.CENTER,
                controls = [
                    ft.Container(
                        padding=10,
                        width=300,
                        content=ft.Column(
                            controls=[
                                ft.ElevatedButton(
                                    text="Simular Código G",
                                    bgcolor = ft.colors.BLACK,
                                    color = ft.colors.WHITE,
                                    on_click=on_simulate_click,
                                ),
                                ft.ElevatedButton(
                                    text="Descargar código G",
                                    bgcolor = ft.colors.BLACK,
                                    color = ft.colors.WHITE,
                                    on_click=lambda e:download_code(name=name,content=data, page=page)
                                ),
                            ]
                        )
                    )
                ]
            )
        )

    #Evento cuando el botón de "Generar código G" es precionado
    def on_generate_click(e):
        nonlocal codigo
        codigo = generar_codigo(coordinates)
        page.go("/generador")
        
    # Evento cuando el botón de "Simular Código G" es presionado
    def on_simulate_click(e):
        nonlocal codigo
        name = file_name_ref.current.value.split(": ")[1]
        name = name.split(".")[0]
        page.set_clipboard(codigo)
        page.launch_url('https://nraynaud.github.io/webgcode/')

    # Definir rutas y contenido
    def route_change(route):
        nonlocal codigo
        page.controls.clear()
        if page.route == "/":
            page.add(home_page())
        elif page.route == "/generador":
            original_file_name = file_name_ref.current.value.split(": ")[1]
            original_file_name = original_file_name.split(".")[0]
            page.add(generador_page(original_file_name, data=codigo))

    # Escuchar cambios de ruta
    page.on_route_change = route_change
    page.go(page.route)

# Ejecutar la aplicación Flet
ft.app(target=main)