import tkinter as tk
import math
import sys
import re

def iniciar_simulador():
    if len(sys.argv) > 4:
        name = sys.argv[1]
        content = sys.argv[2]
        width = sys.argv[3]
        height = sys.argv[4]
    else:
        name = "Simulador de Código G"
        content = "G00 X100 Y100\nG01 X200 Y200\nM30"  # Un ejemplo básico de contenido
        width = 400
        height = 400

    ventana = tk.Tk()
    ventana.title(name)
    
    lienzo = tk.Canvas(ventana, width=int(width), height=int(height), bg='white')
    lienzo.pack()
    
    global x_actual, y_actual, instrucciones, indice_instruccion, movimiento_en_progreso
    x_actual, y_actual = None, None
    paso = 2
    indice_instruccion = 0
    movimiento_en_progreso = False
    
    lienzo.create_oval(0, 0, 0, 0, fill='black')
    
    instrucciones = []
    
    def ejecutar_codigo_g(codigo):
        global instrucciones, x_actual, y_actual
        instrucciones = [re.sub(r'\(.*?\)', '', line).strip() for line in codigo.split('\n') if line.strip() and not line.strip().startswith(';')]
        instrucciones = [line.split(' ', 1)[1] if line.startswith('N') and len(line.split(' ', 1)) > 1 else line for line in instrucciones]
        instrucciones = [line for line in instrucciones if not line.startswith('N')]
        for instruccion in instrucciones:
            partes = instruccion.strip().split()
            if partes[0] in ['G00', 'G01']:
                for sub_instruccion in partes[1:]:
                    if sub_instruccion.startswith('X'):
                        x_actual = float(sub_instruccion[1:])
                    elif sub_instruccion.startswith('Y'):
                        y_actual = float(sub_instruccion[1:])
                break
        
        if x_actual is None or y_actual is None:
            x_actual, y_actual = 100, 100

        lienzo.create_oval(x_actual - 2, y_actual - 2, x_actual + 2, y_actual + 2, fill='black')
        
        siguiente_movimiento()
    
    def siguiente_movimiento():
        global indice_instruccion, movimiento_en_progreso, x_actual, y_actual
        
        if indice_instruccion < len(instrucciones):
            instruccion = instrucciones[indice_instruccion].strip().split()
            comando = instruccion[0]
            
            if comando == 'M30':
                return
            elif comando in ['G00', 'G01']:
                x_destino, y_destino = x_actual, y_actual
                for sub_instruccion in instruccion[1:]:
                    if sub_instruccion.startswith('X'):
                        x_destino = float(sub_instruccion[1:])
                    elif sub_instruccion.startswith('Y'):
                        y_destino = float(sub_instruccion[1:])
                
                movimiento_en_progreso = True
                mover_apuntador_paso_a_paso(x_destino, y_destino)
                return
            elif comando in ['G02', 'G03']:
                x_destino, y_destino = x_actual, y_actual
                i, j = 0, 0
                for sub_instruccion in instruccion[1:]:
                    if sub_instruccion.startswith('X'):
                        x_destino = float(sub_instruccion[1:])
                    elif sub_instruccion.startswith('Y'):
                        y_destino = float(sub_instruccion[1:])
                    elif sub_instruccion.startswith('I'):
                        i = float(sub_instruccion[1:])
                    elif sub_instruccion.startswith('J'):
                        j = float(sub_instruccion[1:])
                
                centro_x = x_actual + i
                centro_y = y_actual + j
                radio = math.sqrt(i**2 + j**2)
                sentido = 1 if comando == 'G02' else -1
                
                # Identificar si es entrada, círculo completo o salida tangente
                if indice_instruccion == 0:  # primer G02: entrada tangente
                    angulo_inicial = math.atan2(y_actual - centro_y, x_actual - centro_x)
                    angulo_final = angulo_inicial + 0.2 * sentido  # Pequeña sección para entrada
                elif indice_instruccion == 1:  # segundo G02: círculo completo
                    angulo_inicial = math.atan2(y_actual - centro_y, x_actual - centro_x)
                    angulo_final = angulo_inicial + 2 * math.pi * sentido  # Círculo completo
                elif indice_instruccion == 2:  # tercer G02: salida tangente
                    angulo_inicial = math.atan2(y_actual - centro_y, x_actual - centro_x)
                    angulo_final = angulo_inicial + 0.2 * sentido  # Pequeña sección para salida

                movimiento_en_progreso = True
                dibujar_circulo_paso_a_paso(centro_x, centro_y, radio, sentido, angulo_inicial, angulo_final)
                return
            elif comando == 'G04':
                for sub_instruccion in instruccion[1:]:
                    if sub_instruccion.startswith('P'):
                        pausa = int(sub_instruccion[1:])
                        indice_instruccion += 1
                        ventana.after(pausa * 1000, siguiente_movimiento)
                        return
            else:
                indice_instruccion += 1
                siguiente_movimiento()

            indice_instruccion += 1
            siguiente_movimiento()
    
    def mover_apuntador_paso_a_paso(x_destino, y_destino):
        global x_actual, y_actual, movimiento_en_progreso, indice_instruccion
        
        distancia = math.sqrt((x_destino - x_actual)**2 + (y_destino - y_actual)**2)
        
        if distancia > paso:
            dx = (x_destino - x_actual) / distancia * paso
            dy = (y_destino - y_actual) / distancia * paso
            
            lienzo.create_line(x_actual, y_actual, x_actual + dx, y_actual + dy, fill='black')
            
            x_actual += dx
            y_actual += dy
            
            ventana.after(20, mover_apuntador_paso_a_paso, x_destino, y_destino)
        else:
            lienzo.create_line(x_actual, y_actual, x_destino, y_destino, fill='black')
            x_actual, y_actual = x_destino, y_destino
            movimiento_en_progreso = False
            
            indice_instruccion += 1
            siguiente_movimiento()
    
    def dibujar_circulo_paso_a_paso(centro_x, centro_y, radio, sentido, angulo_inicial, angulo_final):
        global x_actual, y_actual, indice_instruccion

        num_puntos = 36
        paso_angulo = (angulo_final - angulo_inicial) / num_puntos

        puntos = []
        for i in range(num_puntos + 1):
            angulo = angulo_inicial + i * paso_angulo
            x = centro_x + radio * math.cos(angulo)
            y = centro_y + radio * math.sin(angulo)
            puntos.append((x, y))

        def dibujar_segmento(indice):
            global indice_instruccion, x_actual, y_actual
            if indice < len(puntos) - 1:
                x1, y1 = puntos[indice]
                x2, y2 = puntos[indice + 1]
                lienzo.create_line(x1, y1, x2, y2, fill='black')
                ventana.after(20, dibujar_segmento, indice + 1)
            else:
                x_actual, y_actual = puntos[-1]
                movimiento_en_progreso = False
                indice_instruccion += 1
                siguiente_movimiento()
                
        dibujar_segmento(0)
    
    ejecutar_codigo_g(content)
    
    ventana.mainloop()

if __name__ == "__main__":
    iniciar_simulador()
