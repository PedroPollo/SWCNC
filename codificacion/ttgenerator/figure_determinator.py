import networkx as nx
import numpy as np

def classify_lines(data):
    lines = data['lines']

    # Crear un grafo no dirigido
    G = nx.Graph()

    # Añadir las líneas como aristas en el grafo
    for line in lines:
        G.add_edge(tuple(line[0]), tuple(line[1]))

    # Encontrar todos los ciclos en el grafo
    cycles = list(nx.cycle_basis(G))

    # Crear una lista de figuras (ciclos)
    figures = []
    used_edges = set()  # Para rastrear las líneas que pertenecen a ciclos

    for cycle in cycles:
        figure = []
        for i in range(len(cycle)):
            start = cycle[i]
            end = cycle[(i + 1) % len(cycle)]
            line = [list(start), list(end)]
            figure.append(line)
            used_edges.add((tuple(start), tuple(end)))
            used_edges.add((tuple(end), tuple(start)))  # Para considerar ambas direcciones
        figures.append(figure)

    # Identificar las líneas que no pertenecen a ningún ciclo
    lone_lines = [line for line in lines if (tuple(line[0]), tuple(line[1])) not in used_edges]

    return {'figures': figures, 'lone_lines': lone_lines}

def new_data(orig_coor):
    coordinates = {
        'figures': [],
        'polylines': [],
        'lone_lines': [],
        'circles': [],
        'arcs': [],
    }
    for polyline in orig_coor['polylines']:
        coordinates['polylines'].append(polyline)
    
    for circle in orig_coor['circles']:
        coordinates['circles'].append(circle)
    
    for arc in orig_coor['arcs']:
        coordinates['arcs'].append(arc)
    
    return coordinates

def new_data_estructure(orig_coor, structured_data):
    coordinates = {
        'figures': [],
        'polylines': [],
        'lone_lines': [],
        'circles': [],
        'arcs': [],
    }
    
    for figure in structured_data['figures']:
        coordinates['figures'].append(figure)
    
    for polyline in orig_coor['polylines']:
        coordinates['polylines'].append(polyline)
    
    for line in structured_data['lone_lines']:
        coordinates['lone_lines'].append(line)
    
    for circle in orig_coor['circles']:
        coordinates['circles'].append(circle)
    
    for arc in orig_coor['arcs']:
        coordinates['arcs'].append(arc)
    
    return coordinates

def generate_gcode(coordinates):
    g_code = ""
    
    for figure in coordinates['figures']:
        g_code += f"G00 X{figure[0][0][0]} Y{figure[0][0][1]}\n"
        g_code += "M03\n"
        g_code += "G04 P5\n"
        g_code += "G01 "
        for line in figure:
            g_code += f"X{line[0][0]} Y{line[0][1]}\n"
        g_code += "M05\n"
    
    #print(g_code)