import ezdxf
import sys
import json

def extraer_path():
    if len(sys.argv) > 1:
        return sys.argv[1]
    else:
        print("No se paso el PATH del archivo")
        return None

def extraer_coordenadas(path):
    try:
        doc = ezdxf.readfile(path)
        msp = doc.modelspace()
        coordinates = {
        'lines': [],
        'circles': [],
        'arcs': [],
        'polylines': [],
        'splines': [],
        'ellipses': [],
        'hatches': []
        }
        
        #Recorrer las entidades del archivo DXF
        for entity in msp:
            if entity.dxftype() == 'LINE':  # Líneas
                start = (entity.dxf.start.x, entity.dxf.start.y)  # Coordenada de inicio como lista
                end = (entity.dxf.end.x, entity.dxf.end.y)  # Coordenada final como lista
                
                coordinates['lines'].append([tuple(start), tuple(end)])  # Se agregan como tuplas


            elif entity.dxftype() == 'CIRCLE': #Circulos
                center = (entity.dxf.center.x, entity.dxf.center.y)
                radius = entity.dxf.radius
                coordinates['circles'].append({'center': center, 'radius': radius})

            elif entity.dxftype() == 'ARC': #Arcos
                center = (entity.dxf.center.x, entity.dxf.center.y)
                radius = entity.dxf.radius
                start_angle = entity.dxf.start_angle  # En grados
                end_angle = entity.dxf.end_angle  # En grados
                coordinates['arcs'].append({
                    'center': center,
                    'radius': radius,
                    'start_angle': start_angle,
                    'end_angle': end_angle
                })
                
            elif entity.dxftype() == 'LWPOLYLINE':  # Polilíneas ligeras
                points = []
                for point in entity:
                    points.append((point[0], point[1]))  # X, Y de cada vértice
                coordinates['polylines'].append(points)
            
            elif entity.dxftype() == 'SPLINE':  # Splines
                control_points = []
                for point in entity.control_points:
                    control_points.append((point[0], point[1]))  # X, Y de cada punto de control
                coordinates['splines'].append(control_points)
                
            elif entity.dxftype() == 'ELLIPSE':  # Elipses
                center = (entity.dxf.center.x, entity.dxf.center.y)
                major_axis = (entity.dxf.major_axis.x, entity.dxf.major_axis.y)
                ratio = entity.dxf.ratio  # Proporción del eje mayor al menor
                coordinates['ellipses'].append({
                    'center': center,
                    'major_axis': major_axis,
                    'ratio': ratio
                })
                
            elif entity.dxftype() == 'HATCH':  # Hatches (sombras, rellenos)
                # Un hatch puede tener múltiples loops (bordes cerrados)
                loops = []
                for path in entity.paths:
                    loop_points = []
                    for edge in path.edges:
                        if hasattr(edge, 'start') and hasattr(edge, 'end'):
                            loop_points.append((edge.start.x, edge.start.y))
                            loop_points.append((edge.end.x, edge.end.y))
                    loops.append(loop_points)
                coordinates['hatches'].append(loops)
        
        return coordinates
    except Exception as e:
        print(f"No se pudo leer el documento\nError: {e}")
        return

if __name__ == "__main__":
    path = extraer_path()
    if path:
        coordenadas = extraer_coordenadas(path)
        # Convertir el diccionario a JSON y imprimir para poder capturar en el proceso principal
        print(json.dumps(coordenadas))