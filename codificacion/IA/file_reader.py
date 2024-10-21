import ezdxf

def extract_coordinates_from_dxf(file_path):
    doc = ezdxf.readfile(file_path)
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
    
    # Recorrer entidades del archivo DXF
    for entity in msp:
        if entity.dxftype() == 'LINE':  # Líneas
            start = (entity.dxf.start.x, entity.dxf.start.y)
            end = (entity.dxf.end.x, entity.dxf.end.y)
            coordinates['lines'].append((start, end))

        elif entity.dxftype() == 'CIRCLE':  # Círculos
            center = (entity.dxf.center.x, entity.dxf.center.y)
            radius = entity.dxf.radius
            coordinates['circles'].append({'center': center, 'radius': radius})

        elif entity.dxftype() == 'ARC':  # Arcos
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

        elif entity.dxftype() == 'TEXT':  # Texto simple
            text = entity.dxf.text
            insert_point = (entity.dxf.insert.x, entity.dxf.insert.y)
            coordinates['texts'].append({
                'text': text,
                'insert_point': insert_point
            })

        elif entity.dxftype() == 'MTEXT':  # Texto multilínea
            text = entity.text
            insert_point = (entity.dxf.insert.x, entity.dxf.insert.y)
            coordinates['texts'].append({
                'text': text,
                'insert_point': insert_point
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

# Uso de la función
coordinates = extract_coordinates_from_dxf('EJEMPLO.dxf')
print(coordinates)
