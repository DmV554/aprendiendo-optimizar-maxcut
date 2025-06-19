import numpy as np
from src.MAXCUT import MaxCutInstance


def load_rud_file_to_instance(file_path: str):
    """
    Carga una instancia Max-Cut desde un archivo .rud (formato Gset)
    y crea un objeto MaxCutInstance inicializado con una matriz de pesos.

    Args:
        file_path (str): Ruta al archivo .rud.

    Returns:
        MaxCutInstance: Objeto con los datos cargados.

    Raises:
        FileNotFoundError: Si el archivo no se encuentra.
        ValueError: Si el formato es incorrecto.
    """
    num_vertices = 0
    num_edges_expected = 0
    weights_matrix = None

    try:
        with open(file_path, 'r') as f:
            # Leer la primera línea: num_vertices num_edges
            first_line = f.readline().split()
            if len(first_line) < 2:
                raise ValueError("La primera línea debe contener num_vertices y num_edges")
            num_vertices = int(first_line[0])
            num_edges_expected = int(first_line[1])

            # Inicializar la matriz de pesos con ceros
            weights_matrix = np.zeros((num_vertices, num_vertices), dtype=float) # dtype=int si pesos son enteros

            # Leer las líneas de las aristas y poblar la matriz
            edge_count = 0
            for line in f:
                parts = line.split()
                if len(parts) == 3:
                    # Convertir a enteros y ajustar índices (1-based -> 0-based)
                    node1 = int(parts[0]) - 1
                    node2 = int(parts[1]) - 1
                    # Usar float si los pesos pueden tener decimales, sino int
                    weight = float(parts[2])

                    # Validar índices (0 a num_vertices-1)
                    if 0 <= node1 < num_vertices and 0 <= node2 < num_vertices:
                        # Poblar la matriz (simétrica para grafos no dirigidos)
                        weights_matrix[node1, node2] = weight
                        weights_matrix[node2, node1] = weight
                        edge_count += 1
                    else:
                        print(f"Warning: Edge ({node1+1}, {node2+1}) tiene índices fuera de rango [1, {num_vertices}]. Ignorando línea: {line.strip()}")
                elif line.strip():
                    print(f"Warning: Línea con formato incorrecto ignorada: {line.strip()}")

        # Verificar número de aristas (considerando que contamos cada par una vez al leer)
        if edge_count != num_edges_expected:
             print(f"Warning: Se esperaban {num_edges_expected} aristas, pero se procesaron {edge_count} líneas de arista.")

        # Crear y retornar la instancia usando la matriz de pesos
        return weights_matrix

    except FileNotFoundError:
        print(f"Error: Archivo no encontrado en {file_path}")
        raise
    except ValueError as e:
        print(f"Error: Problema al procesar el archivo {file_path} - {e}")
        raise
    except Exception as e:
        print(f"Error inesperado al leer {file_path}: {e}")
        raise
