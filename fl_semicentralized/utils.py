import csv
import os
import random

def unificar_metricas_csv(node_id):
    """
    Une el archivo CSV (modelo), TXT (f1-scores) y TXT (accuracies),
    saltando el ID del nodo actual en los encabezados de ambos TXT.
    """
    # 1. Nombres de archivos
    csv_filename = f"models_path_{node_id}.csv"
    txt_f1scores = f"f1scores{node_id}.txt"
    txt_accs = f"accs{node_id}.txt"  # <--- Nuevo archivo
    output_filename = f"full_metrics_node_{node_id}.csv"

    # Verificar que existan todos
    if not os.path.exists(csv_filename) or not os.path.exists(txt_f1scores) or not os.path.exists(txt_accs): return

    try:
        with open(csv_filename, mode='r', newline='', encoding='utf-8') as f_csv:
            reader_csv = csv.reader(f_csv)
            try:
                header_csv = next(reader_csv) 
            except StopIteration:
                print(f"[!] El archivo {csv_filename} está vacío.")
                return
            rows_csv = list(reader_csv)

        with open(txt_f1scores, mode='r', newline='', encoding='utf-8') as f_f1:
            reader_f1 = csv.reader(f_f1)
            rows_f1 = list(reader_f1)

        if not rows_f1:
            print(f"[!] El archivo {txt_f1scores} está vacío.")
            return

        with open(txt_accs, mode='r', newline='', encoding='utf-8') as f_acc:
            reader_acc = csv.reader(f_acc)
            rows_acc = list(reader_acc)

        if not rows_acc:
            print(f"[!] El archivo {txt_accs} está vacío.")
            return
        
        # --- Generar Headers para F1 ---
        num_cols_f1 = len(rows_f1[0])
        header_f1 = []
        contador = 1
        while len(header_f1) < num_cols_f1:
            if contador == int(node_id):
                contador += 1
                continue
            header_f1.append(f"f1-score_node_{contador}")
            contador += 1

        # --- Generar Headers para Accuracy (Misma lógica) ---
        num_cols_acc = len(rows_acc[0])
        header_acc = []
        contador = 1
        while len(header_acc) < num_cols_acc:
            if contador == int(node_id):
                contador += 1
                continue
            header_acc.append(f"acc_node_{contador}") # Nombre diferente
            contador += 1
        
        # Unimos las 3 listas de encabezados
        full_header = header_csv + header_f1 + header_acc

        with open(output_filename, mode='w', newline='', encoding='utf-8') as f_out:
            writer = csv.writer(f_out)
            
            # Escribir cabecera total
            writer.writerow(full_header)
            
            # Escribir filas combinadas
            # Usamos zip con 3 listas: CSV, F1 y ACC
            # zip parará cuando la lista más corta se acabe (deberían ser iguales)
            for row_c, row_f1, row_a in zip(rows_csv, rows_f1, rows_acc):
                writer.writerow(row_c + row_f1 + row_a)

        print(f"[✓] Archivo generado: {output_filename}")
        print(f"    Columnas: {len(full_header)}")

    except Exception as e:
        print(f"[!] Error procesando nodo {node_id}: {e}")


def checkConvergence(scores: list[list[float]], patience: int, threshold: float = 0.01) -> bool:
    """
    Comprueba si las métricas han convergido comparando los valores
    de hace `patience` rondas con la ronda más reciente.

    Retorna True si la diferencia absoluta para cada nodo está por
    debajo del `threshold`, False en caso contrario.
    """
    if len(scores) < patience:
        return False

    older_scores = scores[-patience]
    latest_scores = scores[-1]

    # Validación básica de consistencia
    if not older_scores or not latest_scores or len(older_scores) != len(latest_scores):
        return False

    for node in range(len(latest_scores)):
        diff = older_scores[node] - latest_scores[node]
        if abs(diff) > threshold:
            return False

    return True

def select_leader(nodes, round):
    """
    Selecciona un nodo líder basado en probabilidad ponderada por sus capacidades.
    Usa la ronda como semilla para asegurar que todos los nodos elijan al mismo ganador.
    """
    
    # 1. Calcular el puntaje (score) para cada nodo
    scores = []
    for x in nodes:
        score = (
            0.5 * (0.5 * x['net_up'] + 0.5 * x['net_down']) +
            0.3 * x['ram'] +
            0.35 * x['cpu_mhz'] +
            0.2 * int(x['gpu']) +
            0.1 * (1 / int(x['id']))
        )
        scores.append(score)

    # 2. Configurar la semilla compartida
    random.seed(round)

    # 3. Selección ponderada (Weighted Choice)
    ganador = random.choices(nodes, weights=scores, k=1)[0]
    
    random.seed(None) 
    
    return ganador