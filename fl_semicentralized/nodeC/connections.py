import os
from .avg_model import average_models, build_model
import threading
import struct

def recv_exact(sock, n_bytes):
    """Asegura recibir exactamente n_bytes del socket"""
    data = b''
    while len(data) < n_bytes:
        packet = sock.recv(n_bytes - len(data))
        if not packet:
            return None 
        data += packet
    return data

def sendconverge(connections, end_signal):
    for i, (conn, addr) in enumerate(connections):
        try:
            print("Enviando confirmación...", flush=True)
            conn.sendall(b"\x01" if end_signal else b"\x00")
        except Exception as e:
            print(f"   [!] Error enviando al cliente {i+1} ({addr}): {e}", flush=True)

def handle_client(conn, addr, idx, PATH_RECVMODELS, f1scores, accs):
    try:
        model_f1score_bytes = recv_exact(conn, 8)
        if not model_f1score_bytes: return
        model_f1score = struct.unpack('!d', model_f1score_bytes)[0]
        f1scores.append(model_f1score)
        print(f"F1-score del modelo recibido de cliente {idx}, F1-score: {model_f1score}", flush=True)

        model_acc_bytes = recv_exact(conn, 8)
        if not model_acc_bytes: return
        model_acc = struct.unpack('!d', model_acc_bytes)[0]
        accs.append(model_acc)
        print(f"F1-score del modelo recibido de cliente {idx}, Accuracy: {model_acc}", flush=True)

        model_size_bytes = recv_exact(conn, 8)
        if not model_size_bytes: return
        model_size = int.from_bytes(model_size_bytes, 'big')
        print(f"Tamaño del modelo recibido de cliente {idx}: {model_size} bytes", flush=True)

        filename = os.path.join(PATH_RECVMODELS, f'model_node_{idx}.keras')
        with open(filename, "wb") as fo:
            received_bytes = 0
            while received_bytes < model_size:
                data = conn.recv(min(4096, model_size - received_bytes))
                if not data: break
                fo.write(data)
                received_bytes += len(data)
        
        print(f"[✓] Modelo recibido del nodo {idx}, ip: {addr}", flush=True)

    except Exception as e:
        print(f"[!] Error recibiendo modelo del nodo {idx}: {e}", flush=True)

def get_models(connections, idxs, PATH_RECVMODELS, scores_f1, scores_acc):
    print("\n[>] Esperando modelos de los clientes...", flush=True)
    threads = []
    f1scores = []
    accs = []
    for conn, addr, idx in zip([c[0] for c in connections], [c[1] for c in connections], idxs):
        t = threading.Thread(target=handle_client, args=(conn, addr, idx, PATH_RECVMODELS, f1scores, accs))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    scores_f1.append(f1scores)
    scores_acc.append(accs)
    print("[✓] Todos los modelos recibidos", flush=True)

def send_avg_model(connections, PATH_RECVMODELS, PATH_AVGMODELS, ROUND_number, CSV_MODELS):
    print("\n[>] Promediando modelos...", flush=True)
    
    # IMPORTANTE: average_models ahora propaga excepciones, no hace exit()
    try:
        avg_model_path = average_models(PATH_RECVMODELS, PATH_AVGMODELS)
    except Exception as e:
        print(f"[!] Error crítico promediando: {e}", flush=True)
        return

    if avg_model_path is None:
        print("[!] El promediado retornó None.", flush=True)
        return

    with open(CSV_MODELS, 'a') as f:
        if os.path.getsize(CSV_MODELS) == 0:
            f.write("round,avg_model_path\n")
        f.write(f"{ROUND_number},{avg_model_path}\n")
    
    print("[>] Enviando modelo promediado a todos los clientes...", flush=True)
    
    for i, (conn, addr) in enumerate(connections):
        try:
            print("Enviando tamaño del modelo...", flush=True)
            model_size = os.path.getsize(avg_model_path)
            conn.sendall(model_size.to_bytes(8, 'big'))
            print(f"Tamaño enviado!!! -> {model_size}bytes", flush=True)
            with open(avg_model_path, 'rb') as f:
                while chunk := f.read(4096):
                    conn.sendall(chunk)
            
            print(f"   [✓] Modelo enviado al cliente {i+1}", flush=True)
        except Exception as e:
            print(f"   [!] Error enviando al cliente {i+1} ({addr}): {e}", flush=True)
    
    for filemodel in os.listdir(PATH_RECVMODELS):
        os.remove(os.path.join(PATH_RECVMODELS, filemodel))
    
    print("[✓] Todos los clientes actualizados.", flush=True)


def initial(sock, connections, idxs, NCLIENTS, PARAMS, PATH_AVGMODELS, CSV_MODELS):
    """Inicializa las conexiones y envía el modelo inicial"""
    
    # --- CORRECCIÓN CRÍTICA: PREPARAR MODELO ANTES DE ACEPTAR CLIENTES ---
    print("\n[>] Preparando modelo inicial (antes de conectar)...", flush=True)
    first_model = None
    
    if os.path.exists(CSV_MODELS) and os.path.getsize(CSV_MODELS) > 0:
        try:
            with open(CSV_MODELS, 'r') as f:
                lines = f.readlines()
                if len(lines) > 1:
                    last_line = lines[-1]
                    parts = last_line.strip().split(',')
                    if len(parts) >= 2:
                        first_model = parts[1]
        except Exception as e:
            print(f"[!] Error leyendo CSV models: {e}")

    if not first_model or not os.path.exists(first_model):
        print("[>] Creando nuevo modelo inicial...", flush=True)
        first_model = os.path.join(PATH_AVGMODELS, "initial.keras")
        # Esto puede fallar si falta RAM, pero al menos falla ANTES de abrir sockets
        build_model(PARAMS, first_model)
        print(f"[✓] Modelo inicial creado: {first_model}", flush=True)
    else:
        print(f"[✓] Usando modelo existente: {first_model}", flush=True)

    # ---------------------------------------------------------------------

    print(f"\n[>] Esperando {NCLIENTS} cliente(s)...", flush=True)
    
    # Aceptar conexiones
    for i in range(NCLIENTS):
        conn, addr = sock.accept()
        connections.append((conn, addr))
        print(f'[+] Cliente {i+1} conectado desde {addr[0]}:{addr[1]}', flush=True)
    
    print("\n[>] Recibiendo IDs de nodos...", flush=True)
    
    for i, (conn, addr) in enumerate(connections):
        try:
            idx = conn.recv(36).decode('utf-8').strip()
            if not idx:
                print(f"   [!] El cliente {i+1} no envió su ID", flush=True)
                # Generar ID temporal si falla para no romper la lógica
                idx = f"unknown_{i}"
            idxs.append(idx)
            print(f"   [✓] Cliente {i+1} ID: {idx}", flush=True)
        except Exception as e:
            print(f"   [!] Error recibiendo ID: {e}", flush=True)
            idxs.append(f"error_{i}")
    
    print("[>] Enviando modelo inicial a los clientes...", flush=True)
    
    for i, (conn, addr) in enumerate(connections):
        try:
            model_size = os.path.getsize(first_model)
            conn.sendall(model_size.to_bytes(8, 'big'))
            
            with open(first_model, 'rb') as f:
                while chunk := f.read(4096):
                    conn.sendall(chunk)
            
            print(f"   [✓] Modelo inicial enviado al cliente {i+1}", flush=True)
        except Exception as e:
            print(f"   [!] Error enviando al cliente {i+1} ({addr}): {e}", flush=True)