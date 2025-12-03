import socket
import json
import os
import time
import threading
import subprocess
import csv

from utils import select_leader

# --- CONFIGURACIÓN ---
NODE_ID = int(os.getenv("NODE_ID", 1))
BIND_PORT = int(os.getenv("BIND_PORT", 5000))
BUFFER_SIZE = 4096

# Directorios y Rutas
NODE_DIR = f"nodo{NODE_ID}"

# Rutas ABSOLUTAS para evitar problemas con os.chdir repetidos
CSV_METRICS = os.path.join(os.getcwd(), NODE_DIR, 'all_metrics_node.csv')
CLIENT_METRICS_JSON = os.path.join(NODE_DIR, f'client_metrics_{NODE_ID}.json') # El script bash lo genera en el cwd

# --- VARIABLES GLOBALES DE ESTADO ---
n_sent = 0
n_received = 0
csv_lock = threading.Lock()
stop_event = threading.Event() # Bandera para detener el servidor al final de la ronda

# --- FUNCIÓN AUXILIAR PARA GUARDAR EN CSV ---
def guardar_en_csv(sender_id, sender_ip, metrics):
    fila_csv = {
        "node_id": sender_id,
        "ip": sender_ip,
        "ram_disponible_mb": metrics.get('ram_disponible_mb'),
        "disco_disponible_mb": metrics.get('disco_disponible_mb'),
        "cpu_cores": metrics.get('cpu_cores'),
        "cpu_mhz": metrics.get('cpu_mhz'),
        "gpu_activa": metrics.get('gpu_activa'),
        "red_descarga_mbps": metrics.get('red_descarga_mbps'),
        "red_subida_mbps": metrics.get('red_subida_mbps')
    }

    with csv_lock:
        archivo_existe = os.path.isfile(CSV_METRICS)
        try:
            with open(CSV_METRICS, 'a', newline='') as f:
                campos = [
                    "node_id", "ip", 
                    "ram_disponible_mb", "disco_disponible_mb", 
                    "cpu_cores", "cpu_mhz", "gpu_activa", 
                    "red_descarga_mbps", "red_subida_mbps"
                ]
                writer = csv.DictWriter(f, fieldnames=campos)
                if not archivo_existe:
                    writer.writeheader()
                writer.writerow(fila_csv)
                # print(f"[CSV] Datos del Nodo {sender_id} guardados.")
        except Exception as e:
            print(f"[ERROR CSV] {e}", flush=True)

# --- SERVIDOR TCP ---
def iniciar_servidor(peers):
    global n_received
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        server_socket.bind(('0.0.0.0', BIND_PORT))
        server_socket.listen(len(peers))
        # Timeout vital: permite verificar stop_event cada 1 seg para poder cerrar el hilo
        server_socket.settimeout(1.0) 
        print(f"[SERVIDOR] Escuchando en puerto {BIND_PORT}...", flush=True)
        
        while not stop_event.is_set():
            try:
                client_sock, addr = server_socket.accept()
            except socket.timeout:
                continue # Si nadie conecta en 1 seg, verificamos stop_event y seguimos
            except OSError:
                break # El socket se cerró

            try:
                datos_totales = b""
                client_sock.settimeout(2.0)
                while True:
                    try:
                        parte = client_sock.recv(BUFFER_SIZE)
                        if not parte: break 
                        datos_totales += parte
                    except socket.timeout:
                        break
                
                if datos_totales:
                    mensaje = json.loads(datos_totales.decode('utf-8'))
                    sender_id = mensaje.get('node_id')
                    metrics = mensaje.get('metrics', {})

                    guardar_en_csv(sender_id, addr[0], metrics)
                    n_received += 1

                    print(f"[SERVIDOR] Datos recibidos de Nodo {sender_id} ({addr[0]}). Total recibidos: {n_received}, Message: {mensaje}", flush=True)
                        
            except Exception as e:
                print(f"[ERROR SERVER] {e}", flush=True)
            finally:
                client_sock.close()
                
    except Exception as e:
        print(f"[FATAL SERVER] {e}", flush=True)
    finally:
        server_socket.close()
        print("[SERVIDOR] Socket cerrado.", flush=True)

# --- CLIENTE TCP ---
def iniciar_cliente(peers):
    global n_sent
    # Pequeño delay escalonado para no saturar la red al inicio
    time.sleep(NODE_ID * 2)
    
    # Nos aseguramos de correr el script bash
    script_path = "/app/metrics.sh" if os.path.exists("/app/metrics.sh") else "./metrics.sh"
    subprocess.run(["bash", script_path], stdout=subprocess.DEVNULL)
    
    try:
        # metrics.sh genera metrics_node.json en el CWD actual
        # Movemos/Leemos el json generado
        json_path = CLIENT_METRICS_JSON # Nombre que usa tu .sh
        if not os.path.exists(json_path):
             json_path = CLIENT_METRICS_JSON

        with open(json_path, 'r') as f:
            metrics = json.load(f)
            
        # Guardar propios datos
        guardar_en_csv(NODE_ID, "LOCALHOST", metrics)

    except Exception as e:
        print(f"[ERROR CLIENTE] No se pudo leer metrics: {e}", flush=True)
        return

    payload = json.dumps({"node_id": NODE_ID, "metrics": metrics}).encode('utf-8')

    for peer in peers:
        if not peer.strip(): continue
        target_host, target_port = peer.split(':')
        
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(2) 
            s.connect((target_host, int(target_port)))
            s.sendall(payload)
            s.close() 
            n_sent += 1 
        except Exception as e:
            pass 
            print(f" -> Fallo envio a {target_host}: {e}", flush=True)

# --- SELECCIONADOR ---
def seleccionar_servidor(csv_file, round):
    nodos = []
    try:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                nodo = {
                    "id": row['node_id'],
                    "ip": row['ip'],
                    "ram": float(row['ram_disponible_mb']),
                    "net_up": float(row['red_subida_mbps']),
                    "net_down": float(row['red_descarga_mbps']),
                    "cpu_mhz": float(row['cpu_mhz']),
                    "gpu": 1 if row.get('gpu_activa') == 'true' else 0 # Ojo con el string 'true'/'false'
                }
                nodos.append(nodo)
    except Exception:
        return 1 # Default si falla

    if not nodos: return 1

    # Lógica de ganador
    ganador = select_leader(nodos, round)
    return ganador['id']

# --- MAIN COORDINATE FUNCTION ---
def coordinate(peers, round):
    global n_sent, n_received, stop_event
    
    # 1. RESETEAR ESTADO (CRÍTICO PARA MULTIPLES RONDAS)
    n_sent = 0
    n_received = 0
    stop_event.clear()
    
    # 2. LIMPIAR CSV VIEJO
    if os.path.exists(CSV_METRICS):
        os.remove(CSV_METRICS)

    # 3. Iniciar Servidor en hilo
    hilo_servidor = threading.Thread(target=iniciar_servidor, args=(peers,), daemon=True)
    hilo_servidor.start()
    
    # 4. Iniciar Cliente
    iniciar_cliente(peers)
    
    
    print(f"--- Coordinando Ronda... Esperando {len(peers)} peers ---", flush=True)
    
    while True:
        # Salir si tenemos todos los datos o paso el tiempo
        if n_received >= len(peers):
            break
        time.sleep(1)

    # 6. DETENER SERVIDOR LIMPIAMENTE
    stop_event.set() # Esto le dice al hilo que salga del while
    hilo_servidor.join(timeout=2) # Esperamos a que cierre el socket

    # 7. Seleccionar Servidor
    nodo_id = seleccionar_servidor(CSV_METRICS, round)
    
    return str(nodo_id) # Retornamos string porque tu main hace int(nodo_id)