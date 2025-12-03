import socket
import sys
import uuid
import os
from .model_build import FederatedModel
from .connections import *
import traceback

# Configuración
NODE = os.environ.get('NODE_ID', 'default')

PATH_MAIN = '/app/nodex'
PATH_MODELS = os.path.join(PATH_MAIN, f'models_{NODE}')
PATH_DATA = os.path.join("/app/diabetes_divided", f"diabetes_{int(NODE)}.csv")


def run(sock, HOST, PORT, ROUNDS):

    models_info = []

    # Generar ID único del nodo
    node_id = str(uuid.uuid4())
    print(f"\n[>] ID de nodo: {node_id}", flush=True)
    
    # Inicializar modelo federado
    nn = FederatedModel(PATH_DATA)
    
    # Conectar al servidor
    print(f"\n[>] Conectando a {HOST}:{PORT}...", flush=True)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((HOST, PORT))
    print("[✓] Conectado al servidor", flush=True)
    
    # Enviar ID del nodo (debe ser exactamente 36 caracteres)
    print("[>] Enviando ID de nodo...", flush=True)
    node_id_padded = node_id.ljust(36)[:36]  # Asegurar 36 caracteres
    sock.send(node_id_padded.encode('utf-8'))
    print("[✓] ID de nodo enviado", flush=True)
    
    # RONDA 0: Recibir modelo inicial y entrenar
    for round in range(ROUNDS):

        train = not(round == ROUNDS - 1)

        model_info = get_model(sock, nn, round, PATH_MODELS=PATH_MODELS, train=train)
        
        if model_info is None:
            print(f"[!] Error en ronda {round}, abortando...", flush=True)
            sys.exit(1)
            
        models_info.append(model_info)
        
        if round == ROUNDS - 1: break 
        # Enviar modelo entrenado al servidor
        best_model_info = max(models_info, key=lambda x: x['f1_score'])
        send_model(sock, best_model_info)


        print("Recibiendo confirmación...")
        converged = (sock.recv(1)== b"\x01")
        print(f"Confirmación recibida {converged}")

        if converged: return
    
    # Guardar información de modelos
    save_models_info(models_info, NODE, PATH_MAIN)
    
    print("\n" + "="*60, flush=True)
    print("  PROCESO COMPLETADO EXITOSAMENTE", flush=True)
    print("="*60, flush=True)
    print(f"F1-Score final: {models_info[-1]['f1_score']:.4f}", flush=True)
    print("="*60 + "\n", flush=True)



def client(HOST, PORT, ROUNDS):
    
    # Crear directorios necesarios
    os.makedirs(PATH_MODELS, exist_ok=True)
    
    # Validar que existan los datos
    if not os.path.exists(PATH_DATA):
        print(f"[!] Error: No se encontró el archivo de datos: {PATH_DATA}", flush=True)
        sys.exit(1)
    
    print("="*60, flush=True)
    print("      CLIENTE DE APRENDIZAJE FEDERADO", flush=True)
    print("="*60, flush=True)
    print(f"Nodo: {NODE}", flush=True)
    print(f"Servidor: {HOST}:{PORT}", flush=True)
    print("="*60, flush=True)
    
    sock = None
    
    try:

        run(sock, HOST, PORT, ROUNDS)
        
    except ConnectionRefusedError:
        print(f"\n[!] Error: No se pudo conectar a {HOST}:{PORT}", flush=True)
        print("    ¿Está el servidor ejecutándose?", flush=True)
        sys.exit(1)
        
    except FileNotFoundError as e:
        print(f"\n[!] Error: Archivo no encontrado - {e}", flush=True)
        sys.exit(1)
        
    except KeyboardInterrupt:
        print("\n\n[!] Cliente interrumpido por el usuario", flush=True)
        sys.exit(0)
        
    except Exception as e:
        print(f"\n[!] Ocurrió un error: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)
        
    finally:
        # Cerrar socket si está abierto
        if sock:
            try:
                sock.close()
                print("[✓] Conexión cerrada", flush=True)
            except:
                pass