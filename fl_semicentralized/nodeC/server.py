import socket
import os
import sys

from utils import checkConvergence
from .connections import initial, get_models, send_avg_model, sendconverge
import traceback

    
PATH_MODELS = os.path.join('/app/nodeC', 'models')
PATH_RECVMODELS = os.path.join(PATH_MODELS, 'recv')
PATH_AVGMODELS = os.path.join(PATH_MODELS, 'avg')

os.makedirs(PATH_RECVMODELS, exist_ok=True)
os.makedirs(PATH_AVGMODELS, exist_ok=True)


def run(connections, idxs, sock, ROUNDS, NCLIENTS, PARAMS, CSV_MODELS, f1_scores, accs):

    # Fase 1: Inicialización y envío de modelo inicial
    initial(sock, connections, idxs, NCLIENTS, PARAMS, PATH_AVGMODELS, CSV_MODELS)
    for round in range(ROUNDS):
        # Fase 2: Recepción de modelos entrenados
        get_models(connections, idxs, PATH_RECVMODELS, f1_scores, accs)

        converged = checkConvergence(f1_scores, 3)
        sendconverge(connections, converged)
        if converged:
            print(f"Convergencia alcanzada en ronda {round}!!!", flush=True)
            break
        
        # Fase 3: Promediado y envío del modelo global
        send_avg_model(connections, PATH_RECVMODELS, PATH_AVGMODELS, round, CSV_MODELS)
        
        print(f"\n[✓] Ronda {round} completado exitosamente", flush=True)



def server(PORT, ROUNDS, NCLIENTS, PARAMS, f1_scores, accs):

    NODE_ID = os.getenv("NODE_ID")

    HOST = '0.0.0.0'
    CSV_MODELS=f'models_path_{NODE_ID}.csv'


    connections = []
    idxs = []
    
    try:
        print("="*60, flush=True)
        print("      SERVIDOR DE APRENDIZAJE FEDERADO", flush=True)
        print("="*60, flush=True)
        print(f"Host: {HOST}", flush=True)
        print(f"Puerto: {PORT}", flush=True)
        print(f"Clientes esperados: {NCLIENTS}", flush=True)
        print("="*60, flush=True)
        
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((HOST, PORT))
        sock.listen(NCLIENTS)
        run(connections, idxs, sock, ROUNDS, NCLIENTS, PARAMS, CSV_MODELS, f1_scores, accs)
        
    except PermissionError:
        print(f"\n[!] Error: No tienes permisos para usar el puerto {PORT}", flush=True)
        print("    Intenta usar un puerto mayor a 1024 o ejecuta con sudo", flush=True)
        sys.exit(1)
        
    except ConnectionRefusedError as cre:
        print(f"\n[!] Error de conexión: {cre}", flush=True)
        sys.exit(1)
        
    except KeyboardInterrupt:
        print("\n\n[!] Servidor interrumpido por el usuario", flush=True)
        sys.exit(0)
        
    except Exception as e:
        print(f"\n[!] Ocurrió un error: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)
        
    finally:
        # Cerrar todas las conexiones
        for (conn, _) in connections:
            try:
                conn.close()
            except:
                pass
        
        try:
            sock.close()
        except:
            pass
        
        print("\n[✓] Servidor cerrado", flush=True)