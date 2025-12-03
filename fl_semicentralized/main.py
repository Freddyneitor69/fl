from nodeC.server import server
from nodex.client import client
from coordination import coordinate
from utils import unificar_metricas_csv
import os
import time
import sys

#################### MAIN PARAMETERS ###########################

IP = "192.168.0.21"
BIND_PORT = int(os.getenv("BIND_PORT", 5000))
DOCKER_PORT = int(os.getenv("DOCKER_PORT", 5000))
NODE_ID = int(os.getenv("NODE_ID"))
MODE = int(os.getenv("MODE"))
NETWORK_ADDRESSES = [f"{IP}:{port.strip()}" for port in os.getenv("PORTS").split(",")]
DOCKER_ADDRESS = f"{IP}:{DOCKER_PORT}"
PEERS = [port for port in NETWORK_ADDRESSES if port != DOCKER_ADDRESS]

ROUNDS = 1
SUB_ROUNDS = 5

if MODE==0 and ROUNDS > 1:
    print("Centralized modo solo puede tener ROUNDS=1", flush=True)
    sys.exit(1)


NCLIENTS = len(NETWORK_ADDRESSES)
NODE_DIR = f"node{NODE_ID}"

PARAMS = {
    "hidden_layers": [(32, 0.4), (16, 0.3)],
    "activation": "relu",
    "optimizer": "adam"
}

#################################################################

if __name__ == '__main__':
         

    # Espera inicial para que Docker estabilice la red
    time.sleep(3)

    for round in range(ROUNDS): 
        print(f"\n>>> INICIO RONDA {round} <<<", flush=True)
        
        if MODE==1:
            print("Semi-Descentrilized Modo Configurado", flush=True)
            # 1. COORDINACIÓN
            id_nodeserver = coordinate(PEERS, round)
            print(f"Selected node ID: {id_nodeserver} address: {nodo_ip}:{port_ip}", flush=True)
        else:
            print("Centralizado Modo Configurado", flush=True)
            id_nodeserver = 0

        server_ip = NETWORK_ADDRESSES[int(id_nodeserver)]
        port_ip = int(server_ip.split(':')[1])
        nodo_ip = server_ip.split(':')[0]

        time.sleep(2)  # Pequeña espera antes de iniciar la siguiente fase

        f1scores = []
        accs = []
        
        # 2. ENTRENAMIENTO
        if server_ip == DOCKER_ADDRESS:
            # Soy el servidor
            print(f"[MAIN] Iniciando Servidor FL (Esperando {NCLIENTS - 1} clientes)...", flush=True)
            # IMPORTANTE: Usamos BIND_PORT (interno 5000), no DOCKER_PORT (externo)
            server(BIND_PORT, SUB_ROUNDS + 1, NCLIENTS - 1, PARAMS, f1scores, accs)
        
        else:
            time.sleep(5)
            print(f"[MAIN] Conectando al servidor {nodo_ip}:{port_ip}...", flush=True)
            client(nodo_ip, port_ip, SUB_ROUNDS + 1)


        print(f1scores)
        with open(f'f1scores{NODE_ID}.txt', "+a") as f:
            for line in f1scores:
                f.write(','.join([str(l) for l in line]) + '\n')
        print(accs)
        with open(f'accs{NODE_ID}.txt', "+a") as f:
            for line in accs:
                f.write(','.join([str(l) for l in line]) + '\n')

        print(f"Round {round} completed!!!", flush=True)
        print("="*60,'\n', flush=True)

        unificar_metricas_csv(NODE_ID)
    
    print("="*60, '\n')
    print("Federated training completed successfully!!!", flush=True)