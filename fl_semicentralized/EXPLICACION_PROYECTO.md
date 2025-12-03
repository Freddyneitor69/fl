# Explicación del proyecto: Federated (Semi-)Decentralized Learning para Diabetes

## Resumen general
- Propósito: Implementar y comparar arquitecturas de Federated Learning (centralizado y semi-descentralizado) para la predicción de diabetes usando una MLP (TensorFlow/Keras).
- Comunicación: sockets TCP directos entre nodos (no HTTP/gRPC).
- Modos:
  - Centralizado: un servidor fijo (aggregator) agrega modelos.
  - Semi-descentralizado: cada "round" se selecciona un líder/servidor según métricas de hardware; la selección usa la semilla=numero_de_ronda para que todos los nodos lleguen al mismo resultado sin votación de red.
- Flujo: el servidor distribuye un modelo inicial; cada cliente entrena localmente, devuelve su mejor modelo (según F1); el servidor promedia (FedAvg) y redistribuye.

## Estructura del repositorio (archivos/carpetas clave)
- `main.py`: Punto de entrada. Lee variables de entorno (`NODE_ID`, `MODE`, `PORTS`, etc.), orquesta rondas y decide si el proceso actúa como servidor o cliente.
- `coordination.py`: Recolecta métricas de hardware desde todos los peers (ejecutando `metrics.sh` en cada nodo) y selecciona el líder con `select_leader`.
- `utils.py`: Funciones auxiliares:
  - `select_leader(nodes, round)`: calcula scores ponderados por red, RAM, CPU, GPU y usa `random.seed(round)` + `random.choices` para elegir líder.
  - `checkConvergence`, `unificar_metricas_csv`.

- `nodeC/` (lógica del servidor/aggregator):
  - `server.py`: Orquestación del servidor (acepta clientes, llama a inicialización, recibe modelos, promedia, envía modelos promediados).
  - `connections.py`: Implementa el protocolo socket del servidor: `initial`, `get_models`, `send_avg_model`, `sendconverge`, utilidades de envío/recepción.
  - `avg_model.py`: Construcción de modelo inicial y función `average_models` (FedAvg).
  - Directorios de modelos: `nodeC/models/recv/` y `nodeC/models/avg/`.

- `nodex/` (lógica del cliente/edge):
  - `client.py`: Conecta al servidor, envía ID, recibe modelo, entrena localmente (usa `model_build`), envía su mejor modelo.
  - `connections.py`: Protocolo cliente: `get_model`, `send_model`, `save_models_info`.
  - `model_build.py`: Clase `FederatedModel` que carga datos locales (`diabetes_divided/diabetes_N.csv`), normaliza, entrena y evalúa (F1, Accuracy).
  - Rutas de modelos locales: `nodex/models_<NODE>/`.

- `diabetes_divided/`: CSVs particionados (`diabetes_1.csv` ... `diabetes_4.csv`) — cada nodo usa su partición.
- `Dockerfile` y `docker-compose.yaml`: Definen la imagen (base TensorFlow) y un `docker-compose` de ejemplo con 4 nodos.
- `metrics.sh`: Script que mide RAM, disco, CPU MHz, núcleos, presencia de GPU y ancho de banda (opcional) y genera `client_metrics_<NODE>.json`.

## Flujo de ejecución (alto nivel)
1. Lanzar contenedores (recomendado) con `docker compose up` o ejecutar `python main.py` en cada nodo con las variables de entorno correctas.
2. En cada ronda:
   - Si `MODE==1` (semi-descentralizado): `main.py` llama a `coordinate(PEERS, round)`, que:
     - Inicia un servidor temporal para recolectar métricas.
     - Ejecuta cliente que corre `metrics.sh` para generar JSON de métricas locales.
     - Junta las métricas en un CSV y llama a `select_leader` (con semilla `round`) para devolver el ID del líder.
   - `main.py` decide el `server_ip` según `NETWORK_ADDRESSES` y `id_nodeserver`.
   - Si este contenedor es el servidor: ejecuta `nodeC.server.server` (espera clientes y orquesta sub-rounds).
   - Si es cliente: ejecuta `nodex.client.client`, que recibe el modelo, lo entrena localmente y envía su mejor modelo de vuelta.
3. En el servidor (por sub-round):
   - `initial()` prepara o carga un modelo inicial y lo envía a los clientes.
   - `get_models()` recibe modelos de los clientes (envío binario: f1, acc como double + tamaño + bytes del .keras) y los guarda en `nodeC/models/recv/`.
   - `average_models()` promedia pesos (FedAvg) y guarda `avg_YYYYMMDD_HHMMSS.keras` en `nodeC/models/avg/`.
   - `send_avg_model()` envía el modelo promedio a todos los clientes.
   - Si `checkConvergence` detecta convergencia, se envía señal y se termina antes de agotar las rondas.
4. Cliente:
   - Recibe el .keras, entrena con `FederatedModel.train_and_save`, evalúa (F1, Accuracy) y envía su mejor modelo mediante `send_model()`.

## Protocolo de red y formato de mensajes
- Comunicación binaria vía TCP.
- Identificación: cliente envía ID (36 bytes) al conectar.
- Servidor -> cliente: 8 bytes (tamaño del archivo, big-endian) + bytes del `.keras`.
- Cliente -> servidor: 8 bytes (f1 como `!d`), 8 bytes (accuracy como `!d`), 8 bytes (tamaño del modelo), luego bytes del `.keras`.
- Señal de convergencia: servidor envía 1 byte: `\x01` = convergió, `\x00` = continuar.

## Cómo ejecutar (resumen rápido)
- Con Docker (recomendado):
  - Construir imagen (PowerShell):
    ```powershell
    docker build -t federated-semidescentralized_image .
    docker compose up
    ```
  - Logs y resultados se almacenan en carpetas montadas (`nodo1/`, `nodo2/`, ...).

- Local (sin Docker): ejecutar un proceso por nodo (cada uno en su terminal) exportando variables de entorno adecuadas. Ejemplo PowerShell para un nodo:
  ```powershell
  $env:NODE_ID=1; $env:MODE=1; $env:PORTS="5001,5002,5003,5004"; $env:BIND_PORT=5000; $env:DOCKER_PORT=5001
  python main.py
  ```
  Repetir en otros shells cambiando `NODE_ID` y `DOCKER_PORT`.

- Requisitos: instalar paquetes de `requirements.txt` (TensorFlow, scikit-learn, etc.). El `Dockerfile` instala automáticamente las dependencias.

## Observaciones y recomendaciones
- Consistencia de nombres: hay mezcla entre prefijos `node` y `nodo` en rutas y variables. Revisar para evitar errores de path (por ejemplo `NODE_DIR = f"node{NODE_ID}"` vs carpetas `nodo1`).
- `docker-compose.yaml` muestra `NODE_ID=0` para `nodo1` y `NODE_ID=2` para `nodo2` — normaliza los IDs (por ejemplo 1..N) para evitar off-by-one al indexar `NETWORK_ADDRESSES`.
- `metrics.sh` usa utilidades de Linux (`free`, `df`, `nproc`), por lo que en entornos Windows hay que usar Docker o adaptar el script.
- Seguridad: no hay autenticación ni cifrado; si vas a probar en redes no confiables agrega TLS o autenticación simple.
- Compatibilidad de modelos: se asume que todos los modelos tienen la misma arquitectura/input-shape. Añadir validaciones de shape antes de aceptar/usar un modelo evita errores en clientes.

## Archivos de salida y métricas
- Modelos locales: `nodex/models_<NODE>/`.
- Modelos promediados (globales): `nodeC/models/avg/`.
- Modelos recibidos por servidor: `nodeC/models/recv/`.
- CSV de rutas de modelos: `models_path_<NODE>.csv`.
- Métricas consolidadas por nodo: `full_metrics_node_<NODE>.csv` (generado por `utils.unificar_metricas_csv`).

## Próximos pasos sugeridos
- Arreglar consistencias de `NODE_ID`/nombres de carpeta (`node` vs `nodo`).
- Añadir validaciones de forma (input shape) antes de aceptar modelos.
- Añadir autenticación básica o HMAC para integridad de modelos.
- Probar una ejecución mínima local (1 servidor + 1 cliente) y compartir logs para depuración.

---

Si quieres, puedo:
- Ejecutar una prueba mínima local (no-Docker) y mostrar los pasos exactos en PowerShell.
- Normalizar los nombres de carpetas y `NODE_ID` en `docker-compose.yaml` y código.
- Añadir un README corto con comandos de ejecución ya integrados.

¿Qué prefieres que haga ahora?