## Federated Centralized and Semi-Descentralized system
Python implementation of Semi-Decentralized and Centralized Federated Learning architectures for Diabetes prediction using a Multilayer Per
ceptron (MLP). Communication between nodes is handled through raw TCP sockets, enabling distributed training across multiple devices.

## Requirements
To run the project, the libraries listed in `requirements.txt` must be installed.

## Docker Build and Deployment

### Docker Image Build
Run the following command to build the image:
```bash
docker build -t federated-semidescentralized_image .
```
### Docker Containers Deployment
To run multiple nodes, Docker Compose is required. By default, docker-compose.yml is configured to create 4 nodes, but you may modify it to
 scale up.

```Bash
docker compose up
```
## Project Architecture

### Central Node Logic (nodeC)
- **models/**: Contains aggregated .keras models.
- **avg_models.py**: Functions for model aggregation (Federated Averaging).
- **connections.py**: Defines network logic for the server role.
- **server.py**: Handles server-side orchestration.

### Edge Node Logic (nodex)
- **models_#/**: Contains local .keras models for each client.
- **connections.py**: Defines network logic for the client role.
- **model_build.py**: Defines MLP architecture and training functions.
- **client.py**: Handles client-side operations.

### Other Components

- **diabetes_divided/**: Partitioned dataset for diabetes diagnosis.
- **coordination.py**: Logic for dynamic node role selection.
- **utils.py**: Auxiliary functions (convergence, etc.).
- **main.py**: Entry point for the application.
- **Dockerfile**: Image definition.
- **docker-compose.yaml**: Multi-container setup.
- **requirements.txt**: Python dependencies.
- **metrics.sh**: Bash script to measure CPU, RAM, GPU, and network capabilities.

## Main Configuration
Configuration is managed via `main.py` and environment variables in `docker-compose.yaml`.

### Example from `main.py`:

```PY

ROUNDS = 5
SUB_ROUNDS = 5
IP = "172.23.211.39"
BIND_PORT = int(os.getenv("BIND_PORT", 5000))
DOCKER_PORT = int(os.getenv("DOCKER_PORT", 5000))
NODE_ID = int(os.getenv("NODE_ID"))

PARAMS = {
    "hidden_layers": [(32, 0.4), (16, 0.3)],
    "activation": "relu",
    "optimizer": "adam"
}
```
### Variable Descriptions
- **ROUNDS**: Number of times a new central node (server) is selected.
- **SUB_ROUNDS**: Number of local training epochs or iterations per round.
- **IP**: IP address of the node.
- **BIND_PORT**: Port bound by Docker on the host.
- **DOCKER_PORT**: Internal port used within the Docker container.
- **NODE_ID**: Unique identifier for each node.
- **PARAMS**: Hyperparameters for configuring the neural network.
- **MODE**: Defines the system architecture; `0` for Centralized and `1` for Semi-Decentralized.

### System Operation

#### Semi-Decentralized Mode

The system consists of multiple distributed nodes that communicate during initialization to establish roles, as defined in `coordination.py`.

**Leader Selection Policy**:

Each node executes `metrics.sh` to collect data on hardware capabilities. Leader selection is probabilistic, with weights assigned based on node resources:

```PY

def select_leader(nodes, round):
    scores = []
    for x in nodes:
        # Score calculation based on Network, RAM, CPU, GPU, and ID
        score = (
            0.5 * (0.5 * x['net_up'] + 0.5 * x['net_down']) +
            0.3 * x['ram'] +
            0.35 * x['cpu_mhz'] +
            0.2 * int(x['gpu']) +
            0.1 * (1 / int(x['id']))
        )
        scores.append(score)

    # Seed ensures all nodes independently select the same leader
    random.seed(round)
    winner = random.choices(nodes, weights=scores, k=1)[0]
    random.seed(None) 
    
    return winner
```
Using the round number as a seed, all nodes independently determine the designated server without needing a voting phase over the network. The weighted probability ensures load balancing while favoring more powerful nodes.

**Training Cycle:**

- The server node initializes the round and waits for clients.
- The server generates/loads a model and distributes it.
- Clients train locally on their data partition.
- Clients send updates; the server aggregates them (FedAvg).
- This repeats for M sub-rounds before a new coordination step selects a new server.

Convergence Check: The process stops earlier if convergence or performance degradation is detected, as defined in utils.py:

```PY
def checkConvergence(scores:list[list[float, float, float]], patience:int, threshold:float=0.01)->bool:
    if len(scores) < patience:
        return False

    recent_scores = scores[-patience]
    for node in range(len(scores[-1])):
      diff = recent_scores[node] - scores[-1][node]
      if diff > threshold:
          return False
      if diff < -threshold:
          return False
```
#### Centralized Mode

In this mode, the workflow remains similar, but the server role is permanently fixed by assigning `NODE_ID = 0`. The `ROUNDS` variable effectively becomes 1, meaning the loop does not facilitate any role switching, thus maintaining the aggregator's role static.
## Results

- **Models**: Local models are saved in the `nodex/models/` directory, while global models are stored in `nodeC/models/`.
- **Best Model**: The top-performing aggregated models can be found in `nodeC/models/avg/`.
- **Metrics**: Performance metrics, such as F1-score and Accuracy, can be reviewed in the `full_metrics_node_#.csv` files across various rounds.

## Docker Build and Deployment

### Docker Image Build
```bash
docker build -t federated-semidescentralized_image .
```

### Docker Containers Deployment
To run multiple nodes, Docker Compose is required. By default, `docker-compose.yml` is configured to create 4 nodes, but you may modify 
it to scale up.

```bash
docker compose up
```

## Project Architecture

- **nodeC** (Central Node Logic)
    - **models/**: Contains aggregated .keras models.
    - **avg_models.py**: Functions for model aggregation (Federated Averaging).
- **connections.py**: Defines network logic for the server role.
- **server.py**: Handles server-side orchestration.
- **nodeX (Edge Node Logic)**
    - **models_#**: Contains local .keras models for each client.
    - **Connections.py**: Defines network logic for the client role.
    - **Model_build.py**: Defines MLP architecture and training functions.
    - **client.py**: Handles client-side operations.

- **diabetes_divided/**: Partitioned dataset for diabetes diagnosis.
- **coordination.py**: Logic for dynamic node role selection.
- **utils.py**: Auxiliary functions (convergence, etc.).
- **main.py**: Entry point for the application.
- **Dockerfile**: Image definition.
- **docker-compose.yaml**: Multi-container setup.
- **requirements.txt**: Python dependencies.
- **metrics.sh**: Bash script to measure CPU, RAM, GPU, and network capabilities.

## Main Configuration

Configuration is managed via `main.py` and environment variables in `docker-compose.yaml`.

### Example from main.py:
```python
ROUNDS = 5
SUB_ROUNDS = 5
IP = "172.23.211.39"
BIND_PORT = int(os.getenv("BIND_PORT", 5000))
DOCKER_PORT = int(os.getenv("DOCKER_PORT", 5000))
NODE_ID = int(os.getenv("NODE_ID"))

PARAMS = {
    "hidden_layers": [(32, 0.4), (16, 0.3)],
    "activation": "relu",
    "optimizer": "adam"
}
```

### Variable Description
- **ROUNDS**: Number of times a new central node (server) is selected.
- **SUB_ROUNDS**: Training epochs (local iterations) per round.
- **IP**: Node IP address.
- **BIND_PORT**: Host port bound by Docker.
- **DOCKER_PORT**: Internal container port.
- **NODE_ID**: Unique node identifier.
- **PARAMS**: Hyperparameters for the neural network.

### Mode: System Operation

**Semi-Decentralized Mode**
The system consists of N distributed nodes that communicate at initialization to establish roles using the policy defined in 
`coordination.py`.

Leader Selection Policy:
Each node runs `metrics.sh` to gather hardware capabilities. The selection is probabilistic, weighted by the node's resources:

```python
def select_leader(nodes, round):
    scores = []
    for x in nodes:
        # Score calculation based on Network, RAM, CPU, GPU, and ID
        score = (
            0.5 * (0.5 * x['net_up'] + 0.5 * x['net_down']) +
            0.3 * x['ram'] +
            0.35 * x['cpu_mhz'] +
            0.2 * int(x['gpu']) +
            0.1 * (1 / int(x['id']))
        )
        scores.append(score)

    # Seed ensures all nodes independently select the same leader
    random.seed(round)
    winner = random.choices(nodes, weights=scores, k=1)[0]
    random.seed(None) 
    
    return winner

```
By using the round number as a seed, all nodes independently determine the designated server without needing a voting phase over the 
network. The weighted probability ensures load balancing while favoring more powerful nodes.

Training Cycle:

- The server node initializes the round and waits for clients.
- The server generates/loads a model and distributes it.
- Clients train locally on their data partition.
- Clients send the best model of the current subround; the server aggregates them (FedAvg).
- This repeats for M sub-rounds before a new coordination step selects a new server.

**Centralized Mode**
This mode follows a similar workflow but fixes `NODE_ID = 0` as the permanent server. The `ROUNDS` variable is treated as 1 (or the loop 
does not trigger role switching), preventing the dynamic reassignment of the aggregator role.

### Results
- **Models**: Saved in models/ directories within `nodex` (local) and `nodeC` (global).
- **Best Model**: The best aggregated models are found in nodeC/models/avg/.
- **Metrics**: Refer to full metrics files (`full_metrics_node_#.csv`) to analyze performance (F1-score, Accuracy).

