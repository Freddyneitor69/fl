#!/bin/bash

ARCHIVO="nodo$NODE_ID/client_metrics_$NODE_ID.json"

# Si ARCHIVO existe, vacíalo

if [ -f "$ARCHIVO" ]; then
    > "$ARCHIVO"
fi

# 1. Obtener RAM Disponible (en MB)
# Usamos 'free -m', filtramos la linea Mem y tomamos la columna 7 (available)
RAM_AVAIL=$(free -m | awk '/^Mem:/ {print $7}')

# 2. Obtener Disco Disponible en raiz (en MB)
# Usamos 'df -m', tomamos la raiz '/', linea 2, columna 4 (avail)
DISK_AVAIL=$(df -m / | awk 'NR==2 {print $4}')

# 3. Obtener Numero de Procesadores
CORES=$(nproc)

# 4. Obtener Velocidad CPU (MHz)
# Tomamos la velocidad del primer nucleo. Quitamos decimales para facilitar lectura.
CPU_SPEED=$(grep "cpu MHz" /proc/cpuinfo | head -1 | awk -F: '{print $2}' | xargs | cut -d. -f1)

# 5. Detectar GPU (Nvidia)
# Si el comando nvidia-smi funciona, asumimos que hay GPU valida para FL.
# Retorna 'true' o 'false' (texto para JSON)
if command -v nvidia-smi &> /dev/null; then
    HAS_GPU="true"
else
    HAS_GPU="false"
fi

# 6. Ancho de Banda (Requiere speedtest-cli)
# Si no tienes speedtest, ponemos 0. 
# IMPORTANTE: --simple devuelve "Ping: x ms \n Download: y Mbit/s..."
if command -v speedtest-cli &> /dev/null; then
    # Capturamos la salida
    ST_OUTPUT=$(speedtest-cli --simple)
    # Extraemos solo el numero de Download (linea 2, columna 2)
    NET_DOWN=$(echo "$ST_OUTPUT" | awk '/Download/ {print $2}')
    # Extraemos solo el numero de Upload (linea 3, columna 2)
    NET_UP=$(echo "$ST_OUTPUT" | awk '/Upload/ {print $2}')
else
    NET_DOWN=0
    NET_UP=0
fi

# --- GENERAR JSON ---
# Construimos el JSON manualmente para evitar dependencias como 'jq'
cat <<EOF > $ARCHIVO
{
  "ram_disponible_mb": $RAM_AVAIL,
  "disco_disponible_mb": $DISK_AVAIL,
  "cpu_cores": $CORES,
  "cpu_mhz": $CPU_SPEED,
  "gpu_activa": $HAS_GPU,
  "red_descarga_mbps": ${NET_DOWN:-0},
  "red_subida_mbps": ${NET_UP:-0}
}
EOF

# No imprimimos nada en pantalla para mantenerlo silencioso, 
# o un simple echo para confirmar.
echo "Metricas guardadas en $ARCHIVO"