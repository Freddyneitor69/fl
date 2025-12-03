import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import os
import datetime
import traceback
from typing import Any, Optional

# --- TRUCO PARA GPU: Evitar que TF acapare toda la memoria y cause OOM ---
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
# -----------------------------------------------------------------------

def average_models(PATH_RECVMODELS: str, PATH_AVGMODELS: str) -> Optional[str]:
    try:
        model_files = [f for f in os.listdir(PATH_RECVMODELS) if f.endswith('.keras')]
        
        if not model_files:
            print("[!] No se encontraron modelos para promediar", flush=True)
            return None
        
        # ... (Resto de tu lógica de carga igual) ...
        # (Omití el código intermedio para brevedad, es el mismo que tenías)
        
        # Cargar modelos
        models_list = []
        for file_model in model_files:
             # ... carga ...
             filename_base = file_model.split('/')[-1]
             file_path = os.path.join(PATH_RECVMODELS, filename_base)
             model = tf.keras.models.load_model(file_path)
             models_list.append(model)

        if not models_list: return None
        
        base_model = models_list[0]
        
        # Promediar
        new_weights = []
        all_weights = [m.get_weights() for m in models_list]
        for _, weights_tuple in enumerate(zip(*all_weights)):
            layer_avg = np.mean(np.array(weights_tuple), axis=0)
            new_weights.append(layer_avg)
        
        global_model = tf.keras.models.clone_model(base_model)
        global_model.build(base_model.input_shape)
        global_model.set_weights(new_weights)
        
        global_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        avg_name = f'avg_{timestamp}.keras'
        avg_path = os.path.join(PATH_AVGMODELS, avg_name)
        
        global_model.save(avg_path)
        print(f"[✓] Modelo promediado guardado: {avg_path}", flush=True)
        
        return avg_path
        
    except Exception as e:
        print(f"[!] Error al promediar modelos: {e}", flush=True)
        # NO USAR exit(1) AQUI. Lanzar excepción para que server la maneje
        raise e 


def build_model(params: dict[str, Any], filemodel: str, input_dim: int = 21) -> bool:
    try:
        model = models.Sequential()
        model.add(layers.Input(shape=(input_dim,)))
        
        for i, (units, dropout) in enumerate(params["hidden_layers"]):
            model.add(layers.Dense(units, activation=params["activation"]))
            if dropout > 0: model.add(layers.Dropout(dropout))
        
        model.add(layers.Dense(1, activation='sigmoid'))
        
        model.compile(optimizer=params["optimizer"], loss='binary_crossentropy', metrics=['accuracy'])
        
        model.save(filemodel)
        # print(f"[✓] Modelo construido: {filemodel}", flush=True)
        return True
        
    except Exception as e:
        print(f"[!] Error al construir modelo: {e}", flush=True)
        raise e # Propagar el error