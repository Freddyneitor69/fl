import numpy as np
import torch
import torch.nn as nn
import os
import datetime
import traceback
from typing import Any, Optional

def average_models(PATH_RECVMODELS: str, PATH_AVGMODELS: str) -> Optional[str]:
    try:
        model_files = [f for f in os.listdir(PATH_RECVMODELS) if f.endswith('.pth')]
        
        if not model_files:
            print("[!] No se encontraron modelos para promediar", flush=True)
            return None
        
        # Cargar modelos
        models_list = []
        for file_model in model_files:
            filename_base = file_model.split('/')[-1]
            file_path = os.path.join(PATH_RECVMODELS, filename_base)
            model_state = torch.load(file_path, map_location='cpu')
            models_list.append(model_state)

        if not models_list:
            return None
        
        # Promediar pesos (state_dict)
        avg_state_dict = {}
        for key in models_list[0].keys():
            # Promediar cada parámetro del modelo
            stacked_params = torch.stack([model[key].float() for model in models_list])
            avg_state_dict[key] = torch.mean(stacked_params, dim=0)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        avg_name = f'avg_{timestamp}.pth'
        avg_path = os.path.join(PATH_AVGMODELS, avg_name)
        
        torch.save(avg_state_dict, avg_path)
        print(f"[✓] Modelo promediado guardado: {avg_path}", flush=True)
        
        return avg_path
        
    except Exception as e:
        print(f"[!] Error al promediar modelos: {e}", flush=True)
        raise e 


class MLPModel(nn.Module):
    """Red neuronal MLP para clasificación binaria"""
    def __init__(self, input_dim: int, hidden_layers: list, activation: str = 'relu'):
        super(MLPModel, self).__init__()
        layers_list = []
        
        prev_dim = input_dim
        for units, dropout in hidden_layers:
            layers_list.append(nn.Linear(prev_dim, units))
            
            # Activación
            if activation.lower() == 'relu':
                layers_list.append(nn.ReLU())
            elif activation.lower() == 'tanh':
                layers_list.append(nn.Tanh())
            elif activation.lower() == 'sigmoid':
                layers_list.append(nn.Sigmoid())
            
            # Dropout
            if dropout > 0:
                layers_list.append(nn.Dropout(dropout))
            
            prev_dim = units
        
        # Capa de salida
        layers_list.append(nn.Linear(prev_dim, 1))
        layers_list.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers_list)
    
    def forward(self, x):
        return self.network(x)


def build_model(params: dict[str, Any], filemodel: str, input_dim: int = 21) -> bool:
    try:
        model = MLPModel(
            input_dim=input_dim,
            hidden_layers=params["hidden_layers"],
            activation=params["activation"]
        )
        
        # Guardar el state_dict del modelo
        torch.save(model.state_dict(), filemodel)
        # print(f"[✓] Modelo construido: {filemodel}", flush=True)
        return True
        
    except Exception as e:
        print(f"[!] Error al construir modelo: {e}", flush=True)
        raise e