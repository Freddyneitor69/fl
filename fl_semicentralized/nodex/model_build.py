import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import datetime
from typing import Optional, Dict
import traceback
import numpy as np


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


class FederatedModel:
    """
    Clase para manejar el entrenamiento y evaluación de modelos
    en un contexto de aprendizaje federado.
    """
    
    def __init__(self, PATH_DATA: str, test_size: float = 0.3, 
                 val_size: float = 0.1, normalize: bool = True, 
                 random_state: int = 42):
        """
        Inicializa el modelo federado con datos locales.
        
        Args:
            PATH_DATA: Ruta al archivo CSV con los datos
            test_size: Proporción de datos para test (default: 0.3)
            val_size: Proporción de datos de test para validación (default: 0.1)
            normalize: Si se debe normalizar los datos (default: True)
            random_state: Semilla para reproducibilidad (default: 42)
        """
        print(f"[>] Cargando datos desde: {PATH_DATA}")
        
        if not os.path.exists(PATH_DATA):
            raise FileNotFoundError(f"No se encontró el archivo: {PATH_DATA}")
        
        # Cargar datos
        data = pd.read_csv(PATH_DATA)
        print(f"[✓] Datos cargados: {len(data)} muestras", flush=True)
        
        # Verificar que existe la columna target
        if 'Diabetes_binary' not in data.columns:
            raise ValueError("El dataset debe contener la columna 'Diabetes_binary'")
        
        # Separar features y target
        X = data.drop(columns=['Diabetes_binary'])
        y = data['Diabetes_binary']
        
        # Split train/test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, shuffle=True, random_state=random_state, 
            stratify=y
        )
        
        # Split test/validation
        self.X_test, self.X_val, self.y_test, self.y_val = train_test_split(
            self.X_test, self.y_test, test_size=val_size, shuffle=True, 
            random_state=random_state, stratify=self.y_test
        )
        
        # Normalización de datos
        self.scaler = None
        if normalize:
            print("[>] Normalizando datos...", flush=True)
            self.scaler = StandardScaler()
            self.X_train = self.scaler.fit_transform(self.X_train)
            self.X_val = self.scaler.transform(self.X_val)
            self.X_test = self.scaler.transform(self.X_test)
            print("[✓] Datos normalizados", flush=True)
        else:
            # Convertir a numpy arrays
            self.X_train = self.X_train.values
            self.X_val = self.X_val.values
            self.X_test = self.X_test.values
        
        self.y_train = self.y_train.values
        self.y_val = self.y_val.values
        self.y_test = self.y_test.values
        
        # Convertir a tensores PyTorch
        self.X_train = torch.FloatTensor(self.X_train)
        self.X_val = torch.FloatTensor(self.X_val)
        self.X_test = torch.FloatTensor(self.X_test)
        self.y_train = torch.FloatTensor(self.y_train).reshape(-1, 1)
        self.y_val = torch.FloatTensor(self.y_val).reshape(-1, 1)
        self.y_test = torch.FloatTensor(self.y_test).reshape(-1, 1)
        
        # Guardar número de features
        self.n_features = self.X_train.shape[1]
        
        # Device (CPU o GPU si está disponible)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[✓] Usando dispositivo: {self.device}", flush=True)
    
    def train_and_save(self, filemodel: str, train: bool = True, epochs: int = 10, batch_size: int = 32, patience: int = 5, verbose: int = 0) -> Optional[str]:
        """
        Entrena el modelo y guarda los pesos actualizados.
        
        Args:
            filemodel: Ruta del modelo a cargar y entrenar
            train: Si se debe entrenar o solo cargar
            epochs: Número máximo de épocas (default: 10)
            batch_size: Tamaño del batch (default: 32)
            patience: Paciencia para early stopping (default: 5)
            verbose: Nivel de verbosidad (default: 0)
            
        Returns:
            Ruta del modelo entrenado o None si hubo error
        """
        try:
            # Cargar arquitectura (necesitamos reconstruir el modelo)
            # Para PyTorch, guardamos solo state_dict, así que necesitamos la arquitectura
            # La obtenemos del archivo de parámetros o la reconstruimos
            model = MLPModel(
                input_dim=self.n_features,
                hidden_layers=[(32, 0.4), (16, 0.3)],  # Esto debería venir de params
                activation='relu'
            )
            
            print(f"[>] Cargando pesos desde: {filemodel}", flush=True)
            model.load_state_dict(torch.load(filemodel, map_location=self.device))
            model.to(self.device)
            
            if train:
                model.train()
                
                # Crear DataLoader
                train_dataset = TensorDataset(self.X_train, self.y_train)
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                
                val_dataset = TensorDataset(self.X_val, self.y_val)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
                
                # Configurar optimizador y loss
                criterion = nn.BCELoss()
                optimizer = optim.Adam(model.parameters(), lr=0.001)
                
                # Early stopping
                best_val_loss = float('inf')
                patience_counter = 0
                best_model_state = None
                
                print(f"[>] Entrenando modelo ({epochs} épocas máx.)...", flush=True)
                
                for epoch in range(epochs):
                    # Training
                    train_loss = 0.0
                    for batch_X, batch_y in train_loader:
                        batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                        
                        optimizer.zero_grad()
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                        loss.backward()
                        optimizer.step()
                        
                        train_loss += loss.item()
                    
                    # Validation
                    model.eval()
                    val_loss = 0.0
                    with torch.no_grad():
                        for batch_X, batch_y in val_loader:
                            batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                            outputs = model(batch_X)
                            loss = criterion(outputs, batch_y)
                            val_loss += loss.item()
                    
                    val_loss /= len(val_loader)
                    model.train()
                    
                    if verbose > 0:
                        print(f"Época {epoch+1}/{epochs} - Val Loss: {val_loss:.4f}", flush=True)
                    
                    # Early stopping
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        best_model_state = model.state_dict().copy()
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            if verbose > 0:
                                print(f"Early stopping en época {epoch+1}", flush=True)
                            break
                
                # Restaurar mejor modelo
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)

            # Guardar modelo con timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = os.path.splitext(filemodel)[0]
            trained_model_path = f"{base_name}_trained_{timestamp}.pth"
            
            torch.save(model.state_dict(), trained_model_path)
            print(f"[✓] Modelo entrenado guardado en: {trained_model_path}", flush=True)
            return trained_model_path
            
        except Exception as e:
            print(f"[!] Error durante el entrenamiento: {e}", flush=True)
            traceback.print_exc()
            return None
    
    def evaluate(self, filemodel: str, threshold: float = 0.5) -> Dict[str, float]:
        """
        Evalúa el modelo en el conjunto de test.
        
        Args:
            filemodel: Ruta del modelo a evaluar
            threshold: Umbral para clasificación binaria (default: 0.5)
            
        Returns:
            Diccionario con f1 y accuracy
        """
        try:
            # Cargar modelo
            model = MLPModel(
                input_dim=self.n_features,
                hidden_layers=[(32, 0.4), (16, 0.3)],
                activation='relu'
            )
            model.load_state_dict(torch.load(filemodel, map_location=self.device))
            model.to(self.device)
            model.eval()
            
            # Predicciones
            with torch.no_grad():
                y_pred_proba = model(self.X_test.to(self.device)).cpu().numpy()
            
            y_pred = (y_pred_proba > threshold).astype(int).flatten()
            y_true = self.y_test.cpu().numpy().flatten()
            
            # Calcular métricas
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            acc = accuracy_score(y_true, y_pred)
            
            return {
                'f1': f1,
                'accuracy': acc
            }
            
        except Exception as e:
            print(f"[!] Error durante la evaluación: {e}", flush=True)
            traceback.print_exc()
            return {'f1': 0.0, 'accuracy': 0.0}
    
    def get_metrics(self, filemodel: str, threshold: float = 0.5) -> Dict[str, float]:
        """
        Obtiene todas las métricas de evaluación como diccionario.
        
        Args:
            filemodel: Ruta del modelo a evaluar
            threshold: Umbral para clasificación binaria
            
        Returns:
            Diccionario con todas las métricas
        """
        try:
            model = MLPModel(
                input_dim=self.n_features,
                hidden_layers=[(32, 0.4), (16, 0.3)],
                activation='relu'
            )
            model.load_state_dict(torch.load(filemodel, map_location=self.device))
            model.to(self.device)
            model.eval()
            
            with torch.no_grad():
                y_pred_proba = model(self.X_test.to(self.device)).cpu().numpy()
            
            y_pred = (y_pred_proba > threshold).astype(int).flatten()
            y_true = self.y_test.cpu().numpy().flatten()
            
            return {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
                'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
            }
            
        except Exception as e:
            print(f"[!] Error obteniendo métricas: {e}", flush=True)
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0
            }