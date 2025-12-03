import tensorflow as tf
from tensorflow.keras import models, callbacks
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import datetime
from typing import Optional, Dict
import traceback

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    print("GPU Memory Growth habilitado")
  except RuntimeError as e:
    print(e)


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
        
        # Guardar número de features
        self.n_features = self.X_train.shape[1]
    
    def train_and_save(self, filemodel: str, train:bool=True, epochs: int = 10, batch_size: int = 32, patience: int = 5, verbose: int = 0) -> Optional[str]:
        """
        Entrena el modelo y guarda los pesos actualizados.
        
        Args:
            filemodel: Ruta del modelo a cargar y entrenar
            epochs: Número máximo de épocas (default: 10)
            batch_size: Tamaño del batch (default: 32)
            patience: Paciencia para early stopping (default: 5)
            verbose: Nivel de verbosidad (default: 1)
            
        Returns:
            Ruta del modelo entrenado o None si hubo error
        """
        try:
            # Cargar modelo
            print(f"[>] Cargando modelo desde: {filemodel}", flush=True)
            model = models.load_model(filemodel)
            if train:
                model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                
                # Verificar dimensionalidad
                expected_shape = model.input_shape[1]
                if expected_shape != self.n_features:
                    raise ValueError(
                        f"El modelo espera {expected_shape} features pero "
                        f"los datos tienen {self.n_features}"
                    )
                
                # Configurar callbacks
                early_stop = callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=patience,
                    restore_best_weights=True,
                    verbose=1
                )
                
                # Entrenar modelo
                print(f"[>] Entrenando modelo ({epochs} épocas máx.)...", flush=True)
                model.fit(
                    self.X_train, self.y_train,
                    validation_data=(self.X_val, self.y_val),
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[early_stop],
                    verbose=verbose
                )

            # Guardar modelo con timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = os.path.splitext(filemodel)[0]
            trained_model_path = f"{base_name}_trained_{timestamp}.keras"
            
            model.save(trained_model_path)
            print(f"[✓] Modelo entrenado guardado en: {trained_model_path}", flush=True)            
            return trained_model_path
            
        except Exception as e:
            print(f"[!] Error durante el entrenamiento: {e}", flush=True)
            traceback.print_exc()
            return None
    
    def evaluate(self, filemodel: str, threshold: float = 0.5) -> float:
        """
        Evalúa el modelo en el conjunto de test.
        
        Args:
            filemodel: Ruta del modelo a evaluar
            threshold: Umbral para clasificación binaria (default: 0.5)
            verbose: Si se deben imprimir métricas detalladas (default: True)
            
        Returns:
            F1-Score ponderado
        """
        try:
            # Cargar modelo
            model = models.load_model(filemodel)
            
            # Predicciones
            y_pred_proba = model.predict(self.X_test, verbose=0)
            y_pred = (y_pred_proba > threshold).astype(int).flatten()
            
            # Calcular métricas
            f1 = f1_score(self.y_test, y_pred, average='weighted', zero_division=0)
            acc = accuracy_score(self.y_test, y_pred)
            
            return {
                'f1':f1,
                'accuracy':acc
            }
            
        except Exception as e:
            print(f"[!] Error durante la evaluación: {e}", flush=True)
            traceback.print_exc()
            return 0.0
    
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
            model = models.load_model(filemodel)
            y_pred_proba = model.predict(self.X_test, verbose=0)
            y_pred = (y_pred_proba > threshold).astype(int).flatten()
            
            return {
                'accuracy': accuracy_score(self.y_test, y_pred),
                'precision': precision_score(self.y_test, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(self.y_test, y_pred, average='weighted', zero_division=0),
                'f1_score': f1_score(self.y_test, y_pred, average='weighted', zero_division=0)
            }
            
        except Exception as e:
            print(f"[!] Error obteniendo métricas: {e}", flush=True)
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0
            }