import datetime
import os
from .model_build import FederatedModel
import csv
import traceback
import struct

def recv_exact(sock, n_bytes):
    """Asegura recibir exactamente n_bytes del socket"""
    data = b''
    while len(data) < n_bytes:
        packet = sock.recv(n_bytes - len(data))
        if not packet:
            return None # Conexión cerrada
        data += packet
    return data


def send_model(sock, model_info):
    """
    Envía un modelo al servidor.
    
    Args:
        sock: Socket de conexión
        file_model: Ruta del archivo del modelo a enviar
    """
    # Enviar f1-score
    bytes_to_send_f1 = struct.pack('!d', model_info['f1_score'])
    sock.sendall(bytes_to_send_f1)
    print("F1-score del modelo enviado")

    # Enviar accuracy
    bytes_to_send_acc = struct.pack('!d', model_info['accuracy'])
    sock.sendall(bytes_to_send_acc)
    print("Accuracy del modelo enviado")

    file_model = model_info['name']
    # Obtener el tamaño del modelo
    model_size = os.path.getsize(file_model)
    # Enviar el tamaño del modelo al servidor
    sock.sendall(model_size.to_bytes(8, byteorder='big', signed=False))
    try:
        print(f"[>] Enviando modelo: {file_model}\nTamaño: {model_size} bytes", flush=True)
        
        with open(file_model, "rb") as fi:
            bytes_sent = 0
            while bytes_sent < model_size:
                data = fi.read(4096)
                if not data:
                    break
                sock.sendall(data)
                bytes_sent += len(data)
        
        print(f"[✓] Modelo enviado exitosamente ({bytes_sent} bytes)", flush=True)
        
    except FileNotFoundError:
        print(f"[!] Error: No se encontró el archivo {file_model}", flush=True)
        raise
    except IOError as e:
        print(f'[!] Error de I/O al enviar {file_model}: {e}', flush=True)
        raise
    except Exception as e:
        print(f'[!] Error inesperado durante send_model: {e}', flush=True)
        raise


def get_model(sock, nn: FederatedModel, round_num: int, PATH_MODELS:str, train: bool = True):
    """
    Recibe un modelo del servidor, lo entrena y lo evalúa.
    
    Args:
        sock: Socket de conexión
        nn: Instancia de FederatedModel
        round_num: Número de ronda actual
        
    Returns:
        Diccionario con información del modelo entrenado
    """
    try:

        # Recibe el tamaño del modelo
        model_size_data = recv_exact(sock, 8)
        if len(model_size_data) < 8:
            raise ValueError("Datos inválidos para el tamaño del modelo")

        model_size = int.from_bytes(model_size_data, byteorder='big', signed=False)
        print(f"[>] Tamaño del modelo recibido: {model_size} bytes", flush=True)
        print(f"\n{'='*60}", flush=True)
        print(f"  RONDA {round_num}", flush=True)
        print(f"{'='*60}", flush=True)
        print("[>] Esperando modelo del servidor...", flush=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f'model_round{round_num}_{timestamp}.keras'
        received_model = os.path.join(PATH_MODELS, model_name)
        
        # Recibir modelo
        bytes_received = 0
        with open(received_model, "wb") as fo:
            while bytes_received < model_size:
                data = sock.recv(4096)
                if not data:
                    break
                fo.write(data)
                bytes_received += len(data)
        
        print(f"[✓] Modelo recibido ({bytes_received} bytes)", flush=True)
        print(f"[✓] Guardado en: {received_model}", flush=True)
        
        # Entrenar modelo
        print("[>] Entrenando modelo con datos locales...", flush=True)
        trained_model_path = nn.train_and_save(received_model, train, epochs=5)
        
        if trained_model_path is None:
            print("[!] Error: El entrenamiento no retornó un modelo válido", flush=True)
            return None
        
        print(f"[✓] Modelo entrenado guardado en: {trained_model_path}", flush=True)
        
        # Evaluar modelo
        print("[>] Evaluando modelo...", flush=True)
        metrics = nn.evaluate(trained_model_path)
        f1 = metrics['f1']
        acc = metrics['accuracy']

        print(f"[✓] F1-Score: {f1:.4f} | Accuracy: {acc:.4f}", flush=True)
        
        return {
            "date": timestamp,
            "f1_score": f1,
            "accuracy": acc,
            "name": trained_model_path,
            "round": round_num
        }
        
    except IOError as e:
        print(f'[!] Error de I/O al recibir modelo: {e}', flush=True)
        raise
    except Exception as e:
        print(f'[!] Error durante get_model: {e}', flush=True)
        traceback.print_exc()
        raise


def save_models_info(models_info, node_id, PATH_MAIN):
    """
    Guarda información de los modelos en un archivo CSV.
    
    Args:
        models_info: Lista de diccionarios con información de modelos
        node_id: ID del nodo
    """
    try:
        csv_file_path = os.path.join(PATH_MAIN, f'models_info_{node_id}.csv')
        
        with open(csv_file_path, mode='w', newline='') as csvfile:
            fieldnames = ['round', 'date', 'f1_score', 'accuracy', 'name']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for model in models_info:
                if model:  # Solo escribir si el modelo es válido
                    writer.writerow(model)
        
        print(f"\n[✓] Información de modelos guardada en: {csv_file_path}", flush=True)
        
    except IOError as e:
        print(f"[!] Error al escribir archivo CSV: {e}", flush=True)
        raise
