import socket
import time
import torch
from io import BytesIO
import numpy as np

from mm_model import MatrixMultiplyModule


def main(id):
    buffer = BytesIO()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
        client_socket.connect(('localhost', 9999))
        try:
            # Sending data to server
            message = f"{id}"
            client_socket.send(message.encode('utf-8'))

            # Receiving data from server
            while True:
                part = client_socket.recv(1024)
                if not part:
                    break 
                buffer.write(part)

        except Exception as e:
            print(f"Error: {e}")
            return
    
    buffer.seek(0)
    try:
        model_state_dict = torch.load(buffer, map_location=torch.device('cpu'))
    except Exception as e:
        print(f"Error loading model state dict: {e}")
        return

    model = MatrixMultiplyModule()
    model.load_state_dict(model_state_dict)
    
    tensor = torch.eye(5).type(torch.float32)
    out = model(tensor)
    print(out.shape, out)

if __name__ == "__main__":
    id = int(input('id: '))
    main(id)
