import socket
import threading
import torch
import numpy as np
from io import BytesIO

from mm_model import MatrixMultiplyModule

model = MatrixMultiplyModule(5)
torch.save(model.state_dict(), 'models/test_model.pth')


def handle_client(client_socket):
    global model

    received = client_socket.recv(1024).decode('utf-8')
    if not received: return
    print(f"Client id {received} connecting")

    # sending data to client
    buffer = BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0) 
    
    data = buffer.read()
    client_socket.sendall(data)
    client_socket.close()
    return


def main():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('0.0.0.0', 9999))
    server_socket.listen(5)
    print("Listening on port 9999...")
    
    while True:
        client, addr = server_socket.accept()
        print(f"Accepted connection from {addr[0]}:{addr[1]}")
        client_handler = threading.Thread(target=handle_client, args=(client,))
        client_handler.start()

if __name__ == "__main__":
    main()

