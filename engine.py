import socket
import json

HOST = ''
PORT = 50007

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    print('서버가 시작되었습니다.')
    conn, addr = s.accept()
    with conn:
        print(f'클라이언트가 접속했습니다:{addr}')
        while True:
            data = conn.recv(1024).decode('utf-8')
            print(f'데이터:{data}')

            response = "ok"
            conn.sendall(response.encode('utf-8'))
