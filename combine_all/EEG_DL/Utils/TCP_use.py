import socket
import threading
import time

class TCPServer:
    def __init__(self, host='0.0.0.0', port=50007, on_message=None):
        """
        on_message: callback function(msg:str) -> None
        """
        self.host = host
        self.port = port
        self.on_message = on_message
        self.clients = []
        self.stop_event = threading.Event()
        self.server_socket = None
        self.thread = None

        self.start()

    def start(self):
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run_server, daemon=True)
        self.thread.start()
        print(f"🚀 TCP Server started on {self.host}:{self.port}")


    def stop(self):
        self.stop_event.set()
        if self.server_socket:
            self.server_socket.close()
        if self.thread and self.thread.is_alive():
            self.thread.join()
        print("🛑 TCP Server stopped")


    def _run_server(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen()

        while not self.stop_event.is_set():
            try:
                self.server_socket.settimeout(1.0)
                conn, addr = self.server_socket.accept()
                threading.Thread(target=self._handle_client, args=(conn, addr), daemon=True).start()

            except Exception:
                continue


    def _handle_client(self, conn, addr):

        print(f"✅ Client connected from {addr}")
        self.clients.append(conn)

        try:
            while not self.stop_event.is_set():
                data = conn.recv(1024)
                if not data:
                    break
                msg = data.decode('utf-8').strip()
                # print(f"📥 Received from {addr}: {msg}")
                if self.on_message:
                    self.on_message(msg, self)

        except Exception as e:
            print(f"⚠️ Connection error from {addr}: {e}")

        finally:
            conn.close()
            if conn in self.clients:
                self.clients.remove(conn)
            print(f"❌ Disconnected: {addr}")


    def broadcast(self, message: str):
        """廣播訊息給所有連線中的 client"""
        dead_clients = []

        for conn in self.clients:
            try:
                conn.sendall(message.encode('utf-8'))
            except Exception as e:
                print(f"⚠️ 無法傳送給 {conn.getpeername()}: {e}")
                dead_clients.append(conn)
                
        # 移除斷線的 client
        for conn in dead_clients:
            self.clients.remove(conn)



class TCPClient:
    def __init__(self, host='127.0.0.1', port=50007, on_message=None):

        self.host = host
        self.port = port
        self.on_message = on_message
        self.clients = []
        self.stop_event = threading.Event()
        self.client_socket = None
        self.thread = None
        
        self.start()
    
    def start(self):
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run_client, daemon=True)
        self.thread.start()
        print(f"🚀 TCP Client started on {self.host}:{self.port}")
    
    def _run_client(self):

        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            # 1. 連線至 Server
            self.client_socket.connect((self.host, self.port))
            print(f"✅ 已成功連線至 {self.host}:{self.port}")

        except ConnectionRefusedError:
            print("❌ 連線失敗：伺服器尚未啟動或 IP/Port 錯誤")
        except Exception as e:
            print(f"⚠️ 發生錯誤: {e}")


        while not self.stop_event.is_set():
            try:
                self.client_socket.settimeout(1.0)
                
            except Exception:
                continue
            
        

    def stop(self):
        self.stop_event.set()
        if self.client_socket:
            self.client_socket.close()
        if self.thread and self.thread.is_alive():
            self.thread.join()
        print("🛑 TCP Client stopped")


    def send_messages(self, msg):

        # 傳送訊息給 Server
        self.client_socket.sendall(msg.encode('utf-8'))
        time.sleep(0.1) # 稍微停頓方便觀察 log

