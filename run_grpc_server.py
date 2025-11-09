"""Запуск gRPC сервера."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from api.grpc.server import serve
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Запуск gRPC сервера')
    parser.add_argument('--port', type=int, default=50051, help='Порт для запуска сервера')
    args = parser.parse_args()
    
    serve(port=args.port)

