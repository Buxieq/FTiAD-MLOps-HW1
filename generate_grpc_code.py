"""Код из proto файла"""

import subprocess
import os
import sys

def generate_grpc_code():
    proto_file = "api/grpc/service.proto"
    output_dir = "api/grpc"
    
    if not os.path.exists(proto_file):
        print(f"Error: Proto file not found: {proto_file}")
        sys.exit(1)
    
    os.makedirs(output_dir, exist_ok=True)
    
    cmd = [
        sys.executable, "-m", "grpc_tools.protoc",
        f"--proto_path={os.path.dirname(proto_file)}",
        f"--python_out={output_dir}",
        f"--grpc_python_out={output_dir}",
        proto_file
    ]
    
    print(f"Generating gRPC code from {proto_file}...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("gRPC code generated successfully!")
        print(f"Output files should be in: {output_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Error generating gRPC code: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        sys.exit(1)
    except FileNotFoundError:
        print("Error: grpc_tools.protoc not found. Make sure grpcio-tools is installed.")
        print("Install it with: pip install grpcio-tools")
        sys.exit(1)

if __name__ == '__main__':
    generate_grpc_code()

