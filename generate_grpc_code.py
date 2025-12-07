"""Код из proto файла"""

import subprocess
import os
import sys
import shutil

def find_python_with_grpc_tools():
    """Try to find a Python interpreter that has grpc_tools installed."""
    # First, try to use poetry's Python if available
    try:
        result = subprocess.run(
            ["poetry", "run", "python", "-c", "import grpc_tools.protoc"],
            capture_output=True,
            check=True
        )
        # If successful, return poetry run python
        return ["poetry", "run", "python"]
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    # Try current Python
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import grpc_tools.protoc"],
            capture_output=True,
            check=True
        )
        return [sys.executable]
    except subprocess.CalledProcessError:
        pass
    
    # Try to find protoc directly
    protoc_path = shutil.which("protoc")
    if protoc_path:
        # If protoc is available, we might be able to use it with Python plugin
        # But for now, let's just suggest installation
        pass
    
    return None

def generate_grpc_code():
    proto_file = "api/grpc/service.proto"
    output_dir = "api/grpc"
    
    if not os.path.exists(proto_file):
        print(f"Error: Proto file not found: {proto_file}")
        sys.exit(1)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Try to find Python with grpc_tools
    python_cmd = find_python_with_grpc_tools()
    
    if python_cmd is None:
        print("Error: grpc_tools.protoc not found.")
        print("\nPlease install grpcio-tools in one of the following ways:")
        print("1. Using Poetry (recommended):")
        print("   poetry install")
        print("   poetry run python generate_grpc_code.py")
        print("\n2. Using pip:")
        print("   pip install grpcio-tools")
        print("   python generate_grpc_code.py")
        sys.exit(1)
    
    cmd = python_cmd + [
        "-m", "grpc_tools.protoc",
        f"--proto_path={os.path.dirname(proto_file)}",
        f"--python_out={output_dir}",
        f"--grpc_python_out={output_dir}",
        proto_file
    ]
    
    print(f"Generating gRPC code from {proto_file}...")
    print(f"Using: {' '.join(python_cmd)}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("gRPC code generated successfully!")
        print(f"Output files should be in: {output_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Error generating gRPC code: {e}")
        if e.stdout:
            print(f"stdout: {e.stdout}")
        if e.stderr:
            print(f"stderr: {e.stderr}")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"Error: Command not found: {e}")
        print("Make sure grpcio-tools is installed.")
        print("Install it with: poetry install (or pip install grpcio-tools)")
        sys.exit(1)

if __name__ == '__main__':
    generate_grpc_code()

