#!/usr/bin/env python3
"""
Script para instalar pacotes de forma controlada
"""

import subprocess
import sys
import time

def install_package(package, timeout=300):
    """Instala um pacote com timeout"""
    print(f"Instalando {package}...")
    try:
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", package, "--timeout", str(timeout)
        ], capture_output=True, text=True, timeout=timeout+60)
        
        if result.returncode == 0:
            print(f"✅ {package} instalado com sucesso")
            return True
        else:
            print(f"❌ Erro ao instalar {package}: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print(f"⏰ Timeout ao instalar {package}")
        return False
    except Exception as e:
        print(f"❌ Erro inesperado ao instalar {package}: {e}")
        return False

def main():
    """Função principal"""
    print("🚀 Instalando pacotes necessários...")
    
    # Lista de pacotes essenciais (sem PyTorch por enquanto)
    packages = [
        "numpy>=1.21.0",
        "Pillow>=8.3.1", 
        "scikit-learn>=0.24.2",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "opencv-python>=4.5.0"
    ]
    
    # Pacotes opcionais (mais problemáticos)
    optional_packages = [
        "flask==2.0.1",
        "chromadb==0.3.29"
    ]
    
    print("\n📦 Instalando pacotes essenciais...")
    success_count = 0
    
    for package in packages:
        if install_package(package):
            success_count += 1
        time.sleep(2)  # Pausa entre instalações
    
    print(f"\n✅ {success_count}/{len(packages)} pacotes essenciais instalados")
    
    if success_count == len(packages):
        print("\n📦 Tentando instalar pacotes opcionais...")
        optional_success = 0
        
        for package in optional_packages:
            if install_package(package):
                optional_success += 1
            time.sleep(5)  # Pausa maior para pacotes maiores
        
        print(f"✅ {optional_success}/{len(optional_packages)} pacotes opcionais instalados")
    
    print("\n🎉 Instalação concluída!")
    print("\n📋 Para instalar PyTorch manualmente (se necessário):")
    print("py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")

if __name__ == "__main__":
    main() 