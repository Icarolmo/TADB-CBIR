#!/usr/bin/env python3
"""
Script para configurar avaliação adequada do sistema CBIR
Divide o dataset e configura referência/teste sem sobreposição
"""

import os
import shutil
from pathlib import Path
from split_dataset import split_dataset

def setup_proper_evaluation():
    """
    Configura o sistema para avaliação adequada
    """
    print("🔧 Configurando avaliação adequada do sistema CBIR")
    print("=" * 50)
    
    # Verificar se existe dataset original
    original_dataset = Path("image/dataset")
    if not original_dataset.exists():
        print("❌ Erro: Diretório 'image/dataset' não encontrado!")
        print("\n📁 Estrutura esperada:")
        print("   image/dataset/")
        print("   ├── leaf_healthy/")
        print("   │   ├── imagem1.jpg")
        print("   │   └── ...")
        print("   └── leaf_with_disease/")
        print("       ├── imagem1.jpg")
        print("       └── ...")
        return False
    
    # Verificar se existem as categorias
    healthy_dir = original_dataset / "leaf_healthy"
    diseased_dir = original_dataset / "leaf_with_disease"
    
    if not healthy_dir.exists() or not diseased_dir.exists():
        print("❌ Erro: Categorias 'leaf_healthy' ou 'leaf_with_disease' não encontradas!")
        return False
    
    # Dividir dataset
    print("\n🔄 Dividindo dataset...")
    train_dir, test_dir = split_dataset("image/dataset", train_ratio=0.8)
    
    # Limpar banco de dados existente
    print("\n🧹 Limpando banco de dados existente...")
    if Path("database/chroma_db").exists():
        shutil.rmtree("database/chroma_db")
        print("   ✅ Banco de dados removido")
    
    # Preparando sistema com imagens de referência
    print("\n🎯 Preparando sistema com imagens de referência...")
    os.system("python cbir.py --process-only --train-dir image/dataset/train")
    
    # Configurar teste
    print("\n📝 Configurando diretório de teste...")
    test_dataset_dir = Path("image/test_dataset")
    if test_dataset_dir.exists():
        shutil.rmtree(test_dataset_dir)
    
    # Copiar imagens de teste
    shutil.copytree("image/dataset/test", "image/test_dataset")
    print("   ✅ Imagens de teste copiadas para 'image/test_dataset'")
    
    print("\n✅ Configuração concluída!")
    print("\n📊 Estrutura final:")
    print("   image/dataset/train/     - Imagens de referência (80%)")
    print("   image/dataset/test/      - Backup das imagens de teste (20%)")
    print("   image/test_dataset/      - Imagens para avaliação (20%)")
    print("   database/chroma_db       - Banco de dados indexado")
    
    print("\n🚀 Próximos passos:")
    print("   1. Execute: python test_evaluation.py")
    print("   2. Ou execute: python demo_evaluation.py")
    print("   3. Ou use a GUI: python evaluation_gui.py")
    
    return True

def main():
    setup_proper_evaluation()

if __name__ == "__main__":
    main() 