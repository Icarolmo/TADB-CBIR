#!/usr/bin/env python3
"""
Script para dividir o dataset em treinamento e teste
Evita usar as mesmas imagens para treinar e testar
"""

import os
import shutil
import random
from pathlib import Path
import argparse

def split_dataset(source_dir, train_ratio=0.8, random_seed=42):
    """
    Divide o dataset em treinamento e teste
    
    Args:
        source_dir: Diretório com as imagens originais
        train_ratio: Proporção para treinamento (0.8 = 80%)
        random_seed: Semente para reprodutibilidade
    """
    
    # Configurar seed para reprodutibilidade
    random.seed(random_seed)
    
    # Criar diretórios
    train_dir = Path("image/dataset/train")
    test_dir = Path("image/dataset/test")
    
    # Limpar diretórios existentes
    if train_dir.exists():
        shutil.rmtree(train_dir)
    if test_dir.exists():
        shutil.rmtree(test_dir)
    
    # Criar estrutura de diretórios
    for category in ["leaf_healthy", "leaf_with_disease"]:
        (train_dir / category).mkdir(parents=True, exist_ok=True)
        (test_dir / category).mkdir(parents=True, exist_ok=True)
    
    print(f"🔄 Dividindo dataset de: {source_dir}")
    print(f"📊 Proporção: {train_ratio*100}% treinamento, {(1-train_ratio)*100}% teste")
    
    total_images = 0
    train_count = 0
    test_count = 0
    
    # Processar cada categoria
    for category in ["leaf_healthy", "leaf_with_disease"]:
        category_path = Path(source_dir) / category
        
        if not category_path.exists():
            print(f"⚠️  Categoria '{category}' não encontrada em {source_dir}")
            continue
        
        # Listar todas as imagens da categoria
        images = list(category_path.glob("*.jpg")) + list(category_path.glob("*.jpeg")) + list(category_path.glob("*.png"))
        
        if not images:
            print(f"⚠️  Nenhuma imagem encontrada em {category_path}")
            continue
        
        # Embaralhar imagens
        random.shuffle(images)
        
        # Calcular divisão
        split_index = int(len(images) * train_ratio)
        train_images = images[:split_index]
        test_images = images[split_index:]
        
        print(f"\n📁 Categoria: {category}")
        print(f"   Total de imagens: {len(images)}")
        print(f"   Treinamento: {len(train_images)}")
        print(f"   Teste: {len(test_images)}")
        
        # Copiar imagens para treinamento
        for img_path in train_images:
            dest_path = train_dir / category / img_path.name
            shutil.copy2(img_path, dest_path)
            train_count += 1
        
        # Copiar imagens para teste
        for img_path in test_images:
            dest_path = test_dir / category / img_path.name
            shutil.copy2(img_path, dest_path)
            test_count += 1
        
        total_images += len(images)
    
    print(f"\n✅ Divisão concluída!")
    print(f"📊 Resumo:")
    print(f"   Total de imagens: {total_images}")
    print(f"   Treinamento: {train_count} ({train_count/total_images*100:.1f}%)")
    print(f"   Teste: {test_count} ({test_count/total_images*100:.1f}%)")
    print(f"\n📁 Diretórios criados:")
    print(f"   Treinamento: {train_dir}")
    print(f"   Teste: {test_dir}")
    
    return train_dir, test_dir

def main():
    parser = argparse.ArgumentParser(description="Dividir dataset em treinamento e teste")
    parser.add_argument("source_dir", help="Diretório com as imagens originais")
    parser.add_argument("--train-ratio", type=float, default=0.8, 
                       help="Proporção para treinamento (padrão: 0.8)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Semente aleatória (padrão: 42)")
    
    args = parser.parse_args()
    
    if not Path(args.source_dir).exists():
        print(f"❌ Erro: Diretório '{args.source_dir}' não encontrado!")
        return
    
    split_dataset(args.source_dir, args.train_ratio, args.seed)

if __name__ == "__main__":
    main() 