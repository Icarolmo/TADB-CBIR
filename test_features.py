import cv2
import numpy as np
from engine import processing_engine as engine
from database import chroma
import os
from pathlib import Path
import argparse
import traceback

def test_single_image(image_path: str, save_to_db: bool = False):
    """
    Testa a extração de características de uma única imagem.
    
    Args:
        image_path: Caminho para a imagem
        save_to_db: Se True, salva as características no banco de dados
    """
    try:
        print(f"\nTestando extração de características: {image_path}")
        
        # Processar imagem com visualização
        result = engine.process_image(image_path, save_to_db=save_to_db, visualize=True)
        
        if "error" in result:
            print(f"Erro ao processar imagem: {result['error']}")
            return
        
        # Mostrar informações sobre as características extraídas
        features = result['features']
        names = result['feature_names']
        
        print("\nCaracterísticas extraídas:")
        
        # 1. Histograma HSV
        print("\n1. Histograma HSV:")
        print("- Tamanho total:", len(features[:96]))
        print("- Faixas H:", len(features[:32]))
        print("- Faixas S:", len(features[32:64]))
        print("- Faixas V:", len(features[64:96]))
        
        # 2. Estatísticas HSV
        print("\n2. Estatísticas HSV:")
        hsv_stats = features[96:108]  # 12 valores (4 para cada canal)
        stat_names = ['media', 'desvio', 'q25', 'q75']
        channels = ['H', 'S', 'V']
        
        for i, channel in enumerate(channels):
            for j, stat in enumerate(stat_names):
                print(f"- {channel}_{stat}: {hsv_stats[i*4 + j]:.4f}")
        
        # 3. Características GLCM
        print("\n3. Características GLCM:")
        glcm_features = features[108:116]
        glcm_names = [
            'Contraste', 'Correlação', 'Energia', 'Homogeneidade',
            'Dissimilaridade', 'Entropia', 'Cluster_Shade', 'Cluster_Prominence'
        ]
        for name, value in zip(glcm_names, glcm_features):
            print(f"- {name}: {value:.4f}")
        
        # 4. Características LBP
        print("\n4. Características LBP:")
        lbp_features = features[116:120]
        lbp_names = ['Média', 'Desvio', 'Energia', 'Entropia']
        for name, value in zip(lbp_names, lbp_features):
            print(f"- {name}: {value:.4f}")
        
        # 5. Características de Forma
        print("\n5. Características de Forma:")
        shape_features = features[-8:]
        shape_names = [
            'Num_Lesoes', 'Tamanho_Medio', 'Desvio_Tamanho', 
            'Area_Afetada', 'Densidade_Lesoes', 'Circularidade',
            'Dist_Media_Lesoes', 'Desvio_Dist_Lesoes'
        ]
        for name, value in zip(shape_names, shape_features):
            print(f"- {name}: {value:.4f}")
        
        print(f"\nVisualização salva em: {result['visualization_path']}")
        
        if save_to_db:
            try:
                # Garantir que o diretório do banco existe
                os.makedirs("database/chroma_db", exist_ok=True)
                
                # Adicionar ao banco
                chroma.add_embedding(
                    id=os.path.basename(image_path),
                    embedding=features,
                    metadata={"path": image_path}
                )
                
                print("\nCaracterísticas salvas no banco de dados!")
                
                # Mostrar estatísticas atualizadas
                show_database_stats()
                
            except Exception as e:
                print(f"\nErro ao salvar no banco de dados: {str(e)}")
                print("\nStack trace:")
                traceback.print_exc()
                
    except Exception as e:
        print(f"Erro ao processar imagem: {str(e)}")
        print("\nStack trace:")
        traceback.print_exc()

def show_database_stats():
    """Mostra estatísticas do banco de dados"""
    try:
        stats = chroma.get_database_stats()
        
        print("\nEstatísticas do Banco de Dados:")
        print(f"Total de imagens: {stats['total_images']}")
        
        if stats['categories']:
            print("\nImagens por categoria:")
            for category, count in stats['categories'].items():
                print(f"- {category}: {count}")
        
        if stats['last_update']:
            print(f"\nÚltima atualização: {stats['last_update']}")
            
    except Exception as e:
        print(f"\nErro ao buscar estatísticas: {str(e)}")
        print("\nStack trace:")
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description='Teste de Extração de Características')
    parser.add_argument('--image', type=str, help='Caminho para a imagem a ser testada')
    parser.add_argument('--save', action='store_true', help='Salvar características no banco')
    parser.add_argument('--clear-db', action='store_true', help='Limpar banco de dados')
    parser.add_argument('--stats', action='store_true', help='Mostrar estatísticas do banco')
    
    args = parser.parse_args()
    
    try:
        if args.clear_db:
            chroma.clear_database()
            print("Banco de dados limpo!")
            return
        
        if args.stats:
            show_database_stats()
            return
        
        if args.image:
            test_single_image(args.image, save_to_db=args.save)
        else:
            print("Por favor, especifique uma imagem usando --image")
            print("\nExemplo de uso:")
            print("python test_features.py --image image/dataset/Pepper__bell___Bacterial_spot/imagem1.jpg")
            print("python test_features.py --image image/dataset/Pepper__bell___Bacterial_spot/imagem1.jpg --save")
            print("python test_features.py --stats")
            print("python test_features.py --clear-db")
            
    except Exception as e:
        print(f"\nErro no programa: {str(e)}")
        print("\nStack trace:")
        traceback.print_exc()

if __name__ == "__main__":
    main() 