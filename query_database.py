import argparse
from database import chroma
from datetime import datetime
import json

def show_stats():
    """Mostra estatísticas gerais do banco"""
    stats = chroma.get_database_stats()
    print("\n=== Estatísticas do Banco de Dados ===")
    print(f"Total de imagens: {stats['total_images']}")
    
    print("\nDistribuição por categoria:")
    for categoria, quantidade in stats['categories'].items():
        print(f"- {categoria}: {quantidade} imagens")
    
    if stats['last_update']:
        print(f"\nÚltima atualização: {stats['last_update']}")

def list_images():
    """Lista todas as imagens no banco"""
    results = chroma.leaf_collection.get()
    
    print("\n=== Imagens no Banco de Dados ===")
    for i, (id, metadata) in enumerate(zip(results['ids'], results['metadatas']), 1):
        print(f"\nImagem {i}:")
        print(f"ID: {id}")
        print(f"Categoria: {metadata.get('category', 'desconhecido')}")
        print(f"Data Processamento: {metadata.get('processing_date', 'N/A')}")
        print(f"Caminho: {metadata.get('path', 'N/A')}")

def show_image_details(image_id):
    """Mostra detalhes de uma imagem específica"""
    results = chroma.leaf_collection.get(
        ids=[image_id]
    )
    
    if not results['ids']:
        print(f"\nErro: Imagem '{image_id}' não encontrada!")
        return
    
    print(f"\n=== Detalhes da Imagem: {image_id} ===")
    metadata = results['metadatas'][0]
    features = results['embeddings'][0]
    
    print("\nMetadados:")
    for key, value in metadata.items():
        print(f"- {key}: {value}")
    
    print("\nCaracterísticas:")
    print("1. HSV (primeiros 5 valores de cada):")
    print(f"- Matiz (H): {features[:5]}")
    print(f"- Saturação (S): {features[32:37]}")
    print(f"- Valor (V): {features[64:69]}")
    
    print("\n2. Textura:")
    texture_features = features[96:102]
    texture_names = ['Média k3', 'Desvio k3', 'Média k5', 'Desvio k5', 'Média k7', 'Desvio k7']
    for name, value in zip(texture_names, texture_features):
        print(f"- {name}: {value:.4f}")
    
    print("\n3. Forma:")
    shape_features = features[-4:]
    shape_names = ['Num. Manchas', 'Tam. Médio', 'Desvio Tam.', 'Prop. Máx.']
    for name, value in zip(shape_names, shape_features):
        print(f"- {name}: {value:.4f}")

def search_by_category(category):
    """Lista imagens de uma categoria específica"""
    results = chroma.leaf_collection.get()
    
    print(f"\n=== Imagens da Categoria: {category} ===")
    found = False
    
    for id, metadata in zip(results['ids'], results['metadatas']):
        if metadata.get('category') == category:
            found = True
            print(f"\nID: {id}")
            print(f"Data: {metadata.get('processing_date', 'N/A')}")
            print(f"Caminho: {metadata.get('path', 'N/A')}")
    
    if not found:
        print(f"Nenhuma imagem encontrada na categoria '{category}'")

def export_database(output_file):
    """Exporta os dados do banco para um arquivo JSON"""
    results = chroma.leaf_collection.get()
    
    data = {
        'total_images': len(results['ids']),
        'last_update': str(datetime.now()),
        'images': []
    }
    
    for id, metadata, embedding in zip(results['ids'], results['metadatas'], results['embeddings']):
        image_data = {
            'id': id,
            'metadata': metadata,
            'features': {
                'hsv': embedding[:96],
                'texture': embedding[96:102],
                'shape': embedding[-4:]
            }
        }
        data['images'].append(image_data)
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\nBanco de dados exportado para: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Consulta ao Banco de Dados de Doenças em Folhas')
    
    parser.add_argument('--stats', action='store_true',
                      help='Mostra estatísticas gerais do banco')
    parser.add_argument('--list', action='store_true',
                      help='Lista todas as imagens no banco')
    parser.add_argument('--image', type=str,
                      help='Mostra detalhes de uma imagem específica')
    parser.add_argument('--category', type=str,
                      help='Lista imagens de uma categoria específica')
    parser.add_argument('--export', type=str,
                      help='Exporta o banco para um arquivo JSON')
    
    args = parser.parse_args()
    
    if args.stats:
        show_stats()
    elif args.list:
        list_images()
    elif args.image:
        show_image_details(args.image)
    elif args.category:
        search_by_category(args.category)
    elif args.export:
        export_database(args.export)
    else:
        print("\nPor favor, especifique uma opção de consulta. Exemplo:")
        print("python query_database.py --stats")
        print("python query_database.py --list")
        print("python query_database.py --image nome_imagem.jpg")
        print("python query_database.py --category Pepper__bell___Bacterial_spot")
        print("python query_database.py --export dados.json")

if __name__ == "__main__":
    main() 