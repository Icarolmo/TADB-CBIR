import cv2
import numpy as np
from database import chroma
from engine import processing_engine as engine
import os
from pathlib import Path
import shutil
import argparse

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Sistema CBIR para Identificação de Doenças em Folhas')
    parser.add_argument('--process-only', action='store_true', 
                      help='Apenas processa o dataset sem fazer busca')
    parser.add_argument('--clear-db', action='store_true',
                      help='Limpa o banco de dados antes de processar')
    return parser.parse_args()


def process_dataset():
    """Processa todas as imagens do dataset"""
    dataset_dir = Path("image/dataset")
    
    # Verificar se há imagens no diretório dataset
    total_images = sum(1 for _ in dataset_dir.rglob("*") 
                      if _.suffix.lower() in ['.jpg', '.jpeg', '.png'])
    
    if total_images == 0:
        print("\nAVISO: Nenhuma imagem encontrada no dataset.")
        print(os.path.abspath("image/dataset"))
        return
    
    print(f"\nEncontradas {total_images} imagens para processar.")
    
    # Dicionário para armazenar estatísticas por categoria
    stats = {}
    
    # Processar cada categoria separadamente
    for category_dir in dataset_dir.iterdir():
        if not category_dir.is_dir():
            continue
            
        category_name = category_dir.name
        stats[category_name] = {"processed": 0, "errors": 0}
        
        print(f"\nProcessando categoria: {category_name}")
        
        # Processar imagens da categoria
        for img_path in category_dir.glob('*'):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                try:
                    result = engine.process_image(str(img_path), save_to_db=True)
                    if "error" in result:
                        print(f"Erro ao processar {img_path.name}: {result['error']}")
                        stats[category_name]["errors"] += 1
                    else:
                        stats[category_name]["processed"] += 1
                        if stats[category_name]["processed"] % 10 == 0:
                            print(f"Processadas {stats[category_name]['processed']} imagens em {category_name}")
                except Exception as e:
                    print(f"Erro ao processar {img_path.name}: {str(e)}")
                    stats[category_name]["errors"] += 1
    
    # Exibir resumo do processamento
    print("\nResumo do processamento por categoria:")
    print("=" * 50)
    
    total_processed = 0
    total_errors = 0
    
    for category, counts in stats.items():
        processed = counts["processed"]
        errors = counts["errors"]
        total = processed + errors
        success_rate = (processed / total * 100) if total > 0 else 0
        
        print(f"\n{category}:")
        print(f"- Processadas com sucesso: {processed}")
        print(f"- Erros: {errors}")
        print(f"- Taxa de sucesso: {success_rate:.1f}%")
        
        total_processed += processed
        total_errors += errors
    
    print("\nEstatísticas gerais:")
    print("=" * 50)
    print(f"Total de imagens processadas: {total_processed}")
    print(f"Total de erros: {total_errors}")
    total = total_processed + total_errors
    success_rate = (total_processed / total * 100) if total > 0 else 0
    print(f"Taxa de sucesso geral: {success_rate:.1f}%")

def process_query_image(image_path):
    """Processa uma imagem de consulta e retorna os resultados"""
    try:
        # Processar imagem com visualização
        result = engine.process_image(image_path, save_to_db=False, visualize=True)
        
        if "error" in result:
            return {"error": result["error"]}
        
        # Consultar imagens similares
        query_results = chroma.query_embedding(result["features"])
        
        if not query_results:
            return {"error": "Erro ao consultar banco de dados"}
        
        # Analisar resultados
        analysis = chroma.analyze_query_results(query_results)
        
        if not analysis:
            return {"error": "Erro ao analisar resultados"}
        
        # Adicionar caminho da visualização ao resultado
        analysis["visualization_path"] = result.get("visualization_path")
        
        return analysis
        
    except Exception as e:
        return {"error": str(e)}

def main():
    """Função principal"""
    args = parse_arguments()
    
    # Limpar o banco de dados
    if args.clear_db:
        print("\nLimpando banco de dados...")
        if chroma.clear_database():
            print("Banco de dados limpo com sucesso!")
            return
        else:
            print("Erro ao limpar banco de dados!")
            return
    
    # Processar dataset
    if args.process_only:
        print("\nVerificando banco de dados...")
        stats = chroma.get_database_stats()
        
        if stats["total_images"] == 0:
            print("Banco de dados vazio. Iniciando processamento do dataset...")
            process_dataset()
        else:
            print("\nBanco de dados já contém imagens:")
            print(f"Total de imagens: {stats['total_images']}")
            print("\nCategorias:")
            for cat, count in stats["categories"].items():
                print(f"- {cat}: {count} imagens")
        return

    # Verificar se há imagens no banco
    stats = chroma.get_database_stats()
    if stats["total_images"] == 0:
        print("\nNenhuma imagem encontrada no banco de dados!")
        print("Execute primeiro com --process-only para processar o dataset.")
        return
    
    # Processar imagem de consulta
    query_path = os.path.join("image", "uploads", "query_leaf.jpg")
    if not os.path.exists(query_path):
        print("\nNenhuma imagem de consulta encontrada!")
        print("Coloque uma imagem em image/uploads/")
        return
    
    print("\nAnalisando imagem...")
    result = process_query_image(query_path)
    
    if isinstance(result, dict) and "error" in result:
        print(f"\nErro ao processar imagem: {result['error']}")
        return
    
    # Exibir resultados
    print("\nResultados da análise:")
    print("=" * 50)
    
    if "identified_category" in result:
        category = result["identified_category"]
        confidence = result["confidence"]
        print(f"\nDoença Identificada: {category}")
        print(f"Nível de Confiança: {confidence:.1f}%")
        
        print("\nDistribuição de categorias:")
        for cat, perc in result["category_distribution"].items():
            cat_name = cat
            print(f"- {cat_name}: {perc:.1f}%")
        
        # Mostrar as 5 imagens mais similares
        print("\nImagens mais similares encontradas:")
        print("=" * 50)
        for i, img in enumerate(result["similar_images"], 1):
            category = img["category"]
            similarity = img["similarity"]
            print(f"\nImagem #{i}:")
            print(f"- Categoria: {category}")
            print(f"- Similaridade: {similarity:.1f}%")
            print(f"- Caminho: {img['metadata']['path']}")
            
        # Adicionar recomendações baseadas na confiança
        print("\nRecomendações:")
        if confidence >= 80:
            print("✅ Diagnóstico altamente confiável")
            print("1. Consulte um especialista para confirmar o diagnóstico")
            print("2. Pesquise tratamentos específicos para", category)
            print("3. Isole as plantas afetadas para evitar propagação")
        elif confidence >= 50:
            print("⚠️ Diagnóstico provável, mas necessita confirmação")
            print("1. Faça uma inspeção visual detalhada da planta")
            print("2. Tire mais fotos de diferentes ângulos")
            print("3. Consulte um especialista para confirmação")
        else:
            print("❓ Diagnóstico incerto")
            print("1. Tire novas fotos com melhor iluminação e foco")
            print("2. Certifique-se de fotografar a área afetada mais de perto")
            print("3. Consulte um especialista para uma avaliação presencial")
    else:
        print("Não foi possível identificar a doença.")
    
    print("\nImagem de análise salva em:", result.get("visualization_path", "N/A"))

if __name__ == "__main__":
    main() 