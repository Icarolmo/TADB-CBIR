import cv2
import numpy as np
from database import chroma
from engine import processing_engine as engine
import os
from pathlib import Path
import shutil
import argparse
from datetime import datetime
from evaluation_system import CBIREvaluationSystem

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Sistema CBIR para Identificação de Doenças em Folhas')
    parser.add_argument('--process-only', action='store_true', 
                      help='Apenas processa o dataset sem fazer busca')
    parser.add_argument('--clear-db', action='store_true',
                      help='Limpa o banco de dados antes de processar')
    parser.add_argument('--evaluate', action='store_true',
                      help='Executa avaliação do sistema')
    parser.add_argument('--test-dataset', type=str, default='image/test_dataset',
                      help='Caminho para dataset de teste (usado com --evaluate)')
    parser.add_argument('--train-dir', type=str, default='image/dataset',
                      help='Caminho para o diretório de referência (usado com --process-only)')
    parser.add_argument('--generate-report', action='store_true',
                      help='Gera relatório visual da avaliação')
    return parser.parse_args()


def process_dataset(dataset_path):
    """Processa todas as imagens do dataset"""
    dataset_dir = Path(dataset_path)
    
    # Verificar se o diretório existe
    if not dataset_dir.exists():
        print(f"\n❌ ERRO: O diretório de referência '{dataset_path}' não foi encontrado.")
        return
    
    # Verificar se há imagens no diretório dataset
    total_images = sum(1 for _ in dataset_dir.rglob("*") 
                      if _.suffix.lower() in ['.jpg', '.jpeg', '.png'])
    
    if total_images == 0:
        print("\nAVISO: Nenhuma imagem encontrada no conjunto de referência.")
        print(os.path.abspath("image/dataset"))
        return
    
    print(f"\nEncontradas {total_images} imagens para processar no conjunto de referência.")
    
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
        # Processar imagem original para consulta
        query_result = engine.process_image(image_path, save_to_db=False, visualize=False)
        
        if "error" in query_result:
            return {"error": query_result["error"]}
        
        # Processar imagem com visualização (apenas para gerar a imagem de análise)
        result = engine.process_image(image_path, save_to_db=False, visualize=True)
        
        if "error" in result:
            return {"error": result["error"]}
        
        # Adicionar metadados da imagem de consulta
        query_metadata = {
            "path": image_path,
            "type": "leaf_disease",
            "processing_date": str(datetime.now()),
            "category": "query"  # Marcar como imagem de consulta
        }
        
        # Consultar imagens similares usando as características da imagem original
        query_results = chroma.query_embedding(query_result["features"], metadata=query_metadata)
        
        if not query_results:
            return {"error": "Erro ao consultar banco de dados"}
        
        # Analisar resultados
        analysis = chroma.analyze_query_results(query_results)
        
        if not analysis:
            return {"error": "Erro ao analisar resultados"}
        
        # Adicionar caminho da visualização ao resultado
        analysis["visualization_path"] = result.get("visualization_path")
        
        # Atualizar a categoria da imagem de análise para corresponder ao resultado
        analysis_path = result.get("visualization_path")
        if analysis_path:
            # Processar a imagem de análise novamente com a categoria correta
            analysis_metadata = {
                "path": analysis_path,
                "type": "leaf_disease_analysis",
                "processing_date": str(datetime.now()),
                "original_image": image_path,
                "category": analysis["identified_category"]  # Usar a categoria identificada (leaf_healthy ou leaf_with_disease)
            }
            engine.process_image(analysis_path, save_to_db=True, visualize=False, metadata=analysis_metadata)
        
        return analysis
        
    except Exception as e:
        return {"error": str(e)}

def evaluate_system():
    """Executa avaliação completa do sistema"""
    print("\n=== AVALIAÇÃO DO SISTEMA CBIR ===")
    
    # Criar sistema de avaliação
    evaluator = CBIREvaluationSystem()
    
    # Executar avaliação
    evaluation_result = evaluator.evaluate_system_performance(
        args.test_dataset, 
        None  # ground_truth_file
    )
    
    if evaluation_result:
        metrics = evaluation_result["metrics"]
        test_results = evaluation_result["test_results"]
        
        # Exibir resultados
        print("\n" + "="*60)
        print("RESULTADOS DA AVALIAÇÃO".center(60))
        print("="*60)
        
        print(f"\n📊 MÉTRICAS GERAIS:")
        print(f"• Acurácia geral: {metrics['overall_accuracy']:.3f}")
        print(f"• Precisão: {metrics['precision']:.3f}")
        print(f"• Recall: {metrics['recall']:.3f}")
        print(f"• F1-Score: {metrics['f1_score']:.3f}")
        
        print(f"\n📈 ANÁLISE DE CONFIANÇA:")
        print(f"• Confiança média: {metrics['avg_confidence']:.1f}%")
        print(f"• Desvio padrão da confiança: {metrics['std_confidence']:.1f}%")
        print(f"• Score médio de risco: {metrics['avg_risk_score']:.3f}")
        
        print(f"\n🎯 DISTRIBUIÇÃO POR CONFIANÇA:")
        conf_analysis = metrics['confidence_analysis']
        print(f"• Alta confiança (≥80%): {conf_analysis['high_confidence']['count']} imagens")
        if conf_analysis['high_confidence']['count'] > 0:
            print(f"  - Acurácia: {conf_analysis['high_confidence']['accuracy']:.3f}")
        print(f"• Média confiança (60-80%): {conf_analysis['medium_confidence']['count']} imagens")
        print(f"• Baixa confiança (<60%): {conf_analysis['low_confidence']['count']} imagens")
        
        # Análise de padrões de revogação
        revocation_analysis = evaluator.analyze_revocation_patterns(test_results)
        
        print(f"\n⚠️ ANÁLISE DE RISCO DE REVOGAÇÃO:")
        for risk_level, analysis in revocation_analysis.items():
            print(f"• Risco {risk_level}:")
            print(f"  - Quantidade: {analysis['count']} imagens")
            print(f"  - Acurácia: {analysis['accuracy']:.3f}")
            print(f"  - Confiança média: {analysis['avg_confidence']:.1f}%")
            print(f"  - Score de risco médio: {analysis['avg_risk_score']:.3f}")
        
        # Gerar relatório visual
        if args.generate_report:
            print(f"\n📋 Gerando relatório visual...")
            report_path = evaluator.generate_evaluation_report(metrics)
            print(f"Relatório gerado: {report_path}")
        
        print(f"\n✅ Avaliação concluída com sucesso!")
        print(f"Total de imagens testadas: {metrics['total_tests']}")
    
    else:
        print("❌ Falha na avaliação do sistema")

def main():
    """Função principal"""
    global args
    args = parse_arguments()
    
    if args.clear_db:
        print("Limpando banco de dados...")
        chroma.clear_database()
        print("Banco de dados limpo.")

    if args.process_only:
        print(f"Iniciando processamento do conjunto de referência em: {args.train_dir}")
        process_dataset(args.train_dir)
        return

    if args.evaluate:
        evaluate_system()
        return
    
    # Verificar se há imagens no banco
    stats = chroma.get_database_stats()
    if stats["total_images"] == 0:
        print("\nNenhuma imagem encontrada no banco de dados!")
        print("Execute primeiro com --process-only para processar o conjunto de referência.")
        return
    
    # Processar imagem de consulta
    query_path = os.path.join("image", "uploads", "query_leaf.jpg")
    if not os.path.exists(query_path):
        print("\nNenhuma imagem de consulta encontrada!")
        print("Coloque uma imagem em image/uploads/")
        return
    
    print("\n=== SISTEMA DE ANÁLISE DE DOENÇAS EM FOLHAS ===")
    print("Analisando imagem...")
    result = process_query_image(query_path)
    
    if isinstance(result, dict) and "error" in result:
        print(f"\nErro ao processar imagem: {result['error']}")
        return
    
    # Criar sistema de avaliação para previsão de revogação
    evaluator = CBIREvaluationSystem()
    revocation_prediction = evaluator.predict_revocation_risk(result)
    
    # Exibir resultados
    print("\n" + "="*50)
    print("RESULTADOS DA ANÁLISE".center(50))
    print("="*50)
    
    if "identified_category" in result:
        category = result["identified_category"]
        confidence = result["confidence"]
        
        # Exibir diagnóstico
        print("\n📋 DIAGNÓSTICO")
        print("-"*50)
        print(f"Categoria identificada: {category}")
        print(f"Nível de confiança: {confidence:.1f}%")
        
        # Exibir previsão de revogação
        print("\n⚠️ PREVISÃO DE REVOGAÇÃO")
        print("-"*50)
        risk_level = revocation_prediction["revocation_risk"]
        risk_score = revocation_prediction["risk_score"]
        risk_factors = revocation_prediction["risk_factors"]
        
        print(f"Nível de risco: {risk_level}")
        print(f"Score de risco: {risk_score:.3f}")
        
        if risk_factors:
            print("Fatores de risco identificados:")
            for factor in risk_factors:
                print(f"• {factor}")
        else:
            print("Nenhum fator de risco significativo identificado")
        
        # Exibir distribuição de categorias
        print("\n📊 DISTRIBUIÇÃO DE CATEGORIAS")
        print("-"*50)
        for cat, perc in result["category_distribution"].items():
            cat_name = "Folha Saudável" if cat == "leaf_healthy" else "Folha com Doença"
            print(f"• {cat_name}: {perc:.1f}%")
        
        # Exibir imagens similares
        print("\n🔍 IMAGENS MAIS SIMILARES ENCONTRADAS")
        print("-"*50)
        # Ordenar imagens por similaridade em ordem decrescente
        sorted_images = sorted(result["similar_images"], key=lambda x: x["similarity"], reverse=True)
        for i, img in enumerate(sorted_images, 1):
            category = "Folha Saudável" if img["category"] == "leaf_healthy" else "Folha com Doença"
            similarity = img["similarity"]
            print(f"\nImagem #{i}:")
            print(f"• Categoria: {category}")
            print(f"• Similaridade: {similarity:.1f}%")
            print(f"• Caminho: {img['metadata']['path']}")
            
        # Adicionar recomendações baseadas na confiança e risco de revogação
        print("\n💡 RECOMENDAÇÕES")
        print("-"*50)
        
        # Recomendações baseadas na confiança
        if confidence >= 80:
            print("✅ Diagnóstico altamente confiável")
        elif confidence >= 50:
            print("⚠️ Diagnóstico provável, mas necessita confirmação")
        else:
            print("❓ Diagnóstico incerto")
        
        # Recomendações baseadas no risco de revogação
        if risk_level == "ALTO":
            print("🚨 ALTO RISCO DE REVOGAÇÃO - Recomendações especiais:")
            print("1. Consulte um especialista para confirmação")
            print("2. Tire novas fotos com melhor iluminação e ângulos")
            print("3. Considere usar imagens de diferentes partes da planta")
            print("4. Verifique se a imagem está bem focada e sem sombras")
        elif risk_level == "MÉDIO":
            print("⚠️ RISCO MÉDIO DE REVOGAÇÃO:")
            print("1. Faça uma inspeção visual detalhada da planta")
            print("2. Tire mais fotos de diferentes ângulos")
            print("3. Consulte um especialista para confirmação")
        else:
            print("✅ BAIXO RISCO DE REVOGAÇÃO")
            print("O diagnóstico tem alta confiabilidade")
        
        # Ações recomendadas baseadas na categoria
        print("\nAções recomendadas:")
        if "healthy" in category.lower():
            print("1. Continue monitorando a planta regularmente")
            print("2. Mantenha as práticas de cuidado atuais")
            print("3. Tire fotos periódicas para acompanhamento")
        else:
            print("1. Consulte um especialista para confirmar o diagnóstico")
            print("2. Pesquise tratamentos específicos para", category)
            print("3. Isole as plantas afetadas para evitar propagação")
            print("4. Monitore outras plantas próximas")
    else:
        print("Não foi possível identificar a doença.")
    
    print("\n" + "="*50)
    print(f"Imagem de análise salva em: {result.get('visualization_path', 'N/A')}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main() 