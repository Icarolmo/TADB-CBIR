#!/usr/bin/env python3
"""
Demonstração do Sistema de Avaliação CBIR com Previsão de Revogação

Este script demonstra como usar o sistema de avaliação e previsão de revogação
implementado para o CBIR de identificação de doenças em folhas.

Funcionalidades demonstradas:
1. Previsão de risco de revogação
2. Avaliação completa do sistema
3. Geração de relatórios
4. Análise de métricas de performance
"""

import os
import sys
from pathlib import Path
from evaluation_system import CBIREvaluationSystem
from database import chroma
from engine import processing_engine as engine

def print_header(title):
    """Imprime um cabeçalho formatado"""
    print("\n" + "="*60)
    print(title.center(60))
    print("="*60)

def print_section(title):
    """Imprime uma seção formatada"""
    print(f"\n{'-'*40}")
    print(f"📋 {title}")
    print(f"{'-'*40}")

def demo_revocation_prediction():
    """Demonstra a previsão de revogação"""
    print_header("DEMONSTRAÇÃO: PREVISÃO DE REVOGAÇÃO")
    
    # Criar sistema de avaliação
    evaluator = CBIREvaluationSystem()
    
    # Verificar se há imagens no banco
    stats = chroma.get_database_stats()
    if stats["total_images"] == 0:
        print("❌ Nenhuma imagem no banco de dados!")
        print("Execute primeiro: python cbir.py --process-only")
        return
    
    print(f"✅ Banco de dados contém {stats['total_images']} imagens")
    print(f"Categorias: {stats['categories']}")
    
    # Verificar se há imagem de consulta
    query_path = "image/uploads/query_leaf.jpg"
    if not os.path.exists(query_path):
        print(f"❌ Imagem de consulta não encontrada: {query_path}")
        print("Coloque uma imagem em image/uploads/query_leaf.jpg")
        return
    
    print(f"✅ Imagem de consulta encontrada: {query_path}")
    
    # Processar imagem de consulta
    print_section("PROCESSANDO IMAGEM DE CONSULTA")
    try:
        result = engine.process_image(query_path, save_to_db=False, visualize=False)
        
        if "error" in result:
            print(f"❌ Erro ao processar imagem: {result['error']}")
            return
        
        print("✅ Imagem processada com sucesso")
        
        # Consultar banco de dados
        from datetime import datetime
        query_metadata = {
            "path": query_path,
            "type": "demo_query",
            "processing_date": str(datetime.now()),
            "category": "demo"
        }
        
        query_results = chroma.query_embedding(result["features"], metadata=query_metadata)
        
        if not query_results:
            print("❌ Erro ao consultar banco de dados")
            return
        
        # Analisar resultados
        analysis = chroma.analyze_query_results(query_results)
        
        if not analysis:
            print("❌ Erro ao analisar resultados")
            return
        
        print("✅ Análise concluída")
        
        # Prever risco de revogação
        print_section("PREVISÃO DE RISCO DE REVOGAÇÃO")
        revocation_prediction = evaluator.predict_revocation_risk(analysis)
        
        print(f"🎯 Categoria identificada: {analysis.get('identified_category', 'N/A')}")
        print(f"📊 Confiança: {analysis.get('confidence', 0):.1f}%")
        print(f"⚠️ Nível de risco: {revocation_prediction['revocation_risk']}")
        print(f"📈 Score de risco: {revocation_prediction['risk_score']:.3f}")
        
        if revocation_prediction['risk_factors']:
            print("\n🔍 Fatores de risco identificados:")
            for factor in revocation_prediction['risk_factors']:
                print(f"  • {factor}")
        else:
            print("\n✅ Nenhum fator de risco significativo identificado")
        
        # Exibir características extraídas
        features = evaluator.extract_confidence_features(analysis)
        if features:
            print_section("CARACTERÍSTICAS DE CONFIANÇA")
            print(f"• Máxima similaridade: {features['max_similarity']:.1f}%")
            print(f"• Mínima similaridade: {features['min_similarity']:.1f}%")
            print(f"• Média de similaridade: {features['mean_similarity']:.1f}%")
            print(f"• Desvio padrão: {features['std_similarity']:.1f}%")
            print(f"• Consistência de categoria: {features['category_consistency']:.3f}")
            print(f"• Gap de similaridade: {features['similarity_gap']:.1f}%")
            print(f"• Variabilidade de forma: {features['shape_variability']:.3f}")
        
        # Recomendações
        print_section("RECOMENDAÇÕES")
        risk_level = revocation_prediction['revocation_risk']
        
        if risk_level == "ALTO":
            print("🚨 ALTO RISCO DE REVOGAÇÃO")
            print("Recomendações:")
            print("1. Consulte um especialista para confirmação")
            print("2. Tire novas fotos com melhor iluminação")
            print("3. Use imagens de diferentes ângulos")
            print("4. Verifique se a imagem está bem focada")
        elif risk_level == "MÉDIO":
            print("⚠️ RISCO MÉDIO DE REVOGAÇÃO")
            print("Recomendações:")
            print("1. Faça inspeção visual detalhada")
            print("2. Tire mais fotos de diferentes ângulos")
            print("3. Consulte especialista se necessário")
        else:
            print("✅ BAIXO RISCO DE REVOGAÇÃO")
            print("O diagnóstico tem alta confiabilidade")
        
    except Exception as e:
        print(f"❌ Erro durante demonstração: {str(e)}")

def demo_system_evaluation():
    """Demonstra a avaliação completa do sistema"""
    print_header("DEMONSTRAÇÃO: AVALIAÇÃO COMPLETA DO SISTEMA")
    
    # Verificar dataset de teste
    test_dataset = "image/test_dataset"
    if not os.path.exists(test_dataset):
        print(f"❌ Dataset de teste não encontrado: {test_dataset}")
        print("Crie um diretório 'image/test_dataset' com subdiretórios para cada categoria")
        print("Exemplo:")
        print("  image/test_dataset/")
        print("  ├── leaf_healthy/")
        print("  │   ├── healthy1.jpg")
        print("  │   └── healthy2.jpg")
        print("  └── leaf_with_disease/")
        print("      ├── disease1.jpg")
        print("      └── disease2.jpg")
        return
    
    print(f"✅ Dataset de teste encontrado: {test_dataset}")
    
    # Contar imagens de teste
    test_images = []
    for category_dir in Path(test_dataset).iterdir():
        if category_dir.is_dir():
            category_images = list(category_dir.glob('*.jpg')) + list(category_dir.glob('*.png'))
            test_images.extend(category_images)
            print(f"  • {category_dir.name}: {len(category_images)} imagens")
    
    if not test_images:
        print("❌ Nenhuma imagem de teste encontrada")
        return
    
    print(f"✅ Total de imagens de teste: {len(test_images)}")
    
    # Criar sistema de avaliação
    evaluator = CBIREvaluationSystem()
    
    # Executar avaliação
    print_section("EXECUTANDO AVALIAÇÃO")
    print("Isso pode levar alguns minutos...")
    
    try:
        evaluation_result = evaluator.evaluate_system_performance(test_dataset, None)
        
        if not evaluation_result:
            print("❌ Falha na avaliação do sistema")
            return
        
        metrics = evaluation_result["metrics"]
        test_results = evaluation_result["test_results"]
        
        print("✅ Avaliação concluída com sucesso!")
        
        # Exibir resultados
        print_section("RESULTADOS DA AVALIAÇÃO")
        print(f"📊 Acurácia geral: {metrics['overall_accuracy']:.3f}")
        print(f"🎯 Precisão: {metrics['precision']:.3f}")
        print(f"📈 Recall: {metrics['recall']:.3f}")
        print(f"⚖️ F1-Score: {metrics['f1_score']:.3f}")
        
        print(f"\n📊 Análise de confiança:")
        print(f"• Confiança média: {metrics['avg_confidence']:.1f}%")
        print(f"• Desvio padrão: {metrics['std_confidence']:.1f}%")
        print(f"• Score médio de risco: {metrics['avg_risk_score']:.3f}")
        
        # Análise de revogação
        revocation_analysis = evaluator.analyze_revocation_patterns(test_results)
        
        print_section("ANÁLISE DE RISCO DE REVOGAÇÃO")
        for risk_level, analysis in revocation_analysis.items():
            print(f"\n⚠️ Risco {risk_level}:")
            print(f"  • Quantidade: {analysis['count']} imagens")
            print(f"  • Acurácia: {analysis['accuracy']:.3f}")
            print(f"  • Confiança média: {analysis['avg_confidence']:.1f}%")
            print(f"  • Score de risco: {analysis['avg_risk_score']:.3f}")
        
        # Gerar relatório visual
        print_section("GERANDO RELATÓRIO VISUAL")
        try:
            report_path = evaluator.generate_evaluation_report(metrics)
            print(f"✅ Relatório gerado: {report_path}")
        except Exception as e:
            print(f"⚠️ Erro ao gerar relatório: {str(e)}")
        
        print(f"\n✅ Demonstração concluída!")
        print(f"Total de imagens testadas: {metrics['total_tests']}")
        
    except Exception as e:
        print(f"❌ Erro durante avaliação: {str(e)}")

def demo_usage_instructions():
    """Demonstra instruções de uso"""
    print_header("INSTRUÇÕES DE USO")
    
    print_section("1. PREPARAÇÃO DO SISTEMA")
    print("Para usar o sistema de avaliação, você precisa:")
    print("1. Processar um dataset de treinamento:")
    print("   python cbir.py --process-only")
    print("2. Criar um dataset de teste:")
    print("   mkdir -p image/test_dataset/leaf_healthy")
    print("   mkdir -p image/test_dataset/leaf_with_disease")
    print("3. Adicionar imagens de teste nas pastas correspondentes")
    
    print_section("2. PREVISÃO DE REVOGAÇÃO")
    print("Para testar a previsão de revogação:")
    print("1. Coloque uma imagem em image/uploads/query_leaf.jpg")
    print("2. Execute: python cbir.py")
    print("3. O sistema mostrará o risco de revogação")
    
    print_section("3. AVALIAÇÃO COMPLETA")
    print("Para avaliar o sistema completo:")
    print("1. Via linha de comando:")
    print("   python cbir.py --evaluate --test-dataset image/test_dataset")
    print("2. Via interface gráfica:")
    print("   python evaluation_gui.py")
    print("3. Via script de avaliação:")
    print("   python evaluation_system.py --test-dataset image/test_dataset")
    
    print_section("4. INTERFACE GRÁFICA")
    print("Para usar a interface gráfica:")
    print("1. Execute: python evaluation_gui.py")
    print("2. Configure o dataset de teste")
    print("3. Clique em 'Executar Avaliação'")
    print("4. Visualize os resultados nas abas")
    
    print_section("5. INTERPRETAÇÃO DOS RESULTADOS")
    print("• Risco ALTO: Considere revisar o dataset ou melhorar a qualidade das imagens")
    print("• Risco MÉDIO: O sistema está funcionando, mas pode ser melhorado")
    print("• Risco BAIXO: O sistema está funcionando bem")
    print("• Confiança ≥80%: Diagnóstico altamente confiável")
    print("• Confiança 60-80%: Diagnóstico provável, mas necessita confirmação")
    print("• Confiança <60%: Diagnóstico incerto")

def main():
    """Função principal da demonstração"""
    print_header("SISTEMA DE AVALIAÇÃO CBIR - DEMONSTRAÇÃO")
    
    print("Este script demonstra as funcionalidades do sistema de avaliação")
    print("e previsão de revogação implementado para o CBIR.")
    
    while True:
        print("\n" + "="*40)
        print("OPÇÕES DE DEMONSTRAÇÃO:")
        print("1. Previsão de revogação")
        print("2. Avaliação completa do sistema")
        print("3. Instruções de uso")
        print("4. Sair")
        print("="*40)
        
        choice = input("\nEscolha uma opção (1-4): ").strip()
        
        if choice == "1":
            demo_revocation_prediction()
        elif choice == "2":
            demo_system_evaluation()
        elif choice == "3":
            demo_usage_instructions()
        elif choice == "4":
            print("\n👋 Obrigado por usar o sistema de avaliação CBIR!")
            break
        else:
            print("❌ Opção inválida. Escolha 1, 2, 3 ou 4.")

if __name__ == "__main__":
    main() 