#!/usr/bin/env python3
"""
Teste Rápido do Sistema de Avaliação CBIR

Este script testa rapidamente se o sistema de avaliação está funcionando corretamente.
"""

import os
import sys
from pathlib import Path
from evaluation_system import CBIREvaluationSystem
from database import chroma

def test_imports():
    """Testa se todas as dependências estão disponíveis"""
    print("🔍 Testando imports...")
    
    try:
        import numpy as np
        print("✅ NumPy importado")
    except ImportError as e:
        print(f"❌ Erro ao importar NumPy: {e}")
        return False
    
    try:
        import pandas as pd
        print("✅ Pandas importado")
    except ImportError as e:
        print(f"❌ Erro ao importar Pandas: {e}")
        return False
    
    try:
        from sklearn.metrics import accuracy_score
        print("✅ Scikit-learn importado")
    except ImportError as e:
        print(f"❌ Erro ao importar Scikit-learn: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("✅ Matplotlib importado")
    except ImportError as e:
        print(f"❌ Erro ao importar Matplotlib: {e}")
        return False
    
    try:
        import seaborn as sns
        print("✅ Seaborn importado")
    except ImportError as e:
        print(f"❌ Erro ao importar Seaborn: {e}")
        return False
    
    try:
        from evaluation_system import CBIREvaluationSystem
        print("✅ Sistema de avaliação importado")
    except ImportError as e:
        print(f"❌ Erro ao importar sistema de avaliação: {e}")
        return False
    
    try:
        from database import chroma
        print("✅ Módulo de banco de dados importado")
    except ImportError as e:
        print(f"❌ Erro ao importar banco de dados: {e}")
        return False
    
    return True

def test_evaluation_system():
    """Testa a criação do sistema de avaliação"""
    print("\n🔍 Testando sistema de avaliação...")
    
    try:
        evaluator = CBIREvaluationSystem()
        print("✅ Sistema de avaliação criado com sucesso")
        return evaluator
    except Exception as e:
        print(f"❌ Erro ao criar sistema de avaliação: {e}")
        return None

def test_database_connection():
    """Testa a conexão com o banco de dados"""
    print("\n🔍 Testando conexão com banco de dados...")
    
    try:
        stats = chroma.get_database_stats()
        print(f"✅ Conexão com banco estabelecida")
        print(f"   - Total de imagens: {stats['total_images']}")
        print(f"   - Categorias: {stats['categories']}")
        return True
    except Exception as e:
        print(f"❌ Erro ao conectar com banco de dados: {e}")
        return False

def test_revocation_prediction():
    """Testa a previsão de revogação com dados simulados"""
    print("\n🔍 Testando previsão de revogação...")
    
    try:
        evaluator = CBIREvaluationSystem()
        
        # Dados simulados para teste
        mock_result = {
            "similar_images": [
                {"similarity": 85.0, "category": "leaf_healthy", "features": {"shape": {"num_lesions": 0, "disease_coverage": 0, "avg_lesion_size": 0}}},
                {"similarity": 82.0, "category": "leaf_healthy", "features": {"shape": {"num_lesions": 0, "disease_coverage": 0, "avg_lesion_size": 0}}},
                {"similarity": 78.0, "category": "leaf_healthy", "features": {"shape": {"num_lesions": 0, "disease_coverage": 0, "avg_lesion_size": 0}}},
                {"similarity": 75.0, "category": "leaf_healthy", "features": {"shape": {"num_lesions": 0, "disease_coverage": 0, "avg_lesion_size": 0}}},
                {"similarity": 70.0, "category": "leaf_healthy", "features": {"shape": {"num_lesions": 0, "disease_coverage": 0, "avg_lesion_size": 0}}}
            ],
            "confidence": 80.0
        }
        
        prediction = evaluator.predict_revocation_risk(mock_result)
        
        print("✅ Previsão de revogação funcionando")
        print(f"   - Nível de risco: {prediction['revocation_risk']}")
        print(f"   - Score de risco: {prediction['risk_score']:.3f}")
        print(f"   - Fatores de risco: {len(prediction['risk_factors'])}")
        
        return True
    except Exception as e:
        print(f"❌ Erro na previsão de revogação: {e}")
        return False

def test_metrics_calculation():
    """Testa o cálculo de métricas"""
    print("\n🔍 Testando cálculo de métricas...")
    
    try:
        evaluator = CBIREvaluationSystem()
        
        # Dados simulados para teste
        mock_test_results = [
            {
                "true_category": "leaf_healthy",
                "predicted_category": "leaf_healthy",
                "confidence": 85.0,
                "revocation_risk": "BAIXO",
                "risk_score": 0.2
            },
            {
                "true_category": "leaf_healthy",
                "predicted_category": "leaf_healthy",
                "confidence": 78.0,
                "revocation_risk": "MÉDIO",
                "risk_score": 0.5
            },
            {
                "true_category": "leaf_with_disease",
                "predicted_category": "leaf_with_disease",
                "confidence": 82.0,
                "revocation_risk": "BAIXO",
                "risk_score": 0.3
            },
            {
                "true_category": "leaf_with_disease",
                "predicted_category": "leaf_healthy",
                "confidence": 45.0,
                "revocation_risk": "ALTO",
                "risk_score": 0.8
            }
        ]
        
        metrics = evaluator.calculate_performance_metrics(mock_test_results)
        
        print("✅ Cálculo de métricas funcionando")
        print(f"   - Acurácia: {metrics['overall_accuracy']:.3f}")
        print(f"   - Precisão: {metrics['precision']:.3f}")
        print(f"   - Recall: {metrics['recall']:.3f}")
        print(f"   - F1-Score: {metrics['f1_score']:.3f}")
        print(f"   - Confiança média: {metrics['avg_confidence']:.1f}%")
        print(f"   - Score de risco médio: {metrics['avg_risk_score']:.3f}")
        
        return True
    except Exception as e:
        print(f"❌ Erro no cálculo de métricas: {e}")
        return False

def test_directory_structure():
    """Testa se a estrutura de diretórios está correta"""
    print("\n🔍 Testando estrutura de diretórios...")
    
    required_dirs = [
        "image/dataset",
        "image/test_dataset", 
        "image/uploads",
        "database"
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            missing_dirs.append(dir_path)
        else:
            print(f"✅ {dir_path} existe")
    
    if missing_dirs:
        print(f"⚠️ Diretórios faltando: {missing_dirs}")
        print("Execute os seguintes comandos para criar:")
        for dir_path in missing_dirs:
            print(f"   mkdir -p {dir_path}")
        return False
    
    return True

def test_gui_import():
    """Testa se a interface gráfica pode ser importada"""
    print("\n🔍 Testando interface gráfica...")
    
    try:
        import tkinter as tk
        print("✅ Tkinter disponível")
        
        # Testar se matplotlib backend funciona
        import matplotlib
        matplotlib.use('TkAgg')
        print("✅ Matplotlib backend configurado")
        
        return True
    except ImportError as e:
        print(f"❌ Erro ao importar interface gráfica: {e}")
        return False
    except Exception as e:
        print(f"❌ Erro na configuração da interface: {e}")
        return False

def main():
    """Função principal do teste"""
    print("🧪 TESTE RÁPIDO DO SISTEMA DE AVALIAÇÃO CBIR")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Sistema de avaliação", test_evaluation_system),
        ("Conexão com banco", test_database_connection),
        ("Previsão de revogação", test_revocation_prediction),
        ("Cálculo de métricas", test_metrics_calculation),
        ("Estrutura de diretórios", test_directory_structure),
        ("Interface gráfica", test_gui_import)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ Erro inesperado em {test_name}: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 RESULTADO DOS TESTES: {passed}/{total} passaram")
    
    if passed == total:
        print("🎉 Todos os testes passaram! O sistema está funcionando corretamente.")
        print("\n📖 Próximos passos:")
        print("1. Processe um dataset: python cbir.py --process-only")
        print("2. Teste a previsão: python cbir.py")
        print("3. Execute avaliação: python cbir.py --evaluate")
        print("4. Use a interface: python evaluation_gui.py")
        print("5. Veja a demonstração: python demo_evaluation.py")
    else:
        print("⚠️ Alguns testes falharam. Verifique as dependências e configuração.")
        print("\n🔧 Soluções comuns:")
        print("1. Instale dependências: pip install -r requirements.txt")
        print("2. Crie diretórios: mkdir -p image/dataset image/test_dataset image/uploads")
        print("3. Verifique se o banco de dados está configurado")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 