import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import os
from pathlib import Path
from database import chroma
from engine import processing_engine as engine
import warnings
warnings.filterwarnings('ignore')

class CBIREvaluationSystem:
    """
    Sistema de avaliação e previsão de revogação para CBIR
    """
    
    def __init__(self):
        self.revocation_model = None
        self.confidence_threshold = 0.7
        self.anomaly_detector = None
        self.evaluation_history = []
        self.metrics_history = []
        
    def extract_confidence_features(self, query_result):
        """
        Extrai características para previsão de revogação
        """
        if not query_result or "similar_images" not in query_result:
            return None
            
        similar_images = query_result["similar_images"]
        confidence = query_result.get("confidence", 0)
        
        if len(similar_images) < 3:
            return None
            
        # Características baseadas na distribuição de similaridade
        similarities = [img["similarity"] for img in similar_images]
        max_sim = max(similarities)
        min_sim = min(similarities)
        mean_sim = np.mean(similarities)
        std_sim = np.std(similarities)
        
        # Características baseadas na consistência de categoria
        categories = [img["category"] for img in similar_images]
        category_counts = {}
        for cat in categories:
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        dominant_category = max(category_counts.items(), key=lambda x: x[1])
        category_consistency = dominant_category[1] / len(categories)
        
        # Características baseadas nas diferenças entre imagens similares
        top_3_similarities = similarities[:3]
        similarity_gap = max_sim - min(top_3_similarities)
        
        # Características de forma das imagens similares
        shape_features = []
        for img in similar_images[:3]:
            features = img.get("features", {})
            shape = features.get("shape", {})
            shape_features.extend([
                shape.get("num_lesions", 0),
                shape.get("disease_coverage", 0),
                shape.get("avg_lesion_size", 0)
            ])
        
        # Calcular variabilidade das características de forma
        shape_variability = np.std(shape_features) if shape_features else 0
        
        return {
            "confidence": confidence,
            "max_similarity": max_sim,
            "min_similarity": min_sim,
            "mean_similarity": mean_sim,
            "std_similarity": std_sim,
            "category_consistency": category_consistency,
            "similarity_gap": similarity_gap,
            "shape_variability": shape_variability,
            "num_similar_images": len(similar_images)
        }
    
    def predict_revocation_risk(self, query_result):
        """
        Prevê o risco de revogação da classificação
        """
        features = self.extract_confidence_features(query_result)
        if not features:
            return {
                "revocation_risk": "ALTO",
                "confidence": 0,
                "reason": "Dados insuficientes para análise"
            }
        
        # Análise baseada em regras
        risk_factors = []
        risk_score = 0
        
        # Fator 1: Confiança baixa
        if features["confidence"] < 60:
            risk_factors.append("Confiança muito baixa")
            risk_score += 0.4
        
        # Fator 2: Alta variabilidade de similaridade
        if features["std_similarity"] > 15:
            risk_factors.append("Alta variabilidade nas similaridades")
            risk_score += 0.3
        
        # Fator 3: Baixa consistência de categoria
        if features["category_consistency"] < 0.6:
            risk_factors.append("Baixa consistência de categoria")
            risk_score += 0.3
        
        # Fator 4: Gap muito grande entre similaridades
        if features["similarity_gap"] > 20:
            risk_factors.append("Grande diferença entre imagens similares")
            risk_score += 0.2
        
        # Fator 5: Alta variabilidade nas características de forma
        if features["shape_variability"] > 0.5:
            risk_factors.append("Alta variabilidade nas características de forma")
            risk_score += 0.2
        
        # Determinar nível de risco
        if risk_score >= 0.8:
            risk_level = "ALTO"
        elif risk_score >= 0.5:
            risk_level = "MÉDIO"
        else:
            risk_level = "BAIXO"
        
        return {
            "revocation_risk": risk_level,
            "risk_score": risk_score,
            "confidence": features["confidence"],
            "risk_factors": risk_factors,
            "features": features
        }
    
    def evaluate_system_performance(self, test_dataset_path, ground_truth_file=None):
        """
        Avalia a performance do sistema CBIR
        """
        print("=== AVALIAÇÃO DO SISTEMA CBIR ===")
        
        # Obter estatísticas do banco
        db_stats = chroma.get_database_stats()
        print(f"\nBanco de dados atual:")
        print(f"- Total de imagens: {db_stats['total_images']}")
        print(f"- Categorias: {db_stats['categories']}")
        
        # Processar imagens de teste
        test_results = []
        test_dataset = Path(test_dataset_path)
        
        if not test_dataset.exists():
            print(f"Erro: Dataset de teste não encontrado em {test_dataset_path}")
            return None
        
        print(f"\nProcessando dataset de teste: {test_dataset_path}")
        
        # Processar cada categoria
        for category_dir in test_dataset.iterdir():
            if not category_dir.is_dir():
                continue
                
            category_name = category_dir.name
            print(f"\nProcessando categoria: {category_name}")
            
            for img_path in category_dir.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    try:
                        # Processar imagem
                        result = engine.process_image(str(img_path), save_to_db=False, visualize=False)
                        
                        if "error" in result:
                            print(f"Erro ao processar {img_path.name}: {result['error']}")
                            continue
                        
                        # Consultar banco de dados
                        query_metadata = {
                            "path": str(img_path),
                            "type": "test_image",
                            "processing_date": str(datetime.now()),
                            "category": "test"
                        }
                        
                        query_results = chroma.query_embedding(result["features"], metadata=query_metadata)
                        
                        if not query_results:
                            continue
                        
                        # Analisar resultados
                        analysis = chroma.analyze_query_results(query_results)
                        
                        if analysis:
                            # Prever risco de revogação
                            revocation_prediction = self.predict_revocation_risk(analysis)
                            
                            test_results.append({
                                "image_path": str(img_path),
                                "true_category": category_name,
                                "predicted_category": analysis.get("identified_category", "unknown"),
                                "confidence": analysis.get("confidence", 0),
                                "revocation_risk": revocation_prediction["revocation_risk"],
                                "risk_score": revocation_prediction["risk_score"],
                                "analysis": analysis,
                                "revocation_prediction": revocation_prediction
                            })
                            
                            print(f"  {img_path.name}: {analysis.get('identified_category', 'unknown')} "
                                  f"(conf: {analysis.get('confidence', 0):.1f}%, "
                                  f"risco: {revocation_prediction['revocation_risk']})")
                    
                    except Exception as e:
                        print(f"Erro ao processar {img_path.name}: {str(e)}")
        
        # Calcular métricas
        if not test_results:
            print("Nenhum resultado de teste obtido!")
            return None
        
        metrics = self.calculate_performance_metrics(test_results)
        
        # Salvar resultados
        self.save_evaluation_results(test_results, metrics)
        
        return {
            "test_results": test_results,
            "metrics": metrics
        }
    
    def calculate_performance_metrics(self, test_results):
        """
        Calcula métricas de performance
        """
        if not test_results:
            return {}
        
        # Preparar dados para métricas
        y_true = []
        y_pred = []
        confidences = []
        risk_scores = []
        
        for result in test_results:
            true_cat = result["true_category"]
            pred_cat = result["predicted_category"]
            
            # Normalizar categorias
            if "healthy" in true_cat.lower():
                true_cat = "leaf_healthy"
            else:
                true_cat = "leaf_with_disease"
            
            if "healthy" in pred_cat.lower():
                pred_cat = "leaf_healthy"
            else:
                pred_cat = "leaf_with_disease"
            
            y_true.append(true_cat)
            y_pred.append(pred_cat)
            confidences.append(result["confidence"])
            risk_scores.append(result["risk_score"])
        
        # Calcular métricas básicas
        accuracy = accuracy_score(y_true, y_pred)
        
        # Calcular métricas por categoria
        categories = list(set(y_true + y_pred))
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Matriz de confusão
        cm = confusion_matrix(y_true, y_pred, labels=categories)
        
        # Análise de confiança
        avg_confidence = np.mean(confidences)
        std_confidence = np.std(confidences)
        
        # Análise de risco de revogação
        avg_risk_score = np.mean(risk_scores)
        
        # Análise por nível de confiança
        high_conf_results = [r for r in test_results if r["confidence"] >= 80]
        medium_conf_results = [r for r in test_results if 60 <= r["confidence"] < 80]
        low_conf_results = [r for r in test_results if r["confidence"] < 60]
        
        high_conf_accuracy = accuracy_score(
            [r["true_category"] for r in high_conf_results],
            [r["predicted_category"] for r in high_conf_results]
        ) if high_conf_results else 0
        
        return {
            "overall_accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": cm,
            "categories": categories,
            "avg_confidence": avg_confidence,
            "std_confidence": std_confidence,
            "avg_risk_score": avg_risk_score,
            "confidence_analysis": {
                "high_confidence": {
                    "count": len(high_conf_results),
                    "accuracy": high_conf_accuracy
                },
                "medium_confidence": {
                    "count": len(medium_conf_results)
                },
                "low_confidence": {
                    "count": len(low_conf_results)
                }
            },
            "total_tests": len(test_results)
        }
    
    def save_evaluation_results(self, test_results, metrics):
        """
        Salva os resultados da avaliação
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Criar diretório de resultados se não existir
        results_dir = Path("evaluation_results")
        results_dir.mkdir(exist_ok=True)
        
        # Salvar resultados detalhados
        results_file = results_dir / f"evaluation_results_{timestamp}.json"
        
        # Converter numpy arrays para listas para serialização JSON
        serializable_metrics = metrics.copy()
        if "confusion_matrix" in serializable_metrics:
            serializable_metrics["confusion_matrix"] = serializable_metrics["confusion_matrix"].tolist()
        
        results_data = {
            "timestamp": timestamp,
            "test_results": test_results,
            "metrics": serializable_metrics
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        # Salvar resumo em CSV
        summary_file = results_dir / f"evaluation_summary_{timestamp}.csv"
        
        summary_data = []
        for result in test_results:
            summary_data.append({
                "image_path": result["image_path"],
                "true_category": result["true_category"],
                "predicted_category": result["predicted_category"],
                "confidence": result["confidence"],
                "revocation_risk": result["revocation_risk"],
                "risk_score": result["risk_score"],
                "correct": result["true_category"] == result["predicted_category"]
            })
        
        df = pd.DataFrame(summary_data)
        df.to_csv(summary_file, index=False)
        
        print(f"\nResultados salvos em:")
        print(f"- Detalhado: {results_file}")
        print(f"- Resumo: {summary_file}")
        
        return results_file, summary_file
    
    def generate_evaluation_report(self, metrics, output_path=None):
        """
        Gera relatório visual da avaliação
        """
        if not metrics:
            print("Nenhuma métrica disponível para gerar relatório")
            return
        
        # Configurar plot
        plt.style.use('default')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Relatório de Avaliação do Sistema CBIR', fontsize=16, fontweight='bold')
        
        # 1. Métricas gerais
        ax1 = axes[0, 0]
        metrics_names = ['Acurácia', 'Precisão', 'Recall', 'F1-Score']
        metrics_values = [
            metrics['overall_accuracy'],
            metrics['precision'],
            metrics['recall'],
            metrics['f1_score']
        ]
        
        bars = ax1.bar(metrics_names, metrics_values, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'])
        ax1.set_title('Métricas Gerais de Performance')
        ax1.set_ylabel('Score')
        ax1.set_ylim(0, 1)
        
        # Adicionar valores nas barras
        for bar, value in zip(bars, metrics_values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # 2. Matriz de confusão
        ax2 = axes[0, 1]
        cm = np.array(metrics['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=metrics['categories'], 
                   yticklabels=metrics['categories'], ax=ax2)
        ax2.set_title('Matriz de Confusão')
        ax2.set_xlabel('Predito')
        ax2.set_ylabel('Real')
        
        # 3. Distribuição de confiança
        ax3 = axes[0, 2]
        conf_analysis = metrics['confidence_analysis']
        conf_levels = ['Alta (≥80%)', 'Média (60-80%)', 'Baixa (<60%)']
        conf_counts = [
            conf_analysis['high_confidence']['count'],
            conf_analysis['medium_confidence']['count'],
            conf_analysis['low_confidence']['count']
        ]
        
        colors = ['#28a745', '#ffc107', '#dc3545']
        ax3.pie(conf_counts, labels=conf_levels, autopct='%1.1f%%', colors=colors)
        ax3.set_title('Distribuição por Nível de Confiança')
        
        # 4. Acurácia por nível de confiança
        ax4 = axes[1, 0]
        if conf_analysis['high_confidence']['count'] > 0:
            high_conf_acc = conf_analysis['high_confidence']['accuracy']
            ax4.bar(['Alta Confiança'], [high_conf_acc], color='#28a745')
            ax4.set_title('Acurácia - Alta Confiança')
            ax4.set_ylabel('Acurácia')
            ax4.set_ylim(0, 1)
            ax4.text(0, high_conf_acc + 0.01, f'{high_conf_acc:.3f}', 
                    ha='center', va='bottom')
        
        # 5. Estatísticas de confiança
        ax5 = axes[1, 1]
        conf_stats = [metrics['avg_confidence'], metrics['std_confidence']]
        conf_labels = ['Média', 'Desvio Padrão']
        ax5.bar(conf_labels, conf_stats, color=['#007bff', '#6c757d'])
        ax5.set_title('Estatísticas de Confiança')
        ax5.set_ylabel('Valor')
        
        # 6. Score médio de risco de revogação
        ax6 = axes[1, 2]
        risk_score = metrics['avg_risk_score']
        ax6.bar(['Risco Médio'], [risk_score], color='#fd7e14')
        ax6.set_title('Score Médio de Risco de Revogação')
        ax6.set_ylabel('Score')
        ax6.set_ylim(0, 1)
        ax6.text(0, risk_score + 0.01, f'{risk_score:.3f}', 
                ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Salvar relatório
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"evaluation_results/evaluation_report_{timestamp}.png"
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Relatório salvo em: {output_path}")
        return output_path
    
    def analyze_revocation_patterns(self, test_results):
        """
        Analisa padrões de revogação
        """
        if not test_results:
            return {}
        
        # Agrupar por risco de revogação
        risk_groups = {}
        for result in test_results:
            risk = result["revocation_risk"]
            if risk not in risk_groups:
                risk_groups[risk] = []
            risk_groups[risk].append(result)
        
        # Analisar cada grupo
        analysis = {}
        for risk_level, results in risk_groups.items():
            correct_predictions = sum(1 for r in results if r["true_category"] == r["predicted_category"])
            total = len(results)
            accuracy = correct_predictions / total if total > 0 else 0
            
            avg_confidence = np.mean([r["confidence"] for r in results])
            avg_risk_score = np.mean([r["risk_score"] for r in results])
            
            analysis[risk_level] = {
                "count": total,
                "accuracy": accuracy,
                "avg_confidence": avg_confidence,
                "avg_risk_score": avg_risk_score,
                "correct_predictions": correct_predictions,
                "incorrect_predictions": total - correct_predictions
            }
        
        return analysis

def main():
    """
    Função principal para executar avaliação
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Sistema de Avaliação CBIR')
    parser.add_argument('--test-dataset', type=str, default='image/test_dataset',
                      help='Caminho para o dataset de teste')
    parser.add_argument('--ground-truth', type=str, default=None,
                      help='Arquivo com ground truth (opcional)')
    parser.add_argument('--generate-report', action='store_true',
                      help='Gerar relatório visual')
    
    args = parser.parse_args()
    
    # Criar sistema de avaliação
    evaluator = CBIREvaluationSystem()
    
    # Executar avaliação
    print("Iniciando avaliação do sistema CBIR...")
    evaluation_result = evaluator.evaluate_system_performance(
        args.test_dataset, 
        args.ground_truth
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

if __name__ == "__main__":
    main() 