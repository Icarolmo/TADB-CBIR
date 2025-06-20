# Sistema de Avaliação CBIR com Previsão de Revogação

Este documento descreve o sistema de avaliação e previsão de revogação implementado para o CBIR (Content-Based Image Retrieval) de identificação de doenças em folhas.

## 📋 Visão Geral

O sistema implementa duas funcionalidades principais:

1. **Previsão de Revogação**: Analisa o risco de uma classificação estar incorreta
2. **Avaliação do Sistema**: Mede a performance geral do CBIR usando métricas padronizadas

## 🚀 Funcionalidades

### 1. Previsão de Revogação

O sistema analisa múltiplos fatores para determinar o risco de uma classificação estar incorreta:

- **Confiança da classificação**: Quão confiante o sistema está na predição
- **Variabilidade de similaridade**: Quão consistentes são as similaridades das imagens encontradas
- **Consistência de categoria**: Se as imagens similares pertencem à mesma categoria
- **Gap de similaridade**: Diferença entre a melhor e pior similaridade
- **Variabilidade de características**: Quão diferentes são as características de forma das imagens similares

**Níveis de Risco:**
- **BAIXO**: Sistema funcionando bem, alta confiabilidade
- **MÉDIO**: Sistema funcionando, mas pode ser melhorado
- **ALTO**: Alto risco de erro, necessita revisão

### 2. Avaliação do Sistema

Métricas calculadas:
- **Acurácia**: Proporção de predições corretas
- **Precisão**: Proporção de predições positivas que são corretas
- **Recall**: Proporção de casos positivos identificados corretamente
- **F1-Score**: Média harmônica entre precisão e recall
- **Análise de confiança**: Distribuição e estatísticas dos níveis de confiança
- **Análise de risco**: Estatísticas por nível de risco de revogação

## 📁 Estrutura de Arquivos

```
TADB-CBIR/
├── evaluation_system.py      # Sistema principal de avaliação
├── evaluation_gui.py         # Interface gráfica
├── demo_evaluation.py        # Script de demonstração
├── cbir.py                   # CBIR principal (atualizado)
├── requirements.txt          # Dependências (atualizado)
└── README_EVALUATION.md      # Este arquivo
```

## ��️ Instalação

1. **(Opcional) Criar e ativar ambiente virtual:**
```bash
# Criar ambiente virtual
python -m venv venv
# Ativar no Windows:
venv\Scripts\activate
# Ativar no Linux/Mac:
source venv/bin/activate
```

2. **Instalar dependências:**
```bash
pip install -r requirements.txt
```

> **Todos os comandos a seguir devem ser executados dentro da pasta `TADB-CBIR`.**

3. **Preparar o ambiente:**
```bash
# Criar diretórios necessários
mkdir -p image/dataset/leaf_healthy
mkdir -p image/dataset/leaf_with_disease
mkdir -p image/test_dataset/leaf_healthy
mkdir -p image/test_dataset/leaf_with_disease
mkdir -p image/uploads
```

## 📖 Como Usar

### 1. Preparação Inicial

**Processar dataset de treinamento:**
```bash
python cbir.py --process-only
```

**Estrutura do dataset de treinamento:**
```
image/dataset/
├── leaf_healthy/
│   ├── healthy1.jpg
│   ├── healthy2.jpg
│   └── ...
└── leaf_with_disease/
    ├── disease1.jpg
    ├── disease2.jpg
    └── ...
```

### 2. Previsão de Revogação

**Via linha de comando:**
```bash
# Colocar imagem de consulta
cp sua_imagem.jpg image/uploads/query_leaf.jpg

# Executar análise
python cbir.py
```

**Saída esperada:**
```
⚠️ PREVISÃO DE REVOGAÇÃO
--------------------------------------------------
Nível de risco: MÉDIO
Score de risco: 0.450
Fatores de risco identificados:
• Baixa consistência de categoria
• Alta variabilidade nas similaridades
```

### 3. Avaliação Completa do Sistema

**Via linha de comando:**
```bash
python cbir.py --evaluate --test-dataset image/test_dataset --generate-report
```

**Via interface gráfica:**
```bash
python evaluation_gui.py
```

**Via script dedicado:**
```bash
python evaluation_system.py --test-dataset image/test_dataset --generate-report
```

### 4. Demonstração Interativa

```bash
python demo_evaluation.py
```

## 📊 Interpretação dos Resultados

### Níveis de Risco de Revogação

| Risco | Score | Significado | Ação Recomendada |
|-------|-------|-------------|------------------|
| BAIXO | 0.0-0.4 | Sistema funcionando bem | Continuar monitoramento |
| MÉDIO | 0.4-0.7 | Sistema funcionando, mas pode melhorar | Revisar qualidade das imagens |
| ALTO | 0.7-1.0 | Alto risco de erro | Consultar especialista, revisar dataset |

### Níveis de Confiança

| Confiança | Significado | Ação |
|-----------|-------------|------|
| ≥80% | Altamente confiável | Confiar no diagnóstico |
| 60-80% | Provável, mas necessita confirmação | Verificar com especialista |
| <60% | Incerto | Tirar novas fotos, consultar especialista |

### Métricas de Performance

- **Acurácia > 0.8**: Excelente performance
- **Acurácia 0.6-0.8**: Boa performance
- **Acurácia < 0.6**: Necessita melhorias

## 🎯 Casos de Uso

### 1. Validação de Diagnóstico

```python
from evaluation_system import CBIREvaluationSystem

# Criar sistema
evaluator = CBIREvaluationSystem()

# Analisar resultado de consulta
revocation_prediction = evaluator.predict_revocation_risk(query_result)

if revocation_prediction['revocation_risk'] == 'ALTO':
    print("⚠️ Consulte um especialista para confirmação")
```

### 2. Avaliação de Performance

```python
# Avaliar sistema completo
evaluation_result = evaluator.evaluate_system_performance(
    test_dataset_path="image/test_dataset"
)

# Gerar relatório
evaluator.generate_evaluation_report(evaluation_result['metrics'])
```

### 3. Análise de Padrões

```python
# Analisar padrões de revogação
revocation_analysis = evaluator.analyze_revocation_patterns(test_results)

for risk_level, analysis in revocation_analysis.items():
    print(f"Risco {risk_level}: {analysis['accuracy']:.3f} acurácia")
```

## 🔧 Configuração Avançada

### Ajustar Thresholds de Risco

```python
# No arquivo evaluation_system.py
class CBIREvaluationSystem:
    def __init__(self):
        self.confidence_threshold = 0.7  # Ajustar conforme necessário
```

### Personalizar Fatores de Risco

```python
def predict_revocation_risk(self, query_result):
    # Ajustar pesos dos fatores
    if features["confidence"] < 60:  # Threshold ajustável
        risk_score += 0.4  # Peso ajustável
```

## 📈 Relatórios Gerados

O sistema gera automaticamente:

1. **Relatório JSON**: Resultados detalhados em formato estruturado
2. **Relatório CSV**: Resumo em formato tabular
3. **Relatório Visual**: Gráficos e visualizações (PNG)

**Localização dos relatórios:**
```
evaluation_results/
├── evaluation_results_20231201_143022.json
├── evaluation_summary_20231201_143022.csv
└── evaluation_report_20231201_143022.png
```

## 🐛 Solução de Problemas

### Erro: "Nenhuma imagem no banco de dados"
```bash
# Solução: Processar dataset primeiro
python cbir.py --process-only
```

### Erro: "Dataset de teste não encontrado"
```bash
# Solução: Criar estrutura de diretórios
mkdir -p image/test_dataset/leaf_healthy
mkdir -p image/test_dataset/leaf_with_disease
```

### Erro: "Imagem de consulta não encontrada"
```bash
# Solução: Colocar imagem no local correto
cp sua_imagem.jpg image/uploads/query_leaf.jpg
```
