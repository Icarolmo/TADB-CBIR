# Sistema CBIR para Identificação de Doenças em Folhas

Este documento apresenta a documentação completa do sistema CBIR (Content-Based Image Retrieval) para identificação de doenças em folhas, incluindo avaliação, previsão de revogação, processamento de imagens, e uso de interfaces gráficas e scripts utilitários.

---

## 📋 Visão Geral do Sistema

O sistema realiza:
- Processamento de imagens de folhas para extração de características
- Armazenamento e consulta de embeddings em banco ChromaDB
- Diagnóstico de categoria (folha saudável ou com doença)
- Previsão de risco de revogação do diagnóstico
- Avaliação quantitativa do desempenho do sistema
- Geração de relatórios e visualizações

### Principais Componentes
- `cbir.py`: Script principal para processamento, consulta e avaliação
- `evaluation_system.py`: Avaliação quantitativa e previsão de revogação
- `evaluation_gui.py`: Interface gráfica de avaliação
- `demo_evaluation.py`: Demonstração interativa
- `simple_gui.py`: GUI simples para inspeção do banco de dados
- `setup_proper_evaluation.py` e `split_dataset.py`: Utilitários para preparação e divisão do dataset
- `database/chroma.py`: Interface com o banco ChromaDB
- `engine/processing_engine.py`: Extração de características e processamento de imagens

---

## 📁 Estrutura de Diretórios

```
TADB-CBIR/
├── cbir.py                  # Script principal do sistema CBIR
├── evaluation_system.py     # Avaliação e previsão de revogação
├── evaluation_gui.py        # Interface gráfica de avaliação
├── demo_evaluation.py       # Demonstração interativa
├── simple_gui.py            # GUI simples para banco de dados
├── setup_proper_evaluation.py # Prepara avaliação adequada
├── split_dataset.py         # Divide dataset em treino/teste
├── extracao_caracteristicas.py # Extração de features (auxiliar)
├── requirements.txt         # Dependências
├── database/
│   └── chroma.py            # Interface com ChromaDB
├── engine/
│   └── processing_engine.py  # Processamento de imagens
├── image/
│   ├── dataset/             # Imagens de referência
│   │   ├── leaf_healthy/
│   │   └── leaf_with_disease/
│   ├── test_dataset/        # Imagens de teste
│   └── uploads/             # Imagens de consulta
├── evaluation_results/      # Relatórios e métricas gerados
└── README_EVALUATION.md     # Este arquivo
```

---

## 🛠️ Instalação e Preparação do Ambiente

1. **(Opcional) Crie e ative um ambiente virtual:**
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

2. **Instale as dependências:**
```bash
pip install -r requirements.txt
```

3. **Prepare a estrutura de diretórios:**
```bash
mkdir -p image/dataset/leaf_healthy
mkdir -p image/dataset/leaf_with_disease
mkdir -p image/test_dataset/leaf_healthy
mkdir -p image/test_dataset/leaf_with_disease
mkdir -p image/uploads
```

4. **Adicione imagens ao dataset:**
- Coloque imagens de folhas saudáveis em `image/dataset/leaf_healthy/`
- Coloque imagens de folhas com doença em `image/dataset/leaf_with_disease/`

---

## 🚦 Fluxo de Uso do Sistema

### 1. Divisão do Dataset (opcional, recomendado para avaliação justa)
```bash
python setup_proper_evaluation.py
```
- Divide o dataset em referência/teste (80/20%) sem sobreposição
- Indexa as imagens de referência e prepara o diretório de teste

### 2. Processamento do Dataset de Referência
```bash
python cbir.py --process-only --train-dir image/dataset/train
```
- Extrai características e popula o banco de dados de referência

### 3. Consulta de Imagem (Diagnóstico)
```bash
cp sua_imagem.jpg image/uploads/query_leaf.jpg
python cbir.py
```
- Analisa a imagem, retorna categoria, confiança, risco de revogação e recomendações

### 4. Avaliação Quantitativa do Sistema
```bash
python cbir.py --evaluate --test-dataset image/test_dataset --generate-report
```
- Mede acurácia, precisão, recall, F1, análise de risco e gera relatório visual

### 5. Uso das Interfaces Gráficas
- **Avaliação completa (GUI):**
  ```bash
  python evaluation_gui.py
  ```
- **Inspeção do banco de dados (GUI simples):**
  ```bash
  python simple_gui.py
  ```

### 6. Demonstração Interativa
```bash
python demo_evaluation.py
```

---

## 📊 Interpretação dos Resultados

- **Categoria identificada:** folha saudável ou com doença
- **Confiança:** nível de certeza do sistema (em %)
- **Risco de revogação:** BAIXO, MÉDIO ou ALTO, com fatores explicativos
- **Relatórios:** arquivos JSON, CSV e PNG em `evaluation_results/`

### Tabela de Risco de Revogação
| Risco | Score | Significado | Ação Recomendada |
|-------|-------|-------------|------------------|
| BAIXO | 0.0-0.4 | Sistema funcionando bem | Monitorar |
| MÉDIO | 0.4-0.7 | Pode melhorar | Revisar imagens |
| ALTO | 0.7-1.0 | Alto risco de erro | Consultar especialista |

### Tabela de Confiança
| Confiança | Significado | Ação |
|-----------|-------------|------|
| ≥80% | Altamente confiável | Confiar |
| 60-80% | Provável | Confirmar |
| <60% | Incerto | Novas fotos |

---

## 🧩 Scripts e Utilitários

- **cbir.py**: Processa dataset, consulta imagem, avalia sistema, gera relatórios
- **evaluation_system.py**: Avaliação quantitativa, previsão de revogação, análise de padrões
- **evaluation_gui.py**: Interface gráfica para avaliação, geração de relatórios e análise de risco
- **simple_gui.py**: GUI simples para inspeção e limpeza do banco de dados
- **demo_evaluation.py**: Demonstração de uso, exemplos de consulta e avaliação
- **setup_proper_evaluation.py**: Prepara avaliação justa, divide dataset, treina e configura teste
- **split_dataset.py**: Função utilitária para dividir dataset em treino/teste
- **extracao_caracteristicas.py**: Extração de features (HSV, GLCM, LBP, forma)

---

## 🧠 Boas Práticas e Recomendações

- Sempre processe o dataset antes de consultas ou avaliações
- Use `setup_proper_evaluation.py` para evitar vazamento de dados entre treino e teste
- Para melhores resultados, use imagens nítidas, bem iluminadas e sem sombras
- Consulte especialistas em caso de risco ALTO de revogação

---

## 🐛 Solução de Problemas

- **Nenhuma imagem no banco de dados:**
  ```bash
  python cbir.py --process-only
  ```
- **Dataset de teste não encontrado:**
  ```bash
  mkdir -p image/test_dataset/leaf_healthy
  mkdir -p image/test_dataset/leaf_with_disease
  ```
- **Imagem de consulta não encontrada:**
  ```bash
  cp sua_imagem.jpg image/uploads/query_leaf.jpg
  ```
- **Erro de importação em GUIs:**
  - Verifique se está executando na pasta correta
  - Confirme dependências instaladas

---

## 📚 Exemplos de Uso em Código

### Validação de Diagnóstico
```python
from evaluation_system import CBIREvaluationSystem
result = ... # resultado de consulta
revocation = CBIREvaluationSystem().predict_revocation_risk(result)
if revocation['revocation_risk'] == 'ALTO':
    print('⚠️ Consulte um especialista')
```

### Avaliação de Performance
```python
eval_result = CBIREvaluationSystem().evaluate_system_performance('image/test_dataset')
CBIREvaluationSystem().generate_evaluation_report(eval_result['metrics'])
```

### Análise de Padrões
```python
patterns = CBIREvaluationSystem().analyze_revocation_patterns(test_results)
for risk, stats in patterns.items():
    print(f'Risco {risk}: {stats["accuracy"]:.3f} acurácia')
```

---

## 📈 Relatórios Gerados

- **JSON:** resultados detalhados
- **CSV:** resumo tabular
- **PNG:** gráficos e visualizações
- **Local:** `evaluation_results/`

---

## ℹ️ Observações Finais

- O sistema é modular e pode ser expandido para novas categorias ou características
- Para dúvidas, consulte os comentários nos scripts ou abra uma issue
- Recomenda-se manter o dataset organizado e balanceado

---

*Documentação atualizada para uso completo do sistema CBIR de identificação de doenças em folhas.*
