# 🍃 Sistema de Identificação de Doenças em Folhas

## 🛠️ Configuração do Ambiente

### 1. Criar Ambiente Virtual
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 2. Instalar Dependências
```bash
pip install -r requirements.txt
```

## 📊 Consultas ao Banco de Dados

O sistema oferece várias opções para consultar e analisar os dados armazenados no banco. Use o script `query_database.py` com os seguintes comandos:

### 1. Estatísticas Gerais
```bash
python query_database.py --stats
```
Mostra:
- Total de imagens no banco
- Distribuição por categoria
- Data da última atualização

### 2. Listar Todas as Imagens
```bash
python query_database.py --list
```
Exibe uma lista detalhada de todas as imagens, incluindo:
- ID da imagem
- Categoria
- Data de processamento
- Caminho do arquivo

### 3. Detalhes de uma Imagem Específica
```bash
python query_database.py --image nome_imagem.jpg
```
Mostra informações detalhadas de uma imagem, incluindo:
- Metadados completos
- Características HSV (primeiros 5 valores)
- Características de textura
- Características de forma

### 4. Buscar por Categoria
```bash
python query_database.py --category Pepper__bell___Bacterial_spot
```
Lista todas as imagens de uma categoria específica.

### 5. Exportar Banco de Dados
```bash
python query_database.py --export dados.json
```
Exporta todo o banco de dados para um arquivo JSON, útil para:
- Backup dos dados
- Análise externa
- Documentação do dataset 

## 📊 Características Extraídas

O sistema extrai 106 características de cada imagem, divididas em três categorias principais:

### 1. Características de Cor (96 valores)
- **Histograma HSV** (Matiz, Saturação, Valor):
  - **Matiz (H) - 32 valores**: 
    - Representa as cores presentes na folha
    - Útil para detectar descoloração e manchas
    - Cada valor representa a frequência de uma faixa de cor
  
  - **Saturação (S) - 32 valores**:
    - Indica a intensidade/pureza das cores
    - Ajuda a identificar áreas desbotadas ou necrosadas
    - Valores baixos indicam cores acinzentadas/desbotadas
  
  - **Valor (V) - 32 valores**:
    - Representa o brilho/luminosidade
    - Auxilia na detecção de áreas escuras ou claras
    - Importante para identificar lesões necróticas (escuras)

### 2. Características de Textura (6 valores)
Análise multi-escala usando kernels de diferentes tamanhos:

- **Kernel 3x3**:
  - `Média k3`: Média da variação local (detalhes finos)
  - `Desvio k3`: Uniformidade dos detalhes finos

- **Kernel 5x5**:
  - `Média k5`: Média da variação em escala média
  - `Desvio k5`: Uniformidade em escala média

- **Kernel 7x7**:
  - `Média k7`: Média da variação em escala maior
  - `Desvio k7`: Uniformidade dos padrões maiores

Estas características capturam:
- Rugosidade da superfície
- Padrões de manchas
- Texturas características de cada doença

### 3. Características de Forma (4 valores)
Análise das manchas e lesões detectadas:

- **Número de Manchas** (`Num_Manchas`):
  - Quantidade total de manchas detectadas
  - Ajuda a diferenciar doenças pontuais de difusas
  - Valor 0 pode indicar folha saudável

- **Tamanho Médio** (`Tamanho_Medio`):
  - Média da área de todas as manchas
  - Útil para caracterizar o padrão típico da doença
  - Medido em pixels quadrados

- **Desvio do Tamanho** (`Desvio_Tamanho`):
  - Variação no tamanho das manchas
  - Valor alto: manchas de tamanhos muito diferentes
  - Valor baixo: manchas uniformes

- **Proporção Máxima** (`Proporcao_Maxima`):
  - Área da maior mancha / Área total da imagem
  - Varia de 0 (sem manchas) a 1 (toda a folha)
  - Ajuda a identificar lesões extensas

### Exemplo de Interpretação

Para uma folha com ferrugem típica:
- **Cor**: Picos no histograma H para tons marrom-alaranjados
- **Textura**: Valores altos para kernels pequenos (lesões granulares)
- **Forma**: Muitas manchas pequenas com tamanho uniforme

Para uma folha com mancha bacteriana:
- **Cor**: Picos no histograma H para tons marrom-escuros
- **Textura**: Valores médios distribuídos (lesões irregulares)
- **Forma**: Número médio de manchas com tamanhos variados 