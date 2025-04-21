# Use uma imagem base do Python
FROM python:3.9-slim

# Definir diretório de trabalho
WORKDIR /app

# Copiar arquivos necessários
COPY requirements.txt .
COPY database/ ./database/
COPY visualize_db.py .

# Instalar dependências
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install streamlit pandas seaborn

# Expor porta do Streamlit
EXPOSE 8501

# Comando para executar a aplicação
CMD ["streamlit", "run", "visualize_db.py", "--server.port=8501", "--server.address=0.0.0.0"] 