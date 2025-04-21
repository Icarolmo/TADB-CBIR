import streamlit as st
import chromadb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
from chromadb.config import Settings
from multiprocessing import cpu_count

# Configurar página
st.set_page_config(page_title="ChromaDB Viewer", layout="wide")

# Título
st.title("🍃 Visualizador do Banco de Dados de Doenças em Folhas")

# Conectar ao ChromaDB

@st.cache_resource
def get_client():
    num_threads = min(cpu_count(), 2)  # por segurança, usa no máximo 2
    settings = Settings(
        persist_directory="./database/chroma_db",
        anonymized_telemetry=False,
        num_threads=num_threads
    )
    return chromadb.Client(settings)

client = get_client()

# Sidebar
st.sidebar.title("Navegação")
page = st.sidebar.radio("Escolha uma página:", 
    ["📊 Visão Geral", "🔍 Explorar Dados", "📈 Análise de Características"])

if page == "📊 Visão Geral":
    st.header("Visão Geral do Banco")
    
    # Obter coleções
    collections = client.list_collections()
    st.subheader(f"Total de Coleções: {len(collections)}")
    
    # Mostrar detalhes de cada coleção
    for collection in collections:
        st.write("---")
        st.subheader(f"Coleção: {collection.name}")
        
        # Metadata da coleção
        st.write("Metadados da Coleção:")
        st.json(collection.metadata)
        
        # Contar itens
        items = collection.get()
        total_items = len(items["ids"]) if "ids" in items else 0
        st.write(f"Total de Itens: {total_items}")
        
        if total_items > 0:
            # Análise de categorias
            categories = {}
            for metadata in items["metadatas"]:
                category = metadata.get("category", "desconhecido")
                categories[category] = categories.get(category, 0) + 1
            
            # Gráfico de distribuição de categorias
            fig, ax = plt.subplots(figsize=(10, 5))
            plt.bar(categories.keys(), categories.values())
            plt.xticks(rotation=45, ha='right')
            plt.title("Distribuição por Categoria")
            plt.tight_layout()
            st.pyplot(fig)

elif page == "🔍 Explorar Dados":
    st.header("Explorar Dados")
    
    # Selecionar coleção
    collections = client.list_collections()
    collection_names = [c.name for c in collections]
    selected_collection = st.selectbox("Escolha uma coleção:", collection_names)
    
    if selected_collection:
        collection = client.get_collection(selected_collection)
        items = collection.get()
        
        if len(items["ids"]) > 0:
            # Criar DataFrame
            df = pd.DataFrame({
                "ID": items["ids"],
                "Categoria": [m.get("category", "desconhecido") for m in items["metadatas"]],
                "Data Processamento": [m.get("processing_date", "N/A") for m in items["metadatas"]],
                "Caminho": [m.get("path", "N/A") for m in items["metadatas"]]
            })
            
            # Filtros
            st.subheader("Filtros")
            col1, col2 = st.columns(2)
            with col1:
                categoria_filter = st.multiselect(
                    "Filtrar por Categoria:",
                    options=sorted(df["Categoria"].unique())
                )
            
            # Aplicar filtros
            if categoria_filter:
                df = df[df["Categoria"].isin(categoria_filter)]
            
            # Mostrar dados
            st.subheader("Dados")
            st.dataframe(df)
            
            # Download CSV
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download CSV",
                csv,
                "dados_chroma.csv",
                "text/csv",
                key='download-csv'
            )

elif page == "📈 Análise de Características":
    st.header("Análise de Características")
    
    # Selecionar coleção
    collections = client.list_collections()
    collection_names = [c.name for c in collections]
    selected_collection = st.selectbox("Escolha uma coleção:", collection_names)
    
    if selected_collection:
        collection = client.get_collection(selected_collection)
        items = collection.get()
        
        if len(items["ids"]) > 0:
            # Converter embeddings para array numpy
            embeddings = np.array(items["embeddings"])
            
            # Selecionar tipo de característica
            feature_type = st.radio(
                "Escolha o tipo de característica:",
                ["Cor (HSV)", "Textura", "Forma"]
            )
            
            if feature_type == "Cor (HSV)":
                # Mostrar histogramas HSV médios por categoria
                categories = {}
                for embedding, metadata in zip(embeddings, items["metadatas"]):
                    category = metadata.get("category", "desconhecido")
                    if category not in categories:
                        categories[category] = []
                    categories[category].append(embedding[:96])  # HSV features
                
                # Calcular médias
                for category in categories:
                    categories[category] = np.mean(categories[category], axis=0)
                
                # Plotar
                fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
                
                for category, features in categories.items():
                    ax1.plot(features[:32], label=category)
                ax1.set_title("Histograma H (Matiz)")
                ax1.legend()
                
                for category, features in categories.items():
                    ax2.plot(features[32:64], label=category)
                ax2.set_title("Histograma S (Saturação)")
                
                for category, features in categories.items():
                    ax3.plot(features[64:96], label=category)
                ax3.set_title("Histograma V (Valor)")
                
                plt.tight_layout()
                st.pyplot(fig)
            
            elif feature_type == "Textura":
                # Análise de textura
                texture_features = embeddings[:, 96:102]
                texture_names = ['Média k3', 'Desvio k3', 'Média k5', 'Desvio k5', 'Média k7', 'Desvio k7']
                
                # Criar DataFrame
                texture_df = pd.DataFrame(
                    texture_features,
                    columns=texture_names
                )
                texture_df["Categoria"] = [m.get("category", "desconhecido") for m in items["metadatas"]]
                
                # Boxplot
                fig, ax = plt.subplots(figsize=(12, 6))
                sns.boxplot(data=texture_df.melt(id_vars=["Categoria"]), 
                          x="variable", y="value", hue="Categoria")
                plt.xticks(rotation=45)
                plt.title("Distribuição das Características de Textura por Categoria")
                plt.tight_layout()
                st.pyplot(fig)
            
            else:  # Forma
                # Análise de forma
                shape_features = embeddings[:, -4:]
                shape_names = ['Num. Manchas', 'Tam. Médio', 'Desvio Tam.', 'Prop. Máx.']
                
                # Criar DataFrame
                shape_df = pd.DataFrame(
                    shape_features,
                    columns=shape_names
                )
                shape_df["Categoria"] = [m.get("category", "desconhecido") for m in items["metadatas"]]
                
                # Boxplot
                fig, ax = plt.subplots(figsize=(12, 6))
                sns.boxplot(data=shape_df.melt(id_vars=["Categoria"]), 
                          x="variable", y="value", hue="Categoria")
                plt.xticks(rotation=45)
                plt.title("Distribuição das Características de Forma por Categoria")
                plt.tight_layout()
                st.pyplot(fig) 