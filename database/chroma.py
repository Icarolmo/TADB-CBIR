import chromadb
import numpy as np
from chromadb.config import Settings
import os
import multiprocessing
from datetime import datetime
import argparse
import json

# Configurar cliente persistente com configurações otimizadas
client = chromadb.PersistentClient(
    path="database/chroma_db",
    settings=Settings(
        anonymized_telemetry=False,
        allow_reset=True
    )
)

# Criar ou obter coleção
collection = client.get_or_create_collection(
    name="leaf_diseases",
    metadata={
        "hnsw:space": "cosine",
        "description": "Características de doenças em folhas",
        "feature_hsv": "96 valores (32 bins para cada canal H, S, V)",
        "feature_texture": "6 valores (média e desvio para 3 kernels)",
        "feature_shape": "6 valores (num_spots, mean_size, std_size, max_ratio, area_total, circularidade)"
    }
)

def get_database_stats():
    """
    Retorna estatísticas detalhadas do banco de dados.
    
    Returns:
        Dicionário com estatísticas sobre as imagens armazenadas
    """
    try:
        # Pegar todos os itens no banco
        results = collection.get()
        
        if not results or "ids" not in results:
            return {
                "total_images": 0,
                "categories": {},
                "last_update": None,
                "ids": []
            }
        
        # Inicializar estatísticas
        stats = {
            "total_images": len(results["ids"]),
            "categories": {},
            "last_update": None,
            "ids": results["ids"]
        }
        
        # Processar metadados
        if "metadatas" in results and results["metadatas"]:
            for metadata in results["metadatas"]:
                if metadata:
                    # Contar por categoria
                    category = metadata.get("category", "unknown")
                    if category not in stats["categories"]:
                        stats["categories"][category] = 0
                    stats["categories"][category] += 1
                    
                    # Atualizar data da última modificação
                    proc_date = metadata.get("processing_date")
                    if proc_date and (not stats["last_update"] or proc_date > stats["last_update"]):
                        stats["last_update"] = proc_date
        
        return stats
        
    except Exception as e:
        print(f"Erro ao obter estatísticas: {str(e)}")
        return {
            "total_images": 0,
            "categories": {},
            "last_update": None,
            "error": str(e)
        }

def clear_database():
    """Limpa o banco de dados"""
    try:
        client.delete_collection("leaf_diseases")
        collection = client.create_collection(
            name="leaf_diseases",
            metadata={"hnsw:space": "cosine"}
        )
        return True
    except Exception as e:
        print(f"Erro ao limpar banco de dados: {str(e)}")
        return False

def add_embedding(id, embedding, metadata=None):
    """Adiciona um embedding ao banco de dados"""
    try:
        # Normalizar o caminho no metadata
        if metadata and 'path' in metadata:
            metadata['path'] = os.path.normpath(metadata['path'])
            
            # Extrair categoria do caminho
            parts = metadata['path'].split(os.sep)
            category = next((part for part in parts if 'leaf_' in part), None)
            if category:
                metadata['category'] = category

        # Extrair características para exibição
        features = extract_features(embedding)
        
        print(f"\nSalvando imagem no banco de dados:")
        print(f"• Caminho: {metadata.get('path', 'N/A')}")
        print(f"• Categoria: {metadata.get('category', 'N/A')}")
        print(f"• Data: {metadata.get('processing_date', 'N/A')}")
        
        print("\nCaracterísticas extraídas:")
        print(f"• Número de lesões: {features['shape']['num_lesions']:.2f}")
        print(f"• Área afetada: {features['shape']['disease_coverage']:.2%}")
        print(f"• Tamanho médio das lesões: {features['shape']['avg_lesion_size']:.2f}")
        print(f"• Densidade de lesões: {features['shape']['lesion_density']:.2f}")
        
        print("\nEstatísticas de cor (HSV):")
        print(f"• Matiz média: {features['hsv']['h_stats']['mean']:.2f}")
        print(f"• Saturação média: {features['hsv']['s_stats']['mean']:.2f}")
        print(f"• Valor médio: {features['hsv']['v_stats']['mean']:.2f}")
        
        print("\nCaracterísticas de textura:")
        print(f"• Contraste: {features['glcm']['contrast']:.2f}")
        print(f"• Energia: {features['glcm']['energy']:.2f}")
        print(f"• Homogeneidade: {features['glcm']['homogeneity']:.2f}")
        print("-" * 50)

        collection.add(
            embeddings=[embedding],
            ids=[id],
            metadatas=[metadata] if metadata else None
        )
        return True
    except Exception as e:
        print(f"Erro ao adicionar embedding: {str(e)}")
        return False

def query_embedding(query_embedding, n_results=5, metadata=None):
    """Consulta embeddings similares no banco de dados"""
    try:
        # Converter para numpy array
        query_embedding = np.array(query_embedding)
        
        # Definir índices dos diferentes grupos de características
        hsv_hist_indices = slice(0, 96)        # 96 valores dos histogramas HSV
        hsv_stats_indices = slice(96, 108)     # 12 valores das estatísticas HSV
        glcm_indices = slice(108, 116)         # 8 valores do GLCM
        lbp_indices = slice(116, 120)          # 4 valores do LBP
        shape_indices = slice(120, 128)        # 8 valores de forma

        # Extrair características críticas
        shape_features = query_embedding[shape_indices]
        num_lesoes = shape_features[0]         # Número de lesões
        area_afetada = shape_features[3]       # Área afetada (porcentagem)
        
        # Determinar se a imagem tem lesões significativas
        has_lesions = num_lesoes > 0 and area_afetada > 0.001  # Mais de 0.1% da área
        
        # Definir pesos baseados nas características da imagem
        weights = np.ones(len(query_embedding))
        
        # Se a imagem tem lesões, dar mais peso para características relevantes
        if has_lesions:
            weights[hsv_stats_indices] = 2.0  # Estatísticas HSV são importantes para manchas
            weights[shape_indices] = 2.0      # Forma é crucial para lesões
            weights[glcm_indices] = 1.5       # Textura pode indicar necrose
            weights[hsv_hist_indices] = 0.5   # Histograma completo é menos importante
        
        # Aplicar pesos ao embedding
        weighted_query = query_embedding * weights
        
        # Normalizar o embedding ponderado
        norm = np.linalg.norm(weighted_query)
        if norm > 0:
            weighted_query = weighted_query / norm
        
        # Consultar o banco de dados - buscar mais resultados para garantir que temos 5 válidos
        results = collection.query(
            query_embeddings=[weighted_query.tolist()],
            n_results=n_results * 3,  # Buscar mais resultados para garantir 5 válidos
            include=["embeddings", "metadatas", "distances"]
        )
        
        # Adicionar metadados da imagem de consulta
        if results and 'metadatas' in results and results['metadatas']:
            if metadata:
                results['metadatas'][0].insert(0, metadata)
            else:
                results['metadatas'][0].insert(0, {
                    "path": "image/uploads/query_leaf.jpg",
                    "type": "leaf_disease",
                    "processing_date": str(datetime.now())
                })
        
        return results, has_lesions
        
    except Exception as e:
        print(f"Erro ao consultar embedding: {str(e)}")
        return None, False

def calculate_similarity(dist, emb1, emb2):
    """Calcula similaridade considerando características específicas"""
    # Extrair características relevantes
    shape1 = emb1[120:128]
    shape2 = emb2[120:128]
    
    num_lesoes1 = shape1[0]
    num_lesoes2 = shape2[0]
    area_afetada1 = shape1[3]  # Índice para área afetada
    area_afetada2 = shape2[3]
    
    # Verificar se as imagens são saudáveis ou doentes
    is_healthy1 = num_lesoes1 <= 1 and area_afetada1 < 0.02  # Reduzido para 2% de área e máximo 1 lesão
    is_healthy2 = num_lesoes2 <= 1 and area_afetada2 < 0.02
    
    has_significant_lesions1 = num_lesoes1 >= 3 or area_afetada1 > 0.05  # 5% de área ou 3+ lesões
    has_significant_lesions2 = num_lesoes2 >= 3 or area_afetada2 > 0.05
    
    # Se uma é claramente saudável e outra doente, similaridade muito baixa
    if (is_healthy1 and has_significant_lesions2) or (is_healthy2 and has_significant_lesions1):
        return 5.0  # Similaridade mínima de 5%
    
    # Calcular similaridade base usando distância coseno
    base_similarity = 100 * (1 - dist)
    
    # Calcular similaridades específicas com foco em lesões
    lesion_diff = abs(num_lesoes1 - num_lesoes2)
    max_lesoes = max(num_lesoes1, num_lesoes2)
    
    # Similaridade no número de lesões mais gradual
    if max_lesoes > 0:
        lesion_similarity = 100 * (1 - lesion_diff / (max_lesoes + 2))  # +2 para suavizar diferenças
    else:
        lesion_similarity = 100 if lesion_diff == 0 else 0
    
    # Similaridade na área afetada mais gradual
    area_similarity = 100 * (1 - min(1.0, abs(area_afetada1 - area_afetada2) / 0.2))  # Normaliza para 20% de diferença
    
    # Calcular similaridade final com pesos ajustados
    if is_healthy1 and is_healthy2:  # Ambas saudáveis
        final_similarity = (
            0.4 * base_similarity +    # Maior peso para similaridade geral
            0.3 * lesion_similarity +  # Menor peso para lesões (ambas têm poucas)
            0.3 * area_similarity      # Menor peso para área
        )
    elif has_significant_lesions1 and has_significant_lesions2:  # Ambas doentes
        final_similarity = (
            0.2 * base_similarity +    # Menor peso para similaridade geral
            0.5 * lesion_similarity +  # Maior peso para número de lesões
            0.3 * area_similarity      # Peso médio para área afetada
        )
    else:  # Casos intermediários
        final_similarity = (
            0.3 * base_similarity +
            0.4 * lesion_similarity +
            0.3 * area_similarity
        )
    
    # Garantir que a similaridade esteja entre 0 e 100
    final_similarity = max(0, min(100, final_similarity))
    
    # Ajustes graduais baseados nas diferenças
    # Penalidades mais suaves para diferenças no número de lesões
    if lesion_diff > 0:
        penalty = 1.0
        if lesion_diff <= 2:
            penalty = 0.9  # 10% de redução
        elif lesion_diff <= 4:
            penalty = 0.8  # 20% de redução
        elif lesion_diff <= 6:
            penalty = 0.7  # 30% de redução
        else:
            penalty = 0.6  # 40% de redução
        final_similarity *= penalty
    
    # Penalidades mais suaves para diferenças na área afetada
    area_diff = abs(area_afetada1 - area_afetada2)
    if area_diff > 0:
        penalty = 1.0
        if area_diff <= 0.05:  # Diferença de até 5%
            penalty = 0.95  # 5% de redução
        elif area_diff <= 0.10:  # Diferença de até 10%
            penalty = 0.85  # 15% de redução
        elif area_diff <= 0.15:  # Diferença de até 15%
            penalty = 0.75  # 25% de redução
        else:
            penalty = 0.65  # 35% de redução
        final_similarity *= penalty
    
    # Bônus para folhas com características muito similares
    if lesion_diff <= 2 and area_diff <= 0.05:  # Diferenças pequenas
        final_similarity = min(100, final_similarity * 1.2)  # Aumenta em até 20%
    
    # Garantir similaridade mínima para folhas doentes
    if has_significant_lesions1 and has_significant_lesions2:
        final_similarity = max(30, final_similarity)  # Mínimo de 30% para folhas doentes
    
    # Limitar a similaridade máxima para evitar 100%
    final_similarity = min(95, final_similarity)
    
    return round(final_similarity, 1)

def analyze_query_results(results):
    """Analisa os resultados da consulta e retorna estatísticas"""
    if not results or not isinstance(results, tuple) or len(results) != 2:
        return None
    
    results, has_lesions = results
    
    if 'distances' not in results or not results['distances']:
        return None
    
    try:
        distances = results['distances'][0]
        metadatas = results['metadatas'][0]
        ids = results['ids'][0]
        embeddings = results['embeddings'][0]
        
        # Extrair características da imagem de consulta para comparação
        query_emb = embeddings[0]
        query_features = extract_features(query_emb)
        
        # Obter o caminho da imagem de consulta
        query_path = metadatas[0].get('path', 'Caminho desconhecido')
        
        print("\n=== Características da Imagem de Consulta ===")
        print(f"Caminho: {query_path}")
        print(f"Número de lesões: {query_features['shape']['num_lesions']:.2f}")
        print(f"Área afetada: {query_features['shape']['disease_coverage']:.2%}")
        print(f"Tamanho médio das lesões: {query_features['shape']['avg_lesion_size']:.2f}")
        print(f"Densidade de lesões: {query_features['shape']['lesion_density']:.2f}")
        
        # Calcular similaridades considerando características
        similarities = []
        detailed_comparisons = []
        
        # Ignorar a primeira imagem (imagem de consulta) e imagens de análise
        for i, (dist, emb, meta) in enumerate(zip(distances[1:], embeddings[1:], metadatas[1:]), 1):
            # Ignorar imagens de análise
            if meta.get('type') == 'leaf_disease_analysis':
                continue
                
            sim = calculate_similarity(dist, query_emb, emb)
            similarities.append(sim)
            
            # Extrair características da imagem comparada
            comp_features = extract_features(emb)
            
            # Calcular diferenças principais
            shape_diff = abs(query_features['shape']['num_lesions'] - comp_features['shape']['num_lesions'])
            area_diff = abs(query_features['shape']['disease_coverage'] - comp_features['shape']['disease_coverage'])
            texture_diff = abs(query_features['glcm']['contrast'] - comp_features['glcm']['contrast'])
            color_diff = abs(query_features['hsv']['h_stats']['mean'] - comp_features['hsv']['h_stats']['mean'])
            
            # Obter o caminho da imagem comparada
            comp_path = meta.get('path', 'Caminho desconhecido')
            
            # Normalizar a categoria para leaf_healthy ou leaf_with_disease
            category = meta.get('category', 'unknown')
            if 'healthy' in category.lower():
                category = 'leaf_healthy'
            elif category != 'query':
                category = 'leaf_with_disease'
            
            detailed_comparisons.append({
                'index': i,
                'category': category,
                'path': comp_path,
                'similarity': sim,
                'differences': {
                    'shape_diff': shape_diff,
                    'area_diff': area_diff,
                    'texture_diff': texture_diff,
                    'color_diff': color_diff
                },
                'features': comp_features
            })
        
        # Filtrar resultados com similaridade muito baixa
        min_similarity = 40
        valid_indices = [i for i, sim in enumerate(similarities) if sim >= min_similarity]
        
        if not valid_indices:
            return None
        
        # Filtrar resultados
        similarities = [similarities[i] for i in valid_indices]
        categories = [detailed_comparisons[i]['category'] for i in valid_indices]
        filtered_ids = [ids[i+1] for i in valid_indices]
        filtered_metadatas = [metadatas[i+1] for i in valid_indices]
        filtered_embeddings = [embeddings[i+1] for i in valid_indices]
        
        # Ordenar imagens similares por similaridade
        similar_images = [
            {
                'id': id_,
                'category': cat,
                'path': meta.get('path', 'Caminho desconhecido'),
                'similarity': sim,
                'metadata': meta,
                'features': extract_features(emb)
            }
            for id_, cat, sim, meta, emb in zip(filtered_ids, categories, similarities, 
                                               filtered_metadatas, filtered_embeddings)
        ]
        
        # Ordenar por similaridade decrescente
        similar_images.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Pegar as 5 imagens mais similares
        top_5_images = similar_images[:5]
        
        # Calcular estatísticas apenas das 5 imagens mais similares
        category_stats = {}
        total_sim = sum(img['similarity'] for img in top_5_images)
        
        for img in top_5_images:
            cat = img['category']
            sim = img['similarity']
            
            if cat not in category_stats:
                category_stats[cat] = {'count': 0, 'total_sim': 0, 'max_sim': 0}
            
            category_stats[cat]['count'] += 1
            category_stats[cat]['total_sim'] += sim
            category_stats[cat]['max_sim'] = max(category_stats[cat]['max_sim'], sim)
        
        # Calcular métricas finais
        for cat, stats in category_stats.items():
            stats['percentage'] = round((stats['total_sim'] / total_sim * 100), 1)
            stats['avg_similarity'] = round(stats['total_sim'] / stats['count'], 1)
        
        # Determinar categoria baseado nas 5 imagens mais similares
        if len(top_5_images) > 0:
            # Usar a categoria da imagem mais similar como base
            best_category = top_5_images[0]['category']
            
            # Se houver pelo menos 3 imagens da mesma categoria nas top 5, usar essa categoria
            category_counts = {}
            for img in top_5_images:
                cat = img['category']
                category_counts[cat] = category_counts.get(cat, 0) + 1
            
            # Se uma categoria tem maioria (3 ou mais), usar ela
            for cat, count in category_counts.items():
                if count >= 3:
                    best_category = cat
                    break
        else:
            # Fallback para a lógica anterior
            if has_lesions:
                disease_cats = [(cat, stats) for cat, stats in category_stats.items() 
                              if 'healthy' not in cat.lower() and cat != 'unknown']
                if disease_cats:
                    best_category = max(disease_cats, key=lambda x: x[1]['max_sim'])[0]
                else:
                    best_category = max(category_stats.items(), key=lambda x: x[1]['max_sim'])[0]
            else:
                if 'leaf_healthy' in category_stats:
                    best_category = 'leaf_healthy'
                else:
                    best_category = max(category_stats.items(), key=lambda x: x[1]['max_sim'])[0]
        
        # Calcular confiança baseada nas 5 imagens mais similares
        best_stats = category_stats[best_category]
        
        # Fatores para confiança
        similarity_factor = best_stats['max_sim'] / 100  # Normalizado para 0-1
        consistency_factor = best_stats['count'] / len(top_5_images)  # Proporção de imagens na melhor categoria
        
        # Se houver mais de uma categoria, considerar a diferença para a segunda melhor
        if len(category_stats) > 1:
            other_cats = [(cat, stats) for cat, stats in category_stats.items() 
                         if cat != best_category]
            second_best = max(other_cats, key=lambda x: x[1]['max_sim'])
            
            # Diferença normalizada entre as melhores categorias
            category_diff = (best_stats['max_sim'] - second_best[1]['max_sim']) / 100
        else:
            category_diff = 1.0
        
        # Calcular confiança final
        confidence = (
            similarity_factor * 0.4 +    # Peso da similaridade
            consistency_factor * 0.3 +    # Peso da consistência
            category_diff * 0.3          # Peso da diferença entre categorias
        ) * 100
        
        # Ajustar confiança baseado na consistência com características físicas
        if has_lesions and 'healthy' not in best_category.lower():
            confidence = min(100, confidence * 1.2)
        elif not has_lesions and 'healthy' in best_category.lower():
            confidence = min(100, confidence * 1.2)
        
        confidence = round(max(0, min(100, confidence)), 1)
        
        # Mostrar detalhes das 5 imagens mais similares
        print("\n=== Imagens Mais Similares ===")
        for i, img in enumerate(top_5_images, 1):
            features = img['features']
            print(f"\nImagem #{i}")
            print(f"Categoria: {img['category']}")
            print(f"Similaridade: {img['similarity']:.1f}%")
            print(f"Número de lesões: {features['shape']['num_lesions']:.2f}")
            print(f"Diferença de lesões: {abs(query_features['shape']['num_lesions'] - features['shape']['num_lesions']):.2f}")
            print(f"Área afetada: {features['shape']['disease_coverage']:.2%}")
            print(f"Caminho: {img['path']}")
        
        return {
            'identified_category': best_category,
            'confidence': confidence,
            'best_match': round(max(similarities), 1),
            'category_distribution': {
                cat: stats['percentage']
                for cat, stats in category_stats.items()
            },
            'similar_images': top_5_images
        }
            
    except Exception as e:
        print(f"Erro ao analisar resultados: {str(e)}")
        return None

def extract_features(embedding):
    """Extrai e formata as características do embedding"""
    # Converter para numpy array se necessário
    if not isinstance(embedding, np.ndarray):
        embedding = np.array(embedding)
    
    # Definir índices dos diferentes grupos de características
    hsv_hist_indices = slice(0, 96)        # 96 valores dos histogramas HSV
    hsv_stats_indices = slice(96, 108)     # 12 valores das estatísticas HSV
    glcm_indices = slice(108, 116)         # 8 valores do GLCM
    lbp_indices = slice(116, 120)          # 4 valores do LBP
    shape_indices = slice(120, 128)        # 8 valores de forma
    
    # Extrair características HSV
    hsv_stats = embedding[hsv_stats_indices]
    hsv_features = {
        'h_stats': {
            'mean': float(hsv_stats[0]),
            'std': float(hsv_stats[1]),
            'q25': float(hsv_stats[2]),
            'q75': float(hsv_stats[3])
        },
        's_stats': {
            'mean': float(hsv_stats[4]),
            'std': float(hsv_stats[5]),
            'q25': float(hsv_stats[6]),
            'q75': float(hsv_stats[7])
        },
        'v_stats': {
            'mean': float(hsv_stats[8]),
            'std': float(hsv_stats[9]),
            'q25': float(hsv_stats[10]),
            'q75': float(hsv_stats[11])
        }
    }
    
    # Extrair características GLCM
    glcm_features = {
        'contrast': float(embedding[glcm_indices][0]),
        'correlation': float(embedding[glcm_indices][1]),
        'energy': float(embedding[glcm_indices][2]),
        'homogeneity': float(embedding[glcm_indices][3]),
        'dissimilarity': float(embedding[glcm_indices][4]),
        'entropy': float(embedding[glcm_indices][5]),
        'cluster_shade': float(embedding[glcm_indices][6]),
        'cluster_prominence': float(embedding[glcm_indices][7])
    }
    
    # Extrair características LBP
    lbp_features = {
        'mean': float(embedding[lbp_indices][0]),
        'std': float(embedding[lbp_indices][1]),
        'energy': float(embedding[lbp_indices][2]),
        'entropy': float(embedding[lbp_indices][3])
    }
    
    # Extrair características de forma
    shape_features = {
        'num_lesions': float(embedding[shape_indices][0]),
        'avg_lesion_size': float(embedding[shape_indices][1]),
        'lesion_size_std': float(embedding[shape_indices][2]),
        'disease_coverage': float(embedding[shape_indices][3]),
        'lesion_density': float(embedding[shape_indices][4]),
        'avg_compactness': float(embedding[shape_indices][5]),
        'avg_distance': float(embedding[shape_indices][6]),
        'std_distance': float(embedding[shape_indices][7])
    }
    
    return {
        'hsv': hsv_features,
        'glcm': glcm_features,
        'lbp': lbp_features,
        'shape': shape_features
    }

def get_embedding_by_id(image_id):
    """Recupera um embedding e seus metadados pelo ID."""
    try:
        results = collection.get(
            ids=[image_id],
            include=["embeddings", "metadatas"]
        )
        
        if results and results['ids']:
            # Retorna o primeiro (e único) resultado encontrado
            return {
                'id': results['ids'][0],
                'embedding': results['embeddings'][0],
                'metadata': results['metadatas'][0]
            }
        else:
            return None # Nenhum resultado encontrado
            
    except Exception as e:
        print(f"Erro ao obter embedding por ID: {str(e)}")
        return {"error": str(e)}
