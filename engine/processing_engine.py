import threading
import cv2
import numpy as np
from database import chroma
from datetime import datetime
import matplotlib.pyplot as plt
import os

def segment_leaf(image):
    """Segmenta a folha do fundo"""
    # Converter para HSV para melhor segmentação de verde
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Definir range para cor verde (folha)
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([85, 255, 255])
    
    # Criar máscara para região verde
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Aplicar operações morfológicas para limpar a máscara
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    return mask

def extract_disease_features(image, mask, image_path=None):
    """Extrai características específicas para identificação de doenças"""
    # Aplicar máscara na imagem
    leaf_region = cv2.bitwise_and(image, image, mask=mask)
    
    # Converter para diferentes espaços de cor para análise
    hsv = cv2.cvtColor(leaf_region, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(leaf_region, cv2.COLOR_BGR2LAB)
    
    features = []
    feature_names = []
    
    # 1. Características de cor (HSV - melhor para detectar manchas e descoloração)
    h_hist = cv2.calcHist([hsv], [0], mask, [32], [0, 180])
    s_hist = cv2.calcHist([hsv], [1], mask, [32], [0, 256])
    v_hist = cv2.calcHist([hsv], [2], mask, [32], [0, 256])
    
    # Normalizar histogramas pelo número total de pixels
    total_pixels = np.sum(mask) / 255.0
    if total_pixels > 0:
        h_hist_norm = h_hist / total_pixels
        s_hist_norm = s_hist / total_pixels
        v_hist_norm = v_hist / total_pixels
    else:
        h_hist_norm = h_hist
        s_hist_norm = s_hist
        v_hist_norm = v_hist
    
    # Adicionar estatísticas dos canais HSV
    h_channel = hsv[:,:,0]
    s_channel = hsv[:,:,1]
    v_channel = hsv[:,:,2]
    
    valid_mask = mask > 0
    
    # Estatísticas do canal H (matiz)
    h_stats = [
        np.mean(h_channel[valid_mask]),
        np.std(h_channel[valid_mask]),
        np.percentile(h_channel[valid_mask], 25),
        np.percentile(h_channel[valid_mask], 75)
    ]
    
    # Estatísticas do canal S (saturação)
    s_stats = [
        np.mean(s_channel[valid_mask]),
        np.std(s_channel[valid_mask]),
        np.percentile(s_channel[valid_mask], 25),
        np.percentile(s_channel[valid_mask], 75)
    ]
    
    # Estatísticas do canal V (valor)
    v_stats = [
        np.mean(v_channel[valid_mask]),
        np.std(v_channel[valid_mask]),
        np.percentile(v_channel[valid_mask], 25),
        np.percentile(v_channel[valid_mask], 75)
    ]
    
    features.extend(h_hist_norm.flatten())
    features.extend(s_hist_norm.flatten())
    features.extend(v_hist_norm.flatten())
    features.extend(h_stats)
    features.extend(s_stats)
    features.extend(v_stats)
    
    feature_names.extend([f'Matiz_bin_{i}' for i in range(32)])
    feature_names.extend([f'Saturacao_bin_{i}' for i in range(32)])
    feature_names.extend([f'Valor_bin_{i}' for i in range(32)])
    feature_names.extend(['H_media', 'H_desvio', 'H_q25', 'H_q75'])
    feature_names.extend(['S_media', 'S_desvio', 'S_q25', 'S_q75'])
    feature_names.extend(['V_media', 'V_desvio', 'V_q25', 'V_q75'])
    
    # 2. Características de textura melhoradas
    gray = cv2.cvtColor(leaf_region, cv2.COLOR_BGR2GRAY)
    
    # Detectar regiões doentes usando múltiplos critérios
    def detect_disease_regions():
        """Detecta regiões com lesões na folha"""
        # 1. Análise HSV adaptativa
        hsv_mask = np.zeros_like(mask)
        
        # Calcular limiares adaptativos baseados na distribuição de cores
        h_mean = np.mean(h_channel[valid_mask])
        h_std = np.std(h_channel[valid_mask])
        s_mean = np.mean(s_channel[valid_mask])
        s_std = np.std(s_channel[valid_mask])
        v_mean = np.mean(v_channel[valid_mask])
        v_std = np.std(v_channel[valid_mask])
        
        # Critérios mais sensíveis para HSV
        hsv_disease = (
            ((h_channel < (h_mean - 1.0 * h_std)) |  # Tons diferentes do verde (mais sensível)
             (h_channel > 140)) &                     # Tons marrons/amarelados (mais sensível)
            (s_channel < (s_mean - 0.8 * s_std)) &   # Baixa saturação (mais sensível)
            (v_channel < (v_mean - 1.0 * v_std))     # Regiões mais escuras (mais sensível)
        )
        
        hsv_mask[valid_mask] = hsv_disease[valid_mask]
        
        # 2. Análise LAB adaptativa
        lab = cv2.cvtColor(cv2.cvtColor(leaf_region, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2LAB)
        l_channel = lab[:,:,0]  # Luminosidade
        a_channel = lab[:,:,1]  # Verde-Vermelho
        b_channel = lab[:,:,2]  # Azul-Amarelo
        
        # Calcular limiares adaptativos LAB
        l_mean = np.mean(l_channel[valid_mask])
        l_std = np.std(l_channel[valid_mask])
        a_mean = np.mean(a_channel[valid_mask])
        a_std = np.std(a_channel[valid_mask])
        b_mean = np.mean(b_channel[valid_mask])
        b_std = np.std(b_channel[valid_mask])
        
        # Critérios mais sensíveis para LAB
        lab_mask = np.zeros_like(mask)
        lab_disease = (
            (l_channel < (l_mean - 1.0 * l_std)) &   # Regiões mais escuras (mais sensível)
            ((a_channel > (a_mean + 1.0 * a_std)) |  # Desvio para vermelho
             (b_channel > (b_mean + 1.0 * b_std)))   # Desvio para amarelo
        )
        
        lab_mask[valid_mask] = lab_disease[valid_mask]
        
        # 3. Análise de textura local
        gray = cv2.cvtColor(leaf_region, cv2.COLOR_BGR2GRAY)
        texture_mask = np.zeros_like(mask)
        
        # Calcular variação local usando desvio padrão
        kernel_size = 3  # Kernel menor para detectar variações menores
        local_std = cv2.blur((gray - cv2.blur(gray, (kernel_size, kernel_size)))**2, 
                            (kernel_size, kernel_size))
        
        # Limiar de textura mais sensível
        texture_thresh = np.percentile(local_std[valid_mask], 70)  # Percentil mais baixo
        texture_mask[valid_mask] = (local_std > texture_thresh)[valid_mask]
        
        # 4. Combinar as máscaras de forma mais sensível
        # Requer que a região seja detectada em pelo menos uma análise forte ou duas fracas
        hsv_weight = 1.5  # Peso maior para HSV
        lab_weight = 1.2  # Peso médio para LAB
        texture_weight = 1.0  # Peso normal para textura
        
        weighted_sum = (
            hsv_mask.astype(float) * hsv_weight +
            lab_mask.astype(float) * lab_weight +
            texture_mask.astype(float) * texture_weight
        )
        
        # Uma região é considerada doente se tiver peso total >= 2.0
        combined_mask = (weighted_sum >= 2.0).astype(np.uint8) * 255
        
        # 5. Pós-processamento mais sensível
        kernel_small = np.ones((3,3), np.uint8)
        kernel_medium = np.ones((5,5), np.uint8)
        
        # Primeiro dilatar um pouco para conectar regiões próximas
        disease_mask = cv2.dilate(combined_mask, kernel_small, iterations=1)
        # Depois remover ruídos pequenos
        disease_mask = cv2.morphologyEx(disease_mask, cv2.MORPH_OPEN, kernel_small)
        disease_mask = cv2.morphologyEx(disease_mask, cv2.MORPH_CLOSE, kernel_medium)
        
        # 6. Análise de contornos com critérios mais sensíveis
        contours, _ = cv2.findContours(disease_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Criar máscara final
        final_mask = np.zeros_like(mask)
        
        # Parâmetros mais sensíveis para filtragem de contornos
        leaf_area = np.sum(mask) / 255.0
        min_area = max(10, leaf_area * 0.0008)  # 0.08% da folha (mais sensível)
        max_area = leaf_area * 0.35  # 35% da folha
        
        valid_contours = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                # Criar ROI para o contorno
                mask_roi = np.zeros_like(mask)
                cv2.drawContours(mask_roi, [contour], -1, 255, -1)
                
                # Calcular características da região
                roi_pixels = cv2.bitwise_and(gray, gray, mask=mask_roi)
                roi_valid = roi_pixels[mask_roi > 0]
                
                if len(roi_valid) > 0:
                    # Calcular estatísticas da região
                    roi_mean = np.mean(roi_valid)
                    roi_std = np.std(roi_valid)
                    
                    # Critérios mais sensíveis para validação
                    intensity_valid = roi_mean < (np.mean(gray[valid_mask]) - 0.8 * np.std(gray[valid_mask]))
                    texture_valid = np.mean(local_std[mask_roi > 0]) > texture_thresh * 0.8
                    
                    # Verificar forma do contorno
                    perimeter = cv2.arcLength(contour, True)
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    
                    # Critérios mais flexíveis para aceitação
                    if (intensity_valid or texture_valid) and 0.2 < circularity < 0.95:
                        cv2.drawContours(final_mask, [contour], -1, 255, -1)
                        valid_contours += 1
        
        # Validação final mais sensível
        if valid_contours > 0:
            return final_mask
        else:
            return np.zeros_like(mask)
    
    disease_mask = detect_disease_regions()
    
    # Calcular GLCM apenas para regiões doentes
    def calculate_glcm_features(img, disease_regions, distances=[1, 2], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
        height, width = img.shape
        glcm = np.zeros((256, 256))
        
        # Usar apenas pixels dentro das regiões doentes
        for distance in distances:
            for angle in angles:
                dx = int(round(distance * np.cos(angle)))
                dy = int(round(distance * np.sin(angle)))
                
                for i in range(max(0, -dx), min(height, height - dx)):
                    for j in range(max(0, -dy), min(width, width - dy)):
                        # Verificar se ambos os pixels estão em regiões doentes
                        if disease_regions[i,j] and disease_regions[i+dx,j+dy]:
                            glcm[img[i,j], img[i+dx,j+dy]] += 1
        
        # Se não houver pixels doentes, retornar zeros
        if glcm.sum() == 0:
            return [0] * 8
        
        # Normalizar GLCM
        glcm = glcm / glcm.sum()
        
        # Calcular características do GLCM
        indices = np.arange(256)
        i_indices, j_indices = np.meshgrid(indices, indices, indexing='ij')
        
        # Médias e variâncias
        mean_i = np.sum(i_indices * glcm)
        mean_j = np.sum(j_indices * glcm)
        std_i = np.sqrt(np.sum(((i_indices - mean_i) ** 2) * glcm))
        std_j = np.sqrt(np.sum(((j_indices - mean_j) ** 2) * glcm))
        
        # Evitar divisão por zero
        epsilon = 1e-10
        std_prod = std_i * std_j
        if std_prod < epsilon:
            std_prod = epsilon
        
        # Calcular características
        contrast = np.sum(((i_indices - j_indices) ** 2) * glcm)
        correlation = np.sum(((i_indices - mean_i) * (j_indices - mean_j) * glcm) / std_prod)
        energy = np.sum(glcm ** 2)
        homogeneity = np.sum(glcm / (1 + (i_indices - j_indices) ** 2))
        
        # Características adicionais
        dissimilarity = np.sum(np.abs(i_indices - j_indices) * glcm)
        entropy = -np.sum(glcm * np.log2(glcm + epsilon))
        
        # Características de cluster
        cluster_shade = np.sum(((i_indices + j_indices - mean_i - mean_j) ** 3) * glcm)
        cluster_prominence = np.sum(((i_indices + j_indices - mean_i - mean_j) ** 4) * glcm)
        
        return [
            contrast,           # Contraste
            correlation,        # Correlação
            energy,            # Energia
            homogeneity,       # Homogeneidade
            dissimilarity,     # Dissimilaridade
            entropy,           # Entropia
            cluster_shade,     # Sombreamento de cluster
            cluster_prominence # Proeminência de cluster
        ]
    
    # Calcular LBP apenas para regiões doentes
    def calculate_lbp(img, disease_regions):
        lbp = np.zeros_like(img)
        
        # Se não houver regiões doentes, retornar características zero
        if np.sum(disease_regions) == 0:
            return [0, 0, 0, 0]
        
        # Calcular LBP apenas para pixels em regiões doentes
        for i in range(1, img.shape[0]-1):
            for j in range(1, img.shape[1]-1):
                if disease_regions[i,j]:
                    center = img[i,j]
                    code = 0
                    code |= (img[i-1,j-1] >= center) << 7
                    code |= (img[i-1,j] >= center) << 6
                    code |= (img[i-1,j+1] >= center) << 5
                    code |= (img[i,j+1] >= center) << 4
                    code |= (img[i+1,j+1] >= center) << 3
                    code |= (img[i+1,j] >= center) << 2
                    code |= (img[i+1,j-1] >= center) << 1
                    code |= (img[i,j-1] >= center)
                    lbp[i,j] = code
        
        # Calcular histograma apenas para regiões doentes
        hist = cv2.calcHist([lbp], [0], disease_regions, [256], [0,256])
        hist = hist.flatten() / np.sum(hist) if np.sum(hist) > 0 else hist.flatten()
        
        # Extrair características do LBP
        lbp_mean = np.mean(hist)
        lbp_std = np.std(hist)
        lbp_energy = np.sum(np.square(hist))
        lbp_entropy = -np.sum(hist * np.log2(hist + 1e-10))
        
        return [lbp_mean, lbp_std, lbp_energy, lbp_entropy]
    
    # Calcular características de textura apenas para regiões doentes
    glcm_features = calculate_glcm_features(gray, disease_mask)
    features.extend(glcm_features)
    feature_names.extend([
        'GLCM_Contraste', 'GLCM_Correlacao', 'GLCM_Energia', 'GLCM_Homogeneidade',
        'GLCM_Dissimilaridade', 'GLCM_Entropia', 'GLCM_Cluster_Shade', 'GLCM_Cluster_Prominence'
    ])
    
    lbp_features = calculate_lbp(gray, disease_mask)
    features.extend(lbp_features)
    feature_names.extend(['LBP_Media', 'LBP_Desvio', 'LBP_Energia', 'LBP_Entropia'])
    
    # 3. Características de forma e região melhoradas
    # Encontrar contornos das regiões doentes
    contours, _ = cv2.findContours(disease_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filtrar contornos muito pequenos
    min_area = 30
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    
    if valid_contours:
        # Calcular características das regiões doentes
        areas = [cv2.contourArea(cnt) for cnt in valid_contours]
        perimeters = [cv2.arcLength(cnt, True) for cnt in valid_contours]
        total_leaf_area = np.sum(mask) / 255.0
        
        # Características de forma e distribuição
        total_disease_area = sum(areas)
        disease_coverage = total_disease_area / total_leaf_area
        avg_lesion_size = np.mean(areas)
        lesion_size_std = np.std(areas) if len(areas) > 1 else 0
        num_lesions = len(valid_contours)
        lesion_density = num_lesions / total_leaf_area * 1000
        
        # Calcular compacidade e elongação das lesões
        compactness = [4 * np.pi * area / (p * p) for area, p in zip(areas, perimeters)]
        avg_compactness = np.mean(compactness)
        
        # Calcular distribuição espacial das lesões
        centroids = [np.mean(cnt.reshape(-1,2), axis=0) for cnt in valid_contours]
        if len(centroids) > 1:
            centroid_distances = []
            for i in range(len(centroids)):
                for j in range(i+1, len(centroids)):
                    dist = np.linalg.norm(centroids[i] - centroids[j])
                    centroid_distances.append(dist)
            avg_distance = np.mean(centroid_distances)
            std_distance = np.std(centroid_distances)
        else:
            avg_distance = 0
            std_distance = 0
        
        shape_features = [
            num_lesions,          # número de lesões
            avg_lesion_size,      # tamanho médio das lesões
            lesion_size_std,      # variação no tamanho
            disease_coverage,      # proporção de área afetada
            lesion_density,       # densidade de lesões
            avg_compactness,      # circularidade média
            avg_distance,         # distância média entre lesões
            std_distance          # variação na distância entre lesões
        ]
    else:
        shape_features = [0, 0, 0, 0, 0, 0, 0, 0]
    
    features.extend(shape_features)
    feature_names.extend([
        'Num_Lesoes', 'Tamanho_Medio', 'Desvio_Tamanho', 
        'Area_Afetada', 'Densidade_Lesoes', 'Circularidade',
        'Dist_Media_Lesoes', 'Desvio_Dist_Lesoes'
    ])
    
    return np.array(features, dtype=np.float32), feature_names

def visualize_features(image_path, features_dict):
    """Visualiza as características extraídas da imagem"""
    plt.figure(figsize=(15, 12))
    
    # 1. Imagem original
    plt.subplot(331)
    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.title('Imagem Original')
    plt.axis('off')
    
    # 2. Máscara da folha
    plt.subplot(332)
    plt.imshow(features_dict['mask'], cmap='gray')
    plt.title('Máscara da Folha')
    plt.axis('off')
    
    # 3. Região processada
    plt.subplot(333)
    processed = cv2.cvtColor(features_dict['processed_image'], cv2.COLOR_BGR2RGB)
    plt.imshow(processed)
    plt.title('Região Processada')
    plt.axis('off')
    
    # 4. Histogramas HSV
    plt.subplot(334)
    h_values = features_dict['features'][:32]
    s_values = features_dict['features'][32:64]
    v_values = features_dict['features'][64:96]
    
    plt.plot(h_values, 'r-', label='Matiz (H)')
    plt.plot(s_values, 'g-', label='Saturação (S)')
    plt.plot(v_values, 'b-', label='Valor (V)')
    plt.title('Histogramas HSV')
    plt.legend()
    
    # 5. Estatísticas HSV
    plt.subplot(335)
    hsv_stats = features_dict['features'][96:108]  # 12 valores (4 para cada canal)
    stat_names = ['Média', 'Desvio', 'Q25', 'Q75']
    x = np.arange(len(stat_names))
    width = 0.25
    
    plt.bar(x - width, hsv_stats[0:4], width, label='H', color='r', alpha=0.7)
    plt.bar(x, hsv_stats[4:8], width, label='S', color='g', alpha=0.7)
    plt.bar(x + width, hsv_stats[8:12], width, label='V', color='b', alpha=0.7)
    plt.title('Estatísticas HSV')
    plt.xticks(x, stat_names, rotation=45)
    plt.legend()
    
    # 6. Características de Textura (GLCM - Parte 1)
    plt.subplot(336)
    glcm_features1 = features_dict['features'][108:112]  # Primeiros 4 valores GLCM
    plt.bar(['Contraste', 'Correlação', 'Energia', 'Homog.'], glcm_features1)
    plt.title('GLCM (Parte 1)')
    plt.xticks(rotation=45)
    
    # 6b. Características de Textura (GLCM - Parte 2)
    plt.subplot(337)
    glcm_features2 = features_dict['features'][112:116]  # Últimos 4 valores GLCM
    plt.bar(['Dissimil.', 'Entropia', 'Clust.Shade', 'Clust.Prom.'], glcm_features2)
    plt.title('GLCM (Parte 2)')
    plt.xticks(rotation=45)
    
    # 7. Características LBP
    plt.subplot(338)
    lbp_features = features_dict['features'][116:120]  # 4 valores LBP
    plt.bar(['Média', 'Desvio', 'Energia', 'Entropia'], lbp_features)
    plt.title('Características LBP')
    plt.xticks(rotation=45)
    
    # 8. Características de Forma
    plt.subplot(339)
    shape_features = features_dict['features'][-8:]  # 8 valores de forma
    labels = ['Num.Les.', 'Tam.Med.', 'Desv.Tam.', 'Area.Af.',
             'Dens.Les.', 'Circ.', 'Dist.Med.', 'Desv.Dist.']
    plt.bar(range(len(shape_features)), shape_features)
    plt.xticks(range(len(shape_features)), labels, rotation=45)
    plt.title('Características de Forma')
    
    plt.tight_layout()
    
    # Salvar a visualização
    base_path = os.path.splitext(image_path)[0]  # Remove qualquer extensão
    output_path = f"{base_path}_analysis.jpg"
    plt.savefig(output_path)
    plt.close()
    
    return output_path

def process_image(path: str, save_to_db: bool = True, visualize: bool = False):
    """Processa a imagem e opcionalmente salva no banco de dados"""
    # Carregar imagem
    image = cv2.imread(path)
    
    if image is None:
        return {"error": "Erro ao carregar a imagem."}
    
    print(f"\n=== Processando imagem: {path} ===")
    
    # Redimensionar para um tamanho padrão
    image = cv2.resize(image, (224, 224))
    
    # Segmentar a folha
    mask = segment_leaf(image)
    
    # Extrair características
    features, feature_names = extract_disease_features(image, mask, path)
    
    # Preparar resultado
    result = {
        "features": features.tolist(),
        "feature_names": feature_names,
        "processed_image": cv2.bitwise_and(image, image, mask=mask),
        "mask": mask,
        "original_image": image
    }
    
    if save_to_db:
        # Salvar no ChromaDB
        metadata = {
            "path": path,
            "type": "leaf_disease",
            "processing_date": str(datetime.now())
        }
        chroma.add_embedding(
            id=path.split("/")[-1],
            embedding=features.tolist(),
            metadata=metadata
        )
    
    # Apenas gerar visualização se explicitamente solicitado
    if visualize:
        result["visualization_path"] = visualize_features(path, result)
        print(f"Imagem de análise gerada: {result['visualization_path']}")
    
    return result