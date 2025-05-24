import tkinter as tk
from tkinter import ttk, messagebox
import sys
import os
import numpy as np

# Adiciona o diretório raiz do projeto ao sys.path para permitir importações
# Ajuste 'TADB-CBIR' se o nome da pasta raiz for diferente
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'TADB-CBIR'))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from database import chroma
    # from engine import processing_engine # Adicione se quiser incluir processamento/query
except ImportError as e:
    messagebox.showerror("Erro de Importação", f"Não foi possível importar módulos do projeto.\nVerifique se o caminho do projeto está correto.\nErro: {e}")
    sys.exit()

class SimpleDatabaseGUI(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("TADB-CBIR - Apresentação do Banco de Dados")
        self.geometry("600x400")

        self.create_widgets()

    def create_widgets(self):
        # Frame principal
        main_frame = ttk.Frame(self, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        # Título
        title_label = ttk.Label(main_frame, text="Status e Consulta do Banco de Dados ChromaDB", font=('Arial', 14, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=10)

        # Frame para botões de estatísticas/limpeza
        stats_clear_frame = ttk.Frame(main_frame)
        stats_clear_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E))
        stats_clear_frame.columnconfigure(0, weight=1)
        stats_clear_frame.columnconfigure(1, weight=1)

        # Botão para obter estatísticas
        stats_button = ttk.Button(stats_clear_frame, text="Obter Estatísticas do Banco", command=self.display_stats)
        stats_button.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)

        # Botão para limpar o banco de dados
        clear_button = ttk.Button(stats_clear_frame, text="Limpar Banco de Dados", command=self.clear_db)
        clear_button.grid(row=0, column=1, padx=5, pady=5, sticky=tk.E)

        # Frame para consulta por ID
        query_frame = ttk.Frame(main_frame)
        query_frame.grid(row=2, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))
        query_frame.columnconfigure(1, weight=1) # Campo de entrada se expande

        query_label = ttk.Label(query_frame, text="Consultar por ID:")
        query_label.grid(row=0, column=0, padx=5, sticky=tk.W)

        self.id_entry = ttk.Entry(query_frame, width=50)
        self.id_entry.grid(row=0, column=1, padx=5, sticky=(tk.W, tk.E))

        query_button = ttk.Button(query_frame, text="Buscar ID", command=self.search_by_id)
        query_button.grid(row=0, column=2, padx=5, sticky=tk.W)

        # Área para exibir as estatísticas/mensagens/resultados da consulta
        self.stats_text = tk.Text(main_frame, wrap=tk.WORD, height=15, width=70)
        self.stats_text.grid(row=3, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configurar expansão
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(3, weight=1) # Área de texto se expande


    def display_stats(self):
        """Obtém e exibe as estatísticas do banco de dados."""
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(tk.END, "Obtendo estatísticas...\n")

        stats = chroma.get_database_stats()

        if "error" in stats:
            self.stats_text.insert(tk.END, f"Erro ao obter estatísticas: {stats['error']}\n", 'error')
        else:
            self.stats_text.insert(tk.END, "\n--- Estatísticas do Banco de Dados ---\n")
            self.stats_text.insert(tk.END, f"Total de Imagens: {stats.get('total_images', 0)}\n")
            self.stats_text.insert(tk.END, "Categorias:\n")
            categories = stats.get('categories', {})
            if categories:
                for cat, count in categories.items():
                    self.stats_text.insert(tk.END, f"  - {cat}: {count}\n")
            else:
                 self.stats_text.insert(tk.END, "  Nenhuma categoria encontrada.\n")

            self.stats_text.insert(tk.END, f"Última Atualização: {stats.get('last_update', 'N/A')}\n")

            # Exibir IDs
            ids = stats.get('ids', [])
            self.stats_text.insert(tk.END, "IDs Salvos:\n")
            if ids:
                # Exibir apenas os primeiros 10 IDs para não sobrecarregar
                display_ids = ids[:10]
                self.stats_text.insert(tk.END, "  " + "\n  ".join(display_ids) + ("\n  ..." if len(ids) > 10 else "") + "\n")
            else:
                 self.stats_text.insert(tk.END, "  Nenhum ID encontrado.\n")

        self.stats_text.see(tk.END) # Rolha para o final

    def clear_db(self):
        """Limpa o banco de dados."""
        confirm = messagebox.askyesno("Confirmar Limpeza", "Tem certeza que deseja limpar todo o banco de dados?")
        if confirm:
            self.stats_text.delete(1.0, tk.END)
            self.stats_text.insert(tk.END, "Limpando banco de dados...\n")
            success = chroma.clear_database()
            if success:
                self.stats_text.insert(tk.END, "Banco de dados limpo com sucesso!\n", 'success')
                self.display_stats() # Atualiza as estatísticas após a limpeza
            else:
                self.stats_text.insert(tk.END, "Erro ao limpar o banco de dados.\n", 'error')
        self.stats_text.see(tk.END) # Rolha para o final

    def search_by_id(self):
        """Busca um item no banco de dados pelo ID fornecido e exibe os resultados."""
        image_id = self.id_entry.get().strip()
        if not image_id:
            messagebox.showwarning("ID Vazio", "Por favor, insira um ID para buscar.")
            return

        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(tk.END, f"Buscando item com ID: {image_id}...\n")

        item = chroma.get_embedding_by_id(image_id)

        if item is None:
            self.stats_text.insert(tk.END, f"\nNenhum item encontrado com o ID: {image_id}\n", 'warning')
        elif "error" in item:
             self.stats_text.insert(tk.END, f"\nErro ao buscar item: {item['error']}\n", 'error')
        else:
            self.stats_text.insert(tk.END, f"\n--- Item Encontrado (ID: {item['id']}) ---\n")
            
            # Exibir Metadados
            self.stats_text.insert(tk.END, "Metadados:\n")
            metadata = item.get('metadata', {})
            if metadata:
                for key, value in metadata.items():
                    self.stats_text.insert(tk.END, f"  {key}: {value}\n")
            else:
                self.stats_text.insert(tk.END, "  Nenhum metadado encontrado.\n")

            # Exibir Características Detalhadas
            self.stats_text.insert(tk.END, "\n--- Características Detalhadas do Embedding ---\n")
            embedding = item.get('embedding', [])

            if isinstance(embedding, (list, np.ndarray)) and len(embedding) > 0:
                try:
                    # Usar a função existente para extrair características formatadas
                    formatted_features = chroma.extract_features(embedding)

                    # Informar que o Histograma HSV é grande e não será exibido por padrão (Primeiro na ordem do embedding)
                    self.stats_text.insert(tk.END, "\n* O Histograma HSV completo (96 valores) não é exibido detalhadamente aqui por ser extenso.\n")

                    # Exibir Estatísticas HSV (Segundo na ordem do embedding)
                    self.stats_text.insert(tk.END, "\nEstatísticas de Cor (HSV):\n")
                    hsv_features = formatted_features.get('hsv', {})
                    if hsv_features:
                         for channel, stats in hsv_features.items():
                             self.stats_text.insert(tk.END, f"  - Canal {channel.upper().replace('_STATS', '')}:\n")
                             self.stats_text.insert(tk.END, f"      Média: {stats.get('mean', 'N/A'):.4f}, ")
                             self.stats_text.insert(tk.END, f"Desvio Padrão: {stats.get('std', 'N/A'):.4f}, ")
                             self.stats_text.insert(tk.END, f"Q25: {stats.get('q25', 'N/A'):.4f}, ")
                             self.stats_text.insert(tk.END, f"Q75: {stats.get('q75', 'N/A'):.4f}\n")
                    else:
                        self.stats_text.insert(tk.END, "  Nenhuma estatística HSV encontrada.\n")

                    # Exibir Características GLCM (Terceiro na ordem do embedding)
                    self.stats_text.insert(tk.END, "\nCaracterísticas de Textura (GLCM):\n")
                    glcm_features = formatted_features.get('glcm', {})
                    if glcm_features:
                        for key, value in glcm_features.items():
                            # Capitalizar a primeira letra da chave para exibição
                            display_key = key.capitalize()
                            self.stats_text.insert(tk.END, f"  {display_key}: {value:.4f}\n")
                    else:
                        self.stats_text.insert(tk.END, "  Nenhuma característica GLCM encontrada.\n")

                    # Exibir Características LBP (Quarto na ordem do embedding)
                    lbp_features = formatted_features.get('lbp', {})
                    if lbp_features:
                         self.stats_text.insert(tk.END, "\nCaracterísticas LBP:\n")
                         for key, value in lbp_features.items():
                             display_key = key.capitalize()
                             self.stats_text.insert(tk.END, f"  {display_key}: {value:.4f}\n")
                    else:
                        self.stats_text.insert(tk.END, "  Nenhuma característica LBP encontrada.\n")

                    # Exibir características de Forma (Quinto na ordem do embedding)
                    self.stats_text.insert(tk.END, "\nCaracterísticas de Forma:\n")
                    shape_features = formatted_features.get('shape', {})
                    if shape_features:
                        self.stats_text.insert(tk.END, f"  Número de lesões: {shape_features.get('num_lesions', 'N/A'):.2f}\n")
                        self.stats_text.insert(tk.END, f"  Área afetada: {shape_features.get('disease_coverage', 'N/A'):.2%}\n")
                        self.stats_text.insert(tk.END, f"  Tamanho médio das lesões: {shape_features.get('avg_lesion_size', 'N/A'):.2f}\n")
                        self.stats_text.insert(tk.END, f"  Densidade de lesões: {shape_features.get('lesion_density', 'N/A'):.2f}\n")
                        self.stats_text.insert(tk.END, f"  Compactação média: {shape_features.get('avg_compactness', 'N/A'):.2f}\n")
                        self.stats_text.insert(tk.END, f"  Distância média do centro: {shape_features.get('avg_distance', 'N/A'):.2f}\n")
                        self.stats_text.insert(tk.END, f"  Desvio padrão da distância: {shape_features.get('std_distance', 'N/A'):.2f}\n")
                        # lesion_size_std está definido mas não foi pedido para mostrar, podemos adicionar se quiser
                        # self.stats_text.insert(tk.END, f"  Desvio padrão do tamanho da lesão: {shape_features.get('lesion_size_std', 'N/A'):.2f}\n")
                    else:
                        self.stats_text.insert(tk.END, "  Nenhuma característica de forma encontrada.\n")


                except Exception as e:
                    self.stats_text.insert(tk.END, f"\nErro ao formatar ou exibir características: {e}\n", 'error')

                # Adicionar a visualização do embedding bruto por categoria novamente
                self.stats_text.insert(tk.END, "\n---\n") # Separador
                self.stats_text.insert(tk.END, "\nEmbedding completo (Valores brutos por categoria):\n")

                # Configurar NumPy para não usar notação científica (garantir)
                np.set_printoptions(suppress=True, precision=6)

                # Converter para array NumPy se for lista (garantir)
                if isinstance(embedding, list):
                    embedding = np.array(embedding)

                # Definir índices dos diferentes grupos de características (mesmos de database/chroma.py)
                hsv_hist_indices = slice(0, 96)        # 96 valores dos histogramas HSV
                hsv_stats_indices = slice(96, 108)     # 12 valores das estatísticas HSV
                glcm_indices = slice(108, 116)         # 8 valores do GLCM
                lbp_indices = slice(116, 120)          # 4 valores do LBP
                shape_indices = slice(120, 128)        # 8 valores de forma

                # Definir a ordem das seções para corresponder à ordem do embedding
                feature_sections_raw_ordered = [
                     ("Histograma HSV (96 valores)", hsv_hist_indices),
                     ("Estatísticas HSV (12 valores)", hsv_stats_indices),
                     ("Características GLCM (8 valores)", glcm_indices),
                     ("Características LBP (4 valores)", lbp_indices),
                     ("Características de Forma (8 valores)", shape_indices),
                ]

                num_cols = 8 # Manter 8 colunas para visualização

                # Iterar sobre as seções na nova ordem
                for description, indices in feature_sections_raw_ordered:
                   self.stats_text.insert(tk.END, f"\n--- {description} (Valores brutos) ---\n")
                   section_embedding = embedding[indices]
                   section_len = len(section_embedding)
                   num_rows = (section_len + num_cols - 1) // num_cols

                   for i in range(num_rows):
                       start_idx = i * num_cols
                       end_idx = min((i + 1) * num_cols, section_len)
                       row_values = section_embedding[start_idx:end_idx]

                       # Formatar cada valor com 6 casas decimais e alinhar
                       formatted_values = [f"{val:10.6f}" for val in row_values]
                       self.stats_text.insert(tk.END, "  " + " ".join(formatted_values) + "\n")

            else:
                 self.stats_text.insert(tk.END, "  Nenhum embedding encontrado para detalhar.\n")

        self.stats_text.see(tk.END) # Rolha para o final


if __name__ == "__main__":
    app = SimpleDatabaseGUI()
    app.mainloop() 