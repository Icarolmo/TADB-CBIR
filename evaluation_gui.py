import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import os
from pathlib import Path
from evaluation_system import CBIREvaluationSystem
from database import chroma
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

class EvaluationGUI(tk.Tk):
    """
    Interface gráfica para o sistema de avaliação CBIR
    """
    
    def __init__(self):
        super().__init__()
        
        self.title("Sistema de Avaliação CBIR - Previsão de Revogação")
        self.geometry("1200x800")
        
        # Sistema de avaliação
        self.evaluator = CBIREvaluationSystem()
        self.evaluation_results = None
        
        # Configurar interface
        self.create_widgets()
        self.update_database_stats()
        
    def create_widgets(self):
        """Cria os widgets da interface"""
        
        # Frame principal
        main_frame = ttk.Frame(self, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        
        # Título
        title_label = ttk.Label(main_frame, text="Sistema de Avaliação CBIR", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Frame superior - Estatísticas do banco
        stats_frame = ttk.LabelFrame(main_frame, text="Estatísticas do Banco de Dados", padding="10")
        stats_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.stats_text = tk.Text(stats_frame, height=4, width=80, wrap=tk.WORD)
        self.stats_text.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        # Frame de configuração da avaliação
        config_frame = ttk.LabelFrame(main_frame, text="Configuração da Avaliação", padding="10")
        config_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Dataset de teste
        ttk.Label(config_frame, text="Dataset de Teste:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.dataset_var = tk.StringVar(value="image/test_dataset")
        self.dataset_entry = ttk.Entry(config_frame, textvariable=self.dataset_var, width=50)
        self.dataset_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 5))
        
        browse_button = ttk.Button(config_frame, text="Procurar", command=self.browse_dataset)
        browse_button.grid(row=0, column=2, padx=(0, 10))
        
        # Checkbox para gerar relatório
        self.generate_report_var = tk.BooleanVar(value=True)
        report_check = ttk.Checkbutton(config_frame, text="Gerar relatório visual", 
                                     variable=self.generate_report_var)
        report_check.grid(row=0, column=3, padx=(10, 0))
        
        # Botões de ação
        action_frame = ttk.Frame(main_frame)
        action_frame.grid(row=3, column=0, columnspan=3, pady=(0, 10))
        
        self.evaluate_button = ttk.Button(action_frame, text="Executar Avaliação", 
                                        command=self.run_evaluation)
        self.evaluate_button.grid(row=0, column=0, padx=(0, 10))
        
        self.clear_db_button = ttk.Button(action_frame, text="Limpar Banco de Dados", 
                                        command=self.clear_database)
        self.clear_db_button.grid(row=0, column=1, padx=(0, 10))
        
        self.load_results_button = ttk.Button(action_frame, text="Carregar Resultados", 
                                            command=self.load_results)
        self.load_results_button.grid(row=0, column=2)
        
        # Notebook para abas
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        
        # Configurar expansão
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(4, weight=1)
        config_frame.columnconfigure(1, weight=1)
        
        # Aba de resultados
        self.create_results_tab()
        
        # Aba de métricas
        self.create_metrics_tab()
        
        # Aba de análise de revogação
        self.create_revocation_tab()
        
        # Barra de status
        self.status_var = tk.StringVar(value="Pronto")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        
    def create_results_tab(self):
        """Cria a aba de resultados"""
        results_frame = ttk.Frame(self.notebook)
        self.notebook.add(results_frame, text="Resultados")
        
        # Área de texto para resultados
        self.results_text = scrolledtext.ScrolledText(results_frame, wrap=tk.WORD, height=20)
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
    def create_metrics_tab(self):
        """Cria a aba de métricas"""
        metrics_frame = ttk.Frame(self.notebook)
        self.notebook.add(metrics_frame, text="Métricas")
        
        # Frame para métricas gerais
        general_frame = ttk.LabelFrame(metrics_frame, text="Métricas Gerais", padding="10")
        general_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Labels para métricas
        self.accuracy_var = tk.StringVar(value="N/A")
        self.precision_var = tk.StringVar(value="N/A")
        self.recall_var = tk.StringVar(value="N/A")
        self.f1_var = tk.StringVar(value="N/A")
        
        ttk.Label(general_frame, text="Acurácia:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        ttk.Label(general_frame, textvariable=self.accuracy_var, font=('Arial', 10, 'bold')).grid(row=0, column=1, sticky=tk.W)
        
        ttk.Label(general_frame, text="Precisão:").grid(row=0, column=2, sticky=tk.W, padx=(20, 10))
        ttk.Label(general_frame, textvariable=self.precision_var, font=('Arial', 10, 'bold')).grid(row=0, column=3, sticky=tk.W)
        
        ttk.Label(general_frame, text="Recall:").grid(row=1, column=0, sticky=tk.W, padx=(0, 10))
        ttk.Label(general_frame, textvariable=self.recall_var, font=('Arial', 10, 'bold')).grid(row=1, column=1, sticky=tk.W)
        
        ttk.Label(general_frame, text="F1-Score:").grid(row=1, column=2, sticky=tk.W, padx=(20, 10))
        ttk.Label(general_frame, textvariable=self.f1_var, font=('Arial', 10, 'bold')).grid(row=1, column=3, sticky=tk.W)
        
        # Frame para análise de confiança
        confidence_frame = ttk.LabelFrame(metrics_frame, text="Análise de Confiança", padding="10")
        confidence_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.avg_confidence_var = tk.StringVar(value="N/A")
        self.std_confidence_var = tk.StringVar(value="N/A")
        self.risk_score_var = tk.StringVar(value="N/A")
        
        ttk.Label(confidence_frame, text="Confiança Média:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        ttk.Label(confidence_frame, textvariable=self.avg_confidence_var, font=('Arial', 10, 'bold')).grid(row=0, column=1, sticky=tk.W)
        
        ttk.Label(confidence_frame, text="Desvio Padrão:").grid(row=0, column=2, sticky=tk.W, padx=(20, 10))
        ttk.Label(confidence_frame, textvariable=self.std_confidence_var, font=('Arial', 10, 'bold')).grid(row=0, column=3, sticky=tk.W)
        
        ttk.Label(confidence_frame, text="Score de Risco:").grid(row=1, column=0, sticky=tk.W, padx=(0, 10))
        ttk.Label(confidence_frame, textvariable=self.risk_score_var, font=('Arial', 10, 'bold')).grid(row=1, column=1, sticky=tk.W)
        
        # Área para gráficos
        self.metrics_canvas_frame = ttk.Frame(metrics_frame)
        self.metrics_canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
    def create_revocation_tab(self):
        """Cria a aba de análise de revogação"""
        revocation_frame = ttk.Frame(self.notebook)
        self.notebook.add(revocation_frame, text="Análise de Revogação")
        
        # Frame para estatísticas de risco
        risk_frame = ttk.LabelFrame(revocation_frame, text="Estatísticas de Risco de Revogação", padding="10")
        risk_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Treeview para estatísticas de risco
        columns = ('Risco', 'Quantidade', 'Acurácia', 'Confiança Média', 'Score de Risco')
        self.risk_tree = ttk.Treeview(risk_frame, columns=columns, show='headings', height=5)
        
        for col in columns:
            self.risk_tree.heading(col, text=col)
            self.risk_tree.column(col, width=120)
        
        self.risk_tree.pack(fill=tk.X, pady=(0, 10))
        
        # Scrollbar para treeview
        risk_scrollbar = ttk.Scrollbar(risk_frame, orient=tk.VERTICAL, command=self.risk_tree.yview)
        risk_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.risk_tree.configure(yscrollcommand=risk_scrollbar.set)
        
        # Frame para detalhes de revogação
        details_frame = ttk.LabelFrame(revocation_frame, text="Detalhes da Análise", padding="10")
        details_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.revocation_text = scrolledtext.ScrolledText(details_frame, wrap=tk.WORD, height=15)
        self.revocation_text.pack(fill=tk.BOTH, expand=True)
        
    def browse_dataset(self):
        """Abre diálogo para selecionar dataset"""
        dataset_path = filedialog.askdirectory(title="Selecionar Dataset de Teste")
        if dataset_path:
            self.dataset_var.set(dataset_path)
            
    def update_database_stats(self):
        """Atualiza estatísticas do banco de dados"""
        try:
            stats = chroma.get_database_stats()
            
            self.stats_text.delete(1.0, tk.END)
            self.stats_text.insert(tk.END, f"Total de imagens: {stats['total_images']}\n")
            self.stats_text.insert(tk.END, f"Última atualização: {stats.get('last_update', 'N/A')}\n\n")
            
            if stats['categories']:
                self.stats_text.insert(tk.END, "Categorias:\n")
                for cat, count in stats['categories'].items():
                    self.stats_text.insert(tk.END, f"• {cat}: {count} imagens\n")
            else:
                self.stats_text.insert(tk.END, "Nenhuma categoria encontrada\n")
                
        except Exception as e:
            self.stats_text.delete(1.0, tk.END)
            self.stats_text.insert(tk.END, f"Erro ao carregar estatísticas: {str(e)}")
            
    def run_evaluation(self):
        """Executa avaliação em thread separada"""
        if not os.path.exists(self.dataset_var.get()):
            messagebox.showerror("Erro", "Dataset de teste não encontrado!")
            return
            
        # Desabilitar botões durante execução
        self.evaluate_button.config(state='disabled')
        self.status_var.set("Executando avaliação...")
        
        # Executar em thread separada
        thread = threading.Thread(target=self._run_evaluation_thread)
        thread.daemon = True
        thread.start()
        
    def _run_evaluation_thread(self):
        """Executa avaliação em thread separada"""
        try:
            # Executar avaliação
            evaluation_result = self.evaluator.evaluate_system_performance(
                self.dataset_var.get(),
                None
            )
            
            if evaluation_result:
                self.evaluation_results = evaluation_result
                
                # Atualizar interface na thread principal
                self.after(0, self._update_results)
                self.after(0, lambda: self.status_var.set("Avaliação concluída com sucesso!"))
            else:
                self.after(0, lambda: messagebox.showerror("Erro", "Falha na avaliação do sistema"))
                self.after(0, lambda: self.status_var.set("Erro na avaliação"))
                
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Erro", f"Erro durante avaliação: {str(e)}"))
            self.after(0, lambda: self.status_var.set("Erro na avaliação"))
        finally:
            self.after(0, lambda: self.evaluate_button.config(state='normal'))
            
    def _update_results(self):
        """Atualiza interface com resultados"""
        if not self.evaluation_results:
            return
            
        metrics = self.evaluation_results["metrics"]
        test_results = self.evaluation_results["test_results"]
        
        # Atualizar métricas gerais
        self.accuracy_var.set(f"{metrics['overall_accuracy']:.3f}")
        self.precision_var.set(f"{metrics['precision']:.3f}")
        self.recall_var.set(f"{metrics['recall']:.3f}")
        self.f1_var.set(f"{metrics['f1_score']:.3f}")
        
        # Atualizar análise de confiança
        self.avg_confidence_var.set(f"{metrics['avg_confidence']:.1f}%")
        self.std_confidence_var.set(f"{metrics['std_confidence']:.1f}%")
        self.risk_score_var.set(f"{metrics['avg_risk_score']:.3f}")
        
        # Atualizar resultados detalhados
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "=== RESULTADOS DA AVALIAÇÃO ===\n\n")
        
        self.results_text.insert(tk.END, f"📊 MÉTRICAS GERAIS:\n")
        self.results_text.insert(tk.END, f"• Acurácia geral: {metrics['overall_accuracy']:.3f}\n")
        self.results_text.insert(tk.END, f"• Precisão: {metrics['precision']:.3f}\n")
        self.results_text.insert(tk.END, f"• Recall: {metrics['recall']:.3f}\n")
        self.results_text.insert(tk.END, f"• F1-Score: {metrics['f1_score']:.3f}\n\n")
        
        self.results_text.insert(tk.END, f"📈 ANÁLISE DE CONFIANÇA:\n")
        self.results_text.insert(tk.END, f"• Confiança média: {metrics['avg_confidence']:.1f}%\n")
        self.results_text.insert(tk.END, f"• Desvio padrão da confiança: {metrics['std_confidence']:.1f}%\n")
        self.results_text.insert(tk.END, f"• Score médio de risco: {metrics['avg_risk_score']:.3f}\n\n")
        
        # Análise de revogação
        revocation_analysis = self.evaluator.analyze_revocation_patterns(test_results)
        
        self.results_text.insert(tk.END, f"⚠️ ANÁLISE DE RISCO DE REVOGAÇÃO:\n")
        for risk_level, analysis in revocation_analysis.items():
            self.results_text.insert(tk.END, f"• Risco {risk_level}:\n")
            self.results_text.insert(tk.END, f"  - Quantidade: {analysis['count']} imagens\n")
            self.results_text.insert(tk.END, f"  - Acurácia: {analysis['accuracy']:.3f}\n")
            self.results_text.insert(tk.END, f"  - Confiança média: {analysis['avg_confidence']:.1f}%\n")
            self.results_text.insert(tk.END, f"  - Score de risco médio: {analysis['avg_risk_score']:.3f}\n\n")
        
        # Atualizar treeview de risco
        self.risk_tree.delete(*self.risk_tree.get_children())
        for risk_level, analysis in revocation_analysis.items():
            self.risk_tree.insert('', 'end', values=(
                risk_level,
                analysis['count'],
                f"{analysis['accuracy']:.3f}",
                f"{analysis['avg_confidence']:.1f}%",
                f"{analysis['avg_risk_score']:.3f}"
            ))
        
        # Atualizar análise de revogação
        self.revocation_text.delete(1.0, tk.END)
        self.revocation_text.insert(tk.END, "=== ANÁLISE DETALHADA DE REVOGAÇÃO ===\n\n")
        
        for risk_level, analysis in revocation_analysis.items():
            self.revocation_text.insert(tk.END, f"RISCO {risk_level}:\n")
            self.revocation_text.insert(tk.END, f"• Total de imagens: {analysis['count']}\n")
            self.revocation_text.insert(tk.END, f"• Predições corretas: {analysis['correct_predictions']}\n")
            self.revocation_text.insert(tk.END, f"• Predições incorretas: {analysis['incorrect_predictions']}\n")
            self.revocation_text.insert(tk.END, f"• Acurácia: {analysis['accuracy']:.3f}\n")
            self.revocation_text.insert(tk.END, f"• Confiança média: {analysis['avg_confidence']:.1f}%\n")
            self.revocation_text.insert(tk.END, f"• Score de risco médio: {analysis['avg_risk_score']:.3f}\n\n")
            
            # Recomendações baseadas no nível de risco
            if risk_level == "ALTO":
                self.revocation_text.insert(tk.END, "RECOMENDAÇÕES PARA RISCO ALTO:\n")
                self.revocation_text.insert(tk.END, "• Revisar qualidade das imagens de treinamento\n")
                self.revocation_text.insert(tk.END, "• Considerar adicionar mais exemplos da categoria\n")
                self.revocation_text.insert(tk.END, "• Verificar se há inconsistências no dataset\n")
                self.revocation_text.insert(tk.END, "• Implementar validação cruzada\n\n")
            elif risk_level == "MÉDIO":
                self.revocation_text.insert(tk.END, "RECOMENDAÇÕES PARA RISCO MÉDIO:\n")
                self.revocation_text.insert(tk.END, "• Melhorar qualidade das imagens\n")
                self.revocation_text.insert(tk.END, "• Aumentar diversidade do dataset\n")
                self.revocation_text.insert(tk.END, "• Considerar técnicas de data augmentation\n\n")
            else:
                self.revocation_text.insert(tk.END, "O sistema está funcionando bem para este nível de risco.\n\n")
        
        # Gerar gráficos se solicitado
        if self.generate_report_var.get():
            self._generate_plots(metrics)
            
    def _generate_plots(self, metrics):
        """Gera gráficos das métricas"""
        try:
            # Limpar frame anterior
            for widget in self.metrics_canvas_frame.winfo_children():
                widget.destroy()
            
            # Criar figura
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle('Métricas de Avaliação CBIR', fontsize=14, fontweight='bold')
            
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
            ax1.set_title('Métricas Gerais')
            ax1.set_ylabel('Score')
            ax1.set_ylim(0, 1)
            
            for bar, value in zip(bars, metrics_values):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
            
            # 2. Distribuição de confiança
            ax2 = axes[0, 1]
            conf_analysis = metrics['confidence_analysis']
            conf_levels = ['Alta (≥80%)', 'Média (60-80%)', 'Baixa (<60%)']
            conf_counts = [
                conf_analysis['high_confidence']['count'],
                conf_analysis['medium_confidence']['count'],
                conf_analysis['low_confidence']['count']
            ]
            
            colors = ['#28a745', '#ffc107', '#dc3545']
            ax2.pie(conf_counts, labels=conf_levels, autopct='%1.1f%%', colors=colors)
            ax2.set_title('Distribuição por Confiança')
            
            # 3. Matriz de confusão
            ax3 = axes[1, 0]
            cm = np.array(metrics['confusion_matrix'])
            im = ax3.imshow(cm, cmap='Blues', aspect='auto')
            ax3.set_title('Matriz de Confusão')
            ax3.set_xlabel('Predito')
            ax3.set_ylabel('Real')
            
            # Adicionar valores na matriz
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax3.text(j, i, str(cm[i, j]), ha='center', va='center')
            
            # 4. Score de risco
            ax4 = axes[1, 1]
            risk_score = metrics['avg_risk_score']
            ax4.bar(['Risco Médio'], [risk_score], color='#fd7e14')
            ax4.set_title('Score Médio de Risco')
            ax4.set_ylabel('Score')
            ax4.set_ylim(0, 1)
            ax4.text(0, risk_score + 0.01, f'{risk_score:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            
            # Criar canvas
            canvas = FigureCanvasTkAgg(fig, self.metrics_canvas_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao gerar gráficos: {str(e)}")
            
    def clear_database(self):
        """Limpa o banco de dados"""
        if messagebox.askyesno("Confirmar", "Tem certeza que deseja limpar o banco de dados?"):
            try:
                if chroma.clear_database():
                    messagebox.showinfo("Sucesso", "Banco de dados limpo com sucesso!")
                    self.update_database_stats()
                else:
                    messagebox.showerror("Erro", "Erro ao limpar banco de dados!")
            except Exception as e:
                messagebox.showerror("Erro", f"Erro ao limpar banco: {str(e)}")
                
    def load_results(self):
        """Carrega resultados salvos"""
        results_file = filedialog.askopenfilename(
            title="Carregar Resultados",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if results_file:
            try:
                import json
                with open(results_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                self.evaluation_results = {
                    "metrics": data["metrics"],
                    "test_results": data["test_results"]
                }
                
                self._update_results()
                messagebox.showinfo("Sucesso", "Resultados carregados com sucesso!")
                
            except Exception as e:
                messagebox.showerror("Erro", f"Erro ao carregar resultados: {str(e)}")

def main():
    """Função principal"""
    app = EvaluationGUI()
    app.mainloop()

if __name__ == "__main__":
    main() 