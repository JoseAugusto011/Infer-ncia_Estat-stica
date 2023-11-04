import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
import numpy as np
from scipy import stats
from scipy.stats import t
from scipy.stats import norm
import matplotlib.pyplot as plt
import pandas as pd
from math import sqrt

class HypothesisTestApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Teste de Hipótese")
        self.root.geometry("400x400")

        self.test_type = tk.StringVar()
        self.data = []
        self.alpha = 0.05  # Valor alpha padrão
        self.M = 42.6 # Valor média teste Z

        # Tela Inicial
        self.label = tk.Label(root, text="Escolha o teste:")
        self.label.pack()

        self.kstest_button = tk.Radiobutton(root, text="Teste Kolmogorov-Smirnov", variable=self.test_type, value="ks")
        self.kstest_button.pack()

        self.shapiro_button = tk.Radiobutton(root, text="Teste Shapiro-Wilk", variable=self.test_type, value="shapiro")
        self.shapiro_button.pack()
        
        self.Ztest_button = tk.Radiobutton(root, text="Teste Z", variable=self.test_type, value="Z")
        self.Ztest_button.pack()

        self.t_independent_button = tk.Radiobutton(root, text="Teste t-Student para Amostras Independentes", variable=self.test_type, value="t_independent")
        self.t_independent_button.pack()

        self.t_paired_button = tk.Radiobutton(root, text="Teste t-Student para Amostras Emparelhadas", variable=self.test_type, value="t_paired")
        self.t_paired_button.pack()
        
        

        self.alpha_label = tk.Label(root, text="Valor Alpha:")
        self.alpha_label.pack()
        self.alpha_entry = tk.Entry(root)
        self.alpha_entry.insert(0, str(self.alpha	))
        self.alpha_entry.pack()
        
        self.M_Label = tk.Label(root, text="Valor Média teste Z:")
        self.M_Label.pack()
        self.M_Entry = tk.Entry(root)
        self.M_Entry.insert(0, str(self.M))        
        self.M_Entry.pack()
        
        
        
        print("VALOR-->",self.M)

        self.select_button = tk.Button(root, text="Selecionar Dados", command=self.select_file)
        self.select_button.pack()

        self.generate_button = tk.Button(root, text="Gerar Aleatoriamente", command=self.generate_data)
        self.generate_button.pack()

        self.result_text = tk.Text(root, height=6, width=40)
        self.result_text.pack()

        self.plot_button = tk.Button(root, text="Plotar Distribuição", command=self.plot_distribution)
        self.plot_button.pack()
        
        self.plot_t_distribution_button = tk.Button(root, text="Plotar Distribuição t-Student", command=self.plot_t_distribution)
        self.plot_t_distribution_button.pack()

    def select_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            df = pd.read_csv(file_path, header=None)
            if len(df)-1 == 2:
                self.data = df.iloc[1:].values
            else:
                self.data = [df.iloc[0].values]
            self.perform_test()

    def generate_data(self):
        try:
            n = int(simpledialog.askstring("Tamanho do Rol", "Digite o tamanho do rol de dados:"))
            if n <= 0:
                raise ValueError("O tamanho do rol deve ser um número positivo.")
        except (ValueError, TypeError):
            messagebox.showerror("Erro", "Tamanho do rol inválido.")
            return

        try:
            mean = float(simpledialog.askstring("Média", "Digite a média:"))
            std_dev = float(simpledialog.askstring("Desvio Padrão", "Digite o desvio padrão:"))
            self.alpha = float(self.alpha_entry.get())
            self.M = float(self.M_Entry.get())  # Atualiza o valor de self.M
            if self.alpha <= 0 or self.alpha >= 1:
                raise ValueError("O valor Alpha deve estar entre 0 e 1.")
        except (ValueError, TypeError):
            messagebox.showerror("Erro", "Valores inválidos.")
            return

        if self.test_type.get() == "ks":
            self.data = [np.random.normal(loc=mean, scale=std_dev, size=n)]
            self.perform_test()
        elif self.test_type.get() == "shapiro":
            self.data = [np.random.normal(loc=mean, scale=std_dev, size=n)]
            self.perform_test()
        elif self.test_type.get() == "t_independent":
            sample1 = np.random.normal(loc=mean, scale=std_dev, size=n)
            sample2 = np.random.normal(loc=mean, scale=std_dev, size=n)
            self.data = [sample1, sample2]
            self.perform_test()
        elif self.test_type.get() == "t_paired":
            sample1 = np.random.normal(loc=mean, scale=std_dev, size=n)
            sample2 = sample1 + np.random.normal(loc=0, scale=std_dev / 2, size=n)
            self.data = [sample1, sample2]
            self.perform_test()
        elif self.test_type.get() == "SF":
            self.data = [np.random.normal(loc=mean, scale=std_dev, size=n)]
            self.perform_test()
        elif self.test_type.get() == "Z":
            self.data = [np.random.normal(loc=mean, scale=std_dev, size=n)]
            self.perform_test()
            
            

    def perform_test(self):
        interpretation = ""  # Inicializa com uma string vazia
        test_result = ""  # Inicializa com uma string vazia

        if self.test_type.get() == "ks":
            #statistic, p_value = stats.kstest(self.data[0], 'norm')
            statistic, p_value = stats.shapiro(self.data[0])
            if p_value > self.alpha:
                test_result = "H0 aceita"
                interpretation = "p-valor > alpha:\nNão há evidência estatística para rejeitar a hipótese nula\n\nH0 = distribuição dos dados segue uma normal"
            else:
                test_result = "H1 aceita"
                interpretation = "p-valor <= alpha:\nHá evidência estatística para rejeitar a hipótese nula\n\nH1 = distribuição não segue uma normal"
                
                
                
        elif self.test_type.get() == "shapiro":
            
            if len(self.data[0]) < 3:
                test_result = "Teste inválido"
                interpretation = "O teste Shapiro-Wilk requer pelo menos 3 observações"
                p_value = None
            else:
                statistic, p_value = stats.shapiro(self.data[0])
                if p_value > self.alpha:
                    test_result = "H0 aceita"
                    interpretation = "p-valor > alpha:\nNão há evidência estatística para rejeitar a hipótese nula\n\nH0 = distribuição dos dados segue uma normal"
                else:
                    test_result = "H1 aceita"
                    interpretation = "p-valor <= alpha:\nHá evidência estatística para rejeitar a hipótese nula\n\nH1 = distribuição não segue uma normal"
        
        
        elif self.test_type.get() in ["t_independent", "t_paired"]:
            
            if len(self.data) < 2:
                test_result = "Teste inválido"
                interpretation = "O teste t-Student requer duas amostras de dados."
                p_value = None
                
            else:
                
                if self.test_type.get() == "t_independent":
                    
                    test_type_label = "t-Student para amostras independentes"
                    n1,n2 = len(self.data[0]),len(self.data[1])  
                    var1,var2 = self.data[0].var(),self.data[1].var()
                    mi1,mi2 = self.data[0].mean(),self.data[1].mean()
                    
                    if var1 != var2:
                        print("Caso 1 --> Sigma²1 != Sigma²2\n")
                        
                        Tcalc = (mi1 - mi2 )/((var1/n1)+(var2/n2))**2  # Estatistica T        
                        V = ((var1/n1   +  var2/n2)**2)/   ( ((var1/n1)**2)/(n1-1)  +   ((var2/n2)**2)/(n2-1))  # Graus de liberdade        
                        Tc = t.ppf(1-self.alpha/2,V) # Tabelado
                        
                    else:
                        print("Caso 2 --> Sigma²1 == Sigma²2\n")
                        
                        Sp = sqrt(((n1-1)*var1 + (n2-1)*var2)/n1+n2-2)
                        Tcalc = (mi1-mi2)/(Sp*(1/n1 + 1/n2)**1/2)        
                        Tc = t.ppf(1-self.alpha/2,n1+n2-2)
        
                   
                   
                   
                   
                elif self.test_type.get() == "t_paired":
                    
                    
                    test_type_label = "t-student para pareadas"
                    n1,n2 = len(self.data[0]),len(self.data[1])  
                    var1,var2 = self.data[0].var(),self.data[1].var()
                    mi1,mi2 = self.data[0].mean(),self.data[1].mean()
                    
                    if var1 != var2:
                        print("Caso 1 --> Sigma²1 != Sigma²2\n")
                        
                        Tcalc = (mi1 - mi2 )/((var1/n1)+(var2/n2))**2  # Estatistica T        
                        V = ((var1/n1   +  var2/n2)**2)/   ( ((var1/n1)**2)/(n1-1)  +   ((var2/n2)**2)/(n2-1))  # Graus de liberdade        
                        Tc = t.ppf(1-self.alpha/2,V) # Tabelado
                        
                        
                    else:
                        
                        print("Caso 2 --> Sigma²1 == Sigma²2\n")
        
                        Sp = sqrt(((n1-1)*var1 + (n2-1)*var2)/n1+n2-2)
                        Tcalc = (mi1-mi2)/(Sp*(1/n1 + 1/n2)**1/2)        
                        Tc = t.ppf(1-self.alpha/2,n1+n2-2)
                        
                        
                if Tcalc < Tc:
                    test_result = "H0 aceita"
                    interpretation = f"p-valor > alpha:\nNão há evidência estatística para rejeitar a hipótese nula\n"
                    interpretation += f"H0 = as médias das duas amostras {test_type_label} são iguais"
                else:
                    test_result = "H1 aceita"
                    interpretation = f"p-valor <= alpha:\nHá evidência estatística para rejeitar a hipótese nula\n"
                    interpretation += f"H1 = as médias das duas amostras {test_type_label} são diferentes"

        elif self.test_type.get() == "Z":
            
            
            print(self.M,type(self.M))
            sigmaXb = np.std(self.data[0], ddof=1)/np.sqrt(len(self.data[0]))
            zCalc = (np.mean(self.data[0])-self.M)/sigmaXb

            """
            Os valores críticos da estatística (zc) são apresentados na Tabela E do apêndice.
            Essa tabela fornece os valores críticos de zc tal que P(𝑍𝑐𝑎𝑙𝑐 > zc) = α
            (para um teste unilateral à direita).
            """

            zc = t.ppf((1-self.alpha)/2, len(self.data[0])-1)
            
            
            
            

            if abs(zCalc) > zc:
                test_result = "H1 aceita"
                interpretation += "O valor calculado do Z está na região de rejeição\n\nH1 = A média da amostra é diferente da média populacional especificada na hipótese nula"
            else:
                
                test_result = "H0 aceita"
                interpretation += "O valor calculado do Z está na região de aceitação\n\nH0 = A média da amostra é igual à média populacional especificada na hipótese nula"
                

        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, f"Resultado do Teste: {test_result}\n\n")
        self.result_text.insert(tk.END, interpretation)

    def plot_distribution(self):
        if len(self.data) == 1:
            plt.hist(self.data[0], bins=20, color='blue', edgecolor='black')
            plt.title('Distribuição de Dados')
            plt.xlabel('Valores')
            plt.ylabel('Frequência')
            plt.show()
        elif len(self.data) >= 2:
            plt.hist(self.data[0], bins=20, color='blue', alpha=0.5, edgecolor='black', label='Amostra 1')
            plt.hist(self.data[1], bins=20, color='red', alpha=0.5, edgecolor='black', label='Amostra 2')
            plt.title('Distribuição de Dados para Amostras Independentes')
            plt.xlabel('Valores')
            plt.ylabel('Frequência')
            plt.legend()
            plt.show()

    def plot_t_distribution(self):
        if len(self.data) >= 2:
            plt.hist(self.data[0], bins=20, color='blue', alpha=0.5, edgecolor='black', label='Amostra 1')
            plt.hist(self.data[1], bins=20, color='red', alpha=0.5, edgecolor='black', label='Amostra 2')
            plt.title('Distribuição de Dados para Amostras Independentes')
            plt.xlabel('Valores')
            plt.ylabel('Frequência')
            plt.legend()
            plt.show()

if __name__ == "__main__":
    root = tk.Tk()
    app = HypothesisTestApp(root)
    root.mainloop()
