#esse ficheiro trata-se do clustering
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

# essa função serve apenaas paraa indicar a quantidade de grupos q devem ser criados para divisão
def descobrir_n_de_grupos(X: pd.DataFrame):
    print("Encontrando numero de grupos para a divisão...")

    inercia = [] #Número de "tensao" nos grupos ja criados

    #testar cada quantidade de grupos para divisão dos dados
    for k in range(1,11): # testar de 1 a 10 grupos

        #funcao que agrupaa os dados
        #k sera o numero de grupos que a funcao vai criar para agrupar os dados
        modelo_kmeans = KMeans(n_clusters = k, random_state = 42)

        #a funcao estuda a tabela x
        modelo_kmeans.fit(X)

        #ele adiciona a "tensao" registrada nos grupos, quanto mais grupos tiver, menos "tensao" vai ter
        inercia.append(modelo_kmeans.inertia_)

    # --- PARTE DO GRÁFICO ---
    # Codigo para a tabela do matplotlib
    plt.figure(figsize=(8, 5))
    plt.plot(range(1,11), inercia, marker='o', linestyle='--', color='b')
    plt.title('Analise para Descobrir Grupos')
    plt.xlabel('Número de Grupos (K)')
    plt.ylabel('Inércia (Tensão interna)')
    plt.grid(True)
    plt.show()  # Isto vai fazer saltar uma janela com o gráfico no seu ecrã!

#essa função vai servir para formar os grupos
#ele vai receber a tabela de dados limpa e o numero de grupos q deve criar
def agrupar_grupos(X: pd.DataFrame, num_grupos: int):
    print(f"\nA aplicar o algoritmo para criar {num_grupos} grupos...")

    #modelo que vai agrupar os dados
    modelo_final = KMeans(n_clusters = num_grupos, random_state=42)

    #formação dos grupos, ele vai criaar uma lista com uma sequência de numeros
    # cada numero representa a etiqueta de um grupo (grupo 0, grupo 1,..., grupo k)
    etiquetas = modelo_final.fit_predict(X)

    # faz uma copia de segurança da tabelaa de dados, para n estragar a original
    X_com_grupos = X.copy()
    #agora criamos uma coluna nova (Grupos) na tabela copiada
    #essa nova coluna terá as etiquetas criadas anteriormente para cada paciente
    X_com_grupos['Grupos'] = etiquetas

    #de acordo com a etiquetagem o groupby vai reunir os grupos
    #e o mean vai fazer a media de cada grupo
    perfil_grupos = X_com_grupos.groupby('Grupos').mean()

    print("Agrupamento concluído! A análise dos grupos está pronta.")

    # devolvemos a tabela com as etiquetas e a tabela de médias
    return X_com_grupos, perfil_grupos


#----BLOCO DE TESTE---
#BLOCO DE TESTE FEITA POR IA PARA GERAR DADOS FALSOS E TESTARMOS O MODELO PARAC INDENTIFICAÇÕES DE PADRÕES
if __name__ == "__main__":
    print("--- INICIANDO O TESTE DE FOGO COM DADOS REAIS (Pacientes) ---")

    # 1. Ler o ficheiro real (garanta que o ficheiro patients.csv está na mesma pasta,
    # ou ajuste o caminho para '../data/patients.csv' consoante a sua estrutura)
    try:
        df_pacientes = pd.read_csv('patients.csv')

        # 2. O truque para o K-Means não dar erro com dados sujos:
        # Vamos selecionar apenas 3 colunas que sabemos que são números
        colunas_para_testar = ['age', 'baseline_weight_kg', 'baseline_bmi']

        # O .dropna() simplesmente apaga qualquer linha que tenha um valor vazio (NaN)
        # Atenção: Fazemos isto SÓ para testar. Na versão final, os seus colegas já terão tratado isto!
        df_teste = df_pacientes[colunas_para_testar].dropna()

        df_teste = df_teste[df_teste['baseline_weight_kg'] < 150]
        print(f"Vamos testar com {len(df_teste)} pacientes limpos de texto e NaNs.")

        # 3. Chamar o gráfico do cotovelo
        # Feche o gráfico quando ele abrir para o código continuar
        descobrir_n_de_grupos(df_teste)

        # 4. Escolher o número de grupos
        # Assuma o valor que achar melhor ao olhar para o gráfico (vou usar 3 como exemplo)
        numero_escolhido = 3
        tabela_completa, medias_das_tribos = agrupar_grupos(df_teste, numero_escolhido)

        # 5. Imprimir o relatório
        print("\n--- RELATÓRIO DAS TRIBOS REAIS ---")
        print("Perfil médio de cada grupo encontrado:")
        print(medias_das_tribos)

    except FileNotFoundError:
        print("Erro: Não encontrei o ficheiro patients.csv. Verifique onde ele está guardado!")



