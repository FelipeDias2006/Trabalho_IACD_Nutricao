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

    # criar o visual do grafico
    # Codigo para a tabela do matplotlib
    plt.figure(figsize=(8, 5))
    plt.plot(range(1,11), inercia, marker='o', linestyle='--', color='b')
    plt.title('Analise para Descobrir Grupos')
    plt.xlabel('Número de Grupos (K)')
    plt.ylabel('Inércia (Tensão interna)')
    plt.grid(True)
    plt.show()

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
if __name__ == "__main__":
    from sklearn.preprocessing import StandardScaler

    print("Iniciando a descoberta de padrões...")

    try:
        # carregando a tabela inteira e limpa
        df_completo = pd.read_csv('dataset_limpo.csv')

        #tensão dos dados
        descobrir_n_de_grupos(df_completo)

        # selecionando as colunas que vamos usar para comparar os clientes (fisionomia)
        # o K-Means vai usar apenas isto para medir quem é parecido com quem
        colunas_para_agrupar = [
            'age', 'baseline_weight_kg', 'height_cm',
            'sleep_hours', 'mean_adherence_pct', 'motivation_score'
        ]
        df_para_estudar = df_completo[colunas_para_agrupar].copy()

        # normalizando as variaveis numa escala de -1 a 1
        escalador = StandardScaler()
        dados_escalados = escalador.fit_transform(df_para_estudar)

        # indentificar as "tribos" que serão crianas
        # numero de grupos q sserá criado
        numero_escolhido = 4
        modelo = KMeans(n_clusters=numero_escolhido, random_state=42)
        etiquetas_finais = modelo.fit_predict(dados_escalados)

        # indetificar com a etiqueta
        df_completo['Tribo'] = etiquetas_finais

        # calculando a media das colunas
        relatorio_final = df_completo.groupby('Tribo').mean()

        # exportando a tabela para umm ficheiro csv, para n termos problema com a leitura no terminal
        nome_ficheiro = 'relatorio_tribos.csv'
        relatorio_final.to_csv(nome_ficheiro)
        print(f"\n[SUCESSO] A tabela foi guardada no ficheiro '{nome_ficheiro}'.")

        # printar uma pesquena parte da tabela, para ver se funcionou
        print("\nUma pequena amostra das Tribos encontradas (Idade e Peso):")
        print(relatorio_final[['age', 'baseline_weight_kg']].round(2))

    except FileNotFoundError:
        print("Erro: Não encontrei o ficheiro 'dataset_limpo.csv'.")