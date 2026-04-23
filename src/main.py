import pandas as pd
import matplotlib.pyplot as plt
import os


# Função para carregar os dados
def carregar_dados():

    path_pacientes = '../data/patients.csv'
    path_dieta = '../data/diets.csv'
    path_nutricionistas = '../data/nutritionists.csv'
    path_resultados = '../data/outcomes.csv'

    # Carregar cada ficheiro
    df_pacientes = pd.read_csv(path_pacientes)
    df_dieta = pd.read_csv(path_dieta)
    df_nutricionistas = pd.read_csv(path_nutricionistas)
    df_resultados = pd.read_csv(path_resultados)

    return df_pacientes, df_dieta, df_nutricionistas, df_resultados

def valores_em_falta(df: pd.DataFrame) -> pd.DataFrame:

    print(df.isnull().sum())

    return df




def integrar_dados(df_pac, df_dieta, df_nut, df_res):
    # 1. Unir Pacientes com Resultados
    # O 'how=inner' garante que só mantemos quem tem registo em ambos
    m1 = pd.merge(df_pac, df_res, on='patient_id')

    # 2. Unir com dieta
    m2 = pd.merge(m1, df_dieta, on='diet_id')

    # 3. Unir com Nutricionistas
    df_final = pd.merge(m2, df_nut, on='nutritionist_id')

    print(f"Integração concluída! O dataset final tem {df_final.shape[0]} linhas e {df_final.shape[1]} colunas.")
    return df_final


if __name__ == "__main__":
    try:
        print("valores em falta no CSV: patients")
        df_pacientes = pd.read_csv("../data/patients.csv")
        df_limpo_pac = valores_em_falta(df_pacientes)

        print("valores em falta no CSV: nutritionists")
        df_nutricionistas = pd.read_csv('../data/nutritionists.csv')
        df_limpo_nutri = valores_em_falta(df_nutricionistas)

        print("Valores em falta no CSV: diets")
        df_dietas = pd.read_csv('../data/diets.csv')
        df_limpo_dieta = valores_em_falta(df_dietas)

        print("valores em falta no CSV: outcomes")
        df_resultado = pd.read_csv('../data/outcomes.csv')
        df_limpo_resultados = valores_em_falta(df_resultado)


        df_count = df_resultado['patient_id'].value_counts()
        print(df_count)


        p, d, n, r = carregar_dados()
        # Passo 2: Integrar
        dataset_completo = integrar_dados(p, d, n, r)


    except FileNotFoundError as e:
        print(f"Erro: Não foi possível encontrar os ficheiros. Verifique o caminho. {e}")


