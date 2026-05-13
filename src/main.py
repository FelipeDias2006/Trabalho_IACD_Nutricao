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


def consistencia_rep(df: pd.DataFrame) -> pd.DataFrame:
    if "sex" in df.columns:
        # 1. Limpeza básica (o que já tinhas)
        df["sex"] = df["sex"].astype(str).str.strip().str.capitalize()

        # 2. Mapeamento para uniformizar (A SOLUÇÃO)
        mapeamento = {
            'Female': 'F',
            'Male': 'M',
        }

        # O 'replace' substitui os valores longos pelos curtos
        df["sex"] = df["sex"].replace(mapeamento)

        print("\nValores únicos de 'sex' após uniformização:")
        print(df["sex"].unique())

    return df


