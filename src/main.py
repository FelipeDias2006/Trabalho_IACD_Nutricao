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


def outliers_iqr(df: pd.DataFrame):
    print("\n" + "=" * 55)
    print("Deteção de outliers — Método IQR")
    print("=" * 55)

    # Escolha uma coluna numérica real, como 'weight_loss' ou 'age'
    col = "baseline_weight_kg"

    # Correção da lógica Q3 - Q1
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1  # O correto é Q3 menos Q1

    lim_inf = Q1 - 1.5 * IQR
    lim_sup = Q3 + 1.5 * IQR

    # Filtragem de outliers
    outliers = df[(df[col] < lim_inf) | (df[col] > lim_sup)]

    print(f"O limete inferior é:{lim_inf}")
    print(f"O limete superiror é:{lim_sup}")

    print(f"\nOutliers detetados na coluna '{col}':")

    if not outliers.empty:
        print(outliers[["patient_id", col]].to_string(index=False))
    else:
        print("Nenhum outlier detetado.")

    return col, lim_inf, lim_sup, outliers


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
        print("=" * 55)
        print("valores em falta no CSV: patients")
        df_pacientes = pd.read_csv("../data/patients.csv")
        df_limpo_pac = valores_em_falta(df_pacientes)

        print("=" * 55)
        print("valores em falta no CSV: nutritionists")
        df_nutricionistas = pd.read_csv('../data/nutritionists.csv')
        df_limpo_nutri = valores_em_falta(df_nutricionistas)

        print("=" * 55)
        print("Valores em falta no CSV: diets")
        df_dietas = pd.read_csv('../data/diets.csv')
        df_limpo_dieta = valores_em_falta(df_dietas)

        print("=" * 55)
        print("valores em falta no CSV: outcomes")
        df_resultado = pd.read_csv('../data/outcomes.csv')
        df_limpo_resultados = valores_em_falta(df_resultado)


        print("="*55)
        print("Valores repetidos:")
        df_count = df_resultado['program_id'].value_counts()
        print(df_count)

        print("=" * 55)
        print("Valores repetidos:")
        df_count = df_pacientes['patient_id'].value_counts()
        print(df_count)


        p, d, n, r = carregar_dados()

        dataset_completo = integrar_dados(p, d, n, r)
        dataset_completo = consistencia_rep(dataset_completo)

        col, lim_inf, lim_sup, outliers = outliers_iqr(dataset_completo)

        dataset_final = dataset_completo[(dataset_completo[col] >= lim_inf) & (dataset_completo[col] <= lim_sup)]


        print(f"Limpeza de outliers na coluna '{col}' concluída!")
        print(f"Linhas antes: {len(dataset_completo)} | Linhas depois: {len(dataset_final)}")

    except FileNotFoundError as e:
        print(f"Erro: Não foi possível encontrar os ficheiros. Verifique o caminho. {e}")


