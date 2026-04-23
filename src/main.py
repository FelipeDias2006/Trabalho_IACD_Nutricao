import pandas as pd
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


def limpar_e_processar(df):
    # 1. Verificar valores nulos
    print("Valores em falta antes:\n", df.isnull().sum())
    df = df.dropna()  # Opção simples: remover linhas com buracos

    # 2. Remover colunas redundantes (exemplo)
    # Se o merge criou colunas duplicadas como 'id_paciente_x' e 'id_paciente_y'
    # df = df.drop(columns=['coluna_desnecessaria'])

    # 3. Tratar Outliers (Exemplo: Idades negativas ou acima de 120)
    # df = df[(df['idade'] > 0) & (df['idade'] < 120)]

    print("Limpeza concluída!")
    return df


if __name__ == "__main__":
    try:
        # Passo 1: Carregar
        p, d, n, r = carregar_dados()

        # Passo 2: Integrar
        dataset_completo = integrar_dados(p, d, n, r)

        # Passo 3: Ver as primeiras linhas para confirmar


        dados_finais = limpar_e_processar(dataset_completo)

        # 4. Verificar o resultado final da limpeza
        print("--- Primeiras 5 linhas do dataset limpo ---")
        print(dados_finais.head())

        # Guardar o ficheiro unificado para a próxima fase (opcional)
        # dataset_completo.to_csv('../data/dataset_unificado.csv', index=False)

    except FileNotFoundError as e:
        print(f"Erro: Não foi possível encontrar os ficheiros. Verifique o caminho. {e}")


