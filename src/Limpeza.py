import pandas as pd
import matplotlib.pyplot as plt
from main import carregar_dados,consistencia_rep

def dietsCSV_limpeza(df: pd.DataFrame) -> pd.DataFrame:
    # Ver NaN antes de tratar

    print("DIETAS:")
    print("Valores em falta dietsCSV:")
    print(df.isnull().sum())
    print("="*50)
    # Verificar relação entre diet_type e as colunas com NaN
    print(df[['diet_type', 'sodium_limit_mg', 'fiber_target_g']])
    print("=" * 50)
    # Imputar pela média do mesmo tipo de dieta
    # fiber_target_g: valores entre 25-38g conforme literatura (25-35g recomendado).
    # O valor em falta é low_carb foi imputado pela média dos outros low_carb
    # para respeitar o contexto do tipo de dieta.
    df['sodium_limit_mg'] = df.groupby('diet_type')['sodium_limit_mg'].transform(lambda x: x.fillna(x.mean()))
    df['fiber_target_g'] = df.groupby('diet_type')['fiber_target_g'].transform(lambda x: x.fillna(x.mean()))

    # Confirmar que ficou limpo
    print("Valores em falta dietCSV (ATUALIZADO)")
    print(df.isnull().sum())

def nutritionistsCSV_limpeza(df: pd.DataFrame) -> pd.DataFrame:
    #Apagar coluna duplicada e corrigir valor 99 para 9.
    #Dado que existe reforma a probabilidade de uma pessoa que tenha trabalhado trabalhado mais de 90 anos e tenha vivido e mais pouco provavel do que a pessoa ter se enganado
    df = df.drop(columns=['years_experience'])
    df.loc[df['nutritionist_id'] == 'N008', 'experience_years'] = 9

    #Ver Nan antes de tratar

    print("NUTRICIONISTAS:")
    print("Valores em falta nutritionistsCSV")
    print(df.isnull().sum())
    print("=" * 50)

    # Criar colunas binárias para investigar padrão dos NaN
    df['approach_falta'] = df['approach'].isnull().astype(int)
    df['years_falta'] = df['experience_years'].isnull().astype(int)
    df['specialty_num'] = df['specialty'].astype('category').cat.codes
    df['approach_num'] = df['approach'].astype('category').cat.codes

    # Calcular correlação
    correlacao = df.select_dtypes(include='number').corr()

    # Desenhar heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(correlacao, cmap='coolwarm', aspect='auto')
    plt.colorbar()
    plt.xticks(range(len(correlacao.columns)), correlacao.columns, rotation=45)
    plt.yticks(range(len(correlacao.columns)), correlacao.columns)
    plt.title('Correlação — Nutritionists')
    plt.tight_layout()
    plt.show()

    # Imputação experience_years — MCAR
    # Mediana porque é variável numérica e é robusta a outliers.
    # A correlação neutra no heatmap entre years_falta e approach_num confirma
    # que o tipo de approach não influencia se os anos estão em falta.
    # Provável falha na recolha de dados sem relação com o valor em si.
    df['experience_years'] = df['experience_years'].fillna(df['experience_years'].median())

    # Imputação approach — MNAR
    # Criamos categoria 'unknown' em vez de imputar com moda.
    # A abordagem de um nutricionista é influenciada pela emoção humana
    # pode ser demasiado complexa para se identificar numa única palavra.
    # Inventar um approach distorceria a realidade — unknown é mais honesto.
    df['approach'] = df['approach'].fillna('unknown')

    # Apagar colunas auxiliares criadas para análise pois não fazem parte dos dados originais
    df = df.drop(columns=['approach_falta', 'years_falta', 'specialty_num', 'approach_num'])

    # Confirmar que ficou limpo
    print("Valores em falta nutritionistsCSV (ATUALIZADO)")
    print(df.isnull().sum())
    print("=" * 50)
    print(df)

    return df

def patientsCSV_limpeza(df: pd.DataFrame) -> pd.DataFrame:
        # Ver NaN antes de tratar
        print("PACIENTES:")
        print("Valores em falta patientsCSV")
        print(df.isnull().sum())
        print("=" * 50)
        print(df.describe())
        print("=" * 50)

        # Apagar coluna duplicada — bmi_redundant é igual a baseline_bmi
        df = df.drop(columns=['bmi_redundant'])

        # CORRIGIR ERROS DE DIGITAÇÃO ÓBVIOS

        # baseline_weight_kg: P0154=5kg e P0181=500kg → impossíveis
        mediana_peso = df[(df['baseline_weight_kg'] > 10) & (df['baseline_weight_kg'] < 300)][
            'baseline_weight_kg'].median()
        df.loc[df['baseline_weight_kg'] < 10, 'baseline_weight_kg'] = mediana_peso
        df.loc[df['baseline_weight_kg'] > 300, 'baseline_weight_kg'] = mediana_peso

        # height_cm: P0278=300cm e P0755=10cm → impossíveis
        mediana_altura = df[(df['height_cm'] > 50) & (df['height_cm'] < 250)]['height_cm'].median()
        df.loc[df['height_cm'] > 250, 'height_cm'] = mediana_altura
        df.loc[df['height_cm'] < 50, 'height_cm'] = mediana_altura

        # Recalcular baseline_bmi após corrigir erros de peso e altura
        df['baseline_bmi'] = df['baseline_weight_kg'] / ((df['height_cm'] / 100) ** 2)

        # sleep_hours: P0670=30h (>24 impossível) e P0790=-1h (negativo impossível)
        mediana_sono = df[(df['sleep_hours'] >= 0) & (df['sleep_hours'] <= 24)]['sleep_hours'].median()
        df.loc[df['sleep_hours'] > 24, 'sleep_hours'] = mediana_sono
        df.loc[df['sleep_hours'] < 0, 'sleep_hours'] = mediana_sono

        # motivation_score: P0821 e P0827=5.0 → fora da escala 0-1
        mediana_motiv = df[df['motivation_score'] <= 1]['motivation_score'].median()
        df.loc[df['motivation_score'] > 1, 'motivation_score'] = mediana_motiv

        # age: min=0 → impossível num estudo de nutrição de adultos
        mediana_idade = df[df['age'] > 0]['age'].median()
        df.loc[df['age'] == 0, 'age'] = mediana_idade

        # DETEÇÃO DE OUTLIERS VIA IQR (para mostrar o que sobra)

        for col in ['age', 'height_cm', 'baseline_weight_kg', 'baseline_bmi', 'sleep_hours', 'motivation_score']:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
            if not outliers.empty:

                print(f"\nOutliers restantes em {col}:")
                print(outliers[['patient_id', col]])

        # --- ANÁLISE DE CORRELAÇÃO DOS NaNs ---
        df['age_falta'] = df['age'].isnull().astype(int)
        df['sleep_falta'] = df['sleep_hours'].isnull().astype(int)
        df['motivation_falta'] = df['motivation_score'].isnull().astype(int)

        correlacao = df.select_dtypes(include='number').corr()
        plt.figure(figsize=(10, 8))
        plt.imshow(correlacao, cmap='coolwarm', aspect='auto')
        plt.colorbar()
        plt.xticks(range(len(correlacao.columns)), correlacao.columns, rotation=45)
        plt.yticks(range(len(correlacao.columns)), correlacao.columns)
        plt.title('Correlação — Patients')
        plt.tight_layout()
        plt.show()

        #IMPUTAÇÃO DOS NaN REAIS
        # age — MCAR → mediana
        df['age'] = df['age'].fillna(df['age'].median())

        # sleep_hours — MCAR → mediana
        df['sleep_hours'] = df['sleep_hours'].fillna(df['sleep_hours'].median())

        # motivation_score — MNAR (baixa motivação → não preenchem) → mediana
        df['motivation_score'] = df['motivation_score'].fillna(df['motivation_score'].median())

        # Apagar colunas auxiliares
        df = df.drop(columns=['age_falta', 'sleep_falta', 'motivation_falta'])
        print("=" * 50)
        print("Valores em falta patientsCSV (ATUALIZADO)")
        print(df.isnull().sum())
        print("=" * 50)
        print(df)
        return df
def outcomesCSV_limpeza(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    print(df.isnull().sum())
    print(df.describe())

    #Verificar linhas totalmente duplicadas
    print("Linhas Duplicadas:", df.duplicated().sum())

    #Eliminar mesmas linhas
    df = df.drop_duplicates()
    print("Linhas duplicadas após limpeza:", df.duplicated().sum())

    # Verificar se adherence_ratio == mean_adherence_pct / 100
    # Se as colunas forem iguais, a diferença deve ser ~0 em todas as linhas
    df['verificacao'] = (df['adherence_ratio'] - df['mean_adherence_pct'] / 100).abs()
    print(df['verificacao'].describe())
    df = df.drop(columns=['verificacao'])

    # adherence_ratio == mean_adherence_pct / 100 — colunas redundantes, confirmado matematicamente
    df = df.drop(columns=['adherence_ratio'])

    return df

df_pacientes, df_dieta, df_nutricionistas, df_resultados = carregar_dados()
df_resultados = outcomesCSV_limpeza(df_resultados)

df_pacientes, df_dieta, df_nutricionistas, df_resultados = carregar_dados()

df_pacientes = patientsCSV_limpeza(df_pacientes)
print("="*50)
print("="*50)
df_dieta = dietsCSV_limpeza(df_dieta)
print("="*50)
print("="*50)
df_nutricionistas = nutritionistsCSV_limpeza(df_nutricionistas)
