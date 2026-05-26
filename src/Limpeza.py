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

    #necesario fazer uma copia da tabela pois a biblioteca panda nao deixa apagar linhas na tabela principal.
    df = df.copy()

    print(df.isnull().sum())
    print(df.describe())

    #Verificar linhas totalmente duplicadas
    print("Linhas Duplicadas:", df.duplicated().sum())

    #Eliminar mesmas linhas
    df = df.drop_duplicates()
    print("Linhas duplicadas após limpeza:", df.duplicated().sum())

    #Notei num dado que adherence_ratio poderia ser a mesma coisa que mean_adherence se dividisse por 100
    # Verificar se adherence_ratio == mean_adherence_pct / 100
    # Se as colunas forem iguais, a diferença deveser ~0 em todas as linhas pois devido aos valores serem discretos no python ele tende a deixar um resto.
    # Logo se for menor que 0.0001 concluo que é igual
    df['verificacao'] = (df['adherence_ratio'] - df['mean_adherence_pct'] / 100).abs()
    print(df['verificacao'].describe())
    df = df.drop(columns=['verificacao'])

    # adherence_ratio == mean_adherence_pct / 100 — colunas redundantes, confirmado matematicamente
    df = df.drop(columns=['adherence_ratio'])

    print("\n----------------------------------Investigação Dos Outliers em nutritionist_id/program_index/diet_id------------------------------------------------------")

    #Ver o que ha de diferente nos valores se ha alguma irregulariadade
    print("Nº de nutricionistas únicos:", df['nutritionist_id'].nunique())
    print("Nº de progamas únicos:", df['program_index'].nunique())
    print("Dietas únicas:", df['diet_id'].nunique()) #serve para dizer quantos valores ha diferente um dos outros
    print("-------------------------------------------------------------------------------------------------------------------------------------------------------------")
    print("Nutricionistas únicos:", sorted(df['nutritionist_id'].unique()))
    print("Dietas únicas:", sorted(df['diet_id'].unique()))
    print("Nº de progamas únicos:", sorted(df['program_index'].unique())) #serve para me dizer os valores, notei que nao existem valores diferentes decidi avançar para ver os outliers numericos


    print("\n----------------------------------Outliers Dos Mean_Adherence_pct e Weight_change_kg_6m------------------------------------------------------\n")
    #Identificar outlier no outcome
    #Meti motivation para ver se havia valores negativos ou maiores que 1 mas estava tudo bem entao retirei do ciclo for
    #No "program_index' estava tudo bem sem presença de outliers decidi retirar do ciclo for
    #Notei que mean_adherence_pct é a percentagem de adesão da dieta ou seja um paciente que seguiu a dieta em 85% das refeições tem 85.0. Logo valores entre 0 e 100 são normais.


    for col in ['mean_adherence_pct', 'weight_change_kg_6m']:
        print(f"\n--- {col} outliers ---")
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
        for _, row in outliers.iterrows():
            print(f"Outlier em {col}: valor={row[col]} | program_id={row['program_id']} | patient_id={row['patient_id']}")

    # Retirar outliers impossíveis do mean_adherence_pct
    mask_mean = (df['mean_adherence_pct'] < 0) | (df['mean_adherence_pct'] > 100)

    # Retirar todos os outliers IQR do weight_change_kg_6m
    Q1 = df['weight_change_kg_6m'].quantile(0.25)
    Q3 = df['weight_change_kg_6m'].quantile(0.75)
    IQR = Q3 - Q1
    mask_weight = (df['weight_change_kg_6m'] < Q1 - 1.5 * IQR) | (df['weight_change_kg_6m'] > Q3 + 1.5 * IQR)

    # Separar
    df_outliers = df[mask_mean | mask_weight]
    df_limpo = df[~(mask_mean | mask_weight)]


    print("---------Confirmar separação----------")
    print("\nTotal linhas originais:", len(df))
    print("Linhas limpas:", len(df_limpo))
    print("Linhas outliers:", len(df_outliers))
    print("Soma:", len(df_limpo) + len(df_outliers))

    # Transformar valores em falta em binario
    df_limpo = df_limpo.copy()
    df_limpo['motivation_falta'] = df_limpo['motivation_score_program'].isnull().astype(int)
    df_limpo['adherence_falta'] = df_limpo['mean_adherence_pct'].isnull().astype(int)
    df_limpo['weight_falta'] = df_limpo['weight_change_kg_6m'].isnull().astype(int)

    # Heatmap/Correlação
    correlacao = df_limpo.select_dtypes(include='number').corr()
    plt.figure(figsize=(10, 8))
    plt.imshow(correlacao, cmap='coolwarm', aspect='auto')
    plt.colorbar()
    plt.xticks(range(len(correlacao.columns)), correlacao.columns, rotation=45)
    plt.yticks(range(len(correlacao.columns)), correlacao.columns)
    plt.title('Correlação — Outcomes')
    plt.tight_layout()
    plt.show()

    #Depois de ver o heatmap sem os outliers notei:
    #A adeçao a dieta sobia quando a motivaçao sobia tambem
    #O peso descia quando a adeçao a dieta/motivaçao sobiam
    #Cheguei a conclusao que sao dados MAR e que apartir de outra variavel conseguiamo chegar a variavel em falta
    #Mas se ambos faltarem ao mesmo tempo seria MCAR pois significaria que houve algum problema que aconteceu para o qual nao ter conseguido recolher os dadoss
    #Para descobrir onde faltam os dois decidi fazer o seguinte metodo
    print("\n ---------Verificação E Tratamento De MCAR---------------")
    tres_em_falta = df_limpo['motivation_score_program'].isnull() & df_limpo['mean_adherence_pct'].isnull() & df_limpo['weight_change_kg_6m'].isnull()
    print("Os 3 em falta ao mesmo tempo:", tres_em_falta.sum())
    print('Não existe dados tipo MCAR \n')
    #Não ha existencia de MCAR.

    # Decidi imputar primeiro onde esta a faltar uma variavel
    # Pois tenho mais informação e a imputação é mais precisa

    # Caso 1: Só mean_adherence_pct em falta
    # Quero linhas onde adherence falta MAS motivation e weight_change estão presentes
    print("-------Linhas Onde ADHERENCE falta mas M and W estao presentes-----")
    mask_so_adherence = (
            df_limpo['mean_adherence_pct'].isna() &
            df_limpo['motivation_score_program'].notna() &
            df_limpo['weight_change_kg_6m'].notna()
    )
    print(f"Linhas onde só adherence falta: {mask_so_adherence.sum()}")

    # Como motivation e weight_change são variáveis contínuas (decimais)
    # não podemos fazer groupby por valor exato
    # então criamos intervalos com pd.qcut (grupos com igual número de linhas)
    # Fiz colunas com os dados originais apenas e dividi em 5 partes
    # Assim os bins são criados com os valores originais sem interferência das imputações
    # Criar bin para adherence vou necessitar dito para
    df_limpo['adherence_bin'] = pd.qcut(
        df_limpo['mean_adherence_pct'],
        q=5,
        duplicates='drop'
    )

    df_limpo['motivation_bin'] = pd.qcut(
        df_limpo['motivation_score_program'],
        q=5,
        duplicates='drop'
    )

    df_limpo['weight_bin'] = pd.qcut(
        df_limpo['weight_change_kg_6m'],
        q=5,
        duplicates='drop'
    )
    #Verificação para saber que o pd.qcut funcionou
    print(df_limpo['motivation_bin'].value_counts())
    print(df_limpo['weight_bin'].value_counts())
    print(df_limpo['adherence_bin'].value_counts())

    print("\n---------Imputação MAR - Caso 1: Só mean_adherence_pct em falta---------")

    # Máscara — linhas onde só adherence falta
    mask_so_adherence = (
            df_limpo['mean_adherence_pct'].isna() &
            df_limpo['motivation_score_program'].notna() &
            df_limpo['weight_change_kg_6m'].notna()
    )
    print(f"Linhas onde só adherence falta: {mask_so_adherence.sum()}")

    # Calcular mediana por grupo
    mediana_adherence = (
        df_limpo
        .groupby(['motivation_bin', 'weight_bin'], observed=True)['mean_adherence_pct']
        .transform('median')
    )

    # Preencher
    df_limpo.loc[mask_so_adherence, 'mean_adherence_pct'] = mediana_adherence[mask_so_adherence]

    print("NaN restantes no caso 1:", (
            df_limpo['mean_adherence_pct'].isna() &
            df_limpo['motivation_score_program'].notna() &
            df_limpo['weight_change_kg_6m'].notna()
    ).sum())

    print("\n---------Imputação MAR - Caso 2: Só motivation_score_program em falta---------")

    mask_so_motivation = (
            df_limpo['motivation_score_program'].isna() &
            df_limpo['mean_adherence_pct'].notna() &
            df_limpo['weight_change_kg_6m'].notna()
    )
    print(f"Linhas onde só motivation falta: {mask_so_motivation.sum()}")

    # Agrupar por adherence e weight e calcular mediana da motivation
    mediana_motivation = (
        df_limpo
        .groupby(['adherence_bin', 'weight_bin'], observed=True)['motivation_score_program']
        .transform('median')
    )

    df_limpo.loc[mask_so_motivation, 'motivation_score_program'] = mediana_motivation[mask_so_motivation]

    print("NaN restantes no caso 2:", (
            df_limpo['motivation_score_program'].isna() &
            df_limpo['mean_adherence_pct'].notna() &
            df_limpo['weight_change_kg_6m'].notna()
    ).sum())


    #Agora o mesmo aqui
    print("\n\n---------Imputação MAR - Caso 3: Só weight_change_kg_6m em falta---------")

    mask_so_weight = (
            df_limpo['weight_change_kg_6m'].isna() &
            df_limpo['mean_adherence_pct'].notna() &
            df_limpo['motivation_score_program'].notna()
    )
    print(f"Linhas onde só weight_change falta: {mask_so_weight.sum()}")

    mediana_weight = (
        df_limpo
        .groupby(['motivation_bin', 'adherence_bin'], observed=True)['weight_change_kg_6m']
        .transform('median')
    )

    df_limpo.loc[mask_so_weight, 'weight_change_kg_6m'] = mediana_weight[mask_so_weight]


    #Confirmar se nao ha nenhuma linha no caso 3 por imputar
    print("NaN restantes no caso 3:", (
            df_limpo['weight_change_kg_6m'].isna() &
            df_limpo['mean_adherence_pct'].notna() &
            df_limpo['motivation_score_program'].notna()
    ).sum())

    #Ver exatamente qual linha ficou por tratar e analisa-la
    #Depois de analisar notei que é incomum alguém que não se sente motivado mas segue a dieta à risca na mesma por disciplina

    linha_problema = df_limpo[
        df_limpo['weight_change_kg_6m'].isna() &
        df_limpo['mean_adherence_pct'].notna() &
        df_limpo['motivation_score_program'].notna()
        ]
    print(linha_problema[['motivation_score_program', 'mean_adherence_pct', 'weight_change_kg_6m', 'motivation_bin',
                          'adherence_bin']])

    # Fallback: o grupo tinha todas as linhas com weight em falta
    # Fiz uma condiçao em que o trata linhas por tratar pela vairavel que mais impacta que é a adherence.
    # Alarguei para só adherence_bin para ter mais linhas e conseguir calcular mediana
    if linha_problema.shape[0] > 0:
        mediana_fallback = (
            df_limpo.groupby(['adherence_bin'], observed=True)['weight_change_kg_6m'].transform('median')
        )

        # Identificar a linha que ainda tenho por tratar
        ainda_em_falta = (
                df_limpo['weight_change_kg_6m'].isna() &
                df_limpo['mean_adherence_pct'].notna() &
                df_limpo['motivation_score_program'].notna()
        )
        # Agora pela Adherence substituo pela mediana.
        df_limpo.loc[ainda_em_falta, 'weight_change_kg_6m'] = mediana_fallback[ainda_em_falta]


    #Confirmar se ja esta tudo resolvido
    print("NaN restantes após fallback:", (
            df_limpo['weight_change_kg_6m'].isna() &
            df_limpo['mean_adherence_pct'].notna() &
            df_limpo['motivation_score_program'].notna()
    ).sum())

    # Caso 4: mean_adherence e motivation em falta
    print("\n---------Imputação MAR - Caso 4: mean_adherence e motivation em falta---------")

    # Primeira coisa a fazer para ver se tenho de tratar ou não
    mask_adh_mot = (
            df_limpo['mean_adherence_pct'].isna() &
            df_limpo['motivation_score_program'].isna() &
            df_limpo['weight_change_kg_6m'].notna()
    )
    print(f"Linhas onde adherence e motivation faltam: {mask_adh_mot.sum()}")

    # Calcular a mediana com base em weight_bin
    mediana_adherence_c4 = (
        df_limpo
        .groupby(['weight_bin'], observed=True)['mean_adherence_pct']
        .transform('median')
    )
    mediana_motivation_c4 = (
        df_limpo
        .groupby(['weight_bin'], observed=True)['motivation_score_program']
        .transform('median')
    )

    # Imputar valor nas casas vazias
    df_limpo.loc[mask_adh_mot, 'mean_adherence_pct'] = mediana_adherence_c4[mask_adh_mot]
    df_limpo.loc[mask_adh_mot, 'motivation_score_program'] = mediana_motivation_c4[mask_adh_mot]

    #Verificar
    print("NaN restantes no caso 4:", (
            df_limpo['mean_adherence_pct'].isna() &
            df_limpo['motivation_score_program'].isna() &
            df_limpo['weight_change_kg_6m'].notna()
    ).sum())

    # Caso 5: mean_adherence e weight_change
    print("\n---------Imputação MAR - Caso 5: mean_adherence e weight_change em falta---------")

    # Primeira coisa a fazer para ver se tenho de tratar ou não
    mask_adh_wgt = (
            df_limpo['mean_adherence_pct'].isna() &
            df_limpo['motivation_score_program'].notna() &
            df_limpo['weight_change_kg_6m'].isna()
    )
    print(f"Linhas onde adherence e weight_change faltam: {mask_adh_wgt.sum()}")

    # Calcular a mediana com base em motivation_score_program
    mediana_adherence_c5 = (
        df_limpo
        .groupby(['motivation_bin'], observed=True)['mean_adherence_pct']
        .transform('median')
    )
    mediana_weight_c5 = (
        df_limpo
        .groupby(['motivation_bin'], observed=True)['weight_change_kg_6m']
        .transform('median')
    )

    # Imputar valor nas casas vazias
    df_limpo.loc[mask_adh_wgt, 'mean_adherence_pct'] = mediana_adherence_c5[mask_adh_wgt]
    df_limpo.loc[mask_adh_wgt, 'weight_change_kg_6m'] = mediana_weight_c5[mask_adh_wgt]

    #Verificar
    print("NaN restantes no caso 5:", (
            df_limpo['mean_adherence_pct'].isna() &
            df_limpo['motivation_score_program'].notna() &
            df_limpo['weight_change_kg_6m'].isna()
    ).sum())

    # Caso 6: motivation e weight_change
    print("\n---------Imputação MAR - Caso 6: motivation e weight_change em falta---------")

    # Primeira coisa a fazer para ver se tenho de tratar ou não
    mask_mot_wgt = (
            df_limpo['mean_adherence_pct'].notna() &
            df_limpo['motivation_score_program'].isna() &
            df_limpo['weight_change_kg_6m'].isna()
    )
    print(f"Linhas onde motivation e weight_change faltam: { mask_mot_wgt.sum()}")

    # Calcular a mediana com base em motivation_score_program
    mediana_motivation_c6 = (
        df_limpo
        .groupby(['adherence_bin'], observed=True)['motivation_score_program']
        .transform('median')
    )
    mediana_weight_c6 = (
        df_limpo
        .groupby(['adherence_bin'], observed=True)['weight_change_kg_6m']
        .transform('median')
    )

    # Imputar valor nas casas vazias
    df_limpo.loc[mask_mot_wgt, 'mean_adherence_pct'] = mediana_motivation_c6[mask_adh_wgt]
    df_limpo.loc[mask_mot_wgt, 'weight_change_kg_6m'] = mediana_weight_c6[mask_adh_wgt]

    #Verificar
    print("NaN restantes no caso 6:", (
            df_limpo['mean_adherence_pct'].notna() &
            df_limpo['motivation_score_program'].isna() &
            df_limpo['weight_change_kg_6m'].isna()
    ).sum())

    # Pelo medo de ter uma contrariação nos dados adherence
    # Por exemplo: Adherence Alto e Motivação Baixa
    # Decidi analisar os valores de forma a perceber se são valores normais ou contrariados
    print("\n-------Analise Dos Outliers de Adherence----------")
    print(df_outliers[['program_id', 'patient_id', 'mean_adherence_pct','motivation_score_program', 'weight_change_kg_6m']])
    #ficou com um valor NaN no outlier mesma coisa que nao ter motivation ou aherence

    # Voltando ao antes vamos tratar dos outliers mais faceis que sao do adherence.
    # Tinha chegado a acordo que todos os valores entre 0 e 100 eram possiveis
    print("\n---------Tratamento Outliers Impossíveis — mean_adherence_pct---------")

    # Máscara para valores impossíveis de adherence
    mask_impossiveis_adherence = (
            (df_outliers['mean_adherence_pct'] < 0) |
            (df_outliers['mean_adherence_pct'] > 100)
    )
    # Corrigir mean_adherence impossíveis (-10 e 150) no df_outliers
    # Usa a mediana do df_limpo com pacientes de motivation parecida (±0.05)
    mask_impossiveis_adherence = (
            (df_outliers['mean_adherence_pct'] < 0) |
            (df_outliers['mean_adherence_pct'] > 100)
    )

    for idx in df_outliers[mask_impossiveis_adherence].index:
        mot = df_outliers.loc[idx, 'motivation_score_program']
        mediana = df_limpo[
            (df_limpo['motivation_score_program'] >= mot - 0.05) &
            (df_limpo['motivation_score_program'] <= mot + 0.05)
            ]['mean_adherence_pct'].median()
        df_outliers.loc[idx, 'mean_adherence_pct'] = mediana

        # Linha PR02444 tem motivation = NaN, não conseguimos agrupar
        # Usamos a mediana geral como último recurso
        mask_nan_motivation = df_outliers['motivation_score_program'].isna()
        df_outliers.loc[mask_nan_motivation, 'motivation_score_program'] = df_limpo['motivation_score_program'].median()
        df_outliers.loc[mask_nan_motivation, 'mean_adherence_pct'] = df_limpo['mean_adherence_pct'].median()


    # Ver Tabela Outliers
    print(df_outliers[['program_id', 'patient_id', 'mean_adherence_pct', 'motivation_score_program']])

    # Verificar valores corrigidos
    print("\nOutliers impossíveis corrigidos:")
    print(df_outliers[mask_impossiveis_adherence][['program_id', 'patient_id', 'mean_adherence_pct','motivation_score_program', 'weight_change_kg_6m']])

    # Parte mais malandra do projeto pois preciso de relacionar o peso perdido pelo peso inicial do paciente e ver se os valores são normais ou são diferentes
    print("\n---------Tratamento Outliers Impossíveis — weight_change_kg_6m ---------")

    # Cruzar outliers de weight com peso inicial do paciente
    analise_weight = df_outliers[['program_id', 'patient_id', 'weight_change_kg_6m']].merge(
        df_pacientes[['patient_id', 'baseline_weight_kg', 'height_cm']],
        on='patient_id',
        how='left'
    )

    # Calcular peso final estimado e IMC (apartir de 18.5 para baixo, pouco saudavel)
    analise_weight['peso_final_estimado'] = analise_weight['baseline_weight_kg'] + analise_weight['weight_change_kg_6m']
    analise_weight['imc_final'] = analise_weight['peso_final_estimado'] / ((analise_weight['height_cm'] / 100) ** 2)

    print(analise_weight)
    # Deu me IMC muito pequenos valores entre os 13 e 17 arredondamente. Para perceber melhor o valor decidi saber se foi por acaso algum nutricionista
    # Eu vi que um IMC entre 13-16 no google as pessoas estariam num hospital e não
    print(analise_weight[['program_id', 'patient_id', 'weight_change_kg_6m']].merge(
        df_outliers[['program_id', 'nutritionist_id']],
        on='program_id'
    ))


    return df

#se quiserem ver linhas em especifico usem o seguinte metodo -print(df.iloc[...:...])

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
