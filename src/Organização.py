import pandas as pd
import matplotlib.pyplot as plt
#So meti para garantir que todas as colunas aparecem sem o "..." funciona sem mas so para garantir a dobrar
pd.set_option('display.max_columns', None)

#Aumenta a largura em vez de metade da tabela estar em cima e a outra metae estar em baixo
pd.set_option('display.width', None)


# Funcao para carregar os dados
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
        # 1. Limpeza basica (o que ja tinhas)
        df["sex"] = df["sex"].astype(str).str.strip().str.capitalize()

        # 2. Mapeamento para uniformizar (A SOLUCAO)
        mapeamento = {
            'Female': 'F',
            'Male': 'M',
        }

        # O 'replace' substitui os valores longos pelos curtos
        df["sex"] = df["sex"].replace(mapeamento)

        print("\nValores unicos de 'sex' apos uniformizacao:")
        print(df["sex"].unique())

    return df

def dietsCSV_limpeza(df: pd.DataFrame) -> pd.DataFrame:
    # Ver NaN antes de tratar
    print(df.isnull().sum())

    # Verificar relacao entre diet_type e as colunas com NaN
    print(df[['diet_type', 'sodium_limit_mg', 'fiber_target_g']])

    # Imputar pela media do mesmo tipo de dieta
    # fiber_target_g: valores entre 25-38g conforme literatura (25-35g recomendado).
    # O valor em falta e low_carb foi imputado pela media dos outros low_carb
    # para respeitar o contexto do tipo de dieta.
    df['sodium_limit_mg'] = df.groupby('diet_type')['sodium_limit_mg'].transform(lambda x: x.fillna(x.mean()))
    df['fiber_target_g'] = df.groupby('diet_type')['fiber_target_g'].transform(lambda x: x.fillna(x.mean()))

    # Confirmar que ficou limpo
    print(df.isnull().sum())

    # CORRECAO: faltava o return aqui, sem isto a funcao devolvia None e o merge na integracao crashava
    return df

def nutritionistsCSV_limpeza(df: pd.DataFrame) -> pd.DataFrame:
    #Apagar coluna duplicada e corrigir valor 99 para 9.
    #Dado que existe reforma a probabilidade de uma pessoa que tenha trabalhado trabalhado mais de 90 anos e tenha vivido e mais pouco provavel do que a pessoa ter se enganado
    df = df.drop(columns=['years_experience'])
    df.loc[df['nutritionist_id'] == 'N008', 'experience_years'] = 9

    #Ver Nan antes de tratar
    print(df.isnull().sum())

    # Criar colunas binarias para investigar padrao dos NaN
    df['approach_falta'] = df['approach'].isnull().astype(int)
    df['years_falta'] = df['experience_years'].isnull().astype(int)
    df['specialty_num'] = df['specialty'].astype('category').cat.codes
    df['approach_num'] = df['approach'].astype('category').cat.codes

    # Calcular correlacao
    correlacao = df.select_dtypes(include='number').corr()

    # Desenhar heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(correlacao, cmap='coolwarm', aspect='auto')
    plt.colorbar()
    plt.xticks(range(len(correlacao.columns)), correlacao.columns, rotation=45)
    plt.yticks(range(len(correlacao.columns)), correlacao.columns)
    plt.title('Correlacao - Nutritionists')
    plt.tight_layout()
    plt.show()

    # Imputacao experience_years - MCAR
    # Mediana porque e variavel numerica e e robusta a outliers.
    # A correlacao neutra no heatmap entre years_falta e approach_num confirma
    # que o tipo de approach nao influencia se os anos estao em falta.
    # Provavel falha na recolha de dados sem relacao com o valor em si.
    df['experience_years'] = df['experience_years'].fillna(df['experience_years'].median())

    # Imputacao approach - MNAR
    # Criamos categoria 'unknown' em vez de imputar com moda.
    # A abordagem de um nutricionista e influenciada pela emocao humana
    # pode ser demasiado complexa para se identificar numa unica palavra.
    # Inventar um approach distorceria a realidade - unknown e mais honesto.
    df['approach'] = df['approach'].fillna('unknown')

    # Apagar colunas auxiliares criadas para analise pois nao fazem parte dos dados originais
    df = df.drop(columns=['approach_falta', 'years_falta', 'specialty_num', 'approach_num'])

    # Confirmar que ficou limpo
    print(df.isnull().sum())

    print(df)

    return df

def patientsCSV_limpeza(df: pd.DataFrame) -> pd.DataFrame:
        # Ver NaN antes de tratar
        print(df.isnull().sum())
        print(df.describe())

        # Apagar coluna duplicada - bmi_redundant e igual a baseline_bmi
        # CORRECAO: antes de apagar confirmo que sao mesmo iguais, so para o professor ver que verifiquei
        print("bmi_redundant == baseline_bmi?", df['baseline_bmi'].equals(df['bmi_redundant']))
        df = df.drop(columns=['bmi_redundant'])

        # CORRIGIR ERROS DE DIGITACAO OBVIOS

        # baseline_weight_kg: P0154=5kg e P0181=500kg - impossiveis
        mediana_peso = df[(df['baseline_weight_kg'] > 10) & (df['baseline_weight_kg'] < 300)][
            'baseline_weight_kg'].median()
        df.loc[df['baseline_weight_kg'] < 10, 'baseline_weight_kg'] = mediana_peso
        df.loc[df['baseline_weight_kg'] > 300, 'baseline_weight_kg'] = mediana_peso

        # height_cm: P0278=300cm e P0755=10cm - impossiveis
        mediana_altura = df[(df['height_cm'] > 50) & (df['height_cm'] < 250)]['height_cm'].median()
        df.loc[df['height_cm'] > 250, 'height_cm'] = mediana_altura
        df.loc[df['height_cm'] < 50, 'height_cm'] = mediana_altura

        # Recalcular baseline_bmi apos corrigir erros de peso e altura (e desta maneira que e calculada o BMI)
        df['baseline_bmi'] = df['baseline_weight_kg'] / ((df['height_cm'] / 100) ** 2)

        # sleep_hours: P0670=30h (>24 impossivel) e P0790=-1h (negativo impossivel)
        mediana_sono = df[(df['sleep_hours'] >= 0) & (df['sleep_hours'] <= 24)]['sleep_hours'].median()
        df.loc[df['sleep_hours'] > 24, 'sleep_hours'] = mediana_sono
        df.loc[df['sleep_hours'] < 0, 'sleep_hours'] = mediana_sono

        # motivation_score: P0821 e P0827=5.0 - fora da escala 0-1
        mediana_motiv = df[df['motivation_score'] <= 1]['motivation_score'].median()
        df.loc[df['motivation_score'] > 1, 'motivation_score'] = mediana_motiv

        # age: min=0 - impossivel num estudo de nutricao de adultos
        mediana_idade = df[df['age'] > 0]['age'].median()
        df.loc[df['age'] == 0, 'age'] = mediana_idade

        # DETECAO DE OUTLIERS VIA IQR (para mostrar o que sobra)
        # NOTA: os outliers que aparecem aqui (tipo motivation 0.15-0.27) sao valores possiveis e reais
        # uma pessoa pode simplesmente ter pouca motivacao, nao e um erro de digitacao
        # por isso mantenho-os no dataset, o IQR so serve para eu ver o que sobrou nao para apagar tudo
        for col in ['age', 'height_cm', 'baseline_weight_kg', 'baseline_bmi', 'sleep_hours', 'motivation_score']:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
            if not outliers.empty:
                print(f"\nOutliers restantes em {col}:")
                print(outliers[['patient_id', col]])

        # --- ANALISE DE CORRELACAO DOS NaNs ---
        df['age_falta'] = df['age'].isnull().astype(int)
        df['sleep_falta'] = df['sleep_hours'].isnull().astype(int)
        df['motivation_falta'] = df['motivation_score'].isnull().astype(int)

        correlacao = df.select_dtypes(include='number').corr()
        plt.figure(figsize=(10, 8))
        plt.imshow(correlacao, cmap='coolwarm', aspect='auto')
        plt.colorbar()
        plt.xticks(range(len(correlacao.columns)), correlacao.columns, rotation=45)
        plt.yticks(range(len(correlacao.columns)), correlacao.columns)
        plt.title('Correlacao - Patients')
        plt.tight_layout()
        plt.show()

        #IMPUTACAO DOS NaN REAIS
        # age - MCAR - mediana
        df['age'] = df['age'].fillna(df['age'].median())

        # sleep_hours - MCAR - mediana
        df['sleep_hours'] = df['sleep_hours'].fillna(df['sleep_hours'].median())

            # motivation_score - MNAR (baixa motivacao - nao preenchem) - mediana
        df['motivation_score'] = df['motivation_score'].fillna(df['motivation_score'].median())

        # Apagar colunas auxiliares
        df = df.drop(columns=['age_falta', 'sleep_falta', 'motivation_falta'])

        print(df.isnull().sum())
        print(df)

        return df

def outcomesCSV_limpeza(df: pd.DataFrame, df_pacientes: pd.DataFrame) -> pd.DataFrame:
    # CORRECAO: agora recebo o df_pacientes como parametro em vez de usar a variavel global
    # assim a funcao nao depende do que esta la fora, fica mais limpa e nao da problemas se alguem mudar o nome

    #necesario fazer uma copia da tabela pois a biblioteca panda nao deixa apagar linhas na tabela principal.
    df = df.copy()

    print(df.isnull().sum())
    print(df.describe())

    #Verificar linhas totalmente duplicadas
    print("Linhas Duplicadas:", df.duplicated().sum())

    #Eliminar mesmas linhas
    df = df.drop_duplicates()
    print("Linhas duplicadas apos limpeza:", df.duplicated().sum())

    #Notei num dado que adherence_ratio poderia ser a mesma coisa que mean_adherence se dividisse por 100
    # Verificar se adherence_ratio == mean_adherence_pct / 100
    # Se as colunas forem iguais, a diferenca deveser ~0 em todas as linhas pois devido aos valores serem discretos no python ele tende a deixar um resto.
    # Logo se for menor que 0.0001 concluo que e igual
    df['verificacao'] = (df['adherence_ratio'] - df['mean_adherence_pct'] / 100).abs()
    print(df['verificacao'].describe())
    df = df.drop(columns=['verificacao'])

    # adherence_ratio == mean_adherence_pct / 100 - colunas redundantes, confirmado matematicamente
    df = df.drop(columns=['adherence_ratio'])

    print("\n----------------------------------Investigacao Dos Outliers em nutritionist_id/program_index/diet_id------------------------------------------------------")

    #Ver o que ha de diferente nos valores se ha alguma irregulariadade
    print("N de nutricionistas unicos:", df['nutritionist_id'].nunique())
    print("N de progamas unicos:", df['program_index'].nunique())
    print("Dietas unicas:", df['diet_id'].nunique()) #serve para dizer quantos valores ha diferente um dos outros
    print("-------------------------------------------------------------------------------------------------------------------------------------------------------------")
    print("Nutricionistas unicos:", sorted(df['nutritionist_id'].unique()))
    print("Dietas unicas:", sorted(df['diet_id'].unique()))
    print("N de progamas unicos:", sorted(df['program_index'].unique())) #serve para me dizer os valores, notei que nao existem valores diferentes decidi avancar para ver os outliers numericos


    print("\n----------------------------------Outliers Dos Mean_Adherence_pct e Weight_change_kg_6m------------------------------------------------------\n")
    #Identificar outlier no outcome
    #Meti motivation para ver se havia valores negativos ou maiores que 1 mas estava tudo bem entao retirei do ciclo for
    #No "program_index' estava tudo bem sem presenca de outliers decidi retirar do ciclo for
    #Notei que mean_adherence_pct e a percentagem de adesao da dieta ou seja um paciente que seguiu a dieta em 85% das refeicoes tem 85.0. Logo valores entre 0 e 100 sao normais.


    for col in ['mean_adherence_pct', 'weight_change_kg_6m']:
        print(f"\n--- {col} outliers ---")
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
        for _, row in outliers.iterrows():
            print(f"Outlier em {col}: valor={row[col]} | program_id={row['program_id']} | patient_id={row['patient_id']}")

    # Retirar outliers impossiveis do mean_adherence_pct
    mask_mean = (df['mean_adherence_pct'] < 0) | (df['mean_adherence_pct'] > 100)

    # Retirar todos os outliers IQR do weight_change_kg_6m
    Q1 = df['weight_change_kg_6m'].quantile(0.25)
    Q3 = df['weight_change_kg_6m'].quantile(0.75)
    IQR = Q3 - Q1
    mask_weight = (df['weight_change_kg_6m'] < Q1 - 1.5 * IQR) | (df['weight_change_kg_6m'] > Q3 + 1.5 * IQR)

    # Separar
    df_outliers = df[mask_mean | mask_weight]
    df_limpo = df[~(mask_mean | mask_weight)]


    print("---------Confirmar separacao----------")
    print("\nTotal linhas originais:", len(df))
    print("Linhas limpas:", len(df_limpo))
    print("Linhas outliers:", len(df_outliers))
    print("Soma:", len(df_limpo) + len(df_outliers))

    # Transformar valores em falta em binario
    df_limpo = df_limpo.copy()
    df_limpo['motivation_falta'] = df_limpo['motivation_score_program'].isnull().astype(int)
    df_limpo['adherence_falta'] = df_limpo['mean_adherence_pct'].isnull().astype(int)
    df_limpo['weight_falta'] = df_limpo['weight_change_kg_6m'].isnull().astype(int)

    # Heatmap/Correlacao
    correlacao = df_limpo.select_dtypes(include='number').corr()
    plt.figure(figsize=(10, 8))
    plt.imshow(correlacao, cmap='coolwarm', aspect='auto')
    plt.colorbar()
    plt.xticks(range(len(correlacao.columns)), correlacao.columns, rotation=45)
    plt.yticks(range(len(correlacao.columns)), correlacao.columns)
    plt.title('Correlacao - Outcomes')
    plt.tight_layout()
    plt.show()

    #Depois de ver o heatmap sem os outliers notei:
    #A adecao a dieta sobia quando a motivacao sobia tambem
    #O peso descia quando a adecao a dieta/motivacao sobiam
    #Cheguei a conclusao que sao dados MAR e que apartir de outra variavel conseguiamo chegar a variavel em falta
    #Mas se ambos faltarem ao mesmo tempo seria MCAR pois significaria que houve algum problema que aconteceu para o qual nao ter conseguido recolher os dadoss
    #Para descobrir onde faltam os dois decidi fazer o seguinte metodo
    print("\n ---------Verificacao E Tratamento De MCAR---------------")
    tres_em_falta = df_limpo['motivation_score_program'].isnull() & df_limpo['mean_adherence_pct'].isnull() & df_limpo['weight_change_kg_6m'].isnull()
    print("Os 3 em falta ao mesmo tempo:", tres_em_falta.sum())
    print('Nao existe dados tipo MCAR \n')
    #Nao ha existencia de MCAR.

    # Decidi imputar primeiro onde esta a faltar uma variavel
    # Pois tenho mais informacao e a imputacao e mais precisa

    # Caso 1: So mean_adherence_pct em falta
    # Quero linhas onde adherence falta MAS motivation e weight_change estao presentes
    print("-------Linhas Onde ADHERENCE falta mas M and W estao presentes-----")
    mask_so_adherence = (
            df_limpo['mean_adherence_pct'].isna() &
            df_limpo['motivation_score_program'].notna() &
            df_limpo['weight_change_kg_6m'].notna()
    )
    print(f"Linhas onde so adherence falta: {mask_so_adherence.sum()}")

    # Como motivation e weight_change sao variaveis continuas (decimais)
    # nao podemos fazer groupby por valor exato
    # entao criamos intervalos com pd.qcut (grupos com igual numero de linhas)
    # Fiz colunas com os dados originais apenas e dividi em 5 partes
    # Assim os bins sao criados com os valores originais sem interferencia das imputacoes
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
    #Verificacao para saber que o pd.qcut funcionou
    print(df_limpo['motivation_bin'].value_counts())
    print(df_limpo['weight_bin'].value_counts())
    print(df_limpo['adherence_bin'].value_counts())

    print("\n---------Imputacao MAR - Caso 1: So mean_adherence_pct em falta---------")

    # Mascara - linhas onde so adherence falta
    mask_so_adherence = (
            df_limpo['mean_adherence_pct'].isna() &
            df_limpo['motivation_score_program'].notna() &
            df_limpo['weight_change_kg_6m'].notna()
    )
    print(f"Linhas onde so adherence falta: {mask_so_adherence.sum()}")

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

    print("\n---------Imputacao MAR - Caso 2: So motivation_score_program em falta---------")

    mask_so_motivation = (
            df_limpo['motivation_score_program'].isna() &
            df_limpo['mean_adherence_pct'].notna() &
            df_limpo['weight_change_kg_6m'].notna()
    )
    print(f"Linhas onde so motivation falta: {mask_so_motivation.sum()}")

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
    print("\n\n---------Imputacao MAR - Caso 3: So weight_change_kg_6m em falta---------")

    mask_so_weight = (
            df_limpo['weight_change_kg_6m'].isna() &
            df_limpo['mean_adherence_pct'].notna() &
            df_limpo['motivation_score_program'].notna()
    )
    print(f"Linhas onde so weight_change falta: {mask_so_weight.sum()}")

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
    #Depois de analisar notei que e incomum alguem que nao se sente motivado mas segue a dieta a risca na mesma por disciplina

    linha_problema = df_limpo[
        df_limpo['weight_change_kg_6m'].isna() &
        df_limpo['mean_adherence_pct'].notna() &
        df_limpo['motivation_score_program'].notna()
        ]
    print(linha_problema[['motivation_score_program', 'mean_adherence_pct', 'weight_change_kg_6m', 'motivation_bin',
                          'adherence_bin']])

    # Fallback: o grupo tinha todas as linhas com weight em falta
    # Fiz uma condicao em que o trata linhas por tratar pela vairavel que mais impacta que e a adherence.
    # Alarguei para so adherence_bin para ter mais linhas e conseguir calcular mediana
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
    print("NaN restantes apos fallback:", (
            df_limpo['weight_change_kg_6m'].isna() &
            df_limpo['mean_adherence_pct'].notna() &
            df_limpo['motivation_score_program'].notna()
    ).sum())

    # Caso 4: mean_adherence e motivation em falta
    print("\n---------Imputacao MAR - Caso 4: mean_adherence e motivation em falta---------")

    # Primeira coisa a fazer para ver se tenho de tratar ou nao
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
    print("\n---------Imputacao MAR - Caso 5: mean_adherence e weight_change em falta---------")

    # Primeira coisa a fazer para ver se tenho de tratar ou nao
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
    print("\n---------Imputacao MAR - Caso 6: motivation e weight_change em falta---------")

    # Primeira coisa a fazer para ver se tenho de tratar ou nao
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
    df_limpo.loc[mask_mot_wgt, 'motivation_score_program'] = mediana_motivation_c6[mask_mot_wgt]
    df_limpo.loc[mask_mot_wgt, 'weight_change_kg_6m'] = mediana_weight_c6[mask_mot_wgt]

    #Verificar
    print("NaN restantes no caso 6:", (
            df_limpo['mean_adherence_pct'].notna() &
            df_limpo['motivation_score_program'].isna() &
            df_limpo['weight_change_kg_6m'].isna()
    ).sum())

    # Pelo medo de ter uma contrariacao nos dados adherence
    # Por exemplo: Adherence Alto e Motivacao Baixa
    # Decidi analisar os valores de forma a perceber se sao valores normais ou contrariados
    print("\n-------Analise Dos Outliers de Adherence----------")
    print(df_outliers[['program_id', 'patient_id', 'mean_adherence_pct','motivation_score_program', 'weight_change_kg_6m']])
    #ficou com um valor NaN no outlier mesma coisa que nao ter motivation ou aherence

    # Voltando ao antes vamos tratar dos outliers mais faceis que sao do adherence.
    # Tinha chegado a acordo que todos os valores entre 0 e 100 eram possiveis
    print("\n---------Tratamento Outliers Impossiveis - mean_adherence_pct---------")

    # Mascara para valores impossiveis de adherence
    mask_impossiveis_adherence = (
            (df_outliers['mean_adherence_pct'] < 0) |
            (df_outliers['mean_adherence_pct'] > 100)
    )
    # Corrigir mean_adherence impossiveis (-10 e 150) no df_outliers
    # Usa a mediana do df_limpo com pacientes de motivation parecida (+-0.05)
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

    # CORRECAO: este bloco estava dentro do for loop e era executado varias vezes sem necessidade
    # so precisa de correr uma vez depois do for acabar pois trata as linhas onde motivation e NaN
    # Linha PR02444 tem motivation = NaN, nao conseguimos agrupar
    # Usamos a mediana geral como ultimo recurso
    mask_nan_motivation = df_outliers['motivation_score_program'].isna()
    df_outliers.loc[mask_nan_motivation, 'motivation_score_program'] = df_limpo['motivation_score_program'].median()
    df_outliers.loc[mask_nan_motivation, 'mean_adherence_pct'] = df_limpo['mean_adherence_pct'].median()


    # Ver Tabela Outliers
    print(df_outliers[['program_id', 'patient_id', 'mean_adherence_pct', 'motivation_score_program']])

    # Verificar valores corrigidos
    print("\nOutliers impossiveis corrigidos:")
    print(df_outliers[mask_impossiveis_adherence][['program_id', 'patient_id', 'mean_adherence_pct','motivation_score_program', 'weight_change_kg_6m']])

    # Parte mais malandra do projeto pois preciso de relacionar o peso perdido pelo peso inicial do paciente e ver se os valores sao normais ou sao diferentes
    print("\n---------Tratamento Outliers Impossiveis - weight_change_kg_6m ---------")

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
    # Eu vi que um IMC entre 13-16 no google as pessoas estariam num hospital e nao
    print(analise_weight[['program_id', 'patient_id', 'weight_change_kg_6m']].merge(
        df_outliers[['program_id', 'nutritionist_id']],
        on='program_id'
    ))

    # -100 e +100 sao claramente registos errados - nenhum paciente perde ou ganha 100kg em 6 meses
    # CORRECAO: em vez de mediana geral, usamos mediana por grupo tal como fizemos para o adherence impossivel
    # Para cada linha impossivel, procuramos pacientes no df_limpo com motivation e adherence parecidos (+-0.05)
    # Assim ficamos coerentes com a filosofia MAR usada nos NaN — usar contexto do paciente para estimar
    mask_peso_impossivel = (
        (df_outliers['weight_change_kg_6m'] == -100) |
        (df_outliers['weight_change_kg_6m'] == 100)
    )
    for idx in df_outliers[mask_peso_impossivel].index:
        mot = df_outliers.loc[idx, 'motivation_score_program']
        adh = df_outliers.loc[idx, 'mean_adherence_pct']
        mediana = df_limpo[
            (df_limpo['motivation_score_program'] >= mot - 0.05) &
            (df_limpo['motivation_score_program'] <= mot + 0.05) &
            (df_limpo['mean_adherence_pct'] >= adh - 5) &
            (df_limpo['mean_adherence_pct'] <= adh + 5)
        ]['weight_change_kg_6m'].median()
        # Se o grupo for demasiado pequeno e nao tiver valores, cai para a mediana geral como ultimo recurso
        if pd.isna(mediana):
            mediana = df_limpo['weight_change_kg_6m'].median()
        df_outliers.loc[idx, 'weight_change_kg_6m'] = mediana

    # IMC < 18.5 significa desnutricao grave - clinicamente incompativel com um paciente em programa de dieta
    # A hipotese mais provavel e erro de escala: escreveram -45.8 em vez de -4.58
    # Dividimos por 10 para corrigir - desta forma -45.8 passa a -4.58 que e uma perda razoavel em 6 meses
    # Usamos o analise_weight que ja calculamos acima - nao precisamos de calcular de novo
    # Excluimos os -100 e +100 que ja foram corrigidos para nao dividir por 10 um valor que ja e mediana
    programas_ja_corrigidos = df_outliers[mask_peso_impossivel]['program_id']
    programas_escala = analise_weight[
        (analise_weight['imc_final'] < 18.5) &
        (~analise_weight['program_id'].isin(programas_ja_corrigidos))
    ]['program_id']
    mask_escala = df_outliers['program_id'].isin(programas_escala)
    df_outliers.loc[mask_escala, 'weight_change_kg_6m'] = df_outliers.loc[mask_escala, 'weight_change_kg_6m'] / 10

    print("\nOutliers de weight apos correcao:")
    print(df_outliers[['program_id', 'patient_id', 'weight_change_kg_6m']])

    # Remover colunas auxiliares criadas so para analise interna
    # motivation_bin, weight_bin, adherence_bin - criados com pd.qcut para permitir o groupby por intervalos
    # motivation_falta, adherence_falta, weight_falta - criados para o heatmap de correlacao
    # Nenhuma destas colunas pertence aos dados originais, por isso apagamos antes de devolver o dataset
    df_limpo = df_limpo.drop(columns=[
        'motivation_bin', 'weight_bin', 'adherence_bin',
        'motivation_falta', 'adherence_falta', 'weight_falta'
    ])

    # Reunir os dados limpos com os outliers ja tratados num so dataframe
    # ignore_index=True reinicia o indice de 0 para nao ficarem indices repetidos
    df_final = pd.concat([df_limpo, df_outliers], ignore_index=True)

    print("\n---------Outcomes apos limpeza completa---------")
    print("Total de linhas:", len(df_final))
    print("NaN restantes:")
    print(df_final[['mean_adherence_pct', 'motivation_score_program', 'weight_change_kg_6m']].isnull().sum())

    return df_final

def integrar_dados(df_outcomes: pd.DataFrame, df_pacientes: pd.DataFrame,
                   df_nutricionistas: pd.DataFrame, df_dieta: pd.DataFrame) -> pd.DataFrame:

    # Juntar os 4 CSVs todos tratados num unico dataset para treino do modelo de IA
    # O outcomes e a tabela central, tem program_id, patient_id, nutritionist_id e diet_id
    # Fazemos LEFT JOIN a partir do outcomes para trazer a informacao dos outros CSVs
    # LEFT JOIN garante que nao perdemos nenhuma linha do outcomes mesmo que haja IDs sem correspondencia

    df = df_outcomes.merge(df_pacientes, on='patient_id', how='left')
    df = df.merge(df_nutricionistas, on='nutritionist_id', how='left')
    df = df.merge(df_dieta, on='diet_id', how='left')

    print("\n---------Integracao dos 4 CSVs---------")
    print("Total de linhas apos integracao:", len(df))
    print("Colunas disponiveis:", list(df.columns))
    print("NaN por coluna:")
    print(df.isnull().sum())

    return df

#se quiserem ver linhas em especifico usem o seguinte metodo -print(df.iloc[...:...])

# CORRECAO: meti dentro do if __name__ para que quando alguem fizer import deste ficheiro noutro sitio
# (tipo no modelos.py) nao executa tudo automaticamente, so corre se correrem este ficheiro diretamente
if __name__ == "__main__":
    df_pacientes, df_dieta, df_nutricionistas, df_resultados = carregar_dados()
    df_pacientes = patientsCSV_limpeza(df_pacientes)
    # Uniformizar coluna sex: Female->F, Male->M — aplicado depois da limpeza geral dos pacientes
    df_pacientes = consistencia_rep(df_pacientes)
    df_dieta = dietsCSV_limpeza(df_dieta)
    df_nutricionistas = nutritionistsCSV_limpeza(df_nutricionistas)
    df_resultados = outcomesCSV_limpeza(df_resultados, df_pacientes)
    df_final = integrar_dados(df_resultados, df_pacientes, df_nutricionistas, df_dieta)

    # adicionando bloco para criação do ficheiro da tabela com os dados limpos
    # retirando as colunas irrelevantes para a previsão do peso
    colunas_para_remover = ['program_id', 'patient_id', 'record_created_at']
    df_final = df_final.drop(columns=colunas_para_remover, errors='ignore')

    # transofromando as colunas de string como sexo, dieta e etc.. para colunas binárias
    df_pronto_para_ia = pd.get_dummies(df_final)

    # guardando o resultado num ficheiro csv
    df_pronto_para_ia.to_csv('data/dataset_limpo.csv', index=False)
    print("\n[SUCESSO] O dataset FINAL foi gerado!")
