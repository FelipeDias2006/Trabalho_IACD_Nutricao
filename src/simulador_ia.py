import pandas as pd
import joblib


#funçoes para que o modelo não quebre no meio do processo
def pedir_numero_real(mensagem, minimo, maximo):
    while True:
        try:
            valor = float(input(mensagem))
            if minimo <= valor <= maximo:
                return valor
            else:
                print(f" Erro: O valor deve estar entre {minimo} e {maximo}.")
        except ValueError:
            print(" Erro: Por favor, digite apenas números.")


def pedir_opcao(mensagem, opcoes_validas):
    while True:
        valor = input(mensagem).strip().upper()
        if valor in opcoes_validas:
            return valor
        else:
            print(f" Erro: Por favor, digite uma das opções aceites: {opcoes_validas}")


# programa para utilização do usuário
# função principal para rodar o prgrama
def iniciar_simulador():
    print("=" * 60)
    print(" Modelo de previsão (versão r2d2.42) - NUTRIÇÃO ")
    print("=" * 60)
    print("Carregando modelo...\n")

    try:
        # leitura do aruqivo para uma variável
        modelo = joblib.load("models/modelo_floresta_final.pkl")
        colunas_necessarias = joblib.load("models/colunas_do_modelo.pkl")

        # parte dos inputs
        print("--- 1. DADOS DO PACIENTE ---")
        print("--- PERFIL FÍSICO ---")
        idade = pedir_numero_real("Idade (ex: 35): ", 18, 120)
        peso = pedir_numero_real("Peso Inicial em kg (ex: 85.5): ", 30, 300)
        altura = pedir_numero_real("Altura em cm (ex: 170): ", 100, 250)
        sexo = pedir_opcao("Sexo do paciente (M/F): ", ['M', 'F'])
        print("\n--- PERFIL PSICOLÓGICO ---")
        sono = pedir_numero_real("Horas de sono por noite (ex: 7.5): ", 0, 24)
        motivacao = pedir_numero_real("Nível de motivação inicial de 0.0 a 1.0 (ex: 0.8): ", 0.0, 1.0)
        adesao = pedir_numero_real("Taxa de adesão esperada à dieta em % (ex: 80): ", 0.0, 100.0)

        # menu com as dietas
        print("\n--- 2. TIPO DE PLANO NUTRICIONAL ---")
        print("1 - Balanced (Equilibrada)")
        print("2 - High Protein (Rica em Proteína)")
        print("3 - Low Carb (Baixa em Hidratos / Keto)")
        print("4 - Plant-Based (Vegetariana/Vegan)")
        escolha_dieta = pedir_opcao("Escolha o número da dieta (1 a 4): ", ['1', '2', '3', '4'])

        # criar a linha do paciente (ainda sem nutricionista)
        df_paciente_base = pd.DataFrame(0.0, index=[0], columns=colunas_necessarias)

        # preencher os dados comuns
        df_paciente_base.at[0, 'age'] = idade
        df_paciente_base.at[0, 'baseline_weight_kg'] = peso
        df_paciente_base.at[0, 'height_cm'] = altura
        df_paciente_base.at[0, 'baseline_bmi'] = peso / ((altura / 100) ** 2)
        df_paciente_base.at[0, 'sleep_hours'] = sono
        df_paciente_base.at[0, 'motivation_score'] = motivacao
        df_paciente_base.at[0, 'mean_adherence_pct'] = adesao

        if sexo == 'M' and 'sex_M' in colunas_necessarias:
            df_paciente_base.at[0, 'sex_M'] = 1.0
        elif sexo == 'F' and 'sex_F' in colunas_necessarias:
            df_paciente_base.at[0, 'sex_F'] = 1.0

        if escolha_dieta == '1' and 'diet_type_balanced' in colunas_necessarias:
            df_paciente_base.at[0, 'diet_type_balanced'] = 1.0
        elif escolha_dieta == '2' and 'diet_type_high_protein' in colunas_necessarias:
            df_paciente_base.at[0, 'diet_type_high_protein'] = 1.0
        elif escolha_dieta == '3' and 'diet_type_low_carb' in colunas_necessarias:
            df_paciente_base.at[0, 'diet_type_low_carb'] = 1.0
        elif escolha_dieta == '4' and 'diet_type_plant' in colunas_necessarias:
            df_paciente_base.at[0, 'diet_type_plant'] = 1.0

        # motor de recomendação
        #testar todos os nutricionistas
        print("\n" + "=" * 60)
        print("A simular o impacto de CADA nutricionista...")

        resultados = []
        colunas_nutri = [col for col in colunas_necessarias if col.startswith('nutritionist_id_')]

        for nutri in colunas_nutri:
            # clone do paciente
            df_simulacao = df_paciente_base.copy()

            # atribuímos este nutricionista ao clone
            df_simulacao.at[0, nutri] = 1.0

            # o modelo faz a previsão para esse nutricionista
            previsao = modelo.predict(df_simulacao)[0]

            # guardar o id do nutricionista e o resultado
            nome_limpo = nutri.replace('nutritionist_id_', '')
            resultados.append((nome_limpo, previsao))

        # ordenar oss resultados de forma a buscar o menor valor
        # ordenamos de formaa ddecrescente
        resultados.sort(key=lambda x: x[1])

        # printar o podio dos nutricionistas
        print("\n Recomendação do modelo ")
        print("Os melhores profissionais para as características deste paciente:")

        for i in range(3):
            nome, prev = resultados[i]
            print(f"  {i + 1}º Lugar: Nutricionista {nome} -> Previsão: {prev:.2f} kg")

        print("\n ALERTA DE INCOMPATIBILIDADE:")
        pior_nome, pior_prev = resultados[-1]  #o último da lista
        print(f"  Pior Cenário: Nutricionista {pior_nome} -> Previsão: {pior_prev:.2f} kg")
        print("=" * 60)

        #o pq o modelo fez essa previsão
        importancias = modelo.feature_importances_
        df_importancias = pd.DataFrame({
            'Fator': colunas_necessarias,
            'Importancia': importancias * 100
        }).sort_values(by='Importancia', ascending=False)

        print("\n Qual a característica mais relevante para a decisão do modelo:")
        print("O algoritmo baseou estas decisões nas seguintes 'Leis Globais':")
        for index, row in df_importancias.head(4).iterrows():
            if row['Importancia'] > 0:
                nome_formatado = row['Fator'].replace('_', ' ').title()
                print(f"  -> {nome_formatado}: {row['Importancia']:.1f}% de influência na fórmula")
        print("\n")

    except Exception as e:
        print(f"Erro Crítico: {e}")
        print("Verifica se os ficheiros '.pkl' estão gerados e na pasta correta.")


if __name__ == "__main__":
    iniciar_simulador()