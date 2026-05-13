import pandas as pd
from sklearn.model_selection import train_test_split # ferramenta para dividir aleatoriamente quais dados serão teste e treino
from sklearn.ensemble import RandomForestRegressor # tem haver com o treino do modelo
from sklearn.metrics import mean_absolute_error


#criar uma função que separe a variavel que queremos prever das outras
def separar_x_y(df: pd.DataFrame):
    # 1. Isolar a variável alvo (y)
    y = df['weight_change_kg_6m']

    # 2. Criar a lista de colunas que o modelo NÃO deve ver
    colunas_para_ignorar = [
        'patient_id',
        'nutritionist_id',
        'diet_id',
        'program_id',
        'weight_change_kg_6m'
    ]

    # 3. Criar o X apagando as colunas ignoradas
    X = df.drop(columns=colunas_para_ignorar)

    # A função devolve as duas partes separadas
    return X, y


def preparar_dados_treino_teste(X: pd.DataFrame, y: pd.Series):  #separar os dados para treino e para teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #sentido da vida

    return X_train, X_test, y_train, y_test


#função para o modelo aprender
def treinar_modelo(X_train, y_train):
    print("A iniciar o treino do modelo...") #saber se a função iniciou

    #1 - dar um nome para o modelo
    modelo_r2d2 = RandomForestRegressor(random_state=42)
    # random_state pode ser qualquer numero, isso serve apenaas para que ele tome sempre as mesmas decisões aleaórias

    # 2 - ordem do que ele vai olhar e fazer
    #olhar para X_train e aprender a prever y_train
    modelo_r2d2.fit(X_train, y_train)

    print("Treino concluído com sucesso!")

    #devolve o modelo já treinado
    return modelo_r2d2

def avaliar_modelo(modelo_treinado, X_test, y_test):
    print("Avaliando modelo...")

    #o modelo aqui faz as previões do peso
    previsoes = modelo_treinado.predict(X_test)

    # aqui avaliamos as previsoes do modelo com os dados reais
    erro_medio = mean_absolute_error(y_test, previsoes)

    print(f"O modelo erra, em média, as previsões de peso em {erro_medio:.2f} kg.")

    return erro_medio


if __name__ == "__main__":
    # Assumindo que recebe o 'dataset_limpo' dos seus colegas
    # dataset_limpo = pd.read_csv("dados_limpos_pelos_colegas.csv")

    # 1. Separar alvo e características
    X, y = separar_x_y(dataset_limpo)

    # 2. Dividir em Treino e Teste
    X_train, X_test, y_train, y_test = preparar_dados_treino_teste(X, y)

    # 3 e 4. Treinar o Modelo
    modelo_final = treinar_modelo(X_train, y_train)

    # 5. Avaliar o Modelo
    erro = avaliar_modelo(modelo_final, X_test, y_test)