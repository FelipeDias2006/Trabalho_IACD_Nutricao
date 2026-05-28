# Disciplina: Elementos de Inteligência Artificial e Ciência de Dados (2025/26)

## Instituição
**Universidade da Beira Interior - Departamento de Informática**

---

## Equipe

- Pedro Ramos - 57188  
- Guilherme Lima - 57190  
- Felipe Dias - 56793  

---

##  Sobre o Projeto

Este repositório contém a implementação do Trabalho Prático da unidade curricular de IACD. O objetivo principal é analisar dados clínicos e comportamentais para prever o sucesso de planos nutricionais, recorrendo a técnicas de Machine Learning.

O projeto está dividido em quatro fases fundamentais:

### Integração e Limpeza de Dados
Tratamento de dados em falta, codificação de variáveis categóricas (One-Hot Encoding) e remoção de outliers.

### Análise e Padrões (Clustering)
Descoberta de perfis ocultos ("Grupos" de pacientes) utilizando o algoritmo K-Means.

### Previsão (Regressão)
Modelação da perda de peso esperada através do algoritmo Random Forest Regressor.

### Simulador Clínico
Um sistema de recomendação interativo capaz de prever os resultados de novos pacientes em tempo real e recomendar o nutricionista ideal.

---

## Estrutura do Repositório

A arquitetura do projeto foi desenhada para garantir modularidade e reprodutibilidade científica:

### `data/`
Contém os 4 ficheiros CSV originais fornecidos e o `dataset_limpo.csv` gerado após o processamento.

### `models/`
Armazena o cérebro da Inteligencia Artificial exportado (`.pkl`) após o treino, bem como o mapeamento das colunas.

### `src/`
Diretório com o código-fonte desenvolvido pelo grupo.

#### `organizacoes.py`
Script responsável pela integração, limpeza de dados e exportação do dataset final.

#### `padroes.py`
Implementação do K-Means para segmentação comportamental e geração de perfis médios.

#### `modelos.py`
Pipeline de treino, afinação (GridSearchCV), teste (divisão 80/20) e avaliação do RandomForest.

#### `simulador_ia.py`
Interface de linha de comandos interativa com sistema de validação de inputs e motor de recomendação clínica.

---

## Como Executar o Projeto

Para reproduzir os resultados do nosso estudo, os scripts devem ser executados pela seguinte ordem lógica:

### Passo 1: Preparação dos Dados
Executar a rotina de limpeza para processar os dados brutos e gerar o ficheiro `dataset_limpo.csv` na pasta `data/`.

```bash
python src/organizacoes.py

### Passo 2: Descoberta de Padrões
Identificar as tribos de pacientes. Este script irá gerar o relatório relatorio_tribos.csv para posterior análise.

```bash
python src/padroes.py

### Passo 3: Treino do Modelo Preditivo
Afinar e avaliar o Random Forest. No final da execução, o modelo será guardado na pasta models/ para utilização futura.

```bash
python src/modelos.py

### Passo 4: Simulador Interativo
Para testar a capacidade de generalização do modelo e obter o ranking dos melhores nutricionistas para um novo perfil de paciente, inicie o simulador interativo:

```bash
python src/simulador_ia.py


## Principais Descobertas e Transparência

Durante o desenvolvimento, priorizámos a interpretabilidade dos resultados (Glass Box). O nosso modelo identificou que o sucesso de um plano dietético depende mais fortemente da Taxa de Adesão e de fatores inerentes ao paciente (como a Idade e o Sono) do que do tipo específico de dieta prescrita.
