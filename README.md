- [SparkLogiMetrics](#sparklogimetrics)
  - [Execuntado o projeto com Container](#execuntado-o-projeto-com-container)
    - [Iniciando o docker nesse repositório:](#iniciando-o-docker-nesse-repositório)
    - [lista os containers:](#lista-os-containers)
    - [Iniciar o Contêiner Nomeando-o:](#iniciar-o-contêiner-nomeando-o)
    - [Dentro do Docker: Ambiente do jupyter notebook web](#dentro-do-docker-ambiente-do-jupyter-notebook-web)
  - [Verificar os Kernels Disponíveis:](#verificar-os-kernels-disponíveis)
  - [Instalador do poetry](#instalador-do-poetry)
  - [ignorar o pacote raiz:](#ignorar-o-pacote-raiz)
  - [Instalar o Kernel do Poetry:](#instalar-o-kernel-do-poetry)
  - [Ativar o Ambiente virtual do poetry](#ativar-o-ambiente-virtual-do-poetry)
    - [Adicionar o Kernel do Poetry ao Jupyter:](#adicionar-o-kernel-do-poetry-ao-jupyter)
    - [Verificar Diretórios de Kernel](#verificar-diretórios-de-kernel)
    - [Copiar o Kernel para o Diretório Correto](#copiar-o-kernel-para-o-diretório-correto)
    - [Reiniciar o Jupyter Notebook](#reiniciar-o-jupyter-notebook)
    - [Reiniciar o Contêiner Existente:](#reiniciar-o-contêiner-existente)
  - [Passo a Passo para Obter o Token do Jupyter e Usar no VSCode](#passo-a-passo-para-obter-o-token-do-jupyter-e-usar-no-vscode)
      - [1.0 Entre no Contêiner:](#10-entre-no-contêiner)
      - [2.0 Execute o Comando Python:](#20-execute-o-comando-python)
      - [3.0 Encontre a opção de selecionar Kernel:](#30-encontre-a-opção-de-selecionar-kernel)
      - [4.0 Inserir a URL do Jupyter com o Token:](#40-inserir-a-url-do-jupyter-com-o-token)
  - [Project Structure](#project-structure)

# SparkLogiMetrics

"Este projeto utiliza uma imagem pyspark dentro de um container e tem o objetivo de aplicar algumas técnicas de intervalo de confiança e testes de significância em uma base escorada por um modelo de regressão logística"


>Este projeto trabalha com uma estrutura de pastas padrão na qual tenho trabalhado com projetos de Ciência de Dados.
>
>Meu objetivo é manter uma estrutura padrão em todos os meus repositório.

## Execuntado o projeto com Container

### Iniciando o docker nesse repositório:

> INICIAR O DOCKER DESCKTOP MANUALMENTE

### lista os containers:
Use esse comando no seu terminal:
```bash 
docker ps
```

### Iniciar o Contêiner Nomeando-o:
```bash
# meu_container_base: É O nome que deu para o meu container, você pode escolher os eu
docker run -p 8888:8888 -v /caminho/local/do/seu/projeto:/home/jovyan/work --name meu_container_base jupyter/pyspark-notebook:spark-3.3.2
```
Fazendo esse processo acima eu não preciso apagar o conteiner em execução e quando eu abrir o computador eu simplesmente uso os comandos abaixo para iniciar o container

> **CASO NÃO TENHA UM CONTEINER EXISTENTE** - Pode iniciar um jupyter notebook com o comando abaixo

```bash
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

> Após esse processo irá surgir alguns logs com um link para abrir o jupyter notebook no navegador.


### Dentro do Docker: Ambiente do jupyter notebook web

No terminal do jupyter notebook web:

## Verificar os Kernels Disponíveis:
```bash
jupyter kernelspec list
```
## Instalador do poetry

[Poetry](https://python-poetry.org/docs/#installing-with-pipx)

## ignorar o pacote raiz:
Desativa o empacotamento do projeto, utilizando o Poetry apenas para gerenciamento de dependências

```bash
poetry install --no-root
```

## Instalar o Kernel do Poetry:
```bash
poetry add ipykernel
```

## Ativar o Ambiente virtual do poetry

```bash
poetry shell
```

### Adicionar o Kernel do Poetry ao Jupyter:

Após instalar o ipykernel no ambiente virtual do Poetry, você pode adicionar o kernel do Poetry ao Jupyter Notebook com o seguinte comando:

```bash
python -m ipykernel install --user --name=image-spark-project-py3.10 --display-name "Python (Poetry)"
```

* `--name=image-spark-project-py3.10`: Especifique o nome do ambiente virtual do Poetry que você deseja usar como base para o kernel.
* `--display-name "Python (Poetry)"`: Especifique o nome que deseja que apareça na lista de kernels do Jupyter Notebook.
Selecionar o Kernel do Poetry no Jupyter Notebook:
Depois de adicionar o kernel, você pode selecioná-lo ao criar um novo notebook ou alterar o kernel de um notebook existente:

Crie um novo notebook ou abra um notebook existente.
Vá para o menu **"Kernel"** e selecione **"Change Kernel"**.
Selecione **"Python (Poetry)"** ou o nome que você especificou ao adicionar o kernel do Poetry.

Agora, o notebook estará utilizando o kernel associado ao ambiente virtual do Poetry, garantindo que todas as dependências do seu projeto sejam utilizadas corretamente.

Esses passos devem ajudar a configurar e usar o kernel do Poetry no Jupyter Notebook dentro do seu ambiente Docker, garantindo que você esteja trabalhando com as dependências corretas do seu projeto.

### Verificar Diretórios de Kernel

O problema pode ser que o kernel foi instalado em um diretório diferente do que o Jupyter Notebook está verificando. Use os comandos abaixo para verificar os diretórios de kernel do Jupyter:

```bash
jupyter --paths
```
Isso exibirá algo como:

```bash
config:
    /home/jovyan/.jupyter
    /usr/local/etc/jupyter
    /etc/jupyter
data:
    /home/jovyan/.local/share/jupyter
    /usr/local/share/jupyter
    /usr/share/jupyter
runtime:
    /home/jovyan/.local/share/jupyter/runtime
```
Certifique-se de que o kernel está instalado em um dos diretórios data.

### Copiar o Kernel para o Diretório Correto

Se o kernel está em um diretório diferente, você pode copiá-lo para o local correto. 
Supondo que o kernel está em **`/home/jovyan/.local/share/jupyter/kernels/`**:

```bash
cp -r /home/jovyan/.local/share/jupyter/kernels/sparklogimetrics-py3.10 /home/jovyan/work/.venv/share/jupyter/kernels/
```
**`LEMBRA QUE VOCÊ ESTÁ USANDO UM CONTAINER E ESTÃO OLHE O SISTEMA DE PASTAS LINUX`**
Isso move o kernel para o diretório que o Jupyter está usando dentro do seu contêiner.

### Reiniciar o Jupyter Notebook
Após reinstalar ou mover o kernel, reinicie o servidor Jupyter Notebook. No terminal do seu contêiner:

```bash
jupyter notebook stop
jupyter notebook start
```
Ou, se você estiver usando o Docker diretamente:

```bash
docker restart sparklogimetrics
```

### Reiniciar o Contêiner Existente:
```bash
docker start meu_container_base
docker attach meu_container_base
```

- Será necessário abrir terminal dentro do container: 
```bash
docker exec -it meu_container_base bash
```
> **`OBSERVAÇÃO:`**
> 
> O comportamento que você está observando é comum ao usar o comando docker exec para interagir com um contêiner que já está executando o Jupyter Notebook ou JupyterLab. Quando você usa docker exec -it meu_container_base bash, você entra no terminal do contêiner, mas isso não tem influência direta sobre como o Jupyter está acessível ou exibido no navegador ou VSCode. O que ocorre é que, quando você inicia ou interage com o Jupyter, o link padrão pode abrir automaticamente em um navegador padrão do seu sistema, mas pode não se integrar ao VSCode diretamente. Vamos resolver isso para que você possa acessar o Jupyter diretamente no VSCode.
> 

## Passo a Passo para Obter o Token do Jupyter e Usar no VSCode

#### 1.0 Entre no Contêiner:
```bash
docker exec -it meu_container_base bash
```
#### 2.0 Execute o Comando Python:
```bash
jupyter notebook list
```
Isso mostrará uma saída semelhante a esta:
```bash
Currently running servers:
http://0.0.0.0:8888/?token=abc123def456 :: /home/jovyan
```

#### 3.0 Encontre a opção de selecionar Kernel:
Geralmente clico no canto superior direito do jupyter notebook, **`Select Kernel`**
#### 4.0 Inserir a URL do Jupyter com o Token:
```bash
http://127.0.0.1:8888/?token=abc123def456
```
Cole esta URL na caixa de diálogo e pressione Enter.


## Project Structure

```bash
.
├── config                      
│   ├── main.yaml                   # Main configuration file
│   ├── model                       # Configurations for training model
│   │   ├── model1.yaml             # First variation of parameters to train model
│   │   └── model2.yaml             # Second variation of parameters to train model
│   └── process                     # Configurations for processing data
│       ├── process1.yaml           # First variation of parameters to process data
│       └── process2.yaml           # Second variation of parameters to process data
├── data            
│   ├── final                       # data after training the model
│   ├── processed                   # data after processing
│   └── raw                         # raw data
├── docs                            # documentation for your project
├── .gitignore                      # ignore files that cannot commit to Git
├── Makefile                        # store useful commands to set up the environment
├── models                          # store models
├── notebooks                       # store notebooks
│   ├── exploration
│   │   └── .gitkeep
│   ├── modeling
│   │   └── .gitkeep
│   ├── preprocessing
│   │   └── .gitkeep
│   └── reporting
│       └── .gitkeep
├── output                          # store outputs
│   ├── figures
│   │   └── .gitkeep
│   ├── predictions
│   │   └── .gitkeep
│   └── reports
│       └── .gitkeep
├── .pre-commit-config.yaml         # configurations for pre-commit
├── pyproject.toml                  # dependencies for poetry
├── README.md                       # describe your project
├── src                             # store source code
│   ├── __init__.py                 # make src a Python module 
│   ├── process.py                  # process data before training model
│   ├── train_model.py              # train model
│   └── utils.py                    # store helper functions
└── tests                           # store tests
    ├── __init__.py                 # make tests a Python module 
    ├── test_process.py             # test functions for process.py
    └── test_train_model.py         # test functions for train_model.py
```