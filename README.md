- [image\_spark\_docker\_file](#image_spark_docker_file)
  - [Como usar Dockerfile:](#como-usar-dockerfile)
  - [docker build](#docker-build)
  - [Rodando o Container](#rodando-o-container)
    - [Descrição dos Componentes](#descrição-dos-componentes)
    - [Resumo](#resumo)
  - [Passo a Passo para Obter o Token do Jupyter e Usar no VSCode](#passo-a-passo-para-obter-o-token-do-jupyter-e-usar-no-vscode)
    - [1.0 Use o link e token:](#10-use-o-link-e-token)
  - [Caso o poetry não funcione corretamente](#caso-o-poetry-não-funcione-corretamente)
  - [Comandos uteis](#comandos-uteis)
    - [Como acessar o shell do Container](#como-acessar-o-shell-do-container)
    - [Como sair do shell](#como-sair-do-shell)
    - [Como parar o Container:](#como-parar-o-container)
    - [Como voltar a usar o Container:](#como-voltar-a-usar-o-container)
    - [Como remover um container:](#como-remover-um-container)
  - [O que é docker-compose.yml?](#o-que-é-docker-composeyml)
  - [Como o docker-compose.yml se relaciona com o dockerfile?](#como-o-docker-composeyml-se-relaciona-com-o-dockerfile)
  - [Principais Comandos do Docker Compose](#principais-comandos-do-docker-compose)
    - [1. Iniciar serviços definidos no arquivo `docker-compose.yml`](#1-iniciar-serviços-definidos-no-arquivo-docker-composeyml)
    - [2. Iniciar os serviços em modo "detached" (em segundo plano)](#2-iniciar-os-serviços-em-modo-detached-em-segundo-plano)
    - [3. Parar os serviços](#3-parar-os-serviços)
    - [4. Remover containers, redes e volumes criados pelo `up`](#4-remover-containers-redes-e-volumes-criados-pelo-up)
    - [5. Remover containers, redes e volumes (incluindo volumes persistentes)](#5-remover-containers-redes-e-volumes-incluindo-volumes-persistentes)
    - [6. Verificar o status dos serviços](#6-verificar-o-status-dos-serviços)
    - [7. Executar um comando em um container rodando](#7-executar-um-comando-em-um-container-rodando)
    - [8. Ver os logs de todos os serviços](#8-ver-os-logs-de-todos-os-serviços)
    - [9. Ver os logs de um serviço específico](#9-ver-os-logs-de-um-serviço-específico)
    - [10. Recriar os serviços (sem cache)](#10-recriar-os-serviços-sem-cache)
    - [11. Escalar o número de containers de um serviço](#11-escalar-o-número-de-containers-de-um-serviço)
    - [12. Parar e remover containers, redes e volumes temporários criados pelo `up`](#12-parar-e-remover-containers-redes-e-volumes-temporários-criados-pelo-up)
    - [13. Ver as redes criadas pelo Docker Compose](#13-ver-as-redes-criadas-pelo-docker-compose)
    - [14. Listar volumes gerados](#14-listar-volumes-gerados)
    - [15. Executar um container sem precisar iniciar todos os serviços](#15-executar-um-container-sem-precisar-iniciar-todos-os-serviços)
- [Project Structure](#project-structure)

# image_spark_docker_file

Com o **`dockfiler`** presente nesse repositório você poderá criar a sua imagem de ambiente pyspark notebook para os estudos de ciências de dados.
Esse projeto cria uma nova imagem com algumas configurações pré estabelecidas como o **`poetry`** a partir de uma imagem base **`jupyter/pyspark-notebook:spark-3.3.2`**.

## Como usar Dockerfile:

1. Criar a imagem Docker:

Faça o clone desse repositório e use o comando:

```bash
docker build -t my-image-pyspark-notebook .
```
Componentes do Comando

1.1 **`Docker Build`**
* O  comando **`docker buil`** é o comando básico para construir uma nova imagem Docker a partir de um **`Dockerfile`**.

1.2 **`-t my-pyspark-notebook`**

A opção **`-t`** (ou **`--tag`**) é usada para atribuir uma tag (ou nome) à imagem Docker que está sendo criada. Neste caso, **`my-image-pyspark-notebook`** é o nome da imagem. A tag é útil para identificar e referenciar a imagem posteriormente.

1.3 **`.`**

Este é o contexto de construção, que indica onde o Docker deve procurar o **`Dockerfile`** e os arquivos necessários para a construção da imagem. O **`.`** representa o diretório atual. Assim, o Docker irá procurar por um arquivo chamado **`Dockerfile`** no diretório atual e usá-lo para construir a imagem.

1. Rodar o container:

``` bash
docker run -p 9090:8888 -v C:/Users/esped/Documents/Respositorio_git/Repositorio_projetos/image_spark_docker_file:/home/jovyan/work --name meu_container_pyspark  my-image-pyspark-notebook
```
Esse **Dockerfile** automatiza o processo de criação de um ambiente Jupyter com PySpark e Poetry, instalando o kernel e configurando o ambiente para rodar o notebook. Caso o container já exista, ele simplesmente reutiliza o kernel já configurado.

## docker build
Com o comando **`docker build`**, o Docker irá verificar se a **imagem base** (neste caso, `jupyter/pyspark-notebook:spark-3.3.2`) já está presente localmente. Se a imagem já estiver disponível no **cache local**, ela não será baixada novamente. No entanto, se a imagem não estiver localmente ou se você usar a flag **`--no-cache`**, o Docker irá baixar a imagem novamente.

Aqui estão algumas opções para otimizar o processo e evitar o download desnecessário da imagem:

* **`Verificar a presença da imagem`**
O Docker primeiro verifica se a imagem está no cache local antes de tentar baixar. Portanto, se você já baixou a imagem uma vez, ela será reutilizada em builds subsequentes.

* **`Uso de cache:`**
Por padrão, o Docker usa o cache para etapas que já foram executadas antes. O docker build reutiliza camadas previamente construídas sempre que possível.

* **`Evitar --no-cache:`** 
Não utilize a opção `--no-cache` a menos que você realmente precise de uma build limpa, pois isso forçará o Docker a ignorar o cache e baixar a imagem novamente.

Se quiser garantir que o build só baixe a imagem se não estiver presente, o comportamento padrão do Docker já lida bem com isso.


## Rodando o Container

Após a criação da imagem, uso o comando abaixo para criar um container:

```bash
docker run -p <porta_host>:<porta_container> -v <diretorio_host>:<diretorio_container> --name <nome_container> <nome_imagem>
```
Exemplo
```bash
docker run -p 9090:8888 -v C:/Users/esped/Documents/Respositorio_git/Repositorio_projetos/image_spark_docker_file:/home/jovyan/work --name meu_container_pyspark  my-image-pyspark-notebook
```
### Descrição dos Componentes

1. **`docker run:`**

* Esse é o comando usado para criar e iniciar um novo container a partir de uma imagem Docker.

2. **`-p 9090:8888:`**

* Mapeia a porta **`8888`** do container para a porta **`9090`** no host. Isso significa que você pode acessar o Jupyter Notebook no navegador através da URL **`http://127.0.0.1:9090/lab`**.
  
3. **`-v /caminho/local/do/seu/projeto:/home/jovyan/work:`**

Esse parâmetro monta um volume, o que significa que ele faz com que um diretório do seu sistema local (**`/caminho/local/do/seu/projeto`**) esteja disponível dentro do container em um diretório específico (**`/home/jovyan/work`**). Isso permite que você compartilhe arquivos entre o seu sistema local e o container.

4. **`--name meu_container_pyspark:`**

Esse parâmetro atribui um nome ao container (**`meu_container_pyspark`**). Isso facilita a referência ao container em comandos futuros, como **`docker start`**, **`docker stop`**, e **`docker rm`**.

5. **`my-pyspark-notebook:`**

Esse é o nome da imagem Docker que você deseja usar para criar o container. O Docker irá procurar por uma imagem com esse nome e, se não encontrar, tentará baixá-la do registro Docker (Docker Hub) se estiver disponível.

### Resumo
Você está criando um container a partir da imagem **`my-pyspark-notebook`**, mapeando a porta do container para a porta do host para acessar o Jupyter Notebook e montando um volume para compartilhar arquivos. O nome do container é definido como **`meu_container_pyspark`**, o que facilita sua gestão.

Se você está acessando o Jupyter Notebook na URL **`http://127.0.0.1:9090/lab`**, isso indica que a porta do host foi corretamente mapeada para a porta do container, e tudo está funcionando conforme o esperado.

## Passo a Passo para Obter o Token do Jupyter e Usar no VSCode

### 1.0 Use o link e token:

* Use o link com token completo fornecido pelo docker, basta copiar o link com o token (**`http://127.0.0.1:9090/lab/?token=abc123def456`**) e colar na caixa `Existent Jupyter Server`.
* Clique em `Select Kernel` > `Select Another Kernel` > `Existent Jupyter Server`.

## Caso o poetry não funcione corretamente
Abra o shell do container e use o comando
```bash
# Exemplo
python -m ipykernel install --user --name=<container> --display-name "Python (nome_que_desejar)"
```

```bash
# Exemplo
python -m ipykernel install --user --name=sparklogimetrics --display-name "Python (Sparklogimetrics)"
```

## Comandos uteis

### Como acessar o shell do Container
```bash
docker exec -it <nome-container> bash
```
### Como sair do shell

```bash
exit
```

### Como parar o Container:
```bash
docker stop <nome-container>
```

### Como voltar a usar o Container:
```bash
docker start <nome-container>
```

### Como remover um container:
```bash
docker rm <nome-container>
```

## O que é docker-compose.yml?
O arquivo **`docker-compose.yml`** é usado pelo **Docker Compose** para definir e gerenciar aplicativos que utilizam múltiplos containers de forma declarativa. Com ele, você pode descrever a infraestrutura do seu aplicativo, especificando os serviços, redes e volumes necessários para a execução do mesmo.

## Como o docker-compose.yml se relaciona com o dockerfile?
Você pode criar uma **imagem personalizada** usando um **`Dockerfile`** e, em seguida, usar o **`Docker Compose`** para orquestrar containers que utilizam essa **imagem personalizada** junto com outras imagens disponíveis na internet.
## Principais Comandos do Docker Compose

### 1. Iniciar serviços definidos no arquivo `docker-compose.yml`
```bash
docker-compose up
```

### 2. Iniciar os serviços em modo "detached" (em segundo plano)
```bash
docker-compose up -d
```

### 3. Parar os serviços
```bash
docker-compose stop
```

### 4. Remover containers, redes e volumes criados pelo `up`
```bash
docker-compose down
```

### 5. Remover containers, redes e volumes (incluindo volumes persistentes)
```bash
docker-compose down -v
```

### 6. Verificar o status dos serviços
```bash
docker-compose ps
```

### 7. Executar um comando em um container rodando
```bash
docker-compose exec <nome_servico> <comando>
```

### 8. Ver os logs de todos os serviços
```bash
docker-compose logs
```

### 9. Ver os logs de um serviço específico
```bash
docker-compose logs <nome_servico>
```

### 10. Recriar os serviços (sem cache)
```bash
docker-compose up --build --no-cache
```

### 11. Escalar o número de containers de um serviço
```bash
docker-compose up --scale <nome_servico>=<número>
```

### 12. Parar e remover containers, redes e volumes temporários criados pelo `up`
```bash
docker-compose down --rmi all
```

### 13. Ver as redes criadas pelo Docker Compose
```bash
docker network ls
```

### 14. Listar volumes gerados
```bash
docker volume ls
```

### 15. Executar um container sem precisar iniciar todos os serviços
```bash
docker-compose run <nome_servico> <comando>
```

# Project Structure

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