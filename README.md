# Modelo com pyspark

Este projeto configura um ambiente de desenvolvimento baseado em Docker que inclui **Jupyter Notebook**, **PySpark** e o **Poetry** para gerenciar dependências Python. O projeto mapeia a pasta atual para o container, permitindo o desenvolvimento direto em notebooks ou scripts Python.

## Pré-requisitos

Antes de começar, certifique-se de ter o **Docker** e o **Docker Compose** instalados em sua máquina:

- [Instalar Docker](https://docs.docker.com/get-docker/)
- [Instalar Docker Compose](https://docs.docker.com/compose/install/)

## Para construir o ambiente de desenvolvimento

- `Dockerfile`: Define a imagem base com `jupyter/pyspark-notebook` e instala o Poetry.
- `docker-compose.yml`: Configura o serviço para rodar o container com mapeamento de volumes e expõe a porta do Jupyter.


## Passos para usar o ambiente

### 1. Clonar o repositório (ou copiar os arquivos para uma pasta)

```bash
git clone https://github.com/espeditoalves/sparklogimetrics.git
cd sparklogimetrics
```
### 2. Construir e iniciar o container

No diretório onde os arquivos Dockerfile e docker-compose.yml estão localizados, execute o comando:

```bash
docker-compose up --build
```

Isso irá:
- Baixar a imagem base jupyter/pyspark-notebook.
- Construir uma nova imagem com o Poetry instalado.
- Abrir o Jupyter Notebook ou Jupyter Lab na porta 8888.

### 3. Acessar o Jupyter Notebook

Após rodar o comando acima, o Jupyter Notebook estará acessível no navegador:

- Abra seu navegador e acesse: http://localhost:8888
- A chave/token do Jupyter estará visível nos logs do terminal onde o Docker Compose está rodando. Copie e cole no navegador para acessar o ambiente.

### 4. Instalar dependências Python com Poetry

Uma vez dentro do container, você pode usar o Poetry para gerenciar suas dependências:

- Abra um terminal no Jupyter ou conecte-se ao container usando docker exec.
```bash
# <nome-container> está definido dentro do docker-compose
docker exec -it <nome-container> /bin/bash
```
- Para criar um novo projeto Poetry:
```bash
poetry init
```

- Para instalar dependências a partir de um arquivo pyproject.toml:
```bash
poetry install
```

- Para ativar o ambiente virtual do Poetry:
```bash
poetry shell
```
### 5. Usar o kernel do poetry
Dentro do container do docker (shell) use o comando:
```bash
# Exemplo
python -m ipykernel install --user --name=sparklogimetrics --display-name "Python (Sparklogimetrics)"
```
### 5.1 Para usar no VSCode

* Use o link com token completo fornecido pelo docker, basta copiar o link com o token (**`http://127.0.0.1:8888/lab/?token=abc123def456`**) e colar na caixa `Existent Jupyter Server`.
* Clique em `Select Kernel` > `Select Another Kernel` > `Existent Jupyter Server`.
```bash
# Para obter o link e o token
docker-compose logs
```
### 5. Montagem de Volume

O diretório de trabalho padrão do container é **`/home/jovyan/work`**, que está mapeado para a pasta atual em sua máquina local. Assim, todos os arquivos na pasta local estarão disponíveis dentro do container e vice-versa.

### 6. Parar o container

Para parar o container, pressione CTRL + C no terminal onde o Docker Compose está rodando ou execute:

```bash
docker-compose down
```
Para iniciar o container novamente

```bash
docker-compose up
```
Se você quiser que os logs sejam exibidos no terminal, use apenas **`docker-compose up`**.

**Execução em segundo plano:** Se você preferir que os containers sejam executados em segundo plano (modo "detach"), você pode usar:

```bash
docker-compose up -d
```
### Personalização

Se desejar adicionar mais ferramentas ou dependências ao ambiente, você pode modificar o `Dockerfile` e incluir comandos adicionais para instalar bibliotecas ou pacotes.

### Troubleshooting
Se houver problemas com o Jupyter Notebook não iniciar, verifique os logs do Docker Compose.
Certifique-se de que a porta 8888 não está em uso por outro serviço em sua máquina.
