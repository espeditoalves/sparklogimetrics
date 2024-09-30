# Dockerfile

# Baixar a imagem oficial do PySpark com Jupyter Notebook
FROM jupyter/pyspark-notebook:spark-3.4.1

# Instalar o Poetry
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    export PATH="$HOME/.local/bin:$PATH" && \
    poetry --version

# Adicionar o Poetry ao PATH permanentemente
ENV PATH="/home/jovyan/.local/bin:$PATH"

# Definir o diretório de trabalho como a pasta padrão do usuário (work directory)
WORKDIR /home/jovyan/work

# Expor a porta padrão do Jupyter Notebook
EXPOSE 8888
