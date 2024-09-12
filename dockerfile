# Baixar a imagem oficial do PySpark com Jupyter Notebook
FROM jupyter/pyspark-notebook:spark-3.3.2

# Atualizar pacotes e instalar curl e sudo
USER root
RUN apt-get update && \
    apt-get install -y curl sudo && \
    apt-get clean

# Definir a senha para o usuário jovyan existente
RUN echo "jovyan:senhaSegura" | chpasswd

# Configurar sudo para pedir senha
RUN echo 'jovyan ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# Definir diretório de trabalho
WORKDIR /home/jovyan/work

# Copiar todos os arquivos do diretório atual para o diretório de trabalho
COPY . /home/jovyan/work

# Instalar o Poetry
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    echo "export PATH='/home/jovyan/.local/bin:$PATH'" >> /home/jovyan/.bashrc

# Atualizar o PATH para o Poetry no ambiente atual
ENV PATH="/home/jovyan/.local/bin:${PATH}"

# Instalar dependências com o Poetry
RUN poetry install --no-root

# Adicionar o kernel IPython ao Jupyter
RUN poetry add ipykernel && \
    poetry run python -m ipykernel install --user --name=image-spark-project-py3.10 --display-name "Python (Poetry)"

# Ajustar permissões para diretórios existentes
RUN mkdir -p /home/jovyan/.local/share/jupyter/runtime && \
    chown -R jovyan:users /home/jovyan/.local

# Trocar para o usuário jovyan
USER jovyan

# Expor a porta do Jupyter Notebook
EXPOSE 8888

# Comando padrão para iniciar o Jupyter Notebook
CMD ["start-notebook.sh"]
