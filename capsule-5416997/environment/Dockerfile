# hash:sha256:44adf2d24cf24272656bd8a2cea0b2fc9a3de031215a3677db81b126432df80f
FROM registry.codeocean.com/codeocean/py-r:python3.10.12-R4.3.2-JupyterLab4.0.10-RStudiorstudio-server-2023.12.0-369-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN pip install -U --no-cache-dir \
    matplotlib==3.10.0 \
    numpy==1.26.4 \
    pandas==2.2.3 \
    rpy2==3.5.17 \
    seaborn \
    stargazer==0.0.7

RUN Rscript -e 'remotes::install_version("car", "3.1-3")' \
    && Rscript -e 'remotes::install_version("corrplot", "0.95")' \
    && Rscript -e 'remotes::install_version("covr", "3.6.4")' \
    && Rscript -e 'remotes::install_version("doParallel", "1.0.17")' \
    && Rscript -e 'remotes::install_version("fastDummies", "1.7.5")' \
    && Rscript -e 'remotes::install_version("feather", "0.3.5")' \
    && Rscript -e 'remotes::install_version("ggtext", "0.1.2")' \
    && Rscript -e 'remotes::install_version("janitor", "2.2.1")' \
    && Rscript -e 'remotes::install_version("lubridate", "1.9.4")' \
    && Rscript -e 'remotes::install_version("mockr", "0.2.1")' \
    && Rscript -e 'remotes::install_version("psych", "2.4.12")' \
    && Rscript -e 'remotes::install_version("randomizr", "1.0.0")' \
    && Rscript -e 'remotes::install_version("sandwich", "3.1-1")' \
    && Rscript -e 'remotes::install_version("stargazer", "5.2.3")' \
    && Rscript -e 'remotes::install_version("systemfonts", "1.2.1")' \
    && Rscript -e 'remotes::install_version("textshaping", "1.0.0")' \
    && Rscript -e 'remotes::install_version("tidyverse", "2.0.0")'

ADD "https://github.com/coder/code-server/releases/download/v4.95.3/code-server-4.95.3-linux-amd64.tar.gz" /.code-server/code-server.tar.gz
	
RUN cd /.code-server \
    && tar -xvf code-server.tar.gz \
    && rm code-server.tar.gz \
    && ln -s /.code-server/code-server-4.95.3-linux-amd64/bin/code-server  /usr/bin/code-server

RUN mkdir -p /.vscode/extensions \
    && code-server --extensions-dir="/.vscode/extensions" --install-extension REditorSupport.R \
    && code-server --extensions-dir="/.vscode/extensions" --install-extension continue.continue \
    && code-server --extensions-dir="/.vscode/extensions" --install-extension ms-python.python \
    && code-server --extensions-dir="/.vscode/extensions" --install-extension ms-toolsai.jupyter \
    && code-server --extensions-dir="/.vscode/extensions" --install-extension reageyao.bioSyntax \
    && code-server --extensions-dir="/.vscode/extensions" --install-extension saoudrizwan.claude-dev
