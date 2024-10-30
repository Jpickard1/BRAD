FROM ubuntu:22.04

ENV TZ=UTC/GMT \
    DEBIAN_FRONTEND=noninteractive

USER root
SHELL ["/bin/bash", "-c"] 

WORKDIR /usr/src/brad

RUN mkdir -p /usr/src/brad
RUN mkdir -p /usr/src/uploads
RUN mkdir -p /usr/src/RAG_Database


RUN apt-get update

RUN apt-get install --reinstall ca-certificates -y

RUN apt-get install software-properties-common -y
RUN apt-get install curl -y
RUN apt-get install vim -y
RUN add-apt-repository 'ppa:deadsnakes/ppa' -y
RUN apt-get install python3.11 -y
RUN ln -sf python3.11 /usr/bin/python3 
RUN apt-get install python3-dev -y
RUN apt-get install python3-pip -y
# RUN python3 -m ensurepip
RUN pip3 install --no-cache --upgrade pip setuptools

RUN python3 --version

# set python environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# install dependencies
COPY ./requirements.txt .
RUN pip install -r requirements.txt --ignore-installed --no-cache-dir

# copy project
COPY . .

# Install Brad
RUN pip install /usr/src/brad/
RUN pip install langchain==0.3.1

RUN curl https://raw.githubusercontent.com/creationix/nvm/master/install.sh | bash \
    && source ~/.nvm/nvm.sh \
    && nvm install 20.18.0 \
    && nvm use 20.18.0 && npm install --prefix /usr/src/brad/brad-chat

RUN mkdir /logs

COPY ./bin/start.sh /usr/src/start.sh
RUN echo "HI"
CMD ["bash", "/usr/src/start.sh"]
