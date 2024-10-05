FROM ubuntu:22.04

ENV TZ=UTC/GMT \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /usr/src/brad

RUN mkdir -p /usr/src/brad


RUN apt-get update


RUN apt-get install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa -y
RUN apt install python3.11 -y
RUN ln -sf python3.11 /usr/bin/python3 
RUN apt install python3-dev -y
RUN apt install python3-pip -y
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
RUN pip install .
RUN pip install langchain==0.3.1

RUN mkdir /logs

COPY ./bin/start.sh /usr/src/start.sh
RUN echo "HI"
CMD ["sh", "/usr/src/start.sh"]