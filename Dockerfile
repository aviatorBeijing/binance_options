# pull official base image
FROM python:3.11.3-slim-buster

# set work directory
WORKDIR /usr/src/binance_options

# set environment variables
ENV BINANCE_OPTIONS_DIR=/usr/src/binance_options/src
ENV PYTHONPATH=/usr/src/binance_options/src:$PYTHONPATH
ENV PYTHON=python

#ENV ES_USERNAME=
#ENV ES_PASSWORD=
#ENV PG_USERNAME=
#ENV PG_PASSWORD=


# install dependencies
RUN pip install --upgrade pip
COPY ./requirements.txt /usr/src/binance_options/requirements.txt

RUN apt-get update \
    && apt-get -y install libpq-dev gcc \
    && pip install psycopg2

RUN pip install -r requirements.txt

# copy project
COPY . /usr/src/binance_options/

cd src
./price_ranges_indicated_by_binance_options_market-makers.sh
