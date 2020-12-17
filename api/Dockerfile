
FROM python:3.7

#update
RUN apt-get update

#install requirements
COPY ./requirements.txt /tmp/requirements.txt
WORKDIR /tmp
RUN pip3 install -r requirements.txt
RUN pip3 install pickle-mixin

#copy app
COPY . /api
WORKDIR /

CMD ["gunicorn", "-w", "3", "-b", ":1313", "-t", "360", "--reload", "api.wsgi:app"]