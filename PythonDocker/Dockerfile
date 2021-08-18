FROM python:3.8-slim-buster

WORKDIR /PythonDocker

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

EXPOSE 5000
COPY . .
ENTRYPOINT [ "python" ]
CMD [ "app.py" ]