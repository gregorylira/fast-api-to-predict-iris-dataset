FROM python:3

WORKDIR /app
ADD . /app

RUN pip install -r requirements.txt


ENV PORT 8000


CMD ["gunicorn", "-w 4", "-k uvicorn.workers.UvicornWorker", "main:app", "--preload"]
