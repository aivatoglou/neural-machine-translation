FROM python:3.8

WORKDIR /application

COPY ./requirements.txt /application/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /application/requirements.txt

COPY . /application/

CMD ["uvicorn", "app.main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]