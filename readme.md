# Neural Machine Translation 

This is a Neural Machine Translation application using PyTorch, EasyNMT, FastAPI, MongoDB and Docker.

The implementation focuses on the deployment using FastAPI for the RESTful services, 
MongoDB (Mongo Atlas) acting as the database and the Docker framework for the containerization of the application.

For the translation services, the [EasyNMT](https://github.com/UKPLab/EasyNMT) package was used and a solution based
on RNN neural networks from PyTorch tutorials on seq2seq networks.

The data used to train the seq2seq were downloaded from [ManyThings.org](https://www.manythings.org/anki/).

## Install

First of all, create a MongoDB account and edit the core/config.py with your Mongo cloud credentials.
Then, build the container using the intstructions.

```console
docker build -t app .
docker run -d --name app -p 8000:8000 {IMAGEID}
```

## Usage

The FastAPI Swagger UI is located in localhost:8000/docs.
Each request is protected through OAuth2 (so you need to create a user in order to interact with the APIs).
