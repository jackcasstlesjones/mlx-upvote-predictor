# Project Overview

In this project, we will try to predict the upvote score of posts on the [Hacker News website](https://news.ycombinator.com/) using just their titles.

You are allowed (and recommended) to use PyTorch and its libraries.

## Suggested Recipe

### 1. Prepare the Dataset of Hacker News Titles and Upvote Scores

- Obtain the data from the database by connecting to it with the following connection string:  
  `postgres://sy91dhb:g5t49ao@178.156.142.230:5432/hd64m1ki`  
  Use `psql` or another tool to connect.

### 2. Tokenisation

To process natural language like Hacker News titles, we first need to perform **tokenisation**, which involves two main steps:

1. **Breaking up the text into tokens** (e.g. words or sub-words), typically involving:

   - Converting all text to lowercase
   - Applying stemming or lemmatisation (e.g. _"running"_ → _"run"_)
   - Replacing punctuation with special tokens (e.g. `<COMMA>` for `,`)
   - Removing extra whitespace

2. **Mapping each token to a unique token ID**, usually a positive integer.

This transforms text into numerical format, making it suitable input for neural networks.

### 3. Token Embeddings and Word2Vec

#### Token Embeddings

Now that we’ve tokenised our input, we need to convert each token into a **token embedding**, which is a vector of real numbers representing that token.

We do this for two key reasons:

- Neural networks require vector inputs — so our tokens need to be converted into numeric vectors.
- These vectors should capture **semantic** and **syntactic** meaning. To do that, each token must be represented with multiple features, which means high-dimensional vectors.

#### Word2Vec

To generate these embeddings, we use the method described in the [word2vec paper](https://arxiv.org/pdf/1301.3781.pdf), formally titled _Efficient Estimation of Word Representations in Vector Space_.

The process looks like this:

1. Use a tokenizer to convert text into token IDs.
2. Build a neural network with:
   - An embedding layer to map token IDs to vector representations
   - A projection and linear layer for prediction
3. Train this network on a chosen task (described below), updating the embedding weights.
4. Use the trained embedding layer to convert new tokens into embeddings.

#### Training Tasks

The paper outlines two tasks for training embeddings: **CBOW** and **Skip-gram**.

##### CBOW (Continuous Bag of Words)

- For each token, take _C_ tokens on either side (the context window).
- Use the context to predict the original token.

Steps:

- Convert context tokens to embeddings and average them.
- Feed the result through a linear layer + softmax to predict the original token.

##### Skip-gram

- For each token, take _C_ context tokens on either side.
- Use the token to predict each context token.

Steps:

- Convert the token to an embedding.
- Feed it through a linear layer + softmax to predict each token in the context window.

### 4. Implement and Train Word Embeddings

- Use either CBOW or Skip-gram to train token embeddings on the [text8 dataset](https://huggingface.co/datasets/ardMLX/text8).  
  This dataset is a cleaned-up extract of Wikipedia articles. More info [here](https://mattmahoney.net/dc/textdata.html).

### 5. Upvote Prediction

Once we have our token embeddings from the word2vec architecture, we move on to our final task — a regression task to predict upvote scores.

Steps:

- Take in a Hacker News title
- Convert it to token embeddings using the trained word2vec model
- Average the embeddings (average pooling)
- Pass the result through a series of hidden layers (your choice of widths and activations)
- Pass it through a final linear output layer with a single neuron — this produces the predicted upvote score
- Compare predictions to true scores using Mean Squared Error (MSE) loss

You’ll be feeding batches of titles and scores, so ensure your model supports vectorised PyTorch tensors.

Train the model on the prepared data and aim for the best predictive performance. Good luck!

### 6. Extension Ideas

- Train your word embeddings on a different dataset, such as:
  - More Hacker News content (e.g. comments)
  - A different corpus like (parts of) Wikipedia

---

## Project Standards

### Project Structure

Your project should follow this structure:

```bash
.
├── app              # Application Directory
│  ├── __init__.py   # Makes this a Python package
│  └── main.py       # FastAPI Application
├── Dockerfile       # Dockerfile for building the image
└── requirements.txt # Required Python packages
```

### Filenames

- `Dockerfile` must be named exactly like that unless you use the `-f` flag with `docker build`.
- `requirements.txt` is conventional and should list all required packages.
- `main.py` should define your FastAPI app and endpoints.

### FastAPI Endpoints

Your application must implement the following endpoints:

- `GET /ping → str`  
  Healthcheck endpoint, returns `"ok"`.

- `GET /version → {"version": str}`  
  Returns the current model version.

- `GET /logs → {"logs": [str]}`  
  Returns logged inference requests. Logs must be persistent and include:

  - Latency
  - Version
  - Timestamp
  - Input
  - Prediction

- `POST /how_many_upvotes → {"author": str, "title": str, "timestamp": str} -> {"upvotes": number}`  
  Performs inference, logs the request, and returns the predicted upvotes.

### Example Implementation

```python
import os
from fastapi import FastAPI

model_version = "0.1.0"
log_dir_path = "/var/log/app"
log_path = f"{log_dir_path}/V-{model_version}.log"

app = FastAPI()

@app.get("/ping")
def ping():
  return "ok"

@app.get("/version")
def version():
  return {"version": model_version}

@app.get("/logs")
def logs():
  return read_logs(log_path)

@app.post("/how_many_upvotes")
def how_many_upvotes(post):
  start_time = os.times().elapsed
  prediction = predict_upvotes(post)
  end_time = os.times().elapsed
  latency = (end_time - start_time) * 1000

  message = {}
  message["Latency"] = latency
  message["Version"] = model_version
  message["Timestamp"] = end_time
  message["Input"] = post.json()
  message["Prediction"] = prediction

  log_request(log_path, message.json())
  return {"upvotes": prediction}

def predict_upvotes(post):
  return 0

def log_request(log_path, message):
  pass

def read_logs(log_path):
  pass
```

Make sure to mount the `log_dir_path` directory in Docker to persist logs.

---

## Deployment Steps

The suggested workflow will consist of four main steps:

1. **Develop your FastAPI service** that provides inference for your model locally.
2. **Turn your application into a Docker image** and push it to [DockerHub](https://hub.docker.com/).
3. **Pull your image** from DockerHub — either on your local machine or the target inference server — and instantiate a container.
4. **Update deployments** by tearing down old containers and starting new ones whenever you publish a new image with your model.

Reference:  
![Project Outline](https://cdn.mlx.institute/assets/mlx4-week2-project-outline.avif)

---

In the following lessons of this week's Cortex module, we will delve into these steps and the technologies they use.
