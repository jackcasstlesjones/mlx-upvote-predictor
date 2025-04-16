# Backprop Bunch's Hacker News Upvote Predictor

This repository implements a machine learning system to predict upvotes for Hacker News posts. The solution leverages a Skip-gram Word2Vec model to generate word embeddings from raw text data and combines these embeddings with other post features (such as title, domain, and timing information) to produce upvote predictions via a feed-forward neural network built with PyTorch. The API is served with FastAPI, and the project is fully containerized to ensure consistent deployment.

---

## Project Overview

• Word2Vec Pipeline

- Implements a skip-gram model for word embedding generation.
- Includes modules for tokenization, vocabulary building (with subsampling), negative sampling, and training using PyTorch.

• Upvote Prediction Model

- Uses pre-trained embeddings alongside additional features (domain, title length, time features) to predict the upvote count for a Hacker News post.

• API Service

- Provides endpoints to check system health, retrieve model version, and submit post details for upvote prediction.

• Documentation

- Detailed design and architecture guidelines are provided in supporting files such as ADR.md, SPEC.md, REPORT.md, and STANDARDS.md.

---

## Directory Structure

Word2vec Pipeline:  
 word2vec_pipeline/  
 ├── src/ (Python modules for tokenization, vocabulary, dataset, training, and model)  
 ├── test_word_similarity.py  
 ├── train_config.yaml  
 └── train_text8.py

Project Root:  
 ├── .gitignore  
 ├── ADR.md  
 ├── environment.yml  
 ├── jacks_plan.md  
 ├── README.md  
 ├── REPORT.md  
 ├── requirements.txt  
 ├── SPEC.md  
 └── STANDARDS.md

---

## Installation

1. Clone the repository and change into the project directory.
2. Set up the environment:
   • For Conda users:
   - Run: conda env create -f environment.yml
   - Activate with: conda activate mlx-upvote-predictor
     • For Pip users:
   - Run: pip install -r requirements.txt

---

## Usage

Training the Word2Vec Model:

- Tokenization logic is in word2vec_pipeline/src/tokenize.py; tokens are processed line by line to conserve memory.
- The vocabulary is built in word2vec_pipeline/src/vocab.py, including token filtering and subsampling probabilities.
- To train the embeddings, run the script:
  python word2vec_pipeline/train_text8.py --config train_config.yaml
- Checkpoints are saved as specified in the configuration. Use word2vec_pipeline/test_word_similarity.py to evaluate embedding quality.

Running the Upvote Prediction API:

- The API has endpoints for health checks (/ping), model version information (/version), and prediction requests (/how_many_upvotes).
- To launch the API service, use a command similar to:
  uvicorn main:app --host 0.0.0.0 --port 8000
- Ensure the API entrypoint aligns with the specifications outlined in SPEC.md.

---

## Documentation & Additional Resources

• Architectural Decisions: See ADR.md  
• Project Standards: See STANDARDS.md  
• Data Analysis Report: See REPORT.md  
• Project Plan: See jacks_plan.md  
• Functional Specification: See SPEC.md

---

## Contributing

Contributions and improvements are encouraged. Please open an issue or submit a pull request with clear explanations and adhere to the established coding standards.

---

## License

Include your license information (e.g., MIT License) here or state the terms under which the project is made available.

---

## Contact

For additional questions or clarifications, contact the project maintainers.
