# Functional Specification

## For Hacker News Upvote Predictor

Version 0.1  
Prepared by maxitect  
15 April 2025

## Revision History

| Name     | Date       | Reason For Changes    | Version |
| -------- | ---------- | --------------------- | ------- |
| maxitect | 15/04/2025 | Initial specification | 0.1     |

## 1. Introduction

### 1.1 Document Purpose

This document provides a comprehensive overview of the functional requirements and core objectives for the Hacker News Upvote Predictor project, serving as a guide for implementation by the development team.

### 1.2 Product Scope

The Hacker News Upvote Predictor is a machine learning system designed to predict the number of upvotes a post on Hacker News will receive based on its title, domain, and posting time. The project aims to create an accurate prediction model while providing a learning experience for team members new to machine learning.

### 1.3 Definitions, Acronyms and Abbreviations

- **HN**: Hacker News
- **ML**: Machine Learning
- **MSE**: Mean Squared Error
- **API**: Application Programming Interface
- **W&B**: Weights & Biases (ML experimentation platform)
- **Word2Vec**: Word to Vector, a technique to convert words to vector representations
- **CBOW**: Continuous Bag of Words (a Word2Vec training method)
- **Skip-gram**: Another Word2Vec training method

### 1.4 References

- [Project Brief](BRIEF.md)
- [Data Analysis Report](REPORT.md)
- [Hacker News Website](https://news.ycombinator.com/)
- [Word2Vec Paper](https://arxiv.org/pdf/1301.3781.pdf)
- [Text8 Dataset](https://huggingface.co/datasets/afmck/text8)

### 1.5 Document Overview

This document begins with a product overview, followed by detailed requirements covering interfaces, functionality, quality of service, compliance, and design considerations. It concludes with verification approaches and appendices containing supplementary information.

## 2. Product Overview

### 2.1 Product Perspective

The Hacker News Upvote Predictor is a new product designed to serve as a learning platform for machine learning concepts. It operates as a standalone system with a database for data storage, a machine learning model for predictions, and an API interface for interaction.

### 2.2 Product Functions

The product will perform the following major functions:

- Connect to a PostgreSQL database to retrieve training data
- Process and tokenise Hacker News post titles
- Extract features from post domains and timing information
- Train word embeddings using Word2Vec
- Predict upvote scores using a neural network model
- Provide an API for making predictions and retrieving logs
- Log prediction requests and responses
- Store predictions in a database for future reference
- Integrate with Weights & Biases for experiment tracking and model versioning

### 2.3 Product Constraints

- Development timeline of 3 days
- Team is new to machine learning
- System must respond to prediction requests within 1 second
- Deployment on bare metal virtual machine server
- Container-based architecture required

### 2.4 User Characteristics

The system will be used exclusively by the development team, consisting of members who are new to machine learning and seeking to learn through hands-on implementation.

### 2.5 Assumptions and Dependencies

- Access to the PostgreSQL database as specified in the brief
- Availability of PyTorch libraries
- Virtual machine environment with Docker support
- Access to the text8 dataset for training word embeddings

### 2.6 Apportioning of Requirements

Must-have features:

- Title processing and word embedding using Word2Vec
- Domain-based features (domain reputation)
- Content-based features (title length, title content)
- Timing-based features (day of week, hour of posting)
- API endpoints as specified in the brief
- Prediction logging and storage
- Containerisation with proper network isolation
- Weights & Biases integration

Should-have features:

- Detailed error messages and codes
- Simple versioning system

Could-have features:

- Performance optimisation to meet the 1-second response target

Won't-have features:

- Custom monitoring dashboards
- Advanced model architectures
- Author-based features (as per data analysis findings)

## 3. Requirements

### 3.1 External Interfaces

#### 3.1.1 User Interfaces

The system will not have a graphical user interface. All interaction will be through the API endpoints.

#### 3.1.2 Hardware Interfaces

The system will be deployed on a bare metal virtual machine server with sufficient resources to run the containers and process prediction requests.

#### 3.1.3 Software Interfaces

The system will interact with:

- PostgreSQL database for training data and prediction storage
- Weights & Biases platform for experiment tracking and model versioning
- Docker for containerisation

### 3.2 Functional Requirements

#### 3.2.1 Data Retrieval and Preparation

- The system shall connect to the PostgreSQL database using the connection string provided
- The system shall extract and process the following features:
  - Title text (for Word2Vec processing)
  - Domain (extracted from URL without extension, with a special "no_domain" token for posts without URLs)
  - Time of day (extracted from timestamp)
  - Day of week (extracted from timestamp)
  - Title length (character count)

#### 3.2.2 Word Embedding Training

- The system shall implement Word2Vec (using either CBOW or Skip-gram) to train token embeddings
- The system shall use the text8 dataset for initial embedding training
- The system shall tokenise text by converting to lowercase, handling punctuation, and removing extra whitespace

#### 3.2.3 Prediction Model

- The system shall implement a feed-forward neural network model to predict upvote scores
- The network architecture shall consist of:
  - An embedding layer for domains (allowing the model to learn domain importance)
  - An input layer accepting concatenated features (word embeddings, domain embeddings, timing features, etc.)
  - 2-3 hidden layers with ReLU activations
  - A single output neuron with linear activation for the regression task
- The model shall use the following inputs:
  - Averaged word embeddings from the title
  - Domain (using a dedicated embedding layer, including handling for posts with no domain)
  - Time of day
  - Day of week
  - Title length
- The model shall output a single value representing the predicted upvote score
- The model shall be evaluated using Mean Squared Error (MSE)

#### 3.2.4 API Endpoints

The system shall implement the following API endpoints:

1. `GET /ping → str`

   - Returns "ok" to indicate system health

2. `GET /version → {"version": str}`

   - Returns the current model version using semantic versioning (e.g., "0.1.0")

3. `GET /logs → {"logs": [str]}`

   - Returns logged inference requests
   - Logs must include: Latency, Version, Timestamp, Input, Prediction

4. `POST /how_many_upvotes → {"author": str, "title": str, "timestamp": str} -> {"upvotes": number}`
   - Takes post information and returns predicted upvote count
   - Logs the request and its details
   - Stores the prediction in the database

#### 3.2.5 Logging and Storage

- The system shall log all prediction requests with the following details:
  - Latency (processing time in milliseconds)
  - Version (model version used)
  - Timestamp (when the request was processed)
  - Input (complete request data)
  - Prediction (output value)
  - Complete post details
- The system shall store predictions in a database for future reference

#### 3.2.6 Weights & Biases Integration

- The system shall integrate with Weights & Biases for:
  - Experiment tracking during model training
  - Logging metrics (particularly MSE)
  - Model versioning
  - Hyperparameter tracking
  - Collaborative result sharing

### 3.3 Quality of Service

#### 3.3.1 Performance

- The system shall respond to prediction requests within 1 second
- The system shall support concurrent requests from team members

#### 3.3.2 Security

- The database container shall be deployed on a private subnet
- The model container shall be deployed on a private subnet
- The API container shall be the only component with a public IP
- Components shall communicate via a shared Docker network

#### 3.3.3 Reliability

- The system shall provide specific error responses and status codes
- The system shall validate input data before processing

#### 3.3.4 Availability

- The system shall be available during the development and learning period

### 3.4 Compliance

No specific compliance requirements have been identified for this internal learning project.

### 3.5 Design and Implementation

#### 3.5.1 Installation

The system shall be deployed using Docker containers with the following components:

- Database container (private subnet)
- Model container (private subnet)
- API container (public IP)

#### 3.5.2 Distribution

The system will be deployed locally on a bare metal virtual machine server.

#### 3.5.3 Maintainability

- The system shall follow a modular design for ease of understanding
- Code shall be well-commented to facilitate learning
- The project shall follow the structure outlined in the brief

#### 3.5.4 Reusability

- The Word2Vec implementation shall be designed to be reusable
- Feature extraction components shall be modular

#### 3.5.5 Portability

The system shall use containerisation to ensure portability across environments.

#### 3.5.6 Cost

There are no specific cost constraints identified for this internal project.

#### 3.5.7 Deadline

The system shall be completed within 3 days of project initiation.

#### 3.5.8 Proof of Concept

The initial version shall serve as a proof of concept for the machine learning approach.

## 4. Verification

The system shall be verified through:

- Unit tests for individual components
- Integration tests for API endpoints
- Model evaluation using a separate test dataset (15% of available data)
- Performance testing to verify response time
- Comparison of MSE against baseline models

## 5. Appendixes

### Appendix A: Glossary

- **Upvote**: A positive vote given to a post on Hacker News
- **Word2Vec**: A technique for creating word embeddings that capture semantic relationships
- **Token**: A word or sub-word unit after text processing
- **Embedding**: A vector representation of a token in a continuous space
- **Regression**: A type of supervised learning where the output is a continuous value
- **MSE**: Mean Squared Error, a metric that measures the average squared difference between predicted and actual values
- **Docker**: A platform for developing, shipping, and running applications in containers
- **Container**: A lightweight, standalone package that includes everything needed to run a piece of software
- **Subnet**: A logical subdivision of an IP network
- **API**: Application Programming Interface, a set of rules that allow programs to communicate with each other
- **Weights & Biases**: A platform for tracking ML experiments

### Appendix B: Data Analysis Summary

Key findings from data analysis:

**Content Factors**

- Optimal title length: 25-75 characters
- Domain quality: Technical/educational domains perform best
- Content type: News domains slightly outperform non-news

**Timing Factors**

- Day of week: Weekend posts perform better
- Time of day: Noon/early afternoon posts perform best
- Year trends: Average scores have increased over time

**Author Factors** (not used in model)

- Author experience: Minimal impact on scores
- Karma levels: Weak correlation with scores
