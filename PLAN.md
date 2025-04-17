# Implementation Plan

## For Hacker News Upvote Predictor

Version 0.1  
Prepared by maxitect
MLX Institute  
April 16, 2025

## Revision History

| Name     | Date       | Reason For Changes | Version |
| -------- | ---------- | ------------------ | ------- |
| maxitect | 16/04/2025 | Initial draft      | 0.1     |

## Implementation Tracking

### Tracking Mechanism

- Task ID format: [Section].[Subsection].[Task] (e.g., 1.2.3)
- Status tracking (Not Started, In Progress, Completed)
- Dependency tracking to ensure proper build sequence
- Estimated implementation time: ≤1 hour per prompt
- Review time: ≤15 minutes per task

### Completion Criteria

- [ ] All API endpoints implemented and functional
- [ ] Word2Vec embeddings trained and evaluated
- [ ] Prediction model trained and achieving MSE < 20
- [ ] Docker container builds and runs successfully
- [ ] Response time under 1 second for predictions
- [ ] All logs persisted and retrievable
- [ ] Documentation complete and accurate

### Continuous Improvement

- Regular code reviews after each component completion
- Periodic performance evaluation of ML models
- Security scanning of Docker configurations
- Test coverage monitoring
- Dependency updates as needed

## Implementation Plan

### 1. Project Setup

**Objective:** Establish basic project structure and environment

#### 1.1 Basic Project Structure

- [ ] Task 1.1.1 - Create minimal project skeleton

  - Depends on: None
  - Creates: Basic directory structure with app folder, **init**.py, and placeholder main.py

- [ ] Task 1.1.2 - Create requirements.txt

  - Depends on: 1.1.1
  - Creates: Basic requirements file with core dependencies (PyTorch, FastAPI, psycopg)

- [ ] Task 1.1.3 - Create simple .gitignore
  - Depends on: 1.1.1
  - Creates: Basic .gitignore for Python project (\*.pyc, **pycache**, etc.)

#### 1.2 Configuration Management

- [ ] Task 1.2.1 - Create basic config module

  - Depends on: 1.1.1
  - Creates: app/config.py with basic configuration loading

- [ ] Task 1.2.2 - Add database configuration
  - Depends on: 1.2.1
  - Adds: Database connection configuration to config.py

### 2. Database Connectivity

**Objective:** Implement basic database connection and data extraction

#### 2.1 Database Connection

- [ ] Task 2.1.1 - Create database connection module

  - Depends on: 1.2.2
  - Creates: app/utils/db_connection.py with basic connection function

- [ ] Task 2.1.2 - Add connection pooling
  - Depends on: 2.1.1
  - Enhances: Database connection with proper pooling

#### 2.2 Basic Data Extraction

- [ ] Task 2.2.1 - Create function to extract post titles and scores

  - Depends on: 2.1.2
  - Creates: app/utils/data_extraction.py with function to extract basic post data

- [ ] Task 2.2.2 - Add function to extract post metadata

  - Depends on: 2.2.1
  - Adds: Function to extract additional post metadata (author, timestamp, URL)

- [ ] Task 2.2.3 - Create function to split data into train/val/test sets
  - Depends on: 2.2.2
  - Creates: Function to organise data for model training and evaluation

### 3. Text Processing

**Objective:** Implement basic text processing for titles

#### 3.1 Basic Text Cleaning

- [ ] Task 3.1.1 - Create text cleaning module

  - Depends on: 1.1.1
  - Creates: app/utils/text_processing.py with basic text cleaning functions

- [ ] Task 3.1.2 - Add title preprocessing
  - Depends on: 3.1.1
  - Adds: Functions for title-specific preprocessing

#### 3.2 Tokenisation

- [ ] Task 3.2.1 - Create basic tokeniser

  - Depends on: 3.1.2
  - Creates: app/utils/tokeniser.py with functions to split text into tokens

- [ ] Task 3.2.2 - Add vocabulary builder
  - Depends on: 3.2.1
  - Adds: Functions to build vocabulary from tokens

### 4. Word2Vec Model

**Objective:** Implement Word2Vec for word embeddings

#### 4.1 Word2Vec Architecture

- [ ] Task 4.1.1 - Create Word2Vec model skeleton

  - Depends on: 3.2.2
  - Creates: app/models/word2vec.py with basic model class

- [ ] Task 4.1.2 - Implement embedding layer

  - Depends on: 4.1.1
  - Adds: Embedding layer implementation to Word2Vec model

- [ ] Task 4.1.3 - Implement Skip-gram architecture
  - Depends on: 4.1.2
  - Adds: Complete Skip-gram model implementation

#### 4.2 Training Utilities

- [ ] Task 4.2.1 - Create context window extraction

  - Depends on: 3.2.2
  - Creates: Functions to extract context windows for Skip-gram training

- [ ] Task 4.2.2 - Add data preparation for Word2Vec

  - Depends on: 4.2.1
  - Adds: Functions to prepare training data for Word2Vec

- [ ] Task 4.2.3 - Implement basic training loop
  - Depends on: 4.1.3, 4.2.2
  - Creates: Basic training loop for Word2Vec model

### 5. Feature Engineering

**Objective:** Extract features for prediction model

#### 5.1 Title Features

- [ ] Task 5.1.1 - Create title length feature

  - Depends on: 3.1.2
  - Creates: app/features/title_features.py with function to extract title length

- [ ] Task 5.1.2 - Implement title embedding averaging
  - Depends on: 4.2.3
  - Adds: Function to convert title to embedding by averaging word embeddings

#### 5.2 Domain Features

- [ ] Task 5.2.1 - Create domain extraction

  - Depends on: 2.2.2
  - Creates: app/features/domain_features.py with function to extract domains from URLs

- [ ] Task 5.2.2 - Implement domain encoding
  - Depends on: 5.2.1
  - Adds: Function to encode domains as features

#### 5.3 Time Features

- [ ] Task 5.3.1 - Create time feature extraction

  - Depends on: 2.2.2
  - Creates: app/features/time_features.py with functions to extract day and hour

- [ ] Task 5.3.2 - Implement cyclical encoding for time
  - Depends on: 5.3.1
  - Adds: Functions to encode time features cyclically (sin/cos)

### 6. Prediction Model

**Objective:** Implement neural network for upvote prediction

#### 6.1 Model Architecture

- [ ] Task 6.1.1 - Create prediction model skeleton

  - Depends on: 5.1.2, 5.2.2, 5.3.2
  - Creates: app/models/predictor.py with basic model class

- [ ] Task 6.1.2 - Implement input processing

  - Depends on: 6.1.1
  - Adds: Functions to process and combine input features

- [ ] Task 6.1.3 - Add neural network layers
  - Depends on: 6.1.2
  - Adds: Hidden layers and output layer to the model

#### 6.2 Training

- [ ] Task 6.2.1 - Create loss function and optimiser

  - Depends on: 6.1.3
  - Adds: MSE loss function and optimiser setup

- [ ] Task 6.2.2 - Implement basic training loop

  - Depends on: 6.2.1
  - Creates: Basic training loop for prediction model

- [ ] Task 6.2.3 - Add model evaluation
  - Depends on: 6.2.2
  - Adds: Functions to evaluate model on validation data

#### 6.3 Model Persistence

- [ ] Task 6.3.1 - Create model saving and loading

  - Depends on: 6.1.3
  - Adds: Functions to save and load model state

- [ ] Task 6.3.2 - Implement version tracking
  - Depends on: 6.3.1
  - Creates: app/utils/versioning.py for model version tracking

### 7. API Implementation

**Objective:** Create FastAPI application with required endpoints

#### 7.1 Basic API Setup

- [ ] Task 7.1.1 - Create FastAPI application skeleton

  - Depends on: 1.1.1
  - Creates: Basic FastAPI app in app/main.py

- [ ] Task 7.1.2 - Add request/response models
  - Depends on: 7.1.1
  - Creates: app/schemas.py with Pydantic models

#### 7.2 Basic Endpoints

- [ ] Task 7.2.1 - Implement /ping endpoint

  - Depends on: 7.1.1
  - Adds: Simple health check endpoint

- [ ] Task 7.2.2 - Implement /version endpoint
  - Depends on: 6.3.2, 7.1.1
  - Adds: Endpoint to return model version

#### 7.3 Prediction Endpoint

- [ ] Task 7.3.1 - Create prediction service

  - Depends on: 6.3.1
  - Creates: app/services/prediction.py with prediction logic

- [ ] Task 7.3.2 - Implement /how_many_upvotes endpoint
  - Depends on: 7.1.2, 7.3.1
  - Adds: Endpoint for upvote prediction

### 8. Logging System

**Objective:** Implement request and prediction logging

#### 8.1 Logging Setup

- [ ] Task 8.1.1 - Create basic logging configuration

  - Depends on: 1.2.1
  - Creates: app/utils/logging_utils.py with basic setup

- [ ] Task 8.1.2 - Implement file logging
  - Depends on: 8.1.1
  - Adds: Configuration for logging to files

#### 8.2 Request Logging

- [ ] Task 8.2.1 - Create request logging middleware

  - Depends on: 7.1.1, 8.1.2
  - Adds: Middleware to log API requests

- [ ] Task 8.2.2 - Implement prediction logging
  - Depends on: 7.3.2, 8.1.2
  - Adds: Detailed logging for predictions

#### 8.3 Log Retrieval

- [ ] Task 8.3.1 - Create log reading function

  - Depends on: 8.2.2
  - Adds: Function to read and parse log files

- [ ] Task 8.3.2 - Implement /logs endpoint
  - Depends on: 7.1.1, 8.3.1
  - Adds: Endpoint to retrieve logs

### 9. Containerisation

**Objective:** Create Docker configuration for deployment

#### 9.1 Dockerfile

- [ ] Task 9.1.1 - Create basic Dockerfile

  - Depends on: 1.1.2
  - Creates: Basic Dockerfile for the application

- [ ] Task 9.1.2 - Add multi-stage build
  - Depends on: 9.1.1
  - Enhances: Dockerfile with multi-stage build for optimisation

#### 9.2 Docker Compose

- [ ] Task 9.2.1 - Create basic docker-compose.yml

  - Depends on: 9.1.2
  - Creates: Basic Docker Compose configuration

- [ ] Task 9.2.2 - Add volume configuration for logs
  - Depends on: 9.2.1
  - Adds: Volume mount for persistent logs

### 10. Integration and Documentation

**Objective:** Finalise integration and documentation

#### 10.1 Training Scripts

- [ ] Task 10.1.1 - Create Word2Vec training script

  - Depends on: 4.2.3
  - Creates: scripts/train_word2vec.py for word embedding training

- [ ] Task 10.1.2 - Create prediction model training script
  - Depends on: 6.2.3
  - Creates: scripts/train_predictor.py for model training

#### 10.2 Documentation

- [ ] Task 10.2.1 - Create comprehensive README

  - Depends on: All previous tasks
  - Creates: Detailed README.md with project documentation

- [ ] Task 10.2.2 - Add API documentation
  - Depends on: 7.2.1, 7.2.2, 7.3.2, 8.3.2
  - Creates: API documentation with examples

## Implementation Prompts

### Project Setup

#### Task 1.1.1: Create minimal project skeleton

```
Create the minimal project skeleton for the Hacker News Upvote Predictor. Set up:

1. Root directory
2. app/ directory
3. app/__init__.py (empty file)
4. app/main.py (with a placeholder comment)
5. Empty Dockerfile

This is just the basic structure to get started. We'll fill in these files in later steps.
```

#### Task 1.1.2: Create requirements.txt

```
Create a requirements.txt file with the core dependencies for the project:

pytorch>=2.1.0
fastapi>=0.104.0
uvicorn>=0.23.2
psycopg>=3.1.12
pydantic>=2.4.2
python-dotenv>=1.0.0
numpy>=1.24.0
wandb>=0.15.12

Include just these essential packages for now. We'll add more specific dependencies as needed.
```

#### Task 1.1.3: Create simple .gitignore

```
Create a .gitignore file appropriate for this Python project. Include patterns for:

1. Python bytecode and cache files
2. Virtual environments
3. Editor/IDE specific files
4. Local configuration files
5. Model checkpoints and large data files
6. Log files

Keep it focused on the essentials for this project.
```

#### Task 1.2.1: Create basic config module

```
Create a basic config.py module in the app directory that:

1. Imports os and dotenv
2. Has a function to load environment variables
3. Defines default configuration values
4. Has a get_config() function that returns a config dictionary

This should be a minimal implementation focused on loading basic configuration.
```

#### Task 1.2.2: Add database configuration

```
Enhance the config.py module to include database configuration:

1. Add the default database URL from BRIEF.md
2. Handle database connection parameters (host, port, user, password, dbname)
3. Add validation for database configuration
4. Provide a get_db_config() function

Ensure the connection string from BRIEF.md is used as the default.
```

### Database Connectivity

#### Task 2.1.1: Create database connection module

```
Create a db_connection.py module in app/utils/ that:

1. Imports the config module
2. Implements a simple function to create a connection to the PostgreSQL database
3. Has a function to test if the connection works
4. Includes basic error handling

Focus on establishing a basic connection using the configuration from config.py.
```

#### Task 2.1.2: Add connection pooling

```
Enhance the db_connection.py module to add connection pooling:

1. Create a connection pool instead of individual connections
2. Add a function to get a connection from the pool
3. Implement a context manager for safe connection handling
4. Add a function to close the pool properly on shutdown

Keep the implementation focused on proper resource management.
```

#### Task 2.2.1: Create function to extract post titles and scores

```
Create a data_extraction.py module in app/utils/ that:

1. Imports the db_connection module
2. Implements a function to extract post titles and scores from the database
3. Includes parameter validation and SQL injection prevention
4. Has basic error handling
5. Returns results in a simple dictionary format

Focus on safely extracting the core data needed for the model.
```

#### Task 2.2.2: Add function to extract post metadata

```
Enhance the data_extraction.py module to add metadata extraction:

1. Create a function to extract post URLs, authors, and timestamps
2. Add a function to join basic data with metadata
3. Include proper error handling for missing fields
4. Return results in a consistent format

Ensure all the data needed for feature engineering is extracted properly.
```

#### Task 2.2.3: Create function to split data

```
Create a data_utils.py module in app/utils/ that:

1. Imports necessary utilities
2. Implements a function to split data into training, validation, and test sets
3. Uses a 70%/15%/15% split ratio
4. Ensures consistent results with a fixed random seed
5. Returns the split datasets in a suitable format

Keep the implementation simple while ensuring proper data organisation.
```

### Text Processing

#### Task 3.1.1: Create text cleaning module

```
Create a text_processing.py module in app/utils/ that:

1. Implements functions for basic text cleaning:
   - Convert to lowercase
   - Remove extra whitespace
   - Handle basic punctuation
2. Has a clean_text() function that applies all preprocessing steps
3. Includes proper error handling for edge cases

Focus on core text cleaning operations required for tokenisation.
```

#### Task 3.1.2: Add title preprocessing

```
Enhance the text_processing.py module to add title-specific preprocessing:

1. Add functions to normalise common patterns in HN titles
2. Implement handling for special characters and symbols
3. Create a function to extract the title length as a feature
4. Add a preprocess_title() function that applies all title-specific steps

Focus on preparing titles for effective tokenisation and feature extraction.
```

#### Task 3.2.1: Create basic tokeniser

```
Create a tokeniser.py module in app/utils/ that:

1. Imports the text_processing module
2. Implements a function to split text into tokens
3. Handles basic token normalisation
4. Has proper error handling for edge cases
5. Returns a list of tokens for a given text

Focus on creating a simple but effective tokeniser following the BRIEF.md guidelines.
```

#### Task 3.2.2: Add vocabulary builder

```
Enhance the tokeniser.py module to add vocabulary building:

1. Create a Vocabulary class to manage token-to-id mapping
2. Implement functions to build vocabulary from a corpus
3. Add special token handling (PAD, UNK, etc.)
4. Include methods to convert between tokens and IDs
5. Add functions to save and load the vocabulary

Focus on establishing the foundation for Word2Vec input preparation.
```

### Word2Vec Model

#### Task 4.1.1: Create Word2Vec model skeleton

```
Create a word2vec.py module in app/models/ that:

1. Imports PyTorch and necessary utilities
2. Defines a basic Word2Vec class that extends nn.Module
3. Implements __init__ with configuration parameters
4. Adds placeholder methods for forward pass
5. Includes docstrings explaining the intended architecture

Create just the skeleton - we'll implement the layers in subsequent tasks.
```

#### Task 4.1.2: Implement embedding layer

```
Enhance the word2vec.py module to implement the embedding layer:

1. Add an embedding layer in the __init__ method
2. Set up proper initialisation for the embeddings
3. Include configuration for embedding dimension
4. Add a method to get embeddings for specific tokens
5. Include proper error handling

Focus on correctly implementing the embedding layer for the Word2Vec model.
```

#### Task 4.1.3: Implement Skip-gram architecture

```
Complete the word2vec.py module by implementing the Skip-gram architecture:

1. Add the output projection layer
2. Implement the forward method for context prediction
3. Create a method to compute the loss with negative sampling
4. Add a predict_context method for inference
5. Include proper handling of batched inputs

Focus on implementing the core Skip-gram model as described in BRIEF.md.
```

#### Task 4.2.1: Create context window extraction

```
Create a context_window.py module in app/utils/ that:

1. Implements a function to extract context windows from token sequences
2. Creates a function to generate (target, context) pairs for Skip-gram training
3. Handles edge cases like short sequences
4. Includes configurable context window sise
5. Has proper error handling

Focus on preparing the data structure needed for Skip-gram training.
```

#### Task 4.2.2: Add data preparation for Word2Vec

```
Create a word2vec_data.py module in app/utils/ that:

1. Imports the tokeniser and context_window modules
2. Implements a function to prepare training data from text
3. Creates batches of (target, context) pairs
4. Handles conversion to PyTorch tensors
5. Includes proper error handling and validation

Focus on preparing properly formatted data for Word2Vec training.
```

#### Task 4.2.3: Implement basic training loop

```
Create a word2vec_training.py module in app/training/ that:

1. Imports the Word2Vec model and data preparation utilities
2. Implements a basic training loop for Word2Vec
3. Uses Adam optimiser with a configurable learning rate
4. Includes progress tracking and basic logging
5. Has a function to save the trained model

Focus on implementing a functional training loop for the Word2Vec model.
```

### Feature Engineering

#### Task 5.1.1: Create title length feature

```
Create a title_features.py module in app/features/ that:

1. Imports necessary utilities
2. Implements a function to extract title length as a feature
3. Normalises title length based on findings in REPORT.md
4. Handles edge cases (very short/long titles)
5. Returns normalised features in a suitable format

Focus on implementing the title length feature identified in REPORT.md.
```

#### Task 5.1.2: Implement title embedding averaging

```
Enhance the title_features.py module to add title embedding functionality:

1. Import the Word2Vec model utilities
2. Implement a function to convert a title to tokens
3. Create a function to get embeddings for each token
4. Add a function to average word embeddings for a title
5. Include proper handling of out-of-vocabulary words

Focus on converting titles to a fixed-dimensional representation for the prediction model.
```

#### Task 5.2.1: Create domain extraction

```
Create a domain_features.py module in app/features/ that:

1. Implements a function to extract domains from URLs
2. Normalises domains (lowercase, remove www.)
3. Handles posts without URLs with a special token
4. Creates a function to build a domain vocabulary
5. Includes proper error handling for malformed URLs

Focus on implementing domain extraction as per REPORT.md findings.
```

#### Task 5.2.2: Implement domain encoding

```
Enhance the domain_features.py module to add domain encoding:

1. Create a function to convert domains to IDs using a vocabulary
2. Implement a method to get one-hot encoded domains
3. Add functionality for domain reputation based on historical performance
4. Handle unknown domains properly
5. Include functions to save and load domain encodings

Focus on converting domains to a format suitable for the prediction model.
```

#### Task 5.3.1: Create time feature extraction

```
Create a time_features.py module in app/features/ that:

1. Implements functions to extract day of week from timestamps
2. Adds a function to extract hour of day
3. Handles timezone conversion if needed
4. Creates a function to format time features
5. Includes proper error handling for invalid timestamps

Focus on extracting the time-based features identified in REPORT.md.
```

#### Task 5.3.2: Implement cyclical encoding for time

```
Enhance the time_features.py module to add cyclical encoding:

1. Import math utilities (sin, cos)
2. Implement functions for sine/cosine encoding of:
   - Hour of day (0-23)
   - Day of week (0-6)
3. Add a function to combine cyclical features
4. Create a function to get all time features for a timestamp
5. Include proper validation and error handling

Focus on properly representing periodic time features for the prediction model.
```

### Prediction Model

#### Task 6.1.1: Create prediction model skeleton

```
Create a predictor.py module in app/models/ that:

1. Imports PyTorch and necessary utilities
2. Defines a basic UpvotePredictor class that extends nn.Module
3. Implements __init__ with configuration parameters
4. Adds placeholder methods for forward pass
5. Includes docstrings explaining the intended architecture

Create just the skeleton - we'll implement the layers in subsequent tasks.
```

#### Task 6.1.2: Implement input processing

```
Enhance the predictor.py module to implement input processing:

1. Add methods to process different feature types:
   - Title embeddings
   - Domain features
   - Time features
2. Create a function to combine features
3. Implement normalisation for combined features
4. Add proper error handling for invalid inputs
5. Include logging for debugging

Focus on properly handling and combining the various input features.
```

#### Task 6.1.3: Add neural network layers

```
Complete the predictor.py module by implementing the neural network:

1. Create a domain embedding layer
2. Add 2-3 hidden layers with ReLU activations
3. Implement dropout for regularisation
4. Create the final output layer (single neuron)
5. Complete the forward method to process inputs through the network

Focus on implementing the neural network architecture described in SPEC.md.
```

#### Task 6.2.1: Create loss function and optimiser

```
Create a predictor_training.py module in app/training/ that:

1. Imports the UpvotePredictor model
2. Implements MSE loss function setup
3. Creates an optimiser configuration (Adam)
4. Adds learning rate scheduling
5. Includes proper initialisation

Focus on setting up the training components for the prediction model.
```

#### Task 6.2.2: Implement basic training loop

```
Enhance the predictor_training.py module to add the training loop:

1. Implement a function to train for one epoch
2. Create a function for the complete training process
3. Add batch processing of training data
4. Include progress tracking and logging
5. Implement checkpoint saving during training

Focus on implementing a functional training loop for the prediction model.
```

#### Task 6.2.3: Add model evaluation

```
Enhance the predictor_training.py module to add evaluation:

1. Implement a function to evaluate on validation data
2. Add MSE calculation for model performance
3. Create a function to track best model performance
4. Implement early stopping based on validation loss
5. Add functionality to restore the best model

Focus on properly evaluating the model during and after training.
```

#### Task 6.3.1: Create model saving and loading

```
Create a model_io.py module in app/utils/ that:

1. Implements functions to save model state:
   - Save weights
   - Save architecture configuration
   - Save metadata (training info)
2. Creates functions to load a saved model
3. Adds versioning to saved models
4. Includes proper error handling
5. Has functions to check if a model exists

Focus on reliable model persistence for later deployment and inference.
```

#### Task 6.3.2: Implement version tracking

```
Create a versioning.py module in app/utils/ that:

1. Implements a VersionTracker class
2. Uses semantic versioning (major.minor.patch)
3. Stores version history in a JSON file
4. Has functions to get and update the current version
5. Includes proper error handling

Focus on tracking model versions for the API to report correctly.
```

### API Implementation

#### Task 7.1.1: Create FastAPI application skeleton

```
Update the app/main.py file to implement a basic FastAPI application:

1. Import FastAPI and related utilities
2. Create the FastAPI application instance
3. Add basic configuration and metadata
4. Set up error handling
5. Include health check logic

Focus on setting up the foundation for the API endpoints.
```

#### Task 7.1.2: Add request/response models

```
Create a schemas.py module in app/ that:

1. Imports Pydantic and related utilities
2. Defines a model for prediction requests:
   - author field (string)
   - title field (string)
   - timestamp field (string)
3. Creates a model for prediction responses:
   - upvotes field (float)
4. Adds validation for required fields
5. Includes models for other endpoint responses

Focus on properly validating API inputs and outputs.
```

#### Task 7.2.1: Implement /ping endpoint

```
Enhance the app/main.py file to add the /ping endpoint:

1. Import necessary utilities
2. Implement the endpoint:
   - GET method
   - Returns "ok" string
   - Includes docstring for OpenAPI documentation
3. Add simple logging for the endpoint
4. Include proper error handling

Focus on implementing the simple health check endpoint specified in BRIEF.md.
```

#### Task 7.2.2: Implement /version endpoint

```
Enhance the app/main.py file to add the /version endpoint:

1. Import the versioning module
2. Implement the endpoint:
   - GET method
   - Returns {"version": "x.y.z"} format
   - Includes docstring for OpenAPI documentation
3. Add proper error handling
4. Include logging for the endpoint

Focus on implementing the version endpoint specified in BRIEF.md.
```

#### Task 7.3.1: Create prediction service

```
Create a prediction.py module in app/services/ that:

1. Imports the predictor model and feature utilities
2. Implements a function to preprocess prediction inputs
3. Creates a predict_upvotes function that:
   - Takes title, author, timestamp inputs
   - Preprocesses the inputs
   - Runs the prediction model
   - Returns the predicted upvote count
4. Includes proper error handling
5. Adds logging for debugging

Focus on encapsulating the prediction logic for use by the API endpoint.
```

#### Task 7.3.2: Implement /how_many_upvotes endpoint

```
Enhance the app/main.py file to add the /how_many_upvotes endpoint:

1. Import the prediction service and schemas
2. Implement the endpoint:
   - POST method
   - Takes UpvotePredictionRequest
   - Returns UpvotePredictionResponse
   - Includes docstring for OpenAPI documentation
3. Add proper error handling for prediction failures
4. Include timing measurement for latency logging

Focus on implementing the main prediction endpoint specified in BRIEF.md.
```

### Logging System

#### Task 8.1.1: Create basic logging configuration

```
Create a logging_utils.py module in app/utils/ that:

1. Imports the logging module and related utilities
2. Implements a function to configure basic logging
3. Sets up console logging with appropriate format
4. Creates different log levels (DEBUG, INFO, ERROR)
5. Includes proper error handling

Focus on establishing a basic logging foundation for the application.
```

#### Task 8.1.2: Implement file logging

```
Enhance the logging_utils.py module to add file logging:

1. Add a function to set up file logging
2. Implement log rotation based on size
3. Create a function to ensure the log directory exists
4. Add a formatter for structured JSON logs
5. Include proper error handling

Focus on implementing persistent logging to files as specified in BRIEF.md.
```

#### Task 8.2.1: Create request logging middleware

```
Create a middleware.py module in app/ that:

1. Imports FastAPI middleware utilities and logging_utils
2. Implements request logging middleware:
   - Logs request method, path, and headers
   - Measures request processing time
   - Logs response status code
3. Adds the middleware to the application
4. Includes proper error handling

Focus on capturing basic information about all API requests.
```

#### Task 8.2.2: Implement prediction logging

```
Enhance the prediction service to add detailed prediction logging:

1. Import logging utilities
2. Implement a function to log prediction details:
   - Request inputs
   - Prediction output
   - Latency
   - Timestamp
   - Model version
3. Save logs in the format specified in BRIEF.md
4. Include proper error handling

Focus on logging prediction details as specified in BRIEF.md.
```

#### Task 8.3.1: Create log reading function

```
Create a log_reader.py module in app/utils/ that:

1. Implements a function to read log files
2. Parses JSON-formatted logs
3. Adds filtering capabilities
4. Includes proper error handling for missing or corrupt logs
5. Returns logs in a format suitable for the API

Focus on reliably reading and parsing log files for the /logs endpoint.
```

#### Task 8.3.2: Implement /logs endpoint

```
Enhance the app/main.py file to add the /logs endpoint:

1. Import the log_reader module
2. Implement the endpoint:
   - GET method
   - Returns {"logs": [...]} format
   - Includes docstring for OpenAPI documentation
3. Add proper error handling for log reading failures
4. Include logging for the endpoint itself

Focus on implementing the logs endpoint specified in BRIEF.md.
```

### Containerisation

#### Task 9.1.1: Create basic Dockerfile

```
Create a Dockerfile in the project root that:

1. Uses Python 3.13.3 slim as the base image
2. Sets up the working directory
3. Copies and installs requirements
4. Copies the application code
5. Sets up environment variables
6. Defines the command to run the application
7. Includes proper documentation in comments

Focus on creating a basic but functional Dockerfile for the application.
```

#### Task 9.1.2: Add multi-stage build

```
Enhance the Dockerfile to implement a multi-stage build:

1. Use a builder stage for dependencies
2. Optimize the final image sise
3. Implement proper caching
4. Add security hardening
5. Create a non-root user for the application
6. Include proper documentation in comments

Focus on optimising the container for production deployment.
```

#### Task 9.2.1: Create basic docker-compose.yml

```
Create a docker-compose.yml file in the project root that:

1. Defines the application service
2. Sets up port mapping
3. Configures environment variables
4. Implements healthcheck
5. Adds restart policy
6. Includes proper documentation in comments

Focus on creating a basic but functional Docker Compose configuration.
```

#### Task 9.2.2: Add volume configuration for logs

```
Enhance the docker-compose.yml file to add volume configuration:

1. Define a volume for logs
2. Mount the volume in the application service
3. Configure appropriate permissions
4. Add log rotation configuration
5. Implement proper volume naming
6. Include proper documentation in comments

Focus on ensuring logs are persisted across container restarts as specified in BRIEF.md.
```

### Integration and Documentation

#### Task 10.1.1: Create Word2Vec training script

```
Create a train_word2vec.py script in a scripts/ directory that:

1. Imports the Word2Vec model and utilities
2. Implements a function to download and process the text8 dataset
3. Creates a training script with command-line arguments
4. Adds progress tracking and visualisation
5. Implements model saving
6. Includes proper error handling and documentation

Focus on creating a usable script for training the Word2Vec model.
```

#### Task 10.1.2: Create prediction model training script

```
Create a train_predictor.py script in the scripts/ directory that:

1. Imports the prediction model and utilities
2. Implements data loading from the database
3. Creates a training script with command-line arguments
4. Adds progress tracking and visualisation
5. Implements model saving and evaluation
6. Includes proper error handling and documentation

Focus on creating a usable script for training the prediction model.
```

#### Task 10.2.1: Create comprehensive README

```
Create a comprehensive README.md in the project root that:

1. Provides a project overview and problem statement
2. Documents the setup process:
   - Environment creation
   - Database connection
   - Running the application
3. Explains the machine learning approach
4. Documents the API endpoints
5. Describes the Docker deployment
6. Includes troubleshooting tips
7. Adds references and acknowledgments

Focus on creating clear, comprehensive documentation for the project.
```

#### Task 10.2.2: Add API documentation

```
Create an API.md file in the project root that:

1. Documents each API endpoint:
   - /ping
   - /version
   - /logs
   - /how_many_upvotes
2. Includes request and response examples
3. Documents error responses
4. Provides usage tips
5. Adds performance considerations

Focus on creating clear documentation for the API endpoints.
```
