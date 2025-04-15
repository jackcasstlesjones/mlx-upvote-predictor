# Architecture Decision Record

## For Hacker News Upvote Predictor

Version 0.1  
Prepared by System Architect  
MLX Institute  
15 April 2025

## Revision History

| Name             | Date       | Reason For Changes            | Version |
| ---------------- | ---------- | ----------------------------- | ------- |
| System Architect | 15/04/2025 | Initial architecture document | 0.1     |

## 1. Context and Problem Statement

### 1.1 Background

The Hacker News Upvote Predictor project aims to create a machine learning system that can predict the number of upvotes a post will receive on Hacker News. The system must process post titles, extract relevant features, and make accurate predictions via an API.

Key drivers for this project include:

- Creating a learning platform for team members new to machine learning
- Building a functional predictive model with measurable accuracy
- Providing a containerized solution that can be easily deployed
- Completing the project within a tight 3-day timeframe

### 1.2 Problem Definition

The project presents several architectural challenges:

- Implementing the Word2Vec algorithm for word embeddings
- Creating a neural network model for upvote prediction
- Handling various data features (titles, domains, timing)
- Meeting the 1-second response time requirement
- Ensuring proper logging and prediction storage
- Building a system that can be completed by a team new to ML in just 3 days

## 2. Decision Drivers

### 2.1 Technical Constraints

- Timeline of only 3 days for development
- Team's limited experience with machine learning
- 1-second response time requirement for prediction requests
- Need for containerization and proper network isolation
- Integration with Weights & Biases for experiment tracking

### 2.2 Business Constraints

- Focus on learning and practical application of ML concepts
- System must be maintainable and understandable for educational purposes
- Solution must be deployable on a bare metal virtual machine
- No specific budget constraints, but emphasis on using open-source technologies

## 3. Considered Alternatives

### 3.1 Monolithic Architecture

- **Description**: Single application containing all components (database access, model training, prediction, API)
- **Pros**: Simplest to develop quickly; no inter-service communication overhead; easier deployment
- **Cons**: Less modular; harder to maintain; single point of failure
- **Fit**: Well-suited for the tight timeline and team's experience level

### 3.2 Microservices Architecture

- **Description**: Multiple services for data processing, model training, prediction, and API
- **Pros**: Highly modular; independent scaling; better separation of concerns
- **Cons**: Complex to develop and deploy; communication overhead; overkill for project scope
- **Fit**: Poorly aligned with 3-day timeline and team experience

### 3.3 Hybrid Architecture

- **Description**: Two main components - training pipeline (offline) and inference service (online)
- **Pros**: Separates training from serving; some modularity without excessive complexity
- **Cons**: More complex than monolithic; requires clear interface design
- **Fit**: Reasonable compromise but potentially too complex for the timeline

## 4. Decision Outcome

### 4.1 Chosen Alternative

We will implement a monolithic architecture with clear internal module boundaries. This approach will use:

- A single application with distinct logical modules
- Pre-training for Word2Vec embeddings before API deployment
- Clear separation between training and inference code paths
- PyTorch for ML components with FastAPI for the web interface
- Docker for containerization with proper network isolation

This approach prioritizes rapid development while maintaining reasonable code organization. The monolithic design reduces complexity while internal module boundaries allow for future refactoring if needed.

### 4.2 Positive Consequences

- Faster development timeline, critical for the 3-day constraint
- Simpler deployment and testing process
- Reduced communication overhead improves response times
- Easier debugging and troubleshooting for ML newcomers
- Faster iterations on the ML pipeline during development

### 4.3 Negative Consequences

- Limited scaling options (whole application must scale together)
- Potential for less clear boundaries between components
- Possible technical debt if the system grows significantly
- Training and inference sharing resources could impact performance

## 5. Technical Architecture

### 5.1 System Components

The system will consist of the following logical modules within a monolithic application:

1. **Data Access Module**:

   - Database connection handling
   - Data extraction and preprocessing
   - Feature engineering pipeline

2. **Word Embedding Module**:

   - Text tokenization utilities
   - Word2Vec implementation (using PyTorch)
   - Embedding persistence and loading

3. **Model Module**:

   - Neural network definition
   - Training and evaluation logic
   - Model persistence and loading
   - Prediction generation

4. **API Module**:

   - FastAPI implementation
   - Endpoint handlers
   - Request/response processing
   - Input validation

5. **Logging Module**:
   - Request and prediction logging
   - Log persistence and retrieval
   - Performance metrics tracking

### 5.2 Technical Interfaces

1. **Database Interface**:

   - SQLAlchemy for database access
   - PostgreSQL connection pooling
   - Transaction management

2. **Model Interface**:

   - Standard predict() method for inference
   - Input: preprocessed features (title embeddings, domain, timing)
   - Output: predicted upvote score

3. **API Endpoints**:

   - RESTful design following SPEC.md requirements
   - JSON request/response format
   - Proper error handling and status codes

4. **W&B Integration**:
   - Experiment tracking during training
   - Model versioning
   - Metric logging

### 5.3 Performance Considerations

1. **Embedding Precomputation**:

   - Word2Vec model trained offline
   - Embeddings cached for common tokens

2. **Model Optimization**:

   - Model quantization to reduce size and inference time
   - Batch processing for title tokenization
   - PyTorch JIT compilation for inference

3. **Response Time Optimization**:
   - In-memory caching for recent predictions
   - Asynchronous logging to avoid blocking prediction path
   - Database connection pooling

### 5.4 Security Architecture

1. **Network Isolation**:

   - API container: public subnet with restricted ports
   - Database container: private subnet
   - Inter-container communication via Docker network

2. **Input Validation**:
   - API-level validation of all inputs
   - Protection against injection attacks
   - Request rate limiting

## 6. Technology Stack

### 6.1 Frontend Technologies

- No traditional frontend (API-only service)
- Swagger UI for API documentation and testing

### 6.2 Backend Technologies

- **Language**: Python 3.13.3
- **ML Framework**: PyTorch
- **API Framework**: FastAPI
- **Database Access**: psycopg
- **Database**: PostgreSQL
- **Experiment Tracking**: Weights & Biases

### 6.3 Infrastructure

- **Containerisation**: Docker
- **Deployment**: Docker Compose
- **Environment**: Bare metal virtual machine
- **Network**: Docker network with subnet isolation
- **Logs**: Persistent volume mount for log storage

## 7. Monitoring and Observability

### 7.1 Logging Strategy

- **Request Logging**:

  - Structured JSON logs
  - Storage in filesystem with rotation
  - Includes latency, version, timestamp, input, prediction

- **Application Logging**:

  - Standard Python logging with levels
  - Console output in development
  - File output in production

### 7.2 Performance Monitoring

- **Latency Tracking**:

  - Per-request timing measurements
  - Periodic aggregation of statistics
  - Stored alongside request logs

- **Resource Monitoring**:
  - Basic Docker resource monitoring
  - CPU and memory usage tracking

## 8. Future Considerations

### 8.1 Potential Evolutions

- **Architectural Refactoring**:

  - Potential split into training and inference services
  - Extraction of data processing as separate component
  - Implementation of model versioning and serving

- **Model Improvements**:
  - Fine-tuning on Hacker News-specific data
  - Exploration of more complex architectures
  - Integration of additional features

### 8.2 Technical Debt

- **Monolithic Design**:

  - May require refactoring as system grows
  - Potential scaling limitations

- **ML Implementation**:
  - Initial Word2Vec implementation may be simplified
  - Feature engineering might need refinement
  - Hyperparameter tuning likely limited in initial version

## 9. Appendixes

### Appendix A: Architectural Diagrams

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            Docker Network                                       │
│                                                                                 │
│  ┌─────────────────────┐    ┌────────────────────────┐   ┌───────────────────┐  │
│  │  Database Container │    │   Model Container      │   │  API Container    │  │
│  │  (Private Subnet)   │    │   (Private Subnet)     │   │  (Public IP)      │  │
│  │                     │    │                        │   │                   │  │
│  │  ┌───────────────┐  │    │  ┌────────────────┐    │   │  ┌──────────────┐ │  │
│  │  │ PostgreSQL DB │  │    │  │ Word Embedding │    │   │  │ API Module   │ │  │
│  │  └───────────────┘  │    │  │ Module         │    │   │  │ (FastAPI)    │ │  │
│  │                     │    │  └────────┬───────┘    │   │  └──────┬───────┘ │  │
│  │                     │    │           │            │   │         │         │  │
│  │                     │    │           ▼            │   │         │         │  │
│  │                     │    │  ┌────────────────┐    │   │         │         │  │
│  │                     │◄─────►│ Data Access    │◄──────►│         │         │  │
│  │                     │    │  │ Module         │    │   │         │         │  │
│  │                     │    │  └────────┬───────┘    │   │         │         │  │
│  │                     │    │           │            │   │         │         │  │
│  │                     │    │           ▼            │   │         │         │  │
│  │                     │    │  ┌────────────────┐    │   │         ▼         │  │
│  │                     │    │  │ Model Module   │◄──────►│  ┌──────────────┐ │  │
│  │                     │    │  └────────────────┘    │   │  │ Logging      │ │  │
│  │                     │    │                        │   │  │ Module       │ │  │
│  └─────────────────────┘    └────────────────────────┘   │  └──────────────┘ │  │
│                                                          │         │         │  │
│                                                          │         ▼         │  │
│                                                          │  ┌──────────────┐ │  │
│                                                          │  │ Log Storage  │ │  │
│                                                          │  │ (Volume)     │ │  │
│                                                          │  └──────────────┘ │  │
│                                                          └───────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Appendix B: Technology Evaluation Details

**PyTorch vs. TensorFlow**

- PyTorch selected for ease of learning and flexibility
- Better fit for team new to ML
- More intuitive debugging
- Simpler model definition syntax

**FastAPI vs. Flask**

- FastAPI selected for automatic validation and documentation
- Better performance with async support
- Modern typing system reduces errors
- Built-in support for JSON serialization

**Docker vs. Virtual Environment**

- Docker selected for deployment consistency
- Easier network isolation
- Simpler deployment on target VM
- Better resource management
