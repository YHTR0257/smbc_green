# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Docker Environment
- **Start development environment**: `docker-compose up -d`
- **Enter container**: `docker exec -it smbc_learning_app bash`
- **Run code inside container**: All development should be done within the Docker container

### Python Execution
- **Run model training**: `python src/model_training.py`
- **Run data processing**: `python -c "from src.data_processing import load_data, preprocess_data; # your code"`
- **Run tests**: `pytest tests/`

### Data Pipeline
- **Raw data location**: `data/raw/` (contains train.csv, test.csv, feature_description.csv)
- **Processed data location**: `data/processed/` (contains processed train/test datasets)
- **Model outputs**: `models/` (checkpoints and trained models)
- **Reports**: `reports/` (analysis reports and visualizations)

## Architecture Overview

### Core Components
- **src/data_processing.py**: Data loading, preprocessing with discomfort index calculation and temporal scaling
- **src/models.py**: XGBoost model implementation with development mode sampling
- **src/model_training.py**: Training pipeline with config-driven hyperparameters
- **src/feature_engineering.py**: Feature engineering utilities
- **config/config.yaml**: Centralized configuration for models, data paths, and training parameters

### Key Configuration
- **Target variable**: `price_actual`
- **Model types**: Random Forest, Neural Network, Denoising Autoencoder, XGBoost
- **Environment**: Set `APP_ENV=development` for smaller sample sizes during development
- **GPU support**: XGBoost configured with `gpu_hist` and `gpu_predictor`

### Data Flow
1. Raw data in `data/raw/` → preprocessing → `data/processed/`
2. Training uses config-driven parameters from `config/config.yaml`
3. Models saved to `models/` with experiment tracking in `experiments/`
4. Reports and visualizations generated in `reports/`

### Special Features
- **Discomfort index calculation**: Automated weather-based comfort metrics for multiple Spanish cities
- **Temporal scaling**: Expanding mean/std scaling with time-series awareness
- **Multi-seed training**: Random seeds [42, 123, 456] for reproducible results