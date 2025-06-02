# Data Science Project

This project is designed to facilitate data science tasks, including data processing, model training, and evaluation. Below is an overview of the project structure and its components.

## Project Structure

- **data/**: Contains datasets used in the project.
  - **raw/**: Raw datasets, each in its own subdirectory.
  - **processed/**: Processed datasets, organized into train, validation, and test subdirectories.
  - **external/**: External data files.

- **notebooks/**: Jupyter notebooks for various tasks.
  - **01_data_exploration.ipynb**: Data exploration tasks.
  - **02_feature_engineering.ipynb**: Feature engineering processes.
  - **03_model_prototyping.ipynb**: Prototyping machine learning models.

- **src/**: Source code for the project.
  - **data_processing.py**: Functions and classes for data processing tasks.
  - **models.py**: Model architecture and related functions.
  - **training.py**: Handles the training of the models.
  - **pipelines.py**: Defines data processing and model training pipelines.
  - **utils.py**: Utility functions used throughout the project.

- **config/**: Configuration files for various settings.
  - **data_config.yaml**: Configuration settings related to data processing.
  - **model_config.yaml**: Configuration settings for model parameters.
  - **train_config.yaml**: Configuration settings for training parameters.

- **experiments/**: Logs, checkpoints, and metrics for experiments.
  - **experiment_20250602_001/**: First experiment logs and metrics.
  - **experiment_20250602_002/**: Second experiment logs and metrics.

- **models/**: Stores trained models.
  - **best_model_v1.pth**: Best version of the trained model.
  - **latest_model.pth**: Latest version of the trained model.

- **reports/**: Contains project reports and visualizations.
  - **final_report.pdf**: Final report of the project.
  - **visualizations/**: Visualizations related to the project.

- **tests/**: Unit tests for the project.
  - **test_data_processing.py**: Unit tests for data processing functions.
  - **test_models.py**: Unit tests for model functions.

## Getting Started

To get started with this project, clone the repository and install the required packages listed in `requirements.txt`.

```bash
git clone <repository-url>
cd data-science-project
pip install -r requirements.txt
```

## Usage

Follow the notebooks for step-by-step guidance on data exploration, feature engineering, and model prototyping. Use the source code in the `src` directory for implementing custom functionalities.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for any suggestions or improvements.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.