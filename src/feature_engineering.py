import pandas as pd
import yaml
import dotenv
import sys
import os
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_processing import DataProcessor
dotenv.load_dotenv()

def feature_engineering(data:pd.DataFrame, config):
    """Perform feature engineering on the dataset."""
    _pressure_cols = [col for col in data.columns if 'pressure' in col]
    data[_pressure_cols] = data[_pressure_cols].clip(upper=1100, lower=800)
    return data

def describe_data(train_df: pd.DataFrame, test_df: pd.DataFrame, config):
    """Describe the training and testing datasets."""
    test_num_cols = test_df.select_dtypes(include=['number']).columns.tolist()
    train_num_cols = train_df.select_dtypes(include=['number']).columns.tolist()

    # Helper function to convert DataFrame to markdown table
    def dataframe_to_markdown(df):
        headers = "| " + " | ".join(df.columns) + " |"
        separator = "| " + " | ".join(["---"] * len(df.columns)) + " |"
        rows = "\n".join(["| " + " | ".join(map(str, row)) + " |" for row in df.values])
        return f"{headers}\n{separator}\n{rows}"

    # Training data description
    with open(config['data_path']['reports'] + f'train_description_{config["exp_name"]}.md', 'w') as f:
        f.write("# Training Data Description\n\n")
        _desc_df = train_df.describe().transpose()
        f.write("## Summary Statistics\n")
        f.write(dataframe_to_markdown(_desc_df.reset_index()))
        f.write("```\n")
        f.write("\n\n## Data Columns\n")
        f.write(dataframe_to_markdown(pd.DataFrame({
            "Column Name": train_df.columns,
            "Data Type": [str(train_df[col].dtype) for col in train_df.columns]
        })))
        f.write("\n\n## Data Shape\n")
        f.write(f"Rows: {train_df.shape[0]}, Columns: {train_df.shape[1]}\n")
        f.write("\n\n## Missing Values\n")
        f.write(dataframe_to_markdown(train_df.isnull().sum().reset_index(name="Missing Values")))
        f.write("\n\n## Correlation Matrix\n")
        f.write(dataframe_to_markdown(train_df[train_num_cols].corr().reset_index()))

    # Testing data description
    with open(config['data_path']['reports'] + f'test_description_{config["exp_name"]}.md', 'w') as f:
        f.write("# Testing Data Description\n\n")
        _desc_df = test_df.describe().transpose()
        f.write("## Summary Statistics\n")
        f.write(dataframe_to_markdown(_desc_df.reset_index()))
        f.write("```\n")
        f.write("\n\n## Data Columns\n")
        f.write(dataframe_to_markdown(pd.DataFrame({
            "Column Name": test_df.columns,
            "Data Type": [str(test_df[col].dtype) for col in test_df.columns]
        })))
        f.write("\n\n## Data Shape\n")
        f.write(f"Rows: {test_df.shape[0]}, Columns: {test_df.shape[1]}\n")
        f.write("\n\n## Missing Values\n")
        f.write(dataframe_to_markdown(test_df.isnull().sum().reset_index(name="Missing Values")))
        f.write("\n\n## Correlation Matrix\n")
        f.write(dataframe_to_markdown(test_df[test_num_cols].corr().reset_index()))
        f.write("\n===========================================\n")
    
    # Visualize the correlation matrix
    plot_col = ["generation_biomass",
                "generation_fossil_brown_coal/lignite",
                "generation_fossil_gas", "generation_fossil_hard_coal", "generation_fossil_oil",
                "generation_hydro_pumped_storage_consumption", "generation_hydro_run_of_river_and_poundage",
                "generation_hydro_water_reservoir","generation_nuclear","generation_other","generation_other_renewable",
                "generation_solar","generation_waste", "generation_wind_onshore"]
    # rename columns for consistency
    train_df = train_df.rename(columns={col: col.replace("generation_", "") for col in plot_col})
    test_df = test_df.rename(columns={col: col.replace("generation_", "") for col in plot_col})
    plot_col = [col.replace("generation_", "") for col in plot_col]
    train_df = pd.concat([train_df[plot_col], train_df[config['train_config']['general']['target']]], axis=1)
    plo_train = train_df[plot_col].corr()
    plo_test = test_df[plot_col].corr()

    # small font size for labels

    fig, ax = plt.subplots()
    fig.suptitle("Correlation Matrix - Training Data")
    plt.figure(figsize=(20, 20))
    sns.heatmap(plo_train, annot=True, fmt=".2f", cmap="coolwarm", square=True, cbar_kws={"shrink": .8}, ax=ax)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    ax.set_xlabel("Features", fontsize=2)
    ax.set_ylabel("Features", fontsize=2)
    fig.savefig(os.path.join(config['data_path']['reports'], f'train_corr_{config["exp_name"]}.png'))
    plt.close(fig)
    fig, ax = plt.subplots()
    fig.suptitle("Correlation Matrix - Testing Data")
    plt.figure(figsize=(20, 20))
    sns.heatmap(plo_test, annot=True, fmt=".2f", cmap="coolwarm", square=True, cbar_kws={"shrink": .8}, ax=ax)
    ax.set_xlabel("Features", fontsize=2)
    ax.set_ylabel("Features", fontsize=2)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    fig.savefig(os.path.join(config['data_path']['reports'], f'test_corr_{config["exp_name"]}.png'))
    plt.close(fig)

    # pairplot for selected columns
    fig, ax = plt.subplots()
    fig.suptitle("Pairplot - Training Data")
    sns.pairplot(train_df[plot_col], diag_kind='kde')
    plt.savefig(os.path.join(config['data_path']['reports'], f'train_pairplot_{config["exp_name"]}.png'))
    plt.close(fig)
    fig, ax = plt.subplots()
    fig.suptitle("Pairplot - Testing Data")
    sns.pairplot(test_df[plot_col], diag_kind='kde')
    plt.savefig(os.path.join(config['data_path']['reports'], f'test_pairplot_{config["exp_name"]}.png'))
    plt.close(fig)

if __name__ == "__main__":
    # Load configuration
    print(os.getcwd())
    parent_dir = Path(__file__).parent.parent
    config_path = os.path.join(parent_dir, 'config', 'config.yml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}. Please check the path.")
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    config['exp_name'] = "20250618"

    train_data_path = Path(os.path.join(parent_dir, config['data_path']['raw_data'], 'train.csv'))
    test_data_path = Path(os.path.join(parent_dir, config['data_path']['raw_data'], 'test.csv'))
    if not train_data_path.exists():
        raise FileNotFoundError(f"Training data file not found at {train_data_path}. Please check the path.")

    # Load data
    data_processor = DataProcessor(train_file_path=train_data_path, test_file_path=test_data_path, config=config)
    train_df, val_df, test_df = data_processor.process_all()

    # Perform feature engineering
    train_df, val_df, test_df = feature_engineering(train_df, config), feature_engineering(val_df, config), feature_engineering(test_df, config)

    _train_df = pd.concat([train_df, val_df], ignore_index=True)
    # Describe the data
    describe_data(_train_df, test_df, config)

    # Save the processed data
    output_path = Path(os.path.join(config['data_path']['processed_data'], f'dataset_{config["exp_name"]}'))
    data_processor.save_processed_data(output_path)
    print(f"Processed data saved to {output_path}_train.csv and {output_path}_test.csv")
    print("Feature engineering and data description completed successfully.")