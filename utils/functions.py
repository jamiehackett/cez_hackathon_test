import pandas as pd
import torch
from collections import defaultdict

def filter_dataframe_by_date(df: pd.DataFrame, date_str: str, column: str = 'ds', inclusive: bool = True) -> pd.DataFrame:
    """
    Filter DataFrame rows by a specified datetime column based on the given date string.
    
    Args:
        df (pd.DataFrame): Input DataFrame with a datetime column.
        date_str (str): Date string to filter by, e.g. '2021-12-31 23:59:59'.
        column (str): Name of the datetime column to filter on. Default is 'ds'.
        inclusive (bool): If True, keep rows with column >= date_str,
                          else keep rows with column > date_str.
    
    Returns:
        pd.DataFrame: Filtered DataFrame copy.
    """
    if inclusive:
        filtered_df = df[df[column] >= date_str].copy()
    else:
        filtered_df = df[df[column] > date_str].copy()
    return filtered_df

def split_df_by_two_dates_with_encoder(
    df: pd.DataFrame, 
    val_split_date: str, 
    test_split_date: str = None, 
    encoder_length: int = 0, 
    freq: str = '15min', 
    test_dataset: bool = True, 
    column: str = 'ds'
) -> tuple:
    """
    Splits a DataFrame into train, validation, and optionally test sets based on split dates and encoder length.

    Parameters:
    df (pd.DataFrame): The DataFrame with a datetime column.
    val_split_date (str or pd.Timestamp): Start of validation set.
    test_split_date (str or pd.Timestamp, optional): Start of test set. Required if test_dataset is True.
    encoder_length (int): Number of encoder steps to include in validation and test sets.
    freq (str): Frequency of the time series (e.g., '15min', '1H').
    test_dataset (bool): Whether to create a test dataset. Default is True.
    column (str): Name of the datetime column to use. Default is 'ds'.

    Returns:
    tuple: Two DataFrames (train, val) if test_dataset is False, or three DataFrames (train, val, test) if True.
    """
    val_split_date = pd.to_datetime(val_split_date)
    if test_dataset and test_split_date is not None:
        test_split_date = pd.to_datetime(test_split_date)

    if column not in df.columns:
        raise KeyError(f"The DataFrame does not contain a '{column}' column.")

    if not pd.api.types.is_datetime64_any_dtype(df[column]):
        df[column] = pd.to_datetime(df[column])

    df = df.drop_duplicates().reset_index(drop=True)

    # Calculate the timedelta for encoder_length * frequency
    encoder_delta = pd.Timedelta(encoder_length * pd.Timedelta(freq))

    # Train: everything before val_split_date
    df_train = df[df[column] < val_split_date]

    # Validation: from val_split_date minus encoder_delta (to get encoder history) up to test_split_date
    val_start_date = val_split_date - encoder_delta
    df_val = df[(df[column] >= val_start_date) & (df[column] < (test_split_date if test_dataset else df[column].max()))]

    if test_dataset:
        # Test: from test_split_date minus encoder_delta to the end
        if test_split_date is None:
            raise ValueError("test_split_date must be provided when test_dataset is True.")
        test_start_date = test_split_date - encoder_delta
        df_test = df[df[column] >= test_start_date]
        return df_train, df_val, df_test

    return df_train, df_val

def save_model_and_train_config(training, subdirectory_name):
    """
    Save model checkpoint to the specified subdirectory.

    Args:
        training: The training object with a save method.
        subdirectory_name: The directory where files will be saved.
    """
    # Save model checkpoint
    training.save(f"{subdirectory_name}/training_config.pkl")

def process_test_dataloader(tft, test_dataloader):
    """
    Processes the test dataloader and collects predictions and inputs.

    Args:
        tft: The trained model with a `to_prediction` method.
        test_dataloader: The dataloader for testing.

    Returns:
        A dictionary containing merged predictions and inputs.
    """
    tft.eval()
    merged = defaultdict(list)

    for x, y in test_dataloader:
        # Move inputs to the device
        x = {k: v.to(tft.device) for k, v in x.items()}

        # Handle target tensor (y)
        if isinstance(y, tuple):
            y_tensor = y[0].to(tft.device)
            # Collect encoder_target if it exists in the tuple
            if len(y) > 1 and torch.is_tensor(y[1]):
                merged['encoder_target'].append(y[1].cpu())
        else:
            y_tensor = y.to(tft.device)

        # Perform inference
        with torch.no_grad():
            out = tft(x)
            prediction = tft.to_prediction(out, x)

        # Collect predictions
        if isinstance(prediction, torch.Tensor):
            merged["prediction"].append(prediction.cpu())
        elif hasattr(prediction, "_asdict"):
            prediction = prediction._asdict()
            for k, v in prediction.items():
                if isinstance(v, torch.Tensor):
                    merged[k].append(v.cpu())
        elif isinstance(prediction, dict):
            for k, v in prediction.items():
                if isinstance(v, torch.Tensor):
                    merged[k].append(v.cpu())

        # Collect inputs from x if present
        for key in ['decoder_time_idx', 'encoder_time_idx', 'time_idx', 'target', 'decoder_target', 'encoder_target']:
            if key in x:
                merged[key].append(x[key].cpu())

        # Collect target tensor (main target)
        merged['target'].append(y_tensor.cpu())

    # Concatenate all collected tensors
    for k in merged:
        merged[k] = torch.cat(merged[k], dim=0)

    return merged