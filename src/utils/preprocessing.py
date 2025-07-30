import pandas as pd

from sklearn.preprocessing import StandardScaler

def preprocess_dataframe(df: pd.DataFrame):
    """
    Preprocesses the input DataFrame by removing rows with missing values 
    and normalizing the numerical features using StandardScaler.

    This function performs the following steps:
    - Selects only numeric columns
    - Drops rows containing NaN values
    - Applies standard scaling (zero mean, unit variance)
    - Returns the cleaned DataFrame aligned with the transformed data

    Args:
        df (pd.DataFrame): The original input DataFrame.

    Returns:
        tuple:
            - pd.DataFrame: Cleaned DataFrame with original columns but only valid rows.
            - np.ndarray: Scaled numerical data (2D array).
            - pd.Index: Index of the retained rows after dropping NaNs.
    """
    num_df = df.select_dtypes(include="number")
    num_df_clean = num_df.dropna()
    indices = num_df_clean.index

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(num_df_clean)

    return df.loc[indices].copy(), data_scaled, indices