import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def handle_missing_values_nsl(df):
    # Check for missing values in each column
    missing_values = df.isnull().sum()
    print("Missing values in each column:\n", missing_values)

    # For numerical columns, fill missing values with the median
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        if missing_values[col] > 0:
            df[col].fillna(df[col].median(), inplace=True)

    # For categorical columns, fill missing values with the mode
    for col in df.select_dtypes(include=['object']).columns:
        if missing_values[col] > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)

    # Drop columns with a high percentage of missing values
    # (Here assuming 'high' is more than 50% missing)
    for col in df.columns:
        if missing_values[col] / len(df) > 0.5:
            df.drop(col, axis=1, inplace=True)

    # Drop rows with missing values
    df.dropna(inplace=True)

    return df


def remove_duplicates_nsl(df):
    return df.drop_duplicates()


def handle_missing_values_nb(df):
    non_numeric_log = []
    excluded_indices = [0, 2, 4, 5, 13, 47]

    # Process all columns for non-numeric entries, excluding specified indices
    for i, col in enumerate(df.columns):
        if i in excluded_indices:
            continue  # Skip the conversion process for excluded columns

        # Attempt to convert the column to numeric, coercing errors (non-numeric entries) to NaN
        original_non_nan = df[col].notna()
        converted = pd.to_numeric(df[col], errors='coerce')
        introduced_nan = converted.isna() & original_non_nan

        # Log where NaNs were introduced due to non-numeric entries
        non_numeric_entries = df.loc[introduced_nan, col]
        for index, value in non_numeric_entries.items():
            non_numeric_log.append((index, col, value))

        # Replace the original column with the converted one
        df[col] = converted

    # Fill missing values for numerical columns with the median
    for col in df.select_dtypes(include=['number']).columns:
        df[col].fillna(df[col].median(), inplace=True)

    # Fill missing values for categorical columns with the mode
    for col in df.select_dtypes(exclude=['number']).columns:
        mode_value = df[col].mode().dropna()
        if not mode_value.empty:
            df[col].fillna(mode_value[0], inplace=True)

    # Drop rows with any missing values that might still exist
    df.dropna(inplace=True)

    # Print or save the log of non-numeric entries converted to NaN
    # for log_entry in non_numeric_log:
    #     print(
    #         f"Non-numeric entry converted to NaN - Index: {log_entry[0]}, Column: {log_entry[1]}, Value: '{log_entry[2]}'")

    return df



def remove_duplicates_nb(df):
    # Trim leading and trailing whitespace from string columns
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip()

    # Remove duplicates after trimming
    df = df.drop_duplicates()

    return df
