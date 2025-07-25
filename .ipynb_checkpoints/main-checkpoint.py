import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
import numpy as np

# Load both datasets
original = pd.read_csv("PCOS_data.csv")
new = pd.read_csv("pcos_dataset.csv")

def preprocess(df):
    df = df.copy()  # avoid SettingWithCopyWarning

    # 1. Clean column names
    df.columns = df.columns.str.strip() \
                           .str.replace(' ', '_') \
                           .str.replace('(', '') \
                           .str.replace(')', '') \
                           .str.replace('.', '') \
                           .str.replace('-', '_') \
                           .str.replace('/', '_')
    df.rename(columns={'II____beta_HCGmIU_mL': 'II_beta_HCG'}, inplace=True)

    # 2. Drop irrelevant columns
    df.drop(columns=['Sl_No', 'Patient_File_No'], inplace=True, errors='ignore')
    df = df.loc[:, ~df.columns.str.startswith('Unnamed')]

    # 3. Merge Age columns
    if 'Age' not in df.columns and 'Age_yrs' in df.columns:
        df.rename(columns={'Age_yrs': 'Age'}, inplace=True)
    elif 'Age' in df.columns and 'Age_yrs' in df.columns:
        df['Age'] = df['Age'].fillna(df['Age_yrs'])
        df.drop(columns=['Age_yrs'], inplace=True)

    # 4. Merge PCOS diagnosis columns
    if 'PCOS_Diagnosis' in df.columns:
        df.rename(columns={'PCOS_Diagnosis': 'PCOS_Y_N'}, inplace=True)

    # 5. Handle missing values
    if 'Marraige_Status_Yrs' in df.columns:
        df.loc[:, 'Marraige_Status_Yrs'] = df['Marraige_Status_Yrs'].fillna(df['Marraige_Status_Yrs'].median())

    if 'Fast_food_Y_N' not in df.columns and 'Fast_food_YN' in df.columns:
        df.rename(columns={'Fast_food_YN': 'Fast_food_Y_N'}, inplace=True)
    if 'Fast_food_Y_N' in df.columns:
        df.loc[:, 'Fast_food_Y_N'] = df['Fast_food_Y_N'].fillna(df['Fast_food_Y_N'].mode()[0])

    # 6. Convert to numeric and fill missing values
    if 'II_beta_HCG' not in df.columns and 'II_beta_HCGmIU_mL' in df.columns:
        df.rename(columns={'II_beta_HCGmIU_mL': 'II_beta_HCG'}, inplace=True)
    if 'II_beta_HCG' in df.columns:
        df.loc[:, 'II_beta_HCG'] = pd.to_numeric(df['II_beta_HCG'], errors='coerce')
        df['II_beta_HCG'] = df['II_beta_HCG'].astype(float)
        df.loc[:, 'II_beta_HCG'] = df['II_beta_HCG'].fillna(df['II_beta_HCG'].median())

    if 'AMHng_mL' in df.columns:
        df.loc[:, 'AMHng_mL'] = pd.to_numeric(df['AMHng_mL'], errors='coerce')
        df['AMHng_mL'] = df['AMHng_mL'].astype(float)
        df.loc[:, 'AMHng_mL'] = df['AMHng_mL'].fillna(df['AMHng_mL'].median())

    return df
# Apply preprocessing
original_clean = preprocess(original)
new_clean = preprocess(new)

# Ensure consistent columns across both
all_columns = list(set(original_clean.columns).union(set(new_clean.columns)))

# Align both dataframes to same columns, fill missing with NaN
original_aligned = original_clean.reindex(columns=all_columns)
new_aligned = new_clean.reindex(columns=all_columns)

# Concatenate datasets
combined_df = pd.concat([original_aligned, new_aligned], ignore_index=True)

# Separate features and target
X = combined_df.drop(columns=['PCOS_Y_N'])
y = combined_df['PCOS_Y_N']

# Initialize KNNImputer (e.g., with 5 nearest neighbors)
knn_imputer = KNNImputer(n_neighbors=5)

# Fit and transform only on features
X_imputed = knn_imputer.fit_transform(X)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)


#Final confirmation
print(combined_df.columns.tolist())
print("Final features shape:", X_scaled.shape)
print("Target distribution:\n", y.value_counts())
print(f"Merged dataset shape: {combined_df.shape}")
print(f"Number of duplicates: {combined_df.duplicated().sum()}")
print(f"Number of missing values:\n{combined_df.isnull().sum()}")
print("Missing values after KNN imputation:", np.isnan(X_imputed).sum())

print(combined_df.info())
