import pandas as pd
import numpy as np
import io
import scipy.stats as stats
from .plotting import PlottingMethods

try:
    from google.colab import files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

class DataInspector:
    """
    Main engine for data ingestion, sanitization, cleaning, and preparation.
    """
    def __init__(self, dataframe=None):
        self.df = dataframe.copy() if dataframe is not None else None

    @classmethod
    def upload_data(cls):
        """Integrates with Google Colab to upload a local CSV file."""
        if not IN_COLAB:
            print("Not in Google Colab. Please instantiate with a dataframe directly.")
            return None
            
        print("Please upload your CSV file:")
        uploaded = files.upload()
        if not uploaded:
            print("No file uploaded.")
            return None
            
        filename = list(uploaded.keys())[0]
        df = pd.read_csv(io.BytesIO(uploaded[filename]))
        print(f"Successfully loaded {filename}")
        
        instance = cls(df)
        instance._sanitize_data()
        return instance

    def _sanitize_data(self):
        """Internal method: Cleans garbage strings and auto-corrects types."""
        if self.df is None: return
        garbage = ['?', 'n/a', 'NULL', ' ', '', 'NA', 'nan']
        self.df.replace(garbage, np.nan, inplace=True)
        
        for col in self.df.columns:
            converted = pd.to_numeric(self.df[col], errors='coerce')
            if not converted.isna().all():
                self.df[col] = converted

    def data_summary(self):
        """Displays row/col counts, previews data, and breaks down data types."""
        if self.df is None: return
        print("-" * 50)
        print(f"Dataset Shape: {self.df.shape[0]} Rows, {self.df.shape[1]} Columns")
        print("-" * 50)
        
        num_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = self.df.select_dtypes(exclude=[np.number]).columns.tolist()
        
        print(f"Numerical Columns ({len(num_cols)}): {num_cols}")
        print(f"Categorical Columns ({len(cat_cols)}): {cat_cols}")
        print("-" * 50)
        display(self.df.head())

    def handle_missing_values(self, strategy='mean', constant_value=None):
        """Imputes missing values."""
        num_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in self.df.columns:
            if self.df[col].isnull().sum() == 0: continue
            
            if strategy == 'constant' and constant_value is not None:
                self.df[col].fillna(constant_value, inplace=True)
            elif strategy == 'mode':
                self.df[col].fillna(self.df[col].mode()[0], inplace=True)
            elif col in num_cols:
                if strategy == 'mean':
                    self.df[col].fillna(self.df[col].mean(), inplace=True)
                elif strategy == 'median':
                    self.df[col].fillna(self.df[col].median(), inplace=True)

    def remove_duplicates(self):
        """Prunes exact row matches."""
        initial_rows = len(self.df)
        self.df.drop_duplicates(inplace=True)
        print(f"Removed {initial_rows - len(self.df)} duplicate rows.")

    def handle_outliers(self, column, action='flag'):
        """IQR-based outlier detection."""
        if column not in self.df.select_dtypes(include=[np.number]).columns:
            print("Requires a numeric column.")
            return

        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outlier_mask = (self.df[column] < lower_bound) | (self.df[column] > upper_bound)
        
        if action == 'flag':
            self.df[f'{column}_outlier'] = outlier_mask
            print(f"Flagged {outlier_mask.sum()} outliers in '{column}'.")
        elif action == 'delete':
            self.df = self.df[~outlier_mask].reset_index(drop=True)
            print(f"Deleted {outlier_mask.sum()} outliers in '{column}'.")

    # --- Feature Engineering ---
    def extract_normalized_numeric(self, method='standard'):
        """Scales numeric data (minmax, standard, robust)."""
        num_df = self.df.select_dtypes(include=[np.number]).copy()
        for col in num_df.columns:
            if method == 'minmax':
                num_df[col] = (num_df[col] - num_df[col].min()) / (num_df[col].max() - num_df[col].min())
            elif method == 'standard':
                num_df[col] = (num_df[col] - num_df[col].mean()) / num_df[col].std()
            elif method == 'robust':
                q1, q3 = num_df[col].quantile(0.25), num_df[col].quantile(0.75)
                num_df[col] = (num_df[col] - num_df[col].median()) / (q3 - q1)
        return num_df

    def extract_normalized_categorical(self, method='onehot'):
        """Encodes categorical data (onehot, ordinal, uniform)."""
        cat_df = self.df.select_dtypes(exclude=[np.number]).copy()
        if method == 'onehot':
            return pd.get_dummies(cat_df, drop_first=False)
        elif method == 'ordinal':
            for col in cat_df.columns:
                cat_df[col] = pd.factorize(cat_df[col])[0]
            return cat_df
        elif method == 'uniform':
            for col in cat_df.columns:
                factorized = pd.factorize(cat_df[col])[0]
                if factorized.max() > 0:
                    cat_df[col] = factorized / factorized.max()
            return cat_df

    def get_unified_dataframe(self):
        """Merges scaled numeric and encoded categorical data."""
        num_df = self.extract_normalized_numeric()
        cat_df = self.extract_normalized_categorical()
        return pd.concat([num_df, cat_df], axis=1)

    # --- Advanced Visualization & Statistics ---
    def visualize_univariate(self, column):
        PlottingMethods.univariate_subplots(self.df, column)
        
    def visualize_relationship(self, col1, col2):
        PlottingMethods.plot_relationship(self.df, col1, col2)

    def plot_all_associations_heatmap(self):
        """Calculates Pearson/Cramer's V/ANOVA and plots a unified heatmap."""
        cols = self.df.columns
        matrix = pd.DataFrame(index=cols, columns=cols, dtype=float)

        for c1 in cols:
            for c2 in cols:
                if c1 == c2:
                    matrix.loc[c1, c2] = 1.0
                    continue
                
                is_num1 = pd.api.types.is_numeric_dtype(self.df[c1])
                is_num2 = pd.api.types.is_numeric_dtype(self.df[c2])
                
                # Drop NAs for pair comparison
                pair_df = self.df[[c1, c2]].dropna()
                if len(pair_df) < 2:
                    matrix.loc[c1, c2] = np.nan
                    continue
                    
                if is_num1 and is_num2:
                    # Pearson's r
                    matrix.loc[c1, c2] = pair_df[c1].corr(pair_df[c2])
                elif not is_num1 and not is_num2:
                    # Cramer's V
                    confusion_matrix = pd.crosstab(pair_df[c1], pair_df[c2])
                    chi2 = stats.chi2_contingency(confusion_matrix)[0]
                    n = confusion_matrix.sum().sum()
                    r, k = confusion_matrix.shape
                    if min((k-1), (r-1)) == 0:
                        matrix.loc[c1, c2] = np.nan
                    else:
                        matrix.loc[c1, c2] = np.sqrt((chi2 / n) / min((k-1), (r-1)))
                else:
                    # ETA (ANOVA-based) for Mixed Types
                    num_c, cat_c = (c1, c2) if is_num1 else (c2, c1)
                    categories = pair_df[cat_c].unique()
                    groups = [pair_df[pair_df[cat_c] == cat][num_c].values for cat in categories]
                    if len(groups) > 1:
                        f_val, _ = stats.f_oneway(*groups)
                        df_b = len(groups) - 1
                        df_w = len(pair_df) - len(groups)
                        matrix.loc[c1, c2] = np.sqrt((f_val * df_b) / ((f_val * df_b) + df_w))
                    else:
                        matrix.loc[c1, c2] = np.nan

        PlottingMethods.plot_heatmap(matrix.astype(float), "Unified Data Association Heatmap")
