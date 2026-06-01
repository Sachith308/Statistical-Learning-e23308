import io
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats

class DataInspector:
    """
    An automated framework for CSV data ingestion, advanced sanitization,
    feature scaling, and interactive multivariate analysis inside Google Colab.
    """
    
    def __init__(self):
        self.df: pd.DataFrame = None

    # ==========================================
    # 1. Data Ingestion & Sanitization
    # ==========================================
    def upload_data(self) -> None:
        """Triggers native Google Colab file upload portal and sanitizes incoming CSV files."""
        try:
            from google.colab import files
        except ImportError:
            print("❌ This method is designed specifically for Google Colab environments.")
            return

        uploaded = files.upload()
        if not uploaded:
            print("❌ Upload cancelled.")
            return
        
        file_name = list(uploaded.keys())[0]
        # Ingest and map garbage strings directly to real NaN values
        garbage_strings = ['?', 'n/a', 'N/A', 'NULL', 'null', ' ', '']
        self.df = pd.read_csv(io.BytesIO(uploaded[file_name]), na_values=garbage_strings)
        print(f"✅ Successfully loaded '{file_name}' ({self.df.shape[0]} rows, {self.df.shape[1]} columns).")
        self._auto_type_correction()

    def _auto_type_correction(self) -> None:
        """Safely force-converts object columns to numerical types if viable."""
        if self.df is None: return
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                converted = pd.to_numeric(self.df[col], errors='coerce')
                if not converted.isna().all():
                    self.df[col] = converted

    # ==========================================
    # 2. Structural Analysis & Cleaning
    # ==========================================
    def data_summary(self) -> None:
        """Displays data metrics, feature splits, and structural dataframes."""
        if self.df is None:
            print("❌ No active dataset found.")
            return
        
        num_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = self.df.select_dtypes(exclude=[np.number]).columns.tolist()
        
        print("="*60)
        print(f" STRUCTURAL SUMMARY: {self.df.shape[0]} Rows | {self.df.shape[1]} Columns")
        print(f" Numeric Features ({len(num_cols)}): {num_cols}")
        print(f" Categorical Features ({len(cat_cols)}): {cat_cols}")
        print("="*60)
        print("\n📊 FIRST 20 ROWS PREVIEW:")
        display(self.df.head(20))

    def handle_missing_values(self, strategy: str = 'median', fill_value=None) -> None:
        """Implements intelligent row-wise imputation using strategic parameters."""
        if self.df is None: return
        
        for col in self.df.columns:
            if self.df[col].isna().sum() == 0:
                continue
                
            if strategy == 'constant':
                self.df[col] = self.df[col].fillna(fill_value)
            elif self.df[col].dtype in [np.float64, np.int64]:
                if strategy == 'mean':
                    self.df[col] = self.df[col].fillna(self.df[col].mean())
                elif strategy == 'median':
                    self.df[col] = self.df[col].fillna(self.df[col].median())
                elif strategy == 'mode':
                    self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
            else:
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0] if not self.df[col].mode().empty else "Missing")
        print(f"✅ Missing value resolution achieved via strategy: '{strategy}'.")

    def remove_duplicates(self) -> None:
        """Drops exact row duplications from the system memory wrapper."""
        if self.df is None: return
        initial_count = len(self.df)
        self.df.drop_duplicates(inplace=True)
        print(f"🧹 Removed {initial_count - len(self.df)} duplicate rows.")

    def handle_outliers(self, columns: list, action: str = 'flag') -> pd.DataFrame:
        """Uses the Interquartile Range (IQR) method to find anomalies."""
        if self.df is None: return None
        outlier_mask = pd.Series(False, index=self.df.index)
        
        for col in columns:
            if col in self.df.columns and self.df[col].dtype in [np.float64, np.int64]:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outlier_mask |= (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
                
        if action == 'delete':
            initial_len = len(self.df)
            self.df = self.df[~outlier_mask]
            print(f"🗑️ Dropped {initial_len - len(self.df)} rows containing IQR outliers.")
        else:
            self.df['is_outlier'] = outlier_mask.astype(int)
            print("🚩 Outliers flagged successfully in the 'is_outlier' column.")
        return self.df

    def delete_rows(self, indices_str: str) -> None:
        """Prunes targeted rows using comma-separated index strings."""
        if self.df is None: return
        try:
            indices = [int(idx.strip()) for idx in indices_str.split(',') if idx.strip().isdigit()]
            self.df.drop(index=indices, inplace=True, errors='ignore')
            print(f"✅ Successfully pruned rows: {indices}")
        except Exception as e:
            print(f"❌ Operation aborted: {e}")

    def delete_columns(self, cols_str: str) -> None:
        """Prunes targeted columns using comma-separated column names."""
        if self.df is None: return
        cols = [col.strip() for col in cols_str.split(',')]
        self.df.drop(columns=cols, inplace=True, errors='ignore')
        print(f"✅ Successfully dropped columns: {cols}")

    # ==========================================
    # 3. Feature Engineering Preparation
    # ==========================================
    def extract_normalized_numeric_data(self, strategy: str = 'standard') -> pd.DataFrame:
        """Returns a scaled dataframe containing only the numeric columns."""
        if self.df is None: return pd.DataFrame()
        num_df = self.df.select_dtypes(include=[np.number]).copy()
        if 'is_outlier' in num_df.columns: num_df.drop(columns=['is_outlier'], inplace=True)
        
        for col in num_df.columns:
            if strategy == 'minmax':
                min_val, max_val = num_df[col].min(), num_df[col].max()
                num_df[col] = (num_df[col] - min_val) / (max_val - min_val + 1e-9)
            elif strategy == 'standard':
                mean, std = num_df[col].mean(), num_df[col].std()
                num_df[col] = (num_df[col] - mean) / (std + 1e-9)
            elif strategy == 'robust':
                q25, q50, q75 = num_df[col].quantile(0.25), num_df[col].median(), num_df[col].quantile(0.75)
                iqr = q75 - q25
                num_df[col] = (num_df[col] - q50) / (iqr + 1e-9)
        return num_df

    def extract_normalized_categorical_data(self, strategy: str = 'onehot') -> pd.DataFrame:
        """Encodes and structures categorical columns using the chosen strategy."""
        if self.df is None: return pd.DataFrame()
        cat_df = self.df.select_dtypes(exclude=[np.number]).copy()
        if cat_df.empty: return pd.DataFrame()
        
        if strategy == 'onehot':
            return pd.get_dummies(cat_df, dtype=float)
        elif strategy in ['ordinal', 'uniform']:
            for col in cat_df.columns:
                cat_df[col] = cat_df[col].astype('category').cat.codes
                if strategy == 'uniform' and cat_df[col].max() > 0:
                    cat_df[col] = cat_df[col] / cat_df[col].max()
            return cat_df.astype(float)
        return pd.DataFrame()

    def merge_processed_datasets(self, num_strategy: str = 'standard', cat_strategy: str = 'onehot') -> pd.DataFrame:
        """Merges scaled numerical and encoded categorical data into one unified DataFrame."""
        num_part = self.extract_normalized_numeric_data(strategy=num_strategy)
        cat_part = self.extract_normalized_categorical_data(strategy=cat_strategy)
        return pd.concat([num_part, cat_part], axis=1)

    # ==========================================
    # 4. Advanced Interactive Visualization
    # ==========================================
    def plot_univariate_subplots(self, column: str) -> None:
        """Generates a 3-panel layout (Box/Violin, Scatter, Histogram) for a numeric feature."""
        if self.df is None or column not in self.df.columns: return
        
        fig = make_subplots(
            rows=3, cols=1, 
            subplot_titles=(f"Violin & Box Plot: {column}", f"Index Scatter Plot: {column}", f"Distribution Histogram: {column}"),
            shared_xaxes=False, vertical_spacing=0.1
        )
        
        fig.add_trace(go.Violin(x=self.df[column], box_visible=True, meanline_visible=True, name="Distribution"), row=1, col=1)
        fig.add_trace(go.Scatter(x=self.df.index, y=self.df[column], mode='markers', name="Index Value"), row=2, col=1)
        fig.add_trace(go.Histogram(x=self.df[column], name="Frequency Bins"), row=3, col=1)
        
        fig.update_layout(height=800, title_text=f"Multi-Panel Inspection Framework for '{column}'", template="plotly_white", showlegend=False)
        fig.show()

    def plot_relationship(self, col_x: str, col_y: str) -> None:
        """Automatically selects and renders the optimal graph type based on column categories."""
        if self.df is None or col_x not in self.df.columns or col_y not in self.df.columns: return
        
        is_x_num = self.df[col_x].dtype in [np.float64, np.int64]
        is_y_num = self.df[col_y].dtype in [np.float64, np.int64]
        
        if is_x_num and is_y_num:
            fig = px.scatter(self.df, x=col_x, y=col_y, trendline="ols", title=f"Bivariate Analysis: {col_x} vs {col_y}")
        elif not is_x_num and not is_y_num:
            counts = self.df.groupby([col_x, col_y]).size().reset_index(name='Counts')
            fig = px.bar(counts, x=col_x, y='Counts', color=col_y, barmode='group', title=f"Multivariate Frequency Matrix: {col_x} vs {col_y}")
        else:
            cat_col = col_x if not is_x_num else col_y
            num_col = col_y if is_y_num else col_x
            fig = px.box(self.df, x=cat_col, y=num_col, points="all", title=f"Structural Box Map: {num_col} by {cat_col}")
            
        fig.update_layout(template="plotly_white")
        fig.show()

    # ==========================================
    # 5. Deep Statistical Insights
    # ==========================================
    def plot_all_associations_heatmap(self) -> None:
        """Computes a unified association matrix (Pearson, Cramér's V, Eta)."""
        if self.df is None: return
        cols = [c for c in self.df.columns if c != 'is_outlier']
        n = len(cols)
        matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                col1, col2 = cols[i], cols[j]
                is_1_num = self.df[col1].dtype in [np.float64, np.int64]
                is_2_num = self.df[col2].dtype in [np.float64, np.int64]
                
                if i == j:
                    matrix[i, j] = 1.0
                elif is_1_num and is_2_num:
                    val, _ = stats.pearsonr(self.df[col1].dropna(), self.df[col2].dropna()) if len(self.df[[col1, col2]].dropna()) > 1 else (0, 0)
                    matrix[i, j] = abs(val)
                elif not is_1_num and not is_2_num:
                    confusion_matrix = pd.crosstab(self.df[col1], self.df[col2])
                    if confusion_matrix.size > 0 and confusion_matrix.sum().sum() > 0:
                        chi2 = stats.chi2_contingency(confusion_matrix)[0]
                        total = confusion_matrix.sum().sum()
                        phi2 = chi2 / total
                        r, k = confusion_matrix.shape
                        phi2corr = max(0, phi2 - ((k-1)*(r-1))/(total-1))
                        rcorr = r - ((r-1)**2)/(total-1)
                        kcorr = k - ((k-1)**2)/(total-1)
                        matrix[i, j] = np.sqrt(phi2corr / min((kcorr-1), (rcorr-1))) if min((kcorr-1), (rcorr-1)) > 0 else 0
                    else:
                        matrix[i, j] = 0
                else:
                    num_col = col1 if is_1_num else col2
                    cat_col = col2 if is_1_num else col1
                    clean_df = self.df[[num_col, cat_col]].dropna()
                    groups = [group[num_col].values for name, group in clean_df.groupby(cat_col) if len(group) > 0]
                    
                    if len(groups) > 1 and sum(len(g) for g in groups) > len(groups):
                        f_val, _ = stats.f_oneway(*groups)
                        df_num = len(groups) - 1
                        df_den = len(clean_df) - len(groups)
                        if (f_val * df_num + df_den) > 0:
                            matrix[i, j] = np.sqrt((f_val * df_num) / (f_val * df_num + df_den))
                        else:
                            matrix[i, j] = 0
                    else:
                        matrix[i, j] = 0
                        
        fig = go.Figure(data=go.Heatmap(
            z=matrix, x=cols, y=cols,
            colorscale='Blues', zmin=0, zmax=1,
            text=np.round(matrix, 2), texttemplate="%{text}", hoverinfo="z"
        ))
        fig.update_layout(title="Unified Association Heatmap (Pearson / Cramér's V / Eta)", template="plotly_white")
        fig.show()
