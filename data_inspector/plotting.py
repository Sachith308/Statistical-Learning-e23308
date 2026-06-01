import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class PlottingMethods:
    """
    A modular class handling granular chart generation using Plotly.
    """
    @staticmethod
    def univariate_subplots(df, column):
        """Generates a 3-panel subplot for a numeric column."""
        if column not in df.select_dtypes(include=[np.number]).columns:
            print(f"Error: '{column}' is not numeric.")
            return
            
        fig = make_subplots(
            rows=1, cols=3, 
            subplot_titles=("Distribution (Violin)", "Scatter (Index vs Value)", "Histogram")
        )
        
        fig.add_trace(go.Violin(x=df[column], box_visible=True, meanline_visible=True, name=""), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df[column], mode='markers', name=""), row=1, col=2)
        fig.add_trace(go.Histogram(x=df[column], name=""), row=1, col=3)
        
        fig.update_layout(height=400, width=1200, title_text=f"Univariate Analysis: {column}", showlegend=False)
        fig.show()

    @staticmethod
    def plot_relationship(df, col1, col2):
        """Detects data types and plots the appropriate relationship chart."""
        is_num1 = pd.api.types.is_numeric_dtype(df[col1])
        is_num2 = pd.api.types.is_numeric_dtype(df[col2])
        
        if is_num1 and is_num2:
            fig = px.scatter(df, x=col1, y=col2, trendline="ols", title=f"Scatter: {col1} vs {col2}")
        elif not is_num1 and not is_num2:
            counts = df.groupby([col1, col2]).size().reset_index(name='count')
            fig = px.bar(counts, x=col1, y='count', color=col2, barmode='group', title=f"Grouped Bar: {col1} vs {col2}")
        else:
            cat_col, num_col = (col1, col2) if not is_num1 else (col2, col1)
            fig = px.box(df, x=cat_col, y=num_col, points="all", title=f"Box Plot: {cat_col} vs {num_col}")
            
        fig.show()

    @staticmethod
    def plot_heatmap(matrix_df, title="Association Heatmap"):
        """Renders a correlation/association matrix as a Plotly heatmap."""
        fig = px.imshow(
            matrix_df, 
            text_auto=".2f", 
            aspect="auto",
            color_continuous_scale="RdBu_r",
            title=title
        )
        fig.show()
