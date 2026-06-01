import pandas as pd
import plotly.express as px

class PlottingMethods:
    """
    A utility class providing isolated methods for generating granular HTML-wrapped,
    interactive Plotly visualizations.
    """

    @staticmethod
    def generate_bar_chart(df: pd.DataFrame, column: str) -> str:
        """Generates a bar chart displaying raw counts and percentage labels."""
        if df is None or column not in df.columns:
            return "<p>Error: Invalid data or column name.</p>"
        
        counts = df[column].value_counts().reset_index()
        counts.columns = [column, 'Count']
        counts['Percentage'] = (counts['Count'] / counts['Count'].sum() * 100).round(2)
        
        fig = px.bar(
            counts, x=column, y='Count',
            text=counts.apply(lambda r: f"{r['Count']} ({r['Percentage']}%)", axis=1),
            title=f"Frequency Distribution: {column}",
            labels={column: str(column), 'Count': 'Frequency'}
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(xaxis_type='category', template="plotly_white")
        return fig.to_html(include_plotlyjs='cdn', full_html=False)

    @staticmethod
    def generate_pie_chart(df: pd.DataFrame, column: str) -> str:
        """Generates a standard pie chart for categorical distributions."""
        if df is None or column not in df.columns:
            return "<p>Error: Invalid data or column name.</p>"
            
        fig = px.pie(df, names=column, title=f"Pie Composition: {column}", template="plotly_white")
        return fig.to_html(include_plotlyjs='cdn', full_html=False)

    @staticmethod
    def generate_histogram(df: pd.DataFrame, column: str, bins: int = 30) -> str:
        """Generates a simple, clean histogram for continuous features."""
        if df is None or column not in df.columns:
            return "<p>Error: Invalid data or column name.</p>"
            
        fig = px.histogram(df, x=column, nbins=bins, title=f"Histogram: {column}", template="plotly_white")
        return fig.to_html(include_plotlyjs='cdn', full_html=False)
