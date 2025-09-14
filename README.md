# Econometric Tool

A simple web-based econometric tool built with Streamlit that allows users to:

1. Upload CSV files with economic data
2. Explore data series and their properties
3. Perform OLS regression analysis
4. View comprehensive regression statistics
5. Visualize marginal relationships between variables

## Installation

1. Install Python 3.8 or higher
2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the application:
```bash
streamlit run main.py
```

Then open your browser to the displayed URL (typically http://localhost:8501)

## Features

- **File Upload**: Upload CSV files with economic data
- **Data Exploration**: View variable summaries and basic statistics
- **Variable Selection**: Choose dependent and independent variables
- **OLS Regression**: Comprehensive ordinary least squares analysis
- **Results Display**: Statistical outputs including R², p-values, confidence intervals
- **Visualization**: Scatter plots with fitted regression lines

## Dependencies

- Streamlit: Web application framework
- Pandas: Data manipulation and analysis
- NumPy: Numerical computing
- SciPy: Scientific computing
- Statsmodels: Statistical models and econometric analysis
- Matplotlib/Seaborn: Data visualization
- Plotly: Interactive visualizations