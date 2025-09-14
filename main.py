import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Lasso, Ridge, ElasticNet
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Advanced Analytics Platform",
    page_icon="�",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """Main application function"""
    st.title("📈 Econometric Analysis Tool")
    st.markdown("**Upload your data, explore variables, and perform OLS regression analysis**")
    
    # Sidebar for navigation
    st.sidebar.header("Navigation")
    
    # File upload section
    st.header("1. Data Upload")
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type="csv",
        help="Upload a CSV file containing your economic data. The first row should contain column headers."
    )
    
    if uploaded_file is not None:
        try:
            # Load the data
            data = load_data(uploaded_file)
            
            if data is not None:
                st.success(f"✅ Data loaded successfully! Dataset contains {data.shape[0]} observations and {data.shape[1]} variables.")
                
                # Store data in session state
                st.session_state['data'] = data
                
                # Show data preview
                show_data_preview(data)
                
                # Data exploration section
                show_data_exploration(data)
                
                # Variable selection and regression
                perform_regression_analysis(data)
                
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
            st.info("Please ensure your CSV file is properly formatted with headers in the first row.")

def load_data(uploaded_file):
    """Load and validate CSV data"""
    try:
        # Try different encodings
        encodings = ['utf-8', 'latin-1', 'cp1252']
        data = None
        
        for encoding in encodings:
            try:
                uploaded_file.seek(0)  # Reset file pointer
                data = pd.read_csv(uploaded_file, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        
        if data is None:
            st.error("Could not read the file with any common encoding.")
            return None
        
        # Basic validation
        if data.empty:
            st.error("The uploaded file is empty.")
            return None
        
        if data.shape[1] < 2:
            st.error("The dataset must have at least 2 columns for regression analysis.")
            return None
        
        # Check for numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            st.warning("⚠️ Dataset has fewer than 2 numeric columns. Some features may be limited.")
        
        return data
        
    except Exception as e:
        st.error(f"Error reading CSV file: {str(e)}")
        return None

def show_data_preview(data):
    """Display data preview and basic information"""
    st.header("2. Data Preview")
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["📋 Data Sample", "📊 Summary Statistics", "🔍 Data Info"])
    
    with tab1:
        st.subheader("First 10 rows of your data:")
        st.dataframe(data.head(10), use_container_width=True)
    
    with tab2:
        st.subheader("Summary Statistics:")
        numeric_data = data.select_dtypes(include=[np.number])
        if not numeric_data.empty:
            st.dataframe(numeric_data.describe(), use_container_width=True)
        else:
            st.warning("No numeric columns found for summary statistics.")
    
    with tab3:
        st.subheader("Dataset Information:")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Observations", data.shape[0])
            st.metric("Total Variables", data.shape[1])
        
        with col2:
            numeric_cols = len(data.select_dtypes(include=[np.number]).columns)
            categorical_cols = len(data.select_dtypes(include=['object']).columns)
            st.metric("Numeric Variables", numeric_cols)
            st.metric("Categorical Variables", categorical_cols)
        
        # Show data types
        st.subheader("Variable Types:")
        dtype_df = pd.DataFrame({
            'Variable': data.columns,
            'Data Type': data.dtypes,
            'Missing Values': data.isnull().sum(),
            'Missing %': (data.isnull().sum() / len(data) * 100).round(2)
        })
        st.dataframe(dtype_df, use_container_width=True)

def show_data_exploration(data):
    """Show data exploration section"""
    st.header("3. Data Exploration")
    
    numeric_data = data.select_dtypes(include=[np.number])
    
    if numeric_data.empty:
        st.warning("No numeric variables available for exploration.")
        return
    
    # Correlation matrix
    if len(numeric_data.columns) > 1:
        st.subheader("Correlation Matrix")
        fig, ax = plt.subplots(figsize=(10, 8))
        correlation_matrix = numeric_data.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
        plt.title("Variable Correlations")
        st.pyplot(fig)
        plt.close()

def perform_regression_analysis(data):
    """Perform OLS regression analysis"""
    st.header("4. Regression Analysis")
    
    numeric_data = data.select_dtypes(include=[np.number])
    
    if len(numeric_data.columns) < 2:
        st.warning("Need at least 2 numeric variables for regression analysis.")
        return
    
    # Variable selection
    st.subheader("Variable Selection")
    
    col1, col2 = st.columns(2)
    
    with col1:
        dependent_var = st.selectbox(
            "Select Dependent Variable (Y):",
            options=numeric_data.columns,
            help="Choose the variable you want to predict or explain"
        )
    
    with col2:
        independent_vars = st.multiselect(
            "Select Independent Variable(s) (X):",
            options=[col for col in numeric_data.columns if col != dependent_var],
            help="Choose one or more explanatory variables"
        )
    
    if dependent_var and independent_vars:
        # Prepare data for regression
        regression_data = prepare_regression_data(data, dependent_var, independent_vars)
        
        if regression_data is not None:
            # Perform OLS regression
            results = perform_ols_regression(regression_data, dependent_var, independent_vars)
            
            if results is not None:
                # Display results
                display_regression_results(results, dependent_var, independent_vars)
                
                # Create visualization
                create_regression_visualization(regression_data, dependent_var, independent_vars[0], results)

def prepare_regression_data(data, dependent_var, independent_vars):
    """Prepare data for regression analysis"""
    try:
        # Select relevant columns
        regression_cols = [dependent_var] + independent_vars
        regression_data = data[regression_cols].copy()
        
        # Handle missing values
        initial_rows = len(regression_data)
        regression_data = regression_data.dropna()
        final_rows = len(regression_data)
        
        if final_rows < initial_rows:
            st.warning(f"⚠️ Removed {initial_rows - final_rows} rows with missing values. "
                      f"Analysis based on {final_rows} observations.")
        
        if final_rows < 10:
            st.error("Insufficient data for reliable regression analysis (need at least 10 observations).")
            return None
        
        return regression_data
        
    except Exception as e:
        st.error(f"Error preparing data: {str(e)}")
        return None

def perform_ols_regression(data, dependent_var, independent_vars):
    """Perform OLS regression using statsmodels"""
    try:
        # Prepare variables
        y = data[dependent_var]
        X = data[independent_vars]
        
        # Add constant for intercept
        X = sm.add_constant(X)
        
        # Fit the model
        model = sm.OLS(y, X)
        results = model.fit()
        
        return results
        
    except Exception as e:
        st.error(f"Error performing regression: {str(e)}")
        return None

def display_regression_results(results, dependent_var, independent_vars):
    """Display comprehensive regression results"""
    st.subheader("Regression Results")
    
    # Create tabs for different result sections
    tab1, tab2, tab3 = st.tabs(["📊 Main Results", "🔬 Diagnostic Tests", "📈 Model Statistics"])
    
    with tab1:
        # Main regression table
        st.write("**OLS Regression Results**")
        st.write(f"**Dependent Variable:** {dependent_var}")
        st.write(f"**Independent Variables:** {', '.join(independent_vars)}")
        
        # Create results dataframe
        results_df = pd.DataFrame({
            'Variable': results.params.index,
            'Coefficient': results.params.values,
            'Std Error': results.bse.values,
            't-statistic': results.tvalues.values,
            'P-value': results.pvalues.values,
            'CI Lower (2.5%)': results.conf_int()[0].values,
            'CI Upper (97.5%)': results.conf_int()[1].values
        })
        
        # Format the dataframe
        results_df['Coefficient'] = results_df['Coefficient'].round(6)
        results_df['Std Error'] = results_df['Std Error'].round(6)
        results_df['t-statistic'] = results_df['t-statistic'].round(3)
        results_df['P-value'] = results_df['P-value'].round(6)
        results_df['CI Lower (2.5%)'] = results_df['CI Lower (2.5%)'].round(6)
        results_df['CI Upper (97.5%)'] = results_df['CI Upper (97.5%)'].round(6)
        
        st.dataframe(results_df, use_container_width=True)
        
        # Significance indicators
        st.write("**Significance levels:** *** p<0.01, ** p<0.05, * p<0.1")
    
    with tab2:
        # Model diagnostics
        st.write("**Diagnostic Tests**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Breusch-Pagan test for heteroscedasticity
            try:
                bp_test = het_breuschpagan(results.resid, results.model.exog)
                st.metric("Breusch-Pagan Test", f"p-value: {bp_test[1]:.4f}")
                if bp_test[1] < 0.05:
                    st.warning("⚠️ Evidence of heteroscedasticity (p < 0.05)")
                else:
                    st.success("✅ No evidence of heteroscedasticity")
            except:
                st.info("Could not perform Breusch-Pagan test")
        
        with col2:
            # Durbin-Watson test for autocorrelation
            try:
                dw_stat = durbin_watson(results.resid)
                st.metric("Durbin-Watson Statistic", f"{dw_stat:.4f}")
                if dw_stat < 1.5 or dw_stat > 2.5:
                    st.warning("⚠️ Possible autocorrelation (DW not between 1.5-2.5)")
                else:
                    st.success("✅ No strong evidence of autocorrelation")
            except:
                st.info("Could not calculate Durbin-Watson statistic")
    
    with tab3:
        # Model statistics
        st.write("**Model Performance Statistics**")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("R-squared", f"{results.rsquared:.4f}")
            st.metric("Adjusted R-squared", f"{results.rsquared_adj:.4f}")
        
        with col2:
            st.metric("F-statistic", f"{results.fvalue:.2f}")
            st.metric("F-statistic p-value", f"{results.f_pvalue:.6f}")
        
        with col3:
            st.metric("Number of Observations", f"{int(results.nobs)}")
            st.metric("Degrees of Freedom", f"{int(results.df_resid)}")
        
        with col4:
            st.metric("AIC", f"{results.aic:.2f}")
            st.metric("BIC", f"{results.bic:.2f}")

def create_regression_visualization(data, dependent_var, main_independent_var, results):
    """Create visualization of the regression relationship"""
    st.subheader("5. Regression Visualization")
    
    try:
        # Create scatter plot with regression line
        fig = px.scatter(
            data, 
            x=main_independent_var, 
            y=dependent_var,
            title=f"Relationship between {main_independent_var} and {dependent_var}",
            labels={
                main_independent_var: main_independent_var,
                dependent_var: dependent_var
            }
        )
        
        # Add regression line
        X_plot = np.linspace(data[main_independent_var].min(), data[main_independent_var].max(), 100)
        
        # For simple regression, get the coefficient and intercept
        if len(results.params) == 2:  # Simple regression (intercept + one variable)
            intercept = results.params[0]
            slope = results.params[1]
            y_plot = intercept + slope * X_plot
        else:  # Multiple regression - show marginal effect of main variable
            # Set other variables to their means
            mean_values = data[results.model.exog_names[1:]].mean()
            y_plot = results.params[0]  # intercept
            
            for i, var in enumerate(results.model.exog_names[1:]):
                if var == main_independent_var:
                    y_plot += results.params[i+1] * X_plot
                else:
                    y_plot += results.params[i+1] * mean_values[var]
        
        fig.add_trace(
            go.Scatter(
                x=X_plot, 
                y=y_plot, 
                mode='lines', 
                name='Regression Line',
                line=dict(color='red', width=2)
            )
        )
        
        fig.update_layout(
            width=800,
            height=500,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display equation
        if len(results.params) == 2:
            equation = f"{dependent_var} = {results.params[0]:.4f} + {results.params[1]:.4f} × {main_independent_var}"
        else:
            equation = f"{dependent_var} = {results.params[0]:.4f}"
            for i, var in enumerate(results.model.exog_names[1:]):
                coef = results.params[i+1]
                sign = "+" if coef >= 0 else ""
                equation += f" {sign} {coef:.4f} × {var}"
        
        st.write(f"**Regression Equation:** {equation}")
        
        # Residual plot
        st.subheader("Residual Analysis")
        
        fig_resid = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Residuals vs Fitted Values", "Q-Q Plot of Residuals")
        )
        
        # Residuals vs fitted
        fitted_values = results.fittedvalues
        residuals = results.resid
        
        fig_resid.add_trace(
            go.Scatter(
                x=fitted_values,
                y=residuals,
                mode='markers',
                name='Residuals',
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Add horizontal line at y=0
        fig_resid.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
        
        # Q-Q plot (simplified)
        from scipy import stats
        (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm", plot=None)
        
        fig_resid.add_trace(
            go.Scatter(
                x=osm,
                y=osr,
                mode='markers',
                name='Q-Q Plot',
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Add Q-Q line
        fig_resid.add_trace(
            go.Scatter(
                x=osm,
                y=slope * osm + intercept,
                mode='lines',
                name='Q-Q Line',
                line=dict(color='red'),
                showlegend=False
            ),
            row=1, col=2
        )
        
        fig_resid.update_layout(height=400, title_text="Diagnostic Plots")
        fig_resid.update_xaxes(title_text="Fitted Values", row=1, col=1)
        fig_resid.update_yaxes(title_text="Residuals", row=1, col=1)
        fig_resid.update_xaxes(title_text="Theoretical Quantiles", row=1, col=2)
        fig_resid.update_yaxes(title_text="Sample Quantiles", row=1, col=2)
        
        st.plotly_chart(fig_resid, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating visualization: {str(e)}")

if __name__ == "__main__":
    main()