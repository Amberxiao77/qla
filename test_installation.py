#!/usr/bin/env python3
"""
Test script to verify the econometric tool installation and dependencies
"""

import sys
import importlib

def test_imports():
    """Test if all required packages are installed"""
    required_packages = [
        'streamlit',
        'pandas', 
        'numpy',
        'scipy',
        'statsmodels',
        'matplotlib',
        'seaborn',
        'plotly'
    ]
    
    print("Testing package imports...")
    print("=" * 50)
    
    all_good = True
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package} - OK")
        except ImportError as e:
            print(f"❌ {package} - FAILED: {e}")
            all_good = False
    
    print("=" * 50)
    if all_good:
        print("🎉 All packages installed successfully!")
        print("\nTo run the econometric tool:")
        print("1. Open a terminal/command prompt")
        print("2. Navigate to the econometric-tool directory")
        print("3. Run: streamlit run main.py")
        print("4. Open your browser to the displayed URL (typically http://localhost:8501)")
    else:
        print("❌ Some packages are missing. Please install them first.")
    
    return all_good

def test_sample_data():
    """Test loading the sample data"""
    try:
        import pandas as pd
        data = pd.read_csv('sample_data.csv')
        print(f"\n📊 Sample data loaded successfully!")
        print(f"   - Shape: {data.shape}")
        print(f"   - Columns: {list(data.columns)}")
        return True
    except Exception as e:
        print(f"\n❌ Error loading sample data: {e}")
        return False

if __name__ == "__main__":
    print("Econometric Tool - Installation Test")
    print("=" * 50)
    print(f"Python version: {sys.version}")
    print()
    
    imports_ok = test_imports()
    data_ok = test_sample_data()
    
    if imports_ok and data_ok:
        print("\n🚀 Ready to run the econometric tool!")
    else:
        print("\n⚠️  Please fix the issues above before running the tool.")