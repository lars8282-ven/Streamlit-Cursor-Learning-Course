# Oil & Gas Analytics Dashboard

A comprehensive Streamlit application for analyzing oil and gas production data, well performance, reservoir characteristics, and economic metrics.

## Features

### 📊 Overview Dashboard
- Key performance metrics (Total Production, Active Wells, etc.)
- Production trends visualization
- Well status and type distribution

### 📈 Production Analytics
- Production analysis by well
- Time-series production data with multiple aggregation levels
- Water cut analysis
- Production summary tables

### 🔧 Well Performance
- Individual well analysis
- Production history tracking
- Decline curve analysis
- Well-to-well comparison

### 🌊 Reservoir Analysis
- Reservoir properties by zone (Porosity, Permeability, Net Pay)
- Fluid saturation analysis
- Cross-plot analysis of reservoir properties
- Pressure and temperature monitoring

### 💰 Economic Analysis
- Revenue calculations based on oil/gas prices
- Operating cost analysis
- Net Present Value (NPV) calculations
- Revenue contribution by well
- Cumulative financial metrics

### 📅 Time Series Analysis
- Time-series visualization with multiple aggregation levels
- Statistical summary
- Seasonal pattern analysis
- Moving average analysis

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

Start the Streamlit app:
```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

## Data

The application uses sample data generated programmatically. In a production environment, you would replace the data generation functions with actual data loading from databases, CSV files, or APIs.

## Usage

1. **Navigation**: Use the sidebar to switch between different dashboard sections
2. **Filters**: Adjust filters (wells, date ranges, etc.) to focus on specific data
3. **Interactivity**: All charts are interactive - hover for details, zoom, pan, etc.
4. **Economic Parameters**: Adjust oil/gas prices and costs in the Economic Analysis section to see real-time impact

## Customization

- Modify the data generation functions to load your actual data
- Adjust economic parameters and calculations as needed
- Add additional metrics and visualizations
- Customize the color schemes and styling

## Requirements

- Python 3.8+
- Streamlit 1.28.0+
- Pandas 2.0.0+
- NumPy 1.24.0+
- Plotly 5.17.0+

## Notes

- The application uses sample data for demonstration purposes
- All calculations and visualizations are based on the generated sample dataset
- For production use, integrate with your actual data sources

