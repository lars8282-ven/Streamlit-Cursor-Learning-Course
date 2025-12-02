import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Oil & Gas Analytics Dashboard",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🛢️ Oil & Gas Analytics Dashboard</h1>', unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Select Dashboard",
    ["Overview", "Production Analytics", "Well Performance", "Reservoir Analysis", "Economic Analysis", "Time Series Analysis"]
)

# Generate sample data
@st.cache_data
def generate_production_data():
    """Generate sample production data"""
    dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='D')
    np.random.seed(42)
    
    data = {
        'Date': dates,
        'Well_ID': np.random.choice(['Well-A', 'Well-B', 'Well-C', 'Well-D', 'Well-E'], len(dates)),
        'Oil_Production_BBL': np.random.normal(500, 100, len(dates)).clip(0),
        'Gas_Production_MCF': np.random.normal(2000, 400, len(dates)).clip(0),
        'Water_Production_BBL': np.random.normal(100, 30, len(dates)).clip(0),
        'Pressure_PSI': np.random.normal(2500, 200, len(dates)),
        'Temperature_F': np.random.normal(180, 20, len(dates)),
    }
    return pd.DataFrame(data)

@st.cache_data
def generate_well_data():
    """Generate sample well data"""
    wells = ['Well-A', 'Well-B', 'Well-C', 'Well-D', 'Well-E', 'Well-F', 'Well-G', 'Well-H']
    np.random.seed(42)
    
    data = {
        'Well_ID': wells,
        'Location': [f'Field-{i+1}' for i in range(len(wells))],
        'Depth_ft': np.random.normal(8000, 1000, len(wells)),
        'Completion_Date': pd.date_range(start='2018-01-01', periods=len(wells), freq='M'),
        'Initial_Production_BBL': np.random.normal(600, 150, len(wells)),
        'Current_Production_BBL': np.random.normal(400, 100, len(wells)),
        'Cumulative_Production_BBL': np.random.normal(500000, 100000, len(wells)),
        'Status': np.random.choice(['Active', 'Inactive', 'Shut-in'], len(wells), p=[0.6, 0.2, 0.2]),
        'Well_Type': np.random.choice(['Horizontal', 'Vertical'], len(wells), p=[0.7, 0.3]),
    }
    return pd.DataFrame(data)

@st.cache_data
def generate_reservoir_data():
    """Generate sample reservoir data"""
    np.random.seed(42)
    zones = ['Zone-1', 'Zone-2', 'Zone-3', 'Zone-4', 'Zone-5']
    
    data = {
        'Zone': zones,
        'Porosity_%': np.random.normal(15, 3, len(zones)),
        'Permeability_mD': np.random.normal(50, 15, len(zones)),
        'Net_Pay_ft': np.random.normal(100, 20, len(zones)),
        'Oil_Saturation_%': np.random.normal(65, 10, len(zones)),
        'Water_Saturation_%': np.random.normal(35, 10, len(zones)),
        'Reservoir_Type': np.random.choice(['Oil', 'Gas', 'Water'], len(zones), p=[0.5, 0.3, 0.2]),
        'Pressure_PSI': np.random.normal(2500, 300, len(zones)),
    }
    return pd.DataFrame(data)

# Load data
production_df = generate_production_data()
well_df = generate_well_data()
reservoir_df = generate_reservoir_data()

# Overview Page
if page == "Overview":
    st.header("📊 Dashboard Overview")
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_oil = production_df['Oil_Production_BBL'].sum()
    total_gas = production_df['Gas_Production_MCF'].sum()
    active_wells = len(well_df[well_df['Status'] == 'Active'])
    avg_production = production_df.groupby('Well_ID')['Oil_Production_BBL'].mean().mean()
    
    with col1:
        st.metric("Total Oil Production", f"{total_oil:,.0f} BBL", delta="5.2%")
    with col2:
        st.metric("Total Gas Production", f"{total_gas:,.0f} MCF", delta="3.8%")
    with col3:
        st.metric("Active Wells", f"{active_wells}", delta="2")
    with col4:
        st.metric("Avg Daily Production", f"{avg_production:.0f} BBL", delta="-1.2%")
    
    st.divider()
    
    # Production Trend
    st.subheader("Production Trend (Last 30 Days)")
    recent_data = production_df[production_df['Date'] >= production_df['Date'].max() - timedelta(days=30)]
    daily_production = recent_data.groupby('Date').agg({
        'Oil_Production_BBL': 'sum',
        'Gas_Production_MCF': 'sum'
    }).reset_index()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily_production['Date'],
        y=daily_production['Oil_Production_BBL'],
        name='Oil Production (BBL)',
        line=dict(color='#1f77b4', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=daily_production['Date'],
        y=daily_production['Gas_Production_MCF'] / 10,  # Scale for visibility
        name='Gas Production (MCF/10)',
        line=dict(color='#ff7f0e', width=2),
        yaxis='y2'
    ))
    
    fig.update_layout(
        title="Daily Production Trend",
        xaxis_title="Date",
        yaxis_title="Oil Production (BBL)",
        yaxis2=dict(title="Gas Production (MCF/10)", overlaying='y', side='right'),
        hovermode='x unified',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Well Status Distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Well Status Distribution")
        status_counts = well_df['Status'].value_counts()
        fig = px.pie(
            values=status_counts.values,
            names=status_counts.index,
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Well Type Distribution")
        type_counts = well_df['Well_Type'].value_counts()
        fig = px.bar(
            x=type_counts.index,
            y=type_counts.values,
            labels={'x': 'Well Type', 'y': 'Count'},
            color=type_counts.values,
            color_continuous_scale='Blues'
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

# Production Analytics Page
elif page == "Production Analytics":
    st.header("📈 Production Analytics")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        selected_wells = st.multiselect("Select Wells", well_df['Well_ID'].unique(), default=well_df['Well_ID'].unique())
    with col2:
        date_range = st.date_input(
            "Date Range",
            value=(production_df['Date'].min().date(), production_df['Date'].max().date()),
            min_value=production_df['Date'].min().date(),
            max_value=production_df['Date'].max().date()
        )
    with col3:
        aggregation = st.selectbox("Aggregation", ["Daily", "Weekly", "Monthly"])
    
    # Filter data
    filtered_df = production_df[
        (production_df['Well_ID'].isin(selected_wells)) &
        (production_df['Date'] >= pd.to_datetime(date_range[0])) &
        (production_df['Date'] <= pd.to_datetime(date_range[1]))
    ]
    
    if aggregation == "Weekly":
        filtered_df['Period'] = filtered_df['Date'].dt.to_period('W').dt.start_time
    elif aggregation == "Monthly":
        filtered_df['Period'] = filtered_df['Date'].dt.to_period('M').dt.start_time
    else:
        filtered_df['Period'] = filtered_df['Date']
    
    # Production by Well
    st.subheader("Production by Well")
    production_by_well = filtered_df.groupby(['Period', 'Well_ID']).agg({
        'Oil_Production_BBL': 'sum',
        'Gas_Production_MCF': 'sum',
        'Water_Production_BBL': 'sum'
    }).reset_index()
    
    fig = px.line(
        production_by_well,
        x='Period',
        y='Oil_Production_BBL',
        color='Well_ID',
        title="Oil Production by Well",
        labels={'Oil_Production_BBL': 'Oil Production (BBL)', 'Period': 'Date'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Production Summary Table
    st.subheader("Production Summary")
    summary = filtered_df.groupby('Well_ID').agg({
        'Oil_Production_BBL': ['sum', 'mean', 'std'],
        'Gas_Production_MCF': ['sum', 'mean'],
        'Water_Production_BBL': ['sum', 'mean']
    }).round(2)
    summary.columns = ['Total Oil (BBL)', 'Avg Oil (BBL)', 'Std Oil (BBL)', 
                       'Total Gas (MCF)', 'Avg Gas (MCF)', 
                       'Total Water (BBL)', 'Avg Water (BBL)']
    st.dataframe(summary, use_container_width=True)
    
    # Production vs Water Cut
    st.subheader("Production vs Water Cut Analysis")
    filtered_df['Water_Cut_%'] = (filtered_df['Water_Production_BBL'] / 
                                  (filtered_df['Oil_Production_BBL'] + filtered_df['Water_Production_BBL']) * 100)
    
    col1, col2 = st.columns(2)
    with col1:
        fig = px.scatter(
            filtered_df,
            x='Oil_Production_BBL',
            y='Water_Cut_%',
            color='Well_ID',
            title="Oil Production vs Water Cut",
            labels={'Oil_Production_BBL': 'Oil Production (BBL)', 'Water_Cut_%': 'Water Cut (%)'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.histogram(
            filtered_df,
            x='Water_Cut_%',
            nbins=30,
            title="Water Cut Distribution",
            labels={'Water_Cut_%': 'Water Cut (%)', 'count': 'Frequency'}
        )
        st.plotly_chart(fig, use_container_width=True)

# Well Performance Page
elif page == "Well Performance":
    st.header("🔧 Well Performance Analysis")
    
    # Well Selection
    selected_well = st.selectbox("Select Well", well_df['Well_ID'].unique())
    
    # Well Information
    well_info = well_df[well_df['Well_ID'] == selected_well].iloc[0]
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Location", well_info['Location'])
    with col2:
        st.metric("Depth", f"{well_info['Depth_ft']:,.0f} ft")
    with col3:
        st.metric("Status", well_info['Status'])
    with col4:
        st.metric("Well Type", well_info['Well_Type'])
    
    st.divider()
    
    # Production History
    well_production = production_df[production_df['Well_ID'] == selected_well]
    
    st.subheader("Production History")
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=well_production['Date'],
            y=well_production['Oil_Production_BBL'],
            name='Oil',
            line=dict(color='#1f77b4')
        ))
        fig.add_trace(go.Scatter(
            x=well_production['Date'],
            y=well_production['Gas_Production_MCF'] / 10,
            name='Gas (MCF/10)',
            line=dict(color='#ff7f0e'),
            yaxis='y2'
        ))
        fig.update_layout(
            title="Production Over Time",
            xaxis_title="Date",
            yaxis_title="Oil Production (BBL)",
            yaxis2=dict(title="Gas Production (MCF/10)", overlaying='y', side='right'),
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Decline Curve
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=well_production['Date'],
            y=well_production['Oil_Production_BBL'],
            mode='lines+markers',
            name='Actual Production',
            line=dict(color='#1f77b4')
        ))
        # Simple exponential decline fit
        days = (well_production['Date'] - well_production['Date'].min()).dt.days
        if len(days) > 1:
            decline_rate = -np.polyfit(days, np.log(well_production['Oil_Production_BBL'].clip(1)), 1)[0]
            forecast_days = np.arange(0, days.max() + 365)
            forecast_production = well_production['Oil_Production_BBL'].iloc[0] * np.exp(-decline_rate * forecast_days)
            forecast_dates = well_production['Date'].min() + pd.to_timedelta(forecast_days, unit='D')
            fig.add_trace(go.Scatter(
                x=forecast_dates,
                y=forecast_production,
                name='Decline Forecast',
                line=dict(color='red', dash='dash')
            ))
        fig.update_layout(
            title="Decline Curve Analysis",
            xaxis_title="Date",
            yaxis_title="Oil Production (BBL)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Well Comparison
    st.subheader("Well Comparison")
    comparison_wells = st.multiselect("Compare with Wells", 
                                     [w for w in well_df['Well_ID'].unique() if w != selected_well],
                                     default=[])
    
    if comparison_wells:
        comparison_data = production_df[production_df['Well_ID'].isin([selected_well] + comparison_wells)]
        monthly_comparison = comparison_data.groupby(['Well_ID', pd.Grouper(key='Date', freq='M')]).agg({
            'Oil_Production_BBL': 'sum'
        }).reset_index()
        
        fig = px.line(
            monthly_comparison,
            x='Date',
            y='Oil_Production_BBL',
            color='Well_ID',
            title="Monthly Production Comparison",
            labels={'Oil_Production_BBL': 'Oil Production (BBL)', 'Date': 'Date'}
        )
        st.plotly_chart(fig, use_container_width=True)

# Reservoir Analysis Page
elif page == "Reservoir Analysis":
    st.header("🌊 Reservoir Analysis")
    
    st.subheader("Reservoir Properties by Zone")
    
    # Reservoir Properties Visualization
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            reservoir_df,
            x='Zone',
            y='Porosity_%',
            title="Porosity by Zone",
            labels={'Porosity_%': 'Porosity (%)', 'Zone': 'Zone'},
            color='Porosity_%',
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        fig = px.bar(
            reservoir_df,
            x='Zone',
            y='Permeability_mD',
            title="Permeability by Zone",
            labels={'Permeability_mD': 'Permeability (mD)', 'Zone': 'Zone'},
            color='Permeability_mD',
            color_continuous_scale='Greens'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(
            reservoir_df,
            x='Zone',
            y='Net_Pay_ft',
            title="Net Pay by Zone",
            labels={'Net_Pay_ft': 'Net Pay (ft)', 'Zone': 'Zone'},
            color='Net_Pay_ft',
            color_continuous_scale='Oranges'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        fig = px.bar(
            reservoir_df,
            x='Zone',
            y='Pressure_PSI',
            title="Reservoir Pressure by Zone",
            labels={'Pressure_PSI': 'Pressure (PSI)', 'Zone': 'Zone'},
            color='Pressure_PSI',
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Saturation Analysis
    st.subheader("Fluid Saturation Analysis")
    saturation_data = pd.melt(
        reservoir_df,
        id_vars=['Zone'],
        value_vars=['Oil_Saturation_%', 'Water_Saturation_%'],
        var_name='Saturation_Type',
        value_name='Saturation_%'
    )
    
    fig = px.bar(
        saturation_data,
        x='Zone',
        y='Saturation_%',
        color='Saturation_Type',
        barmode='group',
        title="Oil vs Water Saturation by Zone",
        labels={'Saturation_%': 'Saturation (%)', 'Zone': 'Zone'},
        color_discrete_map={'Oil_Saturation_%': '#1f77b4', 'Water_Saturation_%': '#ff7f0e'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Reservoir Properties Table
    st.subheader("Reservoir Properties Summary")
    st.dataframe(reservoir_df.set_index('Zone'), use_container_width=True)
    
    # Cross-plot Analysis
    st.subheader("Reservoir Property Cross-Plots")
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.scatter(
            reservoir_df,
            x='Porosity_%',
            y='Permeability_mD',
            size='Net_Pay_ft',
            color='Oil_Saturation_%',
            hover_name='Zone',
            title="Porosity vs Permeability",
            labels={'Porosity_%': 'Porosity (%)', 'Permeability_mD': 'Permeability (mD)',
                   'Net_Pay_ft': 'Net Pay (ft)', 'Oil_Saturation_%': 'Oil Saturation (%)'},
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.scatter(
            reservoir_df,
            x='Oil_Saturation_%',
            y='Permeability_mD',
            size='Net_Pay_ft',
            color='Pressure_PSI',
            hover_name='Zone',
            title="Oil Saturation vs Permeability",
            labels={'Oil_Saturation_%': 'Oil Saturation (%)', 'Permeability_mD': 'Permeability (mD)',
                   'Net_Pay_ft': 'Net Pay (ft)', 'Pressure_PSI': 'Pressure (PSI)'},
            color_continuous_scale='Plasma'
        )
        st.plotly_chart(fig, use_container_width=True)

# Economic Analysis Page
elif page == "Economic Analysis":
    st.header("💰 Economic Analysis")
    
    # Economic Parameters
    st.subheader("Economic Parameters")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        oil_price = st.number_input("Oil Price ($/BBL)", min_value=0.0, value=70.0, step=1.0)
    with col2:
        gas_price = st.number_input("Gas Price ($/MCF)", min_value=0.0, value=3.0, step=0.1)
    with col3:
        operating_cost = st.number_input("Operating Cost ($/BBL)", min_value=0.0, value=15.0, step=1.0)
    with col4:
        discount_rate = st.number_input("Discount Rate (%)", min_value=0.0, value=10.0, step=0.5)
    
    # Calculate Revenue
    monthly_production = production_df.groupby(pd.Grouper(key='Date', freq='M')).agg({
        'Oil_Production_BBL': 'sum',
        'Gas_Production_MCF': 'sum'
    }).reset_index()
    
    monthly_production['Oil_Revenue'] = monthly_production['Oil_Production_BBL'] * oil_price
    monthly_production['Gas_Revenue'] = monthly_production['Gas_Production_MCF'] * gas_price
    monthly_production['Total_Revenue'] = monthly_production['Oil_Revenue'] + monthly_production['Gas_Revenue']
    monthly_production['Operating_Cost'] = monthly_production['Oil_Production_BBL'] * operating_cost
    monthly_production['Net_Revenue'] = monthly_production['Total_Revenue'] - monthly_production['Operating_Cost']
    
    # Cumulative Metrics
    monthly_production['Cumulative_Revenue'] = monthly_production['Total_Revenue'].cumsum()
    monthly_production['Cumulative_Net_Revenue'] = monthly_production['Net_Revenue'].cumsum()
    
    # NPV Calculation
    months = np.arange(1, len(monthly_production) + 1)
    monthly_production['NPV'] = monthly_production['Net_Revenue'] / ((1 + discount_rate/100) ** (months/12))
    monthly_production['Cumulative_NPV'] = monthly_production['NPV'].cumsum()
    
    # Revenue Visualization
    st.subheader("Revenue Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=monthly_production['Date'],
            y=monthly_production['Total_Revenue'],
            name='Total Revenue',
            line=dict(color='green', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=monthly_production['Date'],
            y=monthly_production['Operating_Cost'],
            name='Operating Cost',
            line=dict(color='red', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=monthly_production['Date'],
            y=monthly_production['Net_Revenue'],
            name='Net Revenue',
            line=dict(color='blue', width=2)
        ))
        fig.update_layout(
            title="Monthly Revenue vs Costs",
            xaxis_title="Date",
            yaxis_title="Revenue ($)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=monthly_production['Date'],
            y=monthly_production['Cumulative_Net_Revenue'],
            name='Cumulative Net Revenue',
            line=dict(color='blue', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=monthly_production['Date'],
            y=monthly_production['Cumulative_NPV'],
            name='Cumulative NPV',
            line=dict(color='purple', width=2, dash='dash')
        ))
        fig.update_layout(
            title="Cumulative Financial Metrics",
            xaxis_title="Date",
            yaxis_title="Cumulative Value ($)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Economic Metrics
    st.subheader("Key Economic Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    total_revenue = monthly_production['Total_Revenue'].sum()
    total_net_revenue = monthly_production['Net_Revenue'].sum()
    total_npv = monthly_production['NPV'].sum()
    avg_monthly_net = monthly_production['Net_Revenue'].mean()
    
    with col1:
        st.metric("Total Revenue", f"${total_revenue:,.0f}")
    with col2:
        st.metric("Total Net Revenue", f"${total_net_revenue:,.0f}")
    with col3:
        st.metric("Net Present Value", f"${total_npv:,.0f}")
    with col4:
        st.metric("Avg Monthly Net", f"${avg_monthly_net:,.0f}")
    
    # Revenue by Well
    st.subheader("Revenue Contribution by Well")
    well_revenue = production_df.groupby('Well_ID').agg({
        'Oil_Production_BBL': 'sum',
        'Gas_Production_MCF': 'sum'
    })
    well_revenue['Total_Revenue'] = (well_revenue['Oil_Production_BBL'] * oil_price + 
                                    well_revenue['Gas_Production_MCF'] * gas_price)
    well_revenue = well_revenue.sort_values('Total_Revenue', ascending=False)
    
    fig = px.bar(
        x=well_revenue.index,
        y=well_revenue['Total_Revenue'],
        title="Total Revenue by Well",
        labels={'x': 'Well ID', 'y': 'Total Revenue ($)'},
        color=well_revenue['Total_Revenue'],
        color_continuous_scale='Greens'
    )
    st.plotly_chart(fig, use_container_width=True)

# Time Series Analysis Page
elif page == "Time Series Analysis":
    st.header("📅 Time Series Analysis")
    
    # Time Series Selection
    col1, col2 = st.columns(2)
    with col1:
        metric = st.selectbox("Select Metric", 
                              ["Oil_Production_BBL", "Gas_Production_MCF", 
                               "Water_Production_BBL", "Pressure_PSI", "Temperature_F"])
    with col2:
        aggregation_level = st.selectbox("Aggregation Level", 
                                         ["Daily", "Weekly", "Monthly", "Quarterly", "Yearly"])
    
    # Aggregate data
    freq_map = {
        "Daily": "D",
        "Weekly": "W",
        "Monthly": "M",
        "Quarterly": "Q",
        "Yearly": "Y"
    }
    
    ts_data = production_df.groupby(pd.Grouper(key='Date', freq=freq_map[aggregation_level])).agg({
        metric: 'sum' if 'Production' in metric else 'mean'
    }).reset_index()
    
    # Time Series Plot
    st.subheader(f"{metric.replace('_', ' ')} Over Time")
    fig = px.line(
        ts_data,
        x='Date',
        y=metric,
        title=f"{metric.replace('_', ' ')} - {aggregation_level} Aggregation",
        labels={metric: metric.replace('_', ' '), 'Date': 'Date'}
    )
    fig.update_traces(line_color='#1f77b4', line_width=2)
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistical Analysis
    st.subheader("Statistical Summary")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Mean", f"{ts_data[metric].mean():.2f}")
        st.metric("Median", f"{ts_data[metric].median():.2f}")
    with col2:
        st.metric("Std Deviation", f"{ts_data[metric].std():.2f}")
        st.metric("Min", f"{ts_data[metric].min():.2f}")
    with col3:
        st.metric("Max", f"{ts_data[metric].max():.2f}")
        st.metric("Trend", f"{((ts_data[metric].iloc[-1] / ts_data[metric].iloc[0] - 1) * 100):.2f}%")
    
    # Seasonal Decomposition (if enough data)
    if len(ts_data) > 12:
        st.subheader("Seasonal Pattern Analysis")
        ts_data['Month'] = ts_data['Date'].dt.month
        ts_data['Year'] = ts_data['Date'].dt.year
        
        monthly_avg = ts_data.groupby('Month')[metric].mean().reset_index()
        monthly_avg['Month_Name'] = pd.to_datetime(monthly_avg['Month'], format='%m').dt.strftime('%B')
        
        fig = px.bar(
            monthly_avg,
            x='Month_Name',
            y=metric,
            title="Average Monthly Pattern",
            labels={metric: f"Avg {metric.replace('_', ' ')}", 'Month_Name': 'Month'},
            color=metric,
            color_continuous_scale='Blues'
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    # Moving Average
    st.subheader("Moving Average Analysis")
    window = st.slider("Moving Average Window", min_value=3, max_value=min(30, len(ts_data)), value=7)
    
    ts_data['Moving_Avg'] = ts_data[metric].rolling(window=window).mean()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ts_data['Date'],
        y=ts_data[metric],
        name='Actual',
        line=dict(color='lightblue', width=1)
    ))
    fig.add_trace(go.Scatter(
        x=ts_data['Date'],
        y=ts_data['Moving_Avg'],
        name=f'{window}-Period Moving Average',
        line=dict(color='red', width=2)
    ))
    fig.update_layout(
        title=f"{metric.replace('_', ' ')} with Moving Average",
        xaxis_title="Date",
        yaxis_title=metric.replace('_', ' '),
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "This Oil & Gas Analytics Dashboard provides comprehensive insights into "
    "production, well performance, reservoir characteristics, and economic metrics."
)

