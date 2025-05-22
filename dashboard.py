import random
import datetime
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from io import BytesIO
from streamlit_autorefresh import st_autorefresh
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(layout="wide")

# Custom CSS for layout, hover effect, and table scrolling
st.markdown("""
    <style>
    .block-container {
        padding: 10px 0.5rem;
        margin: 0 auto;
        max-width: 1400px;
    }
    .stRadio > div {
        flex-direction: row;
    }
    .visual-title {
        text-align: center;
        font-size: 14px;
        font-weight: bold;
        margin-bottom: 2px;
    }
    .stPlotlyChart, .stImage {
        width: 100%;
        height: 200px;
        margin-bottom: 8px;
        transition: transform 0.2s ease;
    }
    .stPlotlyChart:hover, .stImage:hover {
        transform: scale(1.05);
    }
    div[data-testid="stVerticalBlock"] > div {
        margin-bottom: 8px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        white-space: nowrap;
        border-radius: 4px;
        font-size: 14px;
    }
    .stTable {
        margin-bottom: 20px;
        max-width: 100%;
        font-size: 12px;
        max-height: 300px;
        overflow-y: auto;
        display: block;
    }
    .stTable table {
        width: 100%;
        border-collapse: collapse;
        table-layout: fixed;
    }
    .stTable thead {
        position: sticky;
        top: 0;
        background-color: #d0d0d0;
        z-index: 1;
        box-shadow: 0 2px 2px -1px rgba(0, 0, 0, 0.1);
    }
    .stTable th {
        padding: 8px;
        text-align: left;
        border-bottom: 1px solid #ddd;
        font-size: 12px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        font-weight: bold;
        color: #333;
    }
    .stTable td {
        padding: 8px;
        text-align: left;
        border-bottom: 1px solid #ddd;
        font-size: 12px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        color: #000;
    }
    .stTable tbody {
        background-color: #ffffff;
    }
    .stTable tbody tr:hover {
        background-color: #f5f5f5;
    }
    .stTable::-webkit-scrollbar {
        height: 8px;
    }
    .stTable::-webkit-scrollbar-thumb {
        background-color: #ccc;
        border-radius: 4px;
    }
    </style>
""", unsafe_allow_html=True)

def generate_data():
    print("Generating data...")  # Debug log
    countries_regions = {
        'USA': ['North America', 'Central USA'],
        'Canada': ['North America', 'Western Canada'],
        'UK': ['Northern Europe', 'Western Europe'],
        'Germany': ['Central Europe', 'Western Europe'],
        'India': ['South Asia', 'Central India'],
        'Japan': ['East Asia', 'Pacific Rim'],
        'Brazil': ['South America', 'Latin America'],
        'Mexico': ['Central America', 'North America'],
        'Australia': ['Oceania', 'Australasia'],
        'South Africa': ['Southern Africa', 'Sub-Saharan Africa']
    }
    job_types = [
        'Software Developer', 'Data Scientist', 'Sales Engineer', 'Product Manager', 'UX Designer',
        'Systems Analyst', 'QA Tester', 'DevOps Engineer', 'Business Analyst', 'Cloud Architect',
        'Security Specialist', 'AI Engineer', 'Database Admin', 'IT Support', 'Network Engineer',
        'Technical Writer', 'Mobile Developer', 'Game Developer', 'Data Engineer', 'Machine Learning Scientist'
    ]
    service_types = ['SaaS', 'Consulting', 'Support', 'Cloud Storage', 'Analytics', 'Monitoring', 'Training', 'Security', 'AI Platform', 'IoT']
    product_names = [
        'CloudSync Pro', 'DataVault Analytics', 'AI-Predictor', 'SecureNet Firewall', 'StreamFlow CRM',
        'IntelliSuite ERP', 'Visionary AI', 'NexGen Database', 'SmartTrack IoT', 'Quantum Compute',
        'FlowCore SaaS', 'Insight360 Dashboard', 'CyberGuard VPN', 'OptiChain SCM', 'TrueView BI',
        'SkyLink Cloud', 'PulseMonitor', 'DataForge ETL', 'AI-Connect Chat', 'BrightPath LMS'
    ]
    request_categories = ['Technical', 'Billing', 'General', 'Sales', 'Feedback', 'Bug Report', 'Feature Request']
    action_types = ['Viewed', 'Clicked', 'Downloaded', 'Subscribed', 'Shared']
    browsers = ['Chrome', 'Firefox', 'Edge', 'Safari', 'Opera', 'Brave']
    responses = ['Success', 'Error', 'Timeout', 'Redirect']
    referral_sources = ['Google', 'LinkedIn', 'Email Campaign', 'Facebook', 'Twitter', 'Direct', 'Partner']
    assistants = ['ChatGPT', 'Alexa', 'Google Assistant', 'Cortana', 'Siri']
    customer_behaviours = ['Loyal', 'Churned', 'New', 'Frequent Buyer', 'Occasional', 'Inactive']
    company_names = ['TechNova', 'DataSync', 'CloudCore', 'InnoWare', 'NetGenius', 'ByteWorks', 'SoftEdge', 'Nexon', 'CodeCrafters', 'SysNova']

    data = []
    now = datetime.datetime.now()
    for i in range(10000):
        country = random.choice(list(countries_regions.keys()))
        region = random.choice(countries_regions[country])
        job = {
            "Job_ID": i + 1,
            "Company_Name": random.choice(company_names),
            "Country": country,
            "Region": region,
            "Job_Type": random.choice(job_types),
            "Service_Type": random.choice(service_types),
            "Revenue": round(random.uniform(500, 10000), 2),
            "Action_Type": random.choice(action_types),
            "Product_Name": random.choice(product_names),
            "Request_Category": random.choice(request_categories),
            "User_ID": f"user_{random.randint(1000, 9999)}",
            "IP_Address": f"{random.randint(10, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}",
            "Pages_Per_Session": random.randint(1, 10),
            "Session_Duration": round(random.uniform(1, 30), 2),
            "Transaction": random.randint(1, 5),
            "Profit": round(random.uniform(100, 9000), 2),
            "Referral": random.choice(referral_sources),
            "AI_Assistant": random.choice(assistants),
            "Engaged_User": random.choice([True, False]),
            "Browser": random.choice(browsers),
            "Response": random.choice(responses),
            "Request_URL": f"/api/v1/{random.choice(['get', 'post', 'update', 'delete'])}/resource_{random.randint(1, 20)}",
            "Conversion_Status": random.choice([True, False]),
            "Demo_Request": random.choice([True, False]),
            "Customer_Behavior": random.choice(customer_behaviours),
            "Number_of_Sales": random.randint(1, 100),
            "Date": (now - datetime.timedelta(days=random.randint(0, 30))).strftime('%Y-%m-%d'),
            "Timestamp": now.isoformat(),
            "Promotional_Event": random.choices([True, False], weights=[0.2, 0.8])[0]
        }
        data.append(job)
    print(f"Generated {len(data)} entries")  # Debug log
    return data

# Cache data generation to improve performance
@st.cache_data(ttl=60)
def fetch_data():
    try:
        logger.info("Generating data locally")
        data = generate_data()
        if not data:
            logger.warning("No data generated")
            return pd.DataFrame()
        logger.info(f"Generated {len(data)} entries")
        return pd.DataFrame(data)
    except Exception as e:
        logger.error(f"Error generating data: {e}")
        st.error(f"Failed to generate data: {e}. Please upload a dataset.")
        return pd.DataFrame()

# Process uploaded file or generated data
def process_uploaded_data(uploaded_file):
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            logger.info("Successfully processed uploaded CSV file")
            return df
        except Exception as e:
            st.error(f"Error processing uploaded file: {e}. Please upload a valid CSV file.")
            return pd.DataFrame()
    return None

# Validate and convert DataFrame columns, impute missing values
def process_data(df):
    if df.empty:
        return df
    expected_columns = {
        'Date': 'datetime64[ns]',
        'Timestamp': 'datetime64[ns]',
        'Job_ID': 'float64',
        'Revenue': 'float64',
        'Session_Duration': 'float64',
        'Conversion_Status': 'str',
        'Country': 'str',
        'Region': 'str',
        'Job_Type': 'str',
        'Demo_Request': 'bool',
        'Company_Name': 'str',
        'Customer_Behavior': 'str',
        'Product_Name': 'str',
        'Number_of_Sales': 'float64',
        'Service_Type': 'str',
        'Action_Type': 'str',
        'Request_Category': 'str',
        'User_ID': 'str',
        'IP_Address': 'str',
        'Pages_Per_Session': 'float64',
        'Transaction': 'float64',
        'Profit': 'float64',
        'Referral': 'str',
        'AI_Assistant': 'str',
        'Engaged_User': 'bool',
        'Browser': 'str',
        'Response': 'str',
        'Request_URL': 'str',
        'Promotional_Event': 'bool'
    }
    missing_cols = [col for col in expected_columns if col not in df.columns]
    if missing_cols:
        st.warning(f"Missing columns: {missing_cols}. Some visualizations may not work.")
    
    # Impute missing values before type conversion
    imputation_rules = {
        'Revenue': 0,
        'Number_of_Sales': 0,
        'Profit': 0,
        'Session_Duration': 0,
        'Pages_Per_Session': 0,
        'Transaction': 0,
        'Job_ID': 0,
        'Country': 'Unknown',
        'Region': 'Unknown',
        'Job_Type': 'Unknown',
        'Conversion_Status': 'False',
        'Company_Name': 'Unknown',
        'Customer_Behavior': 'Unknown',
        'Product_Name': 'Unknown',
        'Service_Type': 'Unknown',
        'Action_Type': 'Unknown',
        'Request_Category': 'Unknown',
        'User_ID': 'Unknown',
        'IP_Address': 'Unknown',
        'Referral': 'Unknown',
        'AI_Assistant': 'None',
        'Browser': 'Unknown',
        'Response': 'Unknown',
        'Request_URL': 'Unknown',
        'Demo_Request': False,
        'Promotional_Event': False,
        'Engaged_User': False,
        'Date': df['Date'].min() if not df['Date'].isna().all() else '2025-01-01',
        'Timestamp': df['Timestamp'].min() if not df['Timestamp'].isna().all() else '2025-01-01'
    }
    df = df.fillna(imputation_rules)
    
    # Convert data types
    for col, dtype in expected_columns.items():
        if col in df.columns:
            try:
                if dtype.startswith('datetime'):
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                elif dtype in ('float64', 'int64'):
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                else:
                    df[col] = df[col].astype(dtype, errors='ignore')
            except Exception as e:
                st.warning(f"Failed to convert column {col}: {e}")
    
    # Additional validation to prevent "undefined"
    for col in ['Job_Type', 'Country', 'Revenue', 'Number_of_Sales', 'Conversion_Status', 'Service_Type', 'Referral', 'Product_Name', 'Pages_Per_Session']:
        if col in df.columns:
            df[col] = df[col].fillna(imputation_rules.get(col, 'Unknown'))
            if df[col].dtype == 'object' and df[col].str.contains('undefined', na=False).any():
                df[col] = df[col].replace('undefined', 'Unknown')
    
    # Warn if significant imputation occurred
    if df.isna().sum().sum() > 0:
        st.warning("Some values could not be imputed or converted, resulting in remaining NaNs.")
    
    return df

# Generate Excel file for download
def to_excel(data):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        data.to_excel(writer, index=False, sheet_name='Sheet1')
    return output.getvalue()

# Render visualization with error handling
def render_visual(fig, title, height=200, use_container_width=True, is_matplotlib=False):
    try:
        st.markdown(f'<div class="visual-title">{title}</div>', unsafe_allow_html=True)
        if is_matplotlib:
            st.pyplot(fig)
        else:
            st.plotly_chart(fig, use_container_width=use_container_width, height=height)
    except Exception as e:
        st.warning(f"Failed to render {title}: {e}")

# Analyze web server logs
def analyze_web_logs(df):
    if df.empty:
        return None
    
    # Jobs Placed by Country
    jobs_by_country = df.groupby('Country')['Job_ID'].nunique().reset_index(name='Jobs_Placed')
    jobs_by_country = jobs_by_country.sort_values('Jobs_Placed', ascending=False)
    
    # Job Types Distribution
    job_types = df['Job_Type'].value_counts().reset_index(name='Count')
    job_types.columns = ['Job_Type', 'Count']
    
    # Demo Requests by Country
    demo_requests = df[df['Demo_Request'] == True].groupby('Country')['Job_ID'].count().reset_index(name='Demo_Requests')
    
    # Promotional Events by Country
    if 'Promotional_Event' in df.columns:
        promo_events = df[df['Promotional_Event'] == True].groupby('Country')['Job_ID'].count().reset_index(name='Promo_Events')
    else:
        promo_events = pd.DataFrame(columns=['Country', 'Promo_Events'])
        st.warning("No 'Promotional_Event' column for promotional events. Using placeholder.")
    
    # AI Assistant Requests by Country
    if 'AI_Assistant' in df.columns:
        ai_requests = df[df['AI_Assistant'].notnull() & (df['AI_Assistant'] != 'None')].groupby('Country')['Job_ID'].count().reset_index(name='AI_Requests')
    else:
        ai_requests = pd.DataFrame(columns=['Country', 'AI_Requests'])
        st.warning("No 'AI_Assistant' column for AI requests. Using placeholder.")
    
    # Action Type Distribution
    if 'Action_Type' in df.columns:
        action_types = df['Action_Type'].value_counts().reset_index(name='Count')
        action_types.columns = ['Action_Type', 'Count']
    else:
        action_types = pd.DataFrame(columns=['Action_Type', 'Count'])
        st.warning("No 'Action_Type' column for action types. Using placeholder.")
    
    # Promotional Events vs. Demo Requests
    events_vs_demos = df.groupby('Country').agg({
        'Promotional_Event': lambda x: (x == True).sum(),
        'Demo_Request': lambda x: (x == True).sum(),
        'Customer_Behavior': lambda x: x.mode()[0] if not x.empty else 'Unknown'
    }).reset_index()
    events_vs_demos.columns = ['Country', 'Promo_Events', 'Demo_Requests', 'Customer_Behavior']
    
    # Conversion Rate by Country
    conversion_by_country = df.groupby('Country')['Conversion_Status'].apply(
        lambda x: (x == 'True').mean() * 100
    ).reset_index(name='Conversion_Rate')
    
    # Statistics
    stats = {
        'Jobs_Placed': {
            'Mean': jobs_by_country['Jobs_Placed'].mean(),
            'Std': jobs_by_country['Jobs_Placed'].std(),
            'Median': jobs_by_country['Jobs_Placed'].median(),
            'Range': jobs_by_country['Jobs_Placed'].max() - jobs_by_country['Jobs_Placed'].min()
        },
        'Revenue': {
            'Mean': df['Revenue'].mean(),
            'Std': df['Revenue'].std(),
            'Median': df['Revenue'].median(),
            'Range': df['Revenue'].max() - df['Revenue'].min()
        },
        'Demo_Requests': {
            'Mean': demo_requests['Demo_Requests'].mean() if not demo_requests.empty else 0,
            'Std': demo_requests['Demo_Requests'].std() if not demo_requests.empty else 0,
            'Median': demo_requests['Demo_Requests'].median() if not demo_requests.empty else 0,
            'Range': (demo_requests['Demo_Requests'].max() - demo_requests['Demo_Requests'].min()) if not demo_requests.empty else 0
        },
        'Number_of_Sales': {
            'Mean': df['Number_of_Sales'].mean(),
            'Std': df['Number_of_Sales'].std(),
            'Median': df['Number_of_Sales'].median(),
            'Range': df['Number_of_Sales'].max() - df['Number_of_Sales'].min()
        },
        'Profit': {
            'Mean': df['Profit'].mean(),
            'Std': df['Profit'].std(),
            'Median': df['Profit'].median(),
            'Range': df['Profit'].max() - df['Profit'].min()
        },
        'Conversion_Rate': {
            'Mean': conversion_by_country['Conversion_Rate'].mean(),
            'Std': conversion_by_country['Conversion_Rate'].std(),
            'Median': conversion_by_country['Conversion_Rate'].median(),
            'Range': conversion_by_country['Conversion_Rate'].max() - conversion_by_country['Conversion_Rate'].min()
        }
    }
    
    # Overall Conversion Rate
    conversion_rate = df['Conversion_Status'].eq('True').mean() * 100
    
    return {
        'jobs_by_country': jobs_by_country,
        'job_types': job_types,
        'demo_requests': demo_requests,
        'promo_events': promo_events,
        'ai_requests': ai_requests,
        'action_types': action_types,
        'events_vs_demos': events_vs_demos,
        'stats': stats,
        'conversion_rate': conversion_rate
    }

# Main dashboard
def main():
    # Auto-refresh every 60 seconds
    st_autorefresh(interval=60000, key="data_refresh")

    # Load data at the beginning
    raw_df = fetch_data()
    if raw_df.empty:
        st.error("No data available. Please upload a dataset.")
        return
    df = process_data(raw_df.copy())

    # Sidebar Filters
    st.sidebar.header("🔍 Filters")

    st.sidebar.subheader("🗕 Date Range")
    min_date, max_date = df['Date'].min(), df['Date'].max()
    date_range = st.sidebar.date_input("Select date range", [min_date, max_date])
    if len(date_range) == 2:
        start_date, end_date = pd.to_datetime(date_range)
        df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

    st.sidebar.subheader("🌍 Country Filter")
    country_filter = st.sidebar.selectbox("Select Country", ['All'] + sorted(df['Country'].dropna().unique().tolist()))
    filtered_df = df if country_filter == 'All' else df[df['Country'] == country_filter]

    # Automatically select region based on country
    if country_filter != 'All':
        available_regions = sorted(filtered_df['Region'].dropna().unique().tolist())
        selected_region = available_regions[0] if available_regions else 'Unknown'
        filtered_df = filtered_df[filtered_df['Region'] == selected_region]
    else:
        selected_region = 'All'

    st.sidebar.subheader("📐 View Options")
    expanded_view = st.sidebar.checkbox("Expanded View (Show More Categories)", value=False)

    # Upload custom dataset (just above Download Options)
    uploaded_file = st.sidebar.file_uploader("Upload your own dataset (CSV)", type=["csv"])
    if uploaded_file is not None:
        df = process_uploaded_data(uploaded_file)
        if df.empty:
            st.error("Failed to load uploaded dataset. Using generated dataset instead.")
            df = process_data(raw_df.copy())
        else:
            df = process_data(df)

    st.sidebar.subheader('⬇️ Download Options')
    csv = df.to_csv(index=False).encode('utf-8')
    excel_data = to_excel(filtered_df)
    st.sidebar.download_button("Download Full Data (CSV)", csv, 'full_data.csv', 'text/csv')
    st.sidebar.download_button("Download Filtered Data (Excel)", excel_data, 'filtered_data.xlsx', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

    # Set max categories based on view option
    max_categories = 10 if expanded_view else 5

    # Compute analysis before tabs
    analysis = analyze_web_logs(filtered_df)
    if analysis is None:
        st.error("Failed to analyze web logs. Check data and try again.")
        return

    # Main Dashboard
    st.title("📊 Sales Performance Dashboard")
    st.markdown(f"🕒 **Last Updated:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Add summary statistics in a single row (all metrics now dynamic)
    st.subheader("📌 Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total_profit = filtered_df['Profit'].sum() if 'Profit' in filtered_df.columns else 0
        st.metric(label="Total Profit", value=f"${total_profit:,.2f}")
    with col2:
        total_revenue = filtered_df['Revenue'].sum() if 'Revenue' in filtered_df.columns else 0
        st.metric(label="Total Revenue", value=f"${total_revenue:,.2f}")
    with col3:
        total_sales = filtered_df['Number_of_Sales'].sum() if 'Number_of_Sales' in filtered_df.columns else 0
        st.metric(label="Total Sales", value=f"{int(total_sales):,}")
    with col4:
        conversion_rate = filtered_df['Conversion_Status'].eq('True').mean() * 100 if 'Conversion_Status' in filtered_df.columns and not filtered_df['Conversion_Status'].empty else 0
        st.metric(label="Conversion Rate", value=f"{conversion_rate:.2f}%")

    # Tabs for navigation
    tabs = st.tabs(["Overview", "Main", "EDA"])

    # Overview Tab
    with tabs[0]:
        st.subheader("📈 Overview Visuals")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if not filtered_df.empty and 'Country' in filtered_df.columns and not filtered_df['Country'].isna().all():
                jobs_by_country = filtered_df.groupby('Country')['Job_ID'].count().reset_index(name='Count').head(max_categories)
                if not jobs_by_country.empty and jobs_by_country['Count'].sum() > 0:
                    jobs_by_country['Count'] = jobs_by_country['Count'].fillna(0).astype(int)
                    fig1 = px.bar(jobs_by_country, x='Country', y='Count', 
                                  height=200, color='Country', 
                                  color_discrete_sequence=px.colors.qualitative.Bold)
                    fig1.update_layout(
                        xaxis_title='Country', yaxis_title='Jobs Placed',
                        xaxis_tickangle=45, title_font_size=10, font=dict(size=8),
                        margin=dict(l=5, r=5, t=15, b=5)
                    )
                    render_visual(fig1, "Total Jobs Placed by Country")
                else:
                    st.warning("No valid data for Total Jobs Placed by Country")
            else:
                st.warning("No data for Total Jobs Placed by Country")

        with col2:
            if not filtered_df.empty and 'Country' in filtered_df.columns and 'Revenue' in filtered_df.columns and not filtered_df['Country'].isna().all():
                revenue_country = filtered_df.groupby('Country')['Revenue'].sum().reset_index().head(max_categories)
                if not revenue_country.empty and revenue_country['Revenue'].sum() > 0:
                    revenue_country['Revenue'] = revenue_country['Revenue'].fillna(0)
                    fig2 = px.bar(revenue_country, x='Country', y='Revenue', color='Revenue', 
                                  color_continuous_scale='Oranges', height=200)
                    fig2.update_layout(
                        xaxis_title='Country', yaxis_title='Revenue ($)',
                        xaxis_tickangle=45, title_font_size=10, font=dict(size=8),
                        margin=dict(l=5, r=5, t=15, b=5)
                    )
                    render_visual(fig2, "Total Revenue by Country")
                else:
                    st.warning("No valid data for Total Revenue by Country")
            else:
                st.warning("No data for Total Revenue by Country")

        with col3:
            if not filtered_df.empty and 'Country' in filtered_df.columns and 'Number_of_Sales' in filtered_df.columns and not filtered_df['Country'].isna().all():
                sales_country = filtered_df.groupby('Country')['Number_of_Sales'].sum().reset_index().head(max_categories)
                if not sales_country.empty and sales_country['Number_of_Sales'].sum() > 0:
                    sales_country['Number_of_Sales'] = sales_country['Number_of_Sales'].fillna(0)
                    fig3 = px.bar(sales_country, x='Country', y='Number_of_Sales', color='Number_of_Sales', 
                                  color_continuous_scale='Blues', height=200)
                    fig3.update_layout(
                        xaxis_title='Country', yaxis_title='Sales',
                        xaxis_tickangle=45, title_font_size=10, font=dict(size=8),
                        margin=dict(l=5, r=5, t=15, b=5)
                    )
                    render_visual(fig3, "Total Sales by Country")
                else:
                    st.warning("No valid data for Total Sales by Country")
            else:
                st.warning("No data for Total Sales by Country")

        with col4:
            if not filtered_df.empty and 'Conversion_Status' in filtered_df.columns and not filtered_df['Conversion_Status'].isna().all():
                conv_counts = filtered_df['Conversion_Status'].value_counts().reset_index(name='Count')
                conv_counts.columns = ['Conversion_Status', 'Count']
                if not conv_counts.empty and conv_counts['Count'].sum() > 0:
                    conv_counts['Count'] = conv_counts['Count'].fillna(0).astype(int)
                    fig4 = px.pie(conv_counts, names='Conversion_Status', values='Count', 
                                  height=200, color_discrete_sequence=px.colors.qualitative.Set1)
                    fig4.update_traces(textinfo='percent+label', pull=[0.1, 0.1], textfont_size=8)
                    fig4.update_layout(
                        showlegend=True, legend_title_text='Status', legend=dict(font=dict(size=8)),
                        title_font_size=10, font=dict(size=8),
                        margin=dict(l=5, r=5, t=15, b=5)
                    )
                    render_visual(fig4, "Conversion Rate Distribution")
                else:
                    st.warning("No valid data for Conversion Rate Distribution")
            else:
                st.warning("No data for Conversion Rate Distribution")

        # Add Choropleth Map for Job Distribution by Country
        st.subheader("🌍 Job Distribution Map")
        if not filtered_df.empty and 'Country' in filtered_df.columns and not filtered_df['Country'].isna().all():
            jobs_by_country_map = filtered_df.groupby('Country')['Job_ID'].count().reset_index(name='Job_Count')
            if not jobs_by_country_map.empty and jobs_by_country_map['Job_Count'].sum() > 0:
                jobs_by_country_map['Job_Count'] = jobs_by_country_map['Job_Count'].fillna(0).astype(int)
                fig_map = px.choropleth(jobs_by_country_map, 
                                        locations="Country",
                                        locationmode="country names",
                                        color="Job_Count",
                                        color_continuous_scale="Viridis",
                                        title="Job Distribution by Country",
                                        height=400,
                                        labels={'Job_Count': 'Number of Jobs'})
                fig_map.update_layout(
                    geo=dict(showframe=False, projection_type='natural earth'),
                    title_font_size=14,
                    font=dict(size=10),
                    margin=dict(l=5, r=5, t=30, b=5)
                )
                render_visual(fig_map, "Job Distribution by Country", height=400)
            else:
                st.warning("No valid data for Job Distribution by Country")
        else:
            st.warning("No data for Job Distribution by Country")

    # Main Tab
    with tabs[1]:
        st.subheader("📊 Main Insights")
        sub_tabs = st.tabs(["Product & Service Insights", "Customer Behavior & Engagement", "Company Performance & Trends", "Promotions & AI Engagement"])

        # Product & Service Insights Subtab
        with sub_tabs[0]:
            col1, col2 = st.columns(2)
            with col1:
                if not filtered_df.empty and 'Product_Name' in filtered_df.columns and 'Revenue' in filtered_df.columns:
                    sales_by_product = filtered_df.groupby('Product_Name')['Revenue'].sum().reset_index(name='Total Revenue')
                    sales_by_product = sales_by_product.sort_values('Total Revenue', ascending=False).head(5)
                    if not sales_by_product.empty and sales_by_product['Total Revenue'].sum() > 0:
                        fig2 = px.pie(sales_by_product, names='Product_Name', values='Total Revenue', 
                                      hole=0.3, height=200, color_discrete_sequence=px.colors.qualitative.Set1)
                        fig2.update_traces(textinfo='label+percent', pull=[0.05] * len(sales_by_product))
                        fig2.update_layout(
                            showlegend=True, 
                            legend=dict(orientation='v', x=1.1, y=0.5, font=dict(size=8)),
                            title_font_size=10, font=dict(size=8),
                            margin=dict(l=5, r=5, t=15, b=5)
                        )
                        render_visual(fig2, "Top Sales by Product")
                    else:
                        st.warning("No valid data for Top Sales by Product")
                else:
                    st.warning("No data for Top Sales by Product")

            with col2:
                if not filtered_df.empty and 'Product_Name' in filtered_df.columns and 'Conversion_Status' in filtered_df.columns:
                    conversion_by_product = filtered_df.groupby('Product_Name')['Conversion_Status'].apply(
                        lambda x: (x == 'True').mean() * 100
                    ).reset_index(name='Conversion Rate')
                    if not conversion_by_product.empty and conversion_by_product['Conversion Rate'].sum() > 0:
                        fig_conversion = px.bar(conversion_by_product, x='Product_Name', y='Conversion Rate', 
                                                color='Conversion Rate', color_continuous_scale='Purples', height=200)
                        fig_conversion.update_layout(
                            xaxis_title='Product Name', yaxis_title='Conversion Rate (%)',
                            xaxis_tickangle=45, title_font_size=10, font=dict(size=8),
                            margin=dict(l=5, r=5, t=15, b=5)
                        )
                        render_visual(fig_conversion, "Conversion Rate by Product")
                    else:
                        st.warning("No valid data for Conversion Rate by Product")
                else:
                    st.warning("No data for Conversion Rate by Product")

            col3, col4 = st.columns(2)
            with col3:
                if not filtered_df.empty and 'Service_Type' in filtered_df.columns and 'Revenue' in filtered_df.columns:
                    revenue_by_service = filtered_df.groupby('Service_Type')['Revenue'].sum().reset_index()
                    if not revenue_by_service.empty and revenue_by_service['Revenue'].sum() > 0:
                        fig_service = px.bar(revenue_by_service, x='Service_Type', y='Revenue', color='Revenue', 
                                             color_continuous_scale='Greens', height=200)
                        fig_service.update_layout(
                            xaxis_title='Service Type', yaxis_title='Revenue ($)',
                            xaxis_tickangle=45, title_font_size=10, font=dict(size=8),
                            margin=dict(l=5, r=5, t=15, b=5)
                        )
                        render_visual(fig_service, "Revenue by Service Type")
                    else:
                        st.warning("No valid data for Revenue by Service Type")
                else:
                    st.warning("No data for Revenue by Service Type")

            with col4:
                if not analysis['action_types'].empty:
                    fig_action = px.pie(analysis['action_types'], names='Action_Type', values='Count', 
                                        height=200, color_discrete_sequence=px.colors.qualitative.Set2)
                    fig_action.update_traces(textinfo='percent', pull=[0.1] * len(analysis['action_types']))
                    render_visual(fig_action, "Action Type Distribution")
                else:
                    st.warning("No data for Action Type Distribution")

        # Customer Behavior & Engagement Subtab
        with sub_tabs[1]:
            col1, col2 = st.columns(2)
            with col1:
                if not filtered_df.empty and 'Customer_Behavior' in filtered_df.columns and 'Revenue' in filtered_df.columns:
                    revenue_by_behavior = filtered_df.groupby('Customer_Behavior')['Revenue'].sum().reset_index(name='Total Revenue')
                    if not revenue_by_behavior.empty and revenue_by_behavior['Total Revenue'].sum() > 0:
                        fig8 = px.bar(revenue_by_behavior, x='Customer_Behavior', y='Total Revenue', color='Total Revenue', 
                                      color_continuous_scale='Purples', height=200)
                        fig8.update_layout(
                            xaxis_title='Customer Behavior', yaxis_title='Revenue ($)',
                            xaxis_tickangle=45, title_font_size=10, font=dict(size=8),
                            margin=dict(l=5, r=5, t=15, b=5)
                        )
                        render_visual(fig8, "Revenue by Customer Behavior")
                    else:
                        st.warning("No valid data for Revenue by Customer Behavior")
                else:
                    st.warning("No data for Revenue by Customer Behavior")

            with col2:
                if not filtered_df.empty and 'Customer_Behavior' in filtered_df.columns and 'Pages_Per_Session' in filtered_df.columns:
                    fig_pages = px.box(filtered_df, x='Customer_Behavior', y='Pages_Per_Session', 
                                       height=200, color='Customer_Behavior', 
                                       color_discrete_sequence=px.colors.qualitative.Set2)
                    fig_pages.update_layout(
                        xaxis_title='Customer Behavior', yaxis_title='Pages Per Session',
                        title_font_size=10, font=dict(size=8),
                        margin=dict(l=5, r=5, t=15, b=5)
                    )
                    render_visual(fig_pages, "Customer Behavior vs. Pages Per Session")
                else:
                    st.warning("No data for Customer Behavior vs. Pages Per Session")

            col3, col4 = st.columns(2)
            with col3:
                if not filtered_df.empty and 'Referral' in filtered_df.columns and 'Number_of_Sales' in filtered_df.columns:
                    sales_by_referral = filtered_df.groupby('Referral')['Number_of_Sales'].sum().reset_index()
                    if not sales_by_referral.empty and sales_by_referral['Number_of_Sales'].sum() > 0:
                        fig_referral = px.bar(sales_by_referral, x='Referral', y='Number_of_Sales', color='Number_of_Sales', 
                                              color_continuous_scale='Blues', height=200)
                        fig_referral.update_layout(
                            xaxis_title='Referral', yaxis_title='Sales',
                            xaxis_tickangle=45, title_font_size=10, font=dict(size=8),
                            margin=dict(l=5, r=5, t=15, b=5)
                        )
                        render_visual(fig_referral, "Top Referrals by Sales")
                    else:
                        st.warning("No valid data for Top Referrals by Sales")
                else:
                    st.warning("No data for Top Referrals by Sales")

            with col4:
                if not analysis['events_vs_demos'].empty:
                    fig_events_demos = px.scatter(analysis['events_vs_demos'], x='Promo_Events', y='Demo_Requests', 
                                                  color='Customer_Behavior', text='Country', 
                                                  height=200, color_discrete_sequence=px.colors.qualitative.Set3)
                    fig_events_demos.update_traces(textposition='top center')
                    fig_events_demos.update_layout(xaxis_title='Promotional Events', yaxis_title='Demo Requests')
                    render_visual(fig_events_demos, "Promotional Events vs. Demo Requests by Customer Behavior")
                else:
                    st.warning("No data for Promotional Events vs. Demo Requests")

            # Customer Behavior Impact Table
            st.subheader("Customer Behavior Impact")
            behavior_sales = filtered_df.groupby('Customer_Behavior').agg({
                'Revenue': 'sum',
                'Number_of_Sales': 'sum',
                'Conversion_Status': lambda x: (x == 'True').mean() * 100
            }).reset_index()
            behavior_sales.columns = ['Customer_Behavior', 'Total_Revenue', 'Total_Sales', 'Conversion_Rate']
            st.table(behavior_sales.sort_values('Total_Revenue', ascending=False))

        # Company Performance & Trends Subtab
        with sub_tabs[2]:
            col1, col2 = st.columns(2)
            with col1:
                if not filtered_df.empty and 'Company_Name' in filtered_df.columns and 'Number_of_Sales' in filtered_df.columns:
                    sales_by_company = filtered_df.groupby('Company_Name')['Number_of_Sales'].sum().reset_index(name='Total Sales')
                    sales_by_company = sales_by_company.sort_values('Total Sales', ascending=False)
                    if not sales_by_company.empty and sales_by_company['Total Sales'].sum() > 0:
                        fig4 = px.bar(sales_by_company, x='Company_Name', y='Total Sales', color='Total Sales', 
                                      color_continuous_scale='Oranges', height=200)
                        fig4.update_layout(
                            xaxis_title='Company Name', yaxis_title='Sales',
                            xaxis_tickangle=45, title_font_size=10, font=dict(size=8),
                            margin=dict(l=5, r=5, t=15, b=5)
                        )
                        render_visual(fig4, "Total Sales by Company")
                    else:
                        st.warning("No valid data for Total Sales by Company")
                else:
                    st.warning("No data for Total Sales by Company")

            with col2:
                top_companies = filtered_df.groupby('Company_Name')['Revenue'].sum().nlargest(10).reset_index()
                if not top_companies.empty and top_companies['Revenue'].sum() > 0:
                    fig13 = px.bar(top_companies, x='Company_Name', y='Revenue', color='Revenue', 
                                   color_continuous_scale='Blues', height=200)
                    fig13.update_layout(
                        xaxis_title='Company Name', yaxis_title='Revenue ($)',
                        xaxis_tickangle=45, title_font_size=10, font=dict(size=8),
                        margin=dict(l=5, r=5, t=15, b=5)
                    )
                    render_visual(fig13, "Top Revenue-Generating Companies")
                else:
                    st.warning("No valid data for Top Revenue-Generating Companies")

            col3, col4 = st.columns(2)
            with col3:
                if not filtered_df.empty and 'Date' in filtered_df.columns and 'Number_of_Sales' in filtered_df.columns:
                    sales_over_time = filtered_df.groupby(filtered_df['Date'].dt.date)['Number_of_Sales'].sum().reset_index()
                    if not sales_over_time.empty and sales_over_time['Number_of_Sales'].sum() > 0:
                        fig_sales_time = px.line(sales_over_time, x='Date', y='Number_of_Sales', height=200, color_discrete_sequence=['#00CC96'])
                        fig_sales_time.update_layout(
                            xaxis_title='Date', yaxis_title='Sales',
                            title_font_size=10, font=dict(size=8),
                            margin=dict(l=5, r=5, t=15, b=5)
                        )
                        render_visual(fig_sales_time, "Sales Trend Over Time")
                    else:
                        st.warning("No valid data for Sales Trend Over Time")
                else:
                    st.warning("No data for Sales Trend Over Time")

            with col4:
                if not analysis['job_types'].empty:
                    fig_types = px.pie(analysis['job_types'].head(5), names='Job_Type', values='Count', 
                                       height=200, color_discrete_sequence=px.colors.qualitative.Set1)
                    fig_types.update_traces(textinfo='percent', pull=[0.1] * len(analysis['job_types'].head(5)))
                    render_visual(fig_types, "Top Job Types Distribution")
                else:
                    st.warning("No data for Job Types Distribution")

        # Promotions & AI Engagement Subtab
        with sub_tabs[3]:
            col1, col2, col3 = st.columns(3)
            with col1:
                if not analysis['promo_events'].empty:
                    fig_promo = px.bar(analysis['promo_events'], x='Country', y='Promo_Events', 
                                       color='Promo_Events', color_continuous_scale='Oranges', height=200)
                    fig_promo.update_layout(
                        xaxis_title='Country', yaxis_title='Promo Events',
                        xaxis_tickangle=45, title_font_size=10, font=dict(size=8),
                        margin=dict(l=5, r=5, t=15, b=5)
                    )
                    render_visual(fig_promo, "Promotional Events by Country")
                else:
                    st.warning("No data for Promotional Events")

            with col2:
                if not analysis['demo_requests'].empty:
                    scatter_data = analysis['demo_requests'].merge(
                        filtered_df.groupby('Country').agg({'Revenue': 'sum', 'Job_ID': 'count'}).reset_index(),
                        on='Country', how='left'
                    )
                    scatter_data['Revenue'] = scatter_data['Revenue'].fillna(0)
                    scatter_data['Job_ID'] = scatter_data['Job_ID'].fillna(0)
                    fig_scatter = px.scatter(scatter_data, x='Demo_Requests', y='Revenue', 
                                             color='Country', size='Job_ID', 
                                             height=200, color_discrete_sequence=px.colors.qualitative.Set2)
                    fig_scatter.update_layout(xaxis_title='Demo Requests', yaxis_title='Revenue ($)')
                    render_visual(fig_scatter, "Revenue vs. Demo Requests by Country")
                else:
                    st.warning("No data for Revenue vs. Demo Requests")

            with col3:
                if not analysis['ai_requests'].empty:
                    fig_ai = px.bar(analysis['ai_requests'], x='Country', y='AI_Requests', 
                                    color='AI_Requests', color_continuous_scale='Purples', height=200)
                    fig_ai.update_layout(
                        xaxis_title='Country', yaxis_title='AI Requests',
                        xaxis_tickangle=45, title_font_size=10, font=dict(size=8),
                        margin=dict(l=5, r=5, t=15, b=5)
                    )
                    render_visual(fig_ai, "AI Assistant Requests by Country")
                else:
                    st.warning("No data for AI Assistant Requests")

    # EDA Tab
    with tabs[2]:
        st.subheader("📊 Exploratory Data Analysis")

        # Data Quality and Structure
        st.subheader("Data Quality and Structure")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Missing Values in Raw Data** (before cleaning)")
            missing_data = pd.DataFrame({
                'Column': raw_df.columns,
                'Missing Count': raw_df.isna().sum(),
                'Missing Percentage': (raw_df.isna().sum() / len(raw_df) * 100).round(2)
            }).reset_index(drop=True)
            st.table(missing_data)

        with col2:
            st.markdown("**Data Types**")
            dtype_data = pd.DataFrame({
                'Column': filtered_df.columns,
                'Data Type': [str(dtype) for dtype in filtered_df.dtypes]
            }).reset_index(drop=True)
            st.table(dtype_data)

        # Sales Team Performance and Summary Statistics
        st.subheader("Sales Performance and Statistics")
        st.markdown(f"**Overall Conversion Rate**: {analysis['conversion_rate']:.2f}%")
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown("**Sales Team Performance**")
            sales_by_company = filtered_df.groupby(['Company_Name', 'Country']).agg({
                'Revenue': 'sum',
                'Number_of_Sales': 'sum',
                'Conversion_Status': lambda x: (x == 'True').mean() * 100
            }).reset_index()
            sales_by_company.columns = ['Company_Name', 'Country', 'Total_Revenue', 'Total_Sales', 'Conversion_Rate']
            st.table(sales_by_company.sort_values('Total_Revenue', ascending=False))
        
        with col4:
            st.markdown("**Summary Statistics**")
            stats_df = pd.DataFrame([
                {
                    "Metric": "Jobs Placed",
                    "Value": f"{analysis['stats']['Jobs_Placed']['Mean']:.2f}",
                    "Std": f"{analysis['stats']['Jobs_Placed']['Std']:.2f}",
                    "Median": f"{analysis['stats']['Jobs_Placed']['Median']:.2f}",
                    "Range": f"{analysis['stats']['Jobs_Placed']['Range']:.2f}"
                },
                {
                    "Metric": "Revenue",
                    "Value": f"${analysis['stats']['Revenue']['Mean']:,.2f}",
                    "Std": f"${analysis['stats']['Revenue']['Std']:,.2f}",
                    "Median": f"${analysis['stats']['Revenue']['Median']:,.2f}",
                    "Range": f"${analysis['stats']['Revenue']['Range']:,.2f}"
                },
                {
                    "Metric": "Demo Requests",
                    "Value": f"{analysis['stats']['Demo_Requests']['Mean']:.2f}",
                    "Std": f"{analysis['stats']['Demo_Requests']['Std']:.2f}",
                    "Median": f"{analysis['stats']['Demo_Requests']['Median']:.2f}",
                    "Range": f"{analysis['stats']['Demo_Requests']['Range']:.2f}"
                },
                {
                    "Metric": "Number of Sales",
                    "Value": f"{analysis['stats']['Number_of_Sales']['Mean']:.2f}",
                    "Std": f"{analysis['stats']['Number_of_Sales']['Std']:.2f}",
                    "Median": f"{analysis['stats']['Number_of_Sales']['Median']:.2f}",
                    "Range": f"{analysis['stats']['Number_of_Sales']['Range']:.2f}"
                },
                {
                    "Metric": "Profit",
                    "Value": f"${analysis['stats']['Profit']['Mean']:,.2f}",
                    "Std": f"${analysis['stats']['Profit']['Std']:,.2f}",
                    "Median": f"${analysis['stats']['Profit']['Median']:,.2f}",
                    "Range": f"${analysis['stats']['Profit']['Range']:,.2f}"
                },
                {
                    "Metric": "Conversion Rate",
                    "Value": f"{analysis['stats']['Conversion_Rate']['Mean']:.2f}%",
                    "Std": f"{analysis['stats']['Conversion_Rate']['Std']:.2f}%",
                    "Median": f"{analysis['stats']['Conversion_Rate']['Median']:.2f}%",
                    "Range": f"{analysis['stats']['Conversion_Rate']['Range']:.2f}%"
                }
            ])
            st.table(stats_df)

if __name__ == "__main__":
    main()
