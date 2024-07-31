# Import libraries
import streamlit as st
import pandas as pd
import plotly.express as px
#import plotly.graph_objects as go
from dotenv import load_dotenv
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import create_sql_agent,SQLDatabaseToolkit
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from sqlalchemy import create_engine 

load_dotenv()

# Set the page configuration
st.set_page_config(page_title="SF state Salary Data Dashboard", layout="wide", initial_sidebar_state="expanded")

#######################
# CSS styling for the floating button
st.markdown("""
    <style>
    .floating-button {
        position: fixed;
        width: 60px;
        height: 60px;
        bottom: 40px;
        right: 40px;
        background-color: #f39c12;
        color: #FFF;
        border-radius: 50px;
        text-align: center;
        box-shadow: 2px 2px 3px #999;
        cursor: pointer;
        font-size: 30px;
        line-height: 60px;
    }
    </style>
""", unsafe_allow_html=True)

#######################
# CSS styling for the rest of the page
st.markdown("""
<style>
[data-testid="block-container"] {
    padding-left: 2rem;
    padding-right: 2rem;
    padding-top: 1rem;
    padding-bottom: 1rem;
}
[data-testid="stSidebar"] {
    background-color:#393939 ;
}
[data-testid="stMetric"] {
    background-color: #393939;
    text-align: center;
    padding: 15px 0;
}
[data-testid="stMetricLabel"] {
    display: flex;
    justify-content: center;
    align-items: center;
}
[data-testid="stMetricDeltaIcon-Up"] {
    position: relative;
    left: 38%;
    -webkit-transform: translateX(-50%);
    -ms-transform: translateX(-50%);
    transform: translateX(-50%);
}
[data-testid="stMetricDeltaIcon-Down"] {
    position: relative;
    left: 38%;
    -webkit-transform: translateX(-50%);
    -ms-transform: translateX(-50%);
    transform: translateX(-50%);
}
</style>
""", unsafe_allow_html=True)

#######################
# Load data
file_path = 'data/Salaries.csv'
salaries_df = pd.read_csv(file_path)

# Convert mixed type columns to numeric
salaries_df['BasePay'] = pd.to_numeric(salaries_df['BasePay'], errors='coerce')
salaries_df['OvertimePay'] = pd.to_numeric(salaries_df['OvertimePay'], errors='coerce')
salaries_df['OtherPay'] = pd.to_numeric(salaries_df['OtherPay'], errors='coerce')
salaries_df['Benefits'] = pd.to_numeric(salaries_df['Benefits'], errors='coerce')

# Handle missing values
salaries_df['BasePay'].fillna(0, inplace=True)
salaries_df['OvertimePay'].fillna(0, inplace=True)
salaries_df['OtherPay'].fillna(0, inplace=True)
salaries_df['Benefits'].fillna(0, inplace=True)

# Remove unnecessary columns
salaries_df.drop(columns=['Notes', 'Status'], inplace=True)

# Create a subset for selected year and sorting
year_list = list(salaries_df['Year'].unique())[::-1]
llm = ChatOpenAI(temperature=0)
#######################
# Sidebar
with st.sidebar:
    st.title('SF State Salary Data Dashboard')
    
    selected_year = st.selectbox('Select a year', year_list)
    df_selected_year = salaries_df[salaries_df['Year'] == selected_year]
    df_selected_year_sorted = df_selected_year.sort_values(by="TotalPay", ascending=False)

    # Floating button in the sidebar
    st.subheader('Ask Questions About Your Data')
    with st.form(key='llm_form'):
        user_question = st.text_input("Ask your assistant:")
        submit_button = st.form_submit_button(label='Submit')
    
    if submit_button and user_question:
        # Call your LLM agent here
        agent = create_pandas_dataframe_agent(llm, df=salaries_df, verbose=True)
        with st.spinner(text="In Progress.."):
            response = agent.invoke(user_question)
            #response = "Mock response based on your dataframe query"  # Mock response for example
            st.write(response['output'])

    st.sidebar.subheader('Upload CSV or Excel File')
    uploaded_file = st.sidebar.file_uploader("Choose a file", type=['csv', 'xlsx'])        
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            st.success('File uploaded successfully')
        except Exception as e:
            st.error(f'Error: {e}')

    # Display the dataframe if a file is uploaded
    if uploaded_file is not None:
        database_file_path = "./salaries.db"
        engine = create_engine(f'sqlite:///{database_file_path}')
        #file_url = "data/CalCareerData.csv"
        df.to_sql("SalriesData",con=engine,if_exists="replace", index="False")
        st.success("Database created  successfully")

#######################
    st.subheader('Chat With Your SQL')
    with st.form(key='agent_form'):
        query = st.text_input("Ask your about the uploaded docs:")
        submit_button = st.form_submit_button(label='Submit')
        if submit_button and query:
        # Call your LLM agent 
            db = SQLDatabase.from_uri("sqlite:///salaries.db")
            agent_executer = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=True)
            with st.spinner(text="In Progress.."):
                answer = agent_executer.invoke(query)
                
                st.write(answer['output'])

# Plots

# Bar Plot
def make_barplot(input_df, input_x, input_y, input_title, input_color):
    fig = px.bar(input_df, x=input_x, y=input_y, color=input_color, color_continuous_scale="Magma", title=input_title)
    fig.update_layout(template='plotly_dark', margin=dict(l=0, r=0, t=30, b=0))
    return fig

# Histogram for Income Distribution
def make_histogram(input_df, input_column, input_title):
    fig = px.histogram(input_df, x=input_column, nbins=50, title=input_title, color_discrete_sequence=['#636EFA'])
    fig.update_layout(template='plotly_dark', margin=dict(l=0, r=0, t=30, b=0))
    return fig

# Scatter Plot
def make_scatterplot(input_df, input_x, input_y, input_size, input_color, input_title):
    fig = px.scatter(input_df, x=input_x, y=input_y, size=input_size, color=input_color, color_continuous_scale="Magma", title=input_title)
    fig.update_layout(template='plotly_dark', margin=dict(l=0, r=0, t=30, b=0))
    return fig

#######################
# Dashboard Main Panel
st.title('SF State Salary Data Dashboard')

# Create columns for metrics
st.header('Key Metrics')
metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
with metrics_col1:
    st.metric(label="Total Records", value=len(df_selected_year))
with metrics_col2:
    st.metric(label="Highest Salary", value=f"${df_selected_year['TotalPay'].max():,.2f}")
with metrics_col3:
    st.metric(label="Lowest Salary", value=f"${df_selected_year['TotalPay'].min():,.2f}")
with metrics_col4:
    st.metric(label="Average Salary", value=f"${df_selected_year['TotalPay'].mean():,.2f}")

# Create rows for plots
st.header('Visual Insights')

# Row 1
row1_col1, row1_col2 = st.columns(2)
with row1_col1:
    st.subheader('Top 5 Highest Paid Jobs')
    top_5_jobs = df_selected_year_sorted.head(5)
    bar_plot = make_barplot(top_5_jobs, 'JobTitle', 'TotalPay', 'Top 5 Highest Paid Jobs', 'TotalPay')
    st.plotly_chart(bar_plot, use_container_width=True)
    
with row1_col2:
    st.subheader('Income Distribution')
    income_histogram = make_histogram(df_selected_year, 'TotalPay', 'Distribution of Total Pay')
    st.plotly_chart(income_histogram, use_container_width=True)

# Row 2
row2_col1, row2_col2 = st.columns(2)
with row2_col1:
    st.subheader('Best Earners')
    best_earners = df_selected_year_sorted.head(10)
    scatter_plot = make_scatterplot(best_earners, 'BasePay', 'TotalPay', 'TotalPayBenefits', 'JobTitle', 'Best Earners Scatter Plot')
    st.plotly_chart(scatter_plot, use_container_width=True)

with row2_col2:
    st.subheader('Top 10 Job Titles by Total Pay')
    top_10_jobs = df_selected_year.groupby('JobTitle')['TotalPay'].sum().sort_values(ascending=False).head(10).reset_index()
    bar_plot_top10 = make_barplot(top_10_jobs, 'JobTitle', 'TotalPay', 'Top 10 Job Titles by Total Pay', 'TotalPay')
    st.plotly_chart(bar_plot_top10, use_container_width=True)

# Summary statistics
st.header('Summary Statistics')
summary_stats = df_selected_year.describe()
st.dataframe(summary_stats)

# About section
st.header('About')
with st.expander('More Information', expanded=True):
    st.write('''
        - Data: [U.S. Census Bureau](https://www.census.gov/data/datasets/time-series/demo/popest/2010s-state-total.html).
        - **Top 5 Highest Paid Jobs**: Displays the top 5 highest paid jobs for the selected year.
        - **Income Distribution**: Histogram showing the distribution of total pay among employees.
        - **Best Earners**: Scatter plot showing the top 10 earners based on total pay.
        - **Top 10 Job Titles by Total Pay**: Bar chart showing the top 10 job titles by total pay.
        - **Summary Statistics**: Displays summary statistics of the salary data.
    ''')
