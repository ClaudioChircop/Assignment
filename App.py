import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt
from scipy.stats import zscore

# Function to load the dataset
@st.cache_data  # Cache the function to enhance performance of streamlit
def load_data():
    file_path = 'micro_world_139countries.csv'
    df = pd.read_csv(file_path, encoding='ISO-8859-1')

    # Create age group column
    bins = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    labels = ['10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90+']
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)
    
    return df

# Load data into session
df = load_data()

# Creating new colums with categorical varibles instead of numerical

# Making a new colum with the female to gender and map the values 1 to "Female" and 2 to "Male"
gender_mapping = {1: 'Female', 2: 'Male'}
df['gender'] = df['female'].map(gender_mapping)

# Making new colum with the ecucation level describtion 
education_mapping = {1: 'Completed primary school or less', 2: 'Completed secondary school', 3: 'Completed tertiary education or more'}
df['education_level'] = df['educ'].map(education_mapping)




# =============================================================================================================================
# Sidebar settings 
# =============================================================================================================================

# Sidebar economy dropdown
selected_economy = st.sidebar.multiselect('Select Economy', df['economy'].unique(), default=[])

# Sidebar gender dropdown
selected_genders = st.sidebar.multiselect('Select Gender', df['gender'].unique(), default=[])

# Sidebar education level dropdown
selected_educational_level = st.sidebar.multiselect('Select educational level', df['education_level'].unique(), default=[])

# Sidebar Age Slider
st.sidebar.header('Filter by Age')
age_range = st.sidebar.slider('Select Age Range', int(df['age'].min()), int(df['age'].max()), (20, 50))

# Initial filter - apply all conditions cumulatively
filtered_data = df[df['age'].between(age_range[0], age_range[1])]

# Apply economy filter if selections are made
if selected_economy:
    filtered_data = filtered_data[filtered_data['economy'].isin(selected_economy)]

# Apply gender filter if selections are made
if selected_genders:
    filtered_data = filtered_data[filtered_data['gender'].isin(selected_genders)]

# Apply educational level filter if selections are made
if selected_educational_level:
    filtered_data = filtered_data[filtered_data['education_level'].isin(selected_educational_level)]

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Info", "Distribution", "Statistics"])



# =============================================================================================================================
# The App's Pages and their content
# =============================================================================================================================

# Home page
# =============================================================================================================================
if page == "Info":
    st.title("Home Page (Info)")
    st.write("Welcome to the Streamlit Dashboard of the FINDEX dataset! Navigate in the sidebar to the correct page you want to explore.")

    st.write("This dashboard provides insights into global financial inclusion, based on data from the Global Findex 2021 / World Bank survey. It covers various demographics, income, and financial behaviors across multiple countries.")

    st.subheader("Dataset Overview")
    st.write(f"Number of Respondents: {df.shape[0]:,}")
    st.write(f"Number of Countries: {df['economy'].nunique():,}")
    st.write(f"Age Range: {df['age'].min():,} - {df['age'].max():,}")
    st.write(f"Gender Distribution: {df['female'].value_counts()[1]:,} Female, {df['female'].value_counts()[2]:,} Male")


    st.subheader("Key Variable Descriptions")

    st.write("""
    - **Age**: Respondent's age.
    - **Economy**: Country of the respondent.
    - **Gender**: 1 for Female, 2 for Male.
    - **Education Level**: 1 for primary or less, 2 for secondary, 3 for tertiary education.
    - **Has an account at a financial institution**: 1 if yes, 0 if no.
    - **Mobile money usage**: 1 if the respondent uses mobile money, 0 if no.
    """)



# Distribution page 
# =============================================================================================================================
elif page == "Distribution":
    st.title("Exploratory Data Analysis")

    st.write("Be aware this page is not effected by the filters")
    
    st.write("Here is a preview of the Age Distribution:")
    def plot_age_distribution(data):
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(data['age'], kde=True, ax=ax)
        st.pyplot(fig)
    plot_age_distribution(df)


    st.write("Here is a preview of the gender Distribution:")
    def plot_gender_distribution(data):
        # Count the number of occurrences for each gender
        gender_counts = data['gender'].value_counts()

        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot the data as a bar plot
        sns.barplot(x=gender_counts.index, y=gender_counts.values, ax=ax, palette="Blues_d")

        # Add gridlines for better visual separation
        ax.grid(True, linestyle='--', alpha=0.6)

        # Add title and labels
        ax.set_title('Distribution of gender', fontsize=16, fontweight='bold')
        ax.set_xlabel('Gender', fontsize=14)
        ax.set_ylabel('Frequency', fontsize=14)

        # Add annotations on top of the bars
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='baseline', fontsize=12, color='black', xytext=(0, 5),
                        textcoords='offset points')

        # Show the plot in Streamlit
        st.pyplot(fig)

    # Call the function to display the plot
    plot_gender_distribution(df)


    st.write("Here is a preview of the education level Distribution:")
    def plot_education_distribution(data):
        # Count the number of occurrences for each education level
        edu_counts = data['education_level'].value_counts()

        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot the data as a bar plot
        sns.barplot(x=edu_counts.index, y=edu_counts.values, ax=ax, palette="Blues_d")

        # Rotate the x labels for better readability
        plt.xticks(rotation=45, ha='right', fontsize=12)

        # Add gridlines for better visual separation
        ax.grid(True, linestyle='--', alpha=0.6)

        # Add title and labels
        ax.set_title('Distribution of Education Levels', fontsize=16, fontweight='bold')
        ax.set_xlabel('education_level', fontsize=14)
        ax.set_ylabel('Frequency', fontsize=14)

        # Add annotations on top of the bars
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='baseline', fontsize=12, color='black', xytext=(0, 5),
                        textcoords='offset points')

        # Show the plot in Streamlit
        st.pyplot(fig)

    # Call the function to display the plot
    plot_education_distribution(df)




# Statistics page 
# =============================================================================================================================
elif page == "Statistics":
    st.title("Statistics Page")

    # age

    if not filtered_data.empty:
        mean_age = filtered_data['age'].mean()
        median_age = filtered_data['age'].median()
        max_age = filtered_data['age'].max()
        min_age = filtered_data['age'].min()
    else:
        mean_age = median_age = max_age = min_age = None

    st.subheader('Age Statistics')
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric('Mean Age', f"{mean_age:.2f}" if mean_age is not None else "N/A")
    with col2:
        st.metric('Median Age', f"{median_age:.2f}" if median_age is not None else "N/A")
    with col3:
        st.metric('Max Age', f"{max_age:.2f}" if max_age is not None else "N/A")
    with col4:
        st.metric('Min Age', f"{min_age:.2f}" if min_age is not None else "N/A")


    # Add your title
    st.title("Boxplot ")

    # Boxplot before applying the cap and hurdle on age
    st.write("Boxplot of Age - figure showing the distribution")
    plt.figure(figsize=(8, 4))  # Define the size of the figure
    sns.boxplot(x='age', data=filtered_data)  # Create a boxplot based on "age"
    plt.title("Boxplot of Age")  # Title of the plot
    st.pyplot(plt)  # Display the plot in Streamlit
    
    
    # Display filtered data
    st.write('Filtered Data:')
    st.dataframe(filtered_data)