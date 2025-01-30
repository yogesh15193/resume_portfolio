pip install kagglehub
pip install kaggle

#importing the necessary libraries 
import pandas as pd
import numpy as np
import kagglehub
from kagglehub.config import get_kaggle_credentials

import kagglehub

from kagglehub.config import get_kaggle_credentials
kagglehub.login() 
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns 

import kagglehub
from kagglehub.config import get_kaggle_credentials
kagglehub.login() 

import os
os.chdir(r"C:\Users\Yogesh Gupta\TU315\Data Project\final_optimized_project")
print("Current Directory:", os.getcwd())

# Set Kaggle API credentials
os.environ["KAGGLE_USERNAME"] = "yogeshgupt"
os.environ["KAGGLE_KEY"] = "fcf33813989dde8b8ad7567e1c78aee9"

!kaggle datasets download -d blastchar/telco-customer-churn -p ./ --unzip

dataset=pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
print(dataset.info())

############################################ CLEANING DATASET #################################

# drop duplicate rows if any
dataset.drop_duplicates(subset=None, keep='first', inplace=True)

# check for any empty rows in a column
for col in dataset.columns:
    blank_count = (dataset[col].astype(str).str.strip() == "").sum()
    print(f"Column '{col}' - Blank values: {blank_count}")

# finding rows with missing values

# Filter rows where TotalCharges is an empty string or contains whitespace
data_missing_rows = dataset[dataset['TotalCharges'].str.strip() == ""].copy()

# Display the resulting DataFrame
print(data_missing_rows)

# correcting the dtypes of columns which need to be corrected
dataset['TotalCharges'] = pd.to_numeric(dataset['TotalCharges'], errors='coerce')
dataset['gender'] = dataset['gender'].astype('category')

# checking what options we have been given in these columns, based on that, convert them into numeric, Yes-1, No- 0
seniorcitisen=set(list(dataset['SeniorCitizen']))
gender=set(list(dataset['gender']))
OnlineBackup= set(list(dataset['OnlineBackup']))
DeviceProtection=set(list(dataset['DeviceProtection']))
TechSupport=set(list(dataset['TechSupport']))
StreamingTV= set(list(dataset['StreamingTV']))
StreamingMovies= set(list(dataset['StreamingMovies']))
Partner =set(list(dataset['Partner']))
Dependents=set(list(dataset['Dependents']))
churn=set(list(dataset['Churn']))
print("seniorcitizen",seniorcitisen)
print("gender", gender)
print("OnlineBackup", OnlineBackup)
print("DeviceProtection", DeviceProtection)
print("TechSupport", TechSupport)
print("StreamingTV", StreamingTV)
print("StreamingMovies", StreamingMovies)
print("Partner", Partner)
print("Dependents", Dependents)
print("churn", churn)


binary_cols = ['Partner', 'Dependents', 'PhoneService', 'OnlineSecurity', 
               'OnlineBackup', 'DeviceProtection', 'TechSupport', 
               'StreamingTV', 'StreamingMovies', 'PaperlessBilling', 'Churn', 'MultipleLines']

for col in binary_cols:
    if col == 'MultipleLines':
        dataset[col] = dataset[col].map({'Yes': 1, 'No': 0, 'No phone service': 0})
    else:
        dataset[col] = dataset[col].map({'Yes': 1, 'No': 0, 'No internet service': 0})


# checking skewness to input mean or median in the missing rows based on the value obtained

dataset['TotalCharges'] = pd.to_numeric(dataset['TotalCharges'], errors='coerce')
skewness = dataset['TotalCharges'].dropna().skew()
print(f"Skewness: {skewness}")


print(f"Number of missing rows: {dataset['TotalCharges'].isnull().sum()}")
# Calculate median
median_value = dataset['TotalCharges'].median()

# Fill missing values with median
dataset['TotalCharges'] = dataset['TotalCharges'].fillna(median_value)

# Verify no missing values remain
print(f"Remaining missing rows: {dataset['TotalCharges'].isnull().sum()}")



print(dataset['MultipleLines'].isnull().sum())
print(dataset['MultipleLines'].unique())


dataset.to_csv("cleaned_stage1.csv",index=False)


################################################# stage 1  complete #######################################

dataset_cleaned=pd.read_csv("cleaned_stage1.csv")
pd.set_option('display.max_columns', None)
dataset_cleaned.head()
dataset_cleaned.head(5)


import plotly.express as px

for col in ['tenure', 'MonthlyCharges', 'TotalCharges']:
    fig = px.box(dataset_cleaned, y=col, title=f"Interactive Boxplot for {col}")
    fig.show()

Q1_m = dataset_cleaned['MonthlyCharges'].quantile(0.25)
Q3_m = dataset_cleaned['MonthlyCharges'].quantile(0.75)
IQR_m = Q3_m - Q1_m
print('IRQ',IQR_m)
lower_fence_m = Q1_m - (1.5 * IQR_m)
upper_fence_m = Q3_m + (1.5 * IQR_m)

print(f"Lower Fence: {lower_fence_m}")
print(f"Upper Fence: {upper_fence_m}")

outliers_m = dataset_cleaned[(dataset_cleaned['MonthlyCharges'] < lower_fence_m) | 
                           (dataset_cleaned['MonthlyCharges'] > upper_fence_m)]

print(f"Number of outliers: {len(outliers_m)}")
print(outliers_m[['MonthlyCharges']])

dataset_cleaned = dataset_cleaned[dataset_cleaned['MonthlyCharges'] <= upper_fence_m]


Q1_t = dataset_cleaned['tenure'].quantile(0.25)
Q3_t = dataset_cleaned['tenure'].quantile(0.75)
IQR_t = Q3_t - Q1_t
print("q1",Q1_t)
print(Q3_t)
print('IQR for tenure:', IQR_t)

lower_fence_t = Q1_t - 1.5 * IQR_t
upper_fence_t = Q3_t + 1.5 * IQR_t

print(f"Lower Fence: {lower_fence_t}")
print(f"Upper Fence: {upper_fence_t}")

outliers_tenure_t = dataset_cleaned[(dataset_cleaned['tenure'] < lower_fence_t) | 
                                  (dataset_cleaned['tenure'] > upper_fence_t)]

print(f"Number of outliers for tenure: {len(outliers_tenure_t)}")
print(outliers_tenure_t[['tenure']])

dataset_cleaned = dataset_cleaned[dataset_cleaned['tenure'] <= upper_fence_t]

Q1_tc = dataset_cleaned['TotalCharges'].quantile(0.25)
print("Q1_tc",Q1_tc)
Q3_tc = dataset_cleaned['TotalCharges'].quantile(0.75)
print("Q3_tc",Q3_tc)
IQR_tc = Q3_tc - Q1_tc
print('IRQ',IQR_tc)
lower_fence_tc = Q1_tc - (1.5 * IQR_tc)
upper_fence_tc = Q3_tc + (1.5 * IQR_tc)

print(f"Lower Fence: {lower_fence_tc}")
print(f"Upper Fence: {upper_fence_tc}")

outliers_tc= dataset_cleaned[(dataset_cleaned['MonthlyCharges'] < lower_fence_tc) | 
                           (dataset_cleaned['MonthlyCharges'] > upper_fence_tc)]

print(f"Number of outliers: {len(outliers_tc)}")
print(outliers_tc[['MonthlyCharges']])

dataset_cleaned = dataset_cleaned[dataset_cleaned['MonthlyCharges'] <= upper_fence_tc]

dataset_cleaned.to_csv("cleaned_stage2.csv",index=False)

########################################################################## CLEANING COMPLETE###################################
####################################################### EDA ##########################################################
cleaned_stage2_dataset=pd.read_csv("cleaned_stage2.csv")
print(cleaned_stage2_dataset)

## line graph representing the customer count between the 6 month intervals. it will give me an idea of how the customer tenure is of telco.
#Cut tenure into bins
bins = range(0, dataset_cleaned['tenure'].max() + 6, 6)
labels = [f'{i//12}y {i%12}m - {(i+6)//12}y {(i+6)%12}m' for i in bins[:-1]]  # Descriptive labels

cleaned_stage2_dataset['tenure_bins'] = pd.cut(cleaned_stage2_dataset['tenure'], bins=bins, labels=labels, right=False)
tenure_counts = cleaned_stage2_dataset['tenure_bins'].value_counts().sort_index()
plt.figure(figsize=(12, 6))
plt.plot(tenure_counts.index.astype(str), tenure_counts.values, marker='o')
plt.title('Line Plot of Tenure (6-Month Bins)')
plt.xlabel('Tenure (Binned)')
plt.ylabel('Customer Count')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


import pandas as pd
import plotly.express as px

# Define custom bins for 0-6, 7-12, 13-18, ...
bins = [0] + [x + 1 for x in range(6, dataset_cleaned['tenure'].max() + 6, 6)]  # Adjust bins for non-overlap
labels = [f"{bins[i]}-{bins[i+1]-1}" for i in range(len(bins) - 1)]  # Create proper labels

# Bin the tenure data
cleaned_stage2_dataset['tenure_bins'] = pd.cut(
    cleaned_stage2_dataset['tenure'],
    bins=bins,
    labels=labels,
    right=False,  # Left-inclusive bins, e.g., [0-6), [7-12)
    include_lowest=True
)

# Count customers in each bin
tenure_bin_counts = cleaned_stage2_dataset['tenure_bins'].value_counts().reindex(labels).reset_index()
tenure_bin_counts.columns = ['tenure_bins', 'count']

# Create the interactive bar chart
fig_hist = px.bar(
    tenure_bin_counts,
    x="tenure_bins",
    y="count",
    title="Interactive Histogram of Tenure (Custom Bins)",
    labels={'tenure_bins': 'Tenure (Binned)', 'count': 'Frequency'},
    text="count"  # Show bin counts on bars
)
fig_hist.update_layout(
    xaxis_title="Tenure (Binned)",
    yaxis_title="Frequency",
    xaxis=dict(tickangle=45)
)
fig_hist.show()



import pandas as pd
import plotly.express as px

# Assuming you've already binned the tenure data into 'tenure_bins'

# Group by tenure bins and contract type to get the count of customers
tenure_contract_counts = cleaned_stage2_dataset.groupby(['tenure_bins', 'Contract']).size().reset_index(name='customer_count')

# Create the interactive grouped bar chart
fig = px.bar(
    tenure_contract_counts,
    x="tenure_bins", 
    y="customer_count", 
    color="Contract",  # Use 'Contract' column for color
    title="Customer Distribution by Tenure and Contract Type",
    labels={'tenure_bins': 'Tenure (Binned)', 'customer_count': 'Customer Count'},
    barmode='group',  # Group bars to compare contract types side-by-side
)

fig.update_layout(
    xaxis_title="Tenure Bins",
    yaxis_title="Customer Count",
    xaxis=dict(tickangle=45),
    legend_title="Contract Type"
)

fig.show()


customer_count_6_month_interval=list(tenure_bin_counts['count'])

''' tenure bins for 3 month intervals'''

import pandas as pd
import plotly.express as px

# Step 1: Define bins for 3-month intervals
bins = list(range(0, cleaned_stage2_dataset['tenure'].max() + 1, 3))  # Create bins with 3-month intervals
bins[-1] += 1  # Ensure the last bin includes the maximum value
labels = [f"{bins[i]}-{bins[i+1]-1}" for i in range(len(bins) - 1)]  # Generate labels like "0-3", "4-6"

# Step 2: Bin the tenure data using pd.cut
cleaned_stage2_dataset['tenure_bins_3_months'] = pd.cut(
    cleaned_stage2_dataset['tenure'],
    bins=bins,
    labels=labels,
    right=False,  # Left-inclusive bins: [0-3), [3-6), ...
    include_lowest=True
)

# Step 3: Count customers in each bin
tenure_bin_counts_3_months = cleaned_stage2_dataset['tenure_bins_3_months'].value_counts().reindex(labels).reset_index()
tenure_bin_counts_3_months.columns = ['tenure_bins', 'count']

# Step 4: Create the interactive bar chart
fig_hist_3_months = px.bar(
    tenure_bin_counts_3_months,
    x="tenure_bins",
    y="count",
    title="Interactive Histogram of Tenure (3-Month Bins)",
    labels={'tenure_bins': 'Tenure (3-Month Binned)', 'count': 'Frequency'},
    text="count"  # Display counts directly on bars
)
fig_hist_3_months.update_layout(
    xaxis_title="Tenure (3-Month Binned)",
    yaxis_title="Frequency",
    xaxis=dict(tickangle=45)  # Rotate x-axis labels for better readability
)
fig_hist_3_months.show()

''' this code snippet will output the mean, median, minimum and maximum monthly charges paid by the customer'''
import pandas as pd
import plotly.express as px

# Define bins for 3-month intervals
bins = list(range(0, cleaned_stage2_dataset['tenure'].max() + 1, 3))  # Create bins with 3-month intervals
bins[-1] += 1  # Ensure the last bin includes the maximum value
labels = [f"{bins[i]}-{bins[i+1]-1}" for i in range(len(bins) - 1)]  # Generate labels like "0-2", "3-5", etc.

# Bin the tenure data using pd.cut
cleaned_stage2_dataset['tenure_bins_3_months'] = pd.cut(
    cleaned_stage2_dataset['tenure'],
    bins=bins,
    labels=labels,
    right=False,  # Left-inclusive bins
    include_lowest=True
)

tenure_bin_counts_3_months = cleaned_stage2_dataset['tenure_bins_3_months'].value_counts().reindex(labels).reset_index()

# Check bin counts for debugging
print("Counts in each 3-month bin:")
print(cleaned_stage2_dataset['tenure_bins_3_months'].value_counts())

# Group MonthlyCharges by tenure bins and calculate summary stats
monthly_charges_summary_3_months = cleaned_stage2_dataset.groupby('tenure_bins_3_months')['MonthlyCharges'].agg(
    ['mean', 'median', 'min', 'max']).reset_index()
monthly_charges_summary_3_months.columns = ['tenure_bins_3_months', 'mean_monthly_charges',
                                            'median_monthly_charges', 'min_monthly_charges',
                                            'max_monthly_charges']

# Display the summary statistics
print("Summary statistics for MonthlyCharges per 3-month bin:")
print(monthly_charges_summary_3_months)

# Create an interactive bar chart for mean MonthlyCharges by 3-month tenure bins
fig_mean_3_months = px.bar(
    monthly_charges_summary_3_months,
    x="tenure_bins_3_months",
    y="mean_monthly_charges",
    title="Mean Monthly Charges by 3-Month Tenure Bins",
    labels={'tenure_bins_3_months': 'Tenure (3-Month Binned)', 'mean_monthly_charges': 'Mean Monthly Charges'},
    text="mean_monthly_charges"  # Display values on bars
)
fig_mean_3_months.update_layout(
    xaxis_title="Tenure (3-Month Binned)",
    yaxis_title="Mean Monthly Charges",
    xaxis=dict(tickangle=45)  # Rotate x-axis labels for better readability
)
fig_mean_3_months.show()


# Calculate churn percentage for 3-month bins
churn_analysis_3_months = cleaned_stage2_dataset.groupby('tenure_bins_3_months', observed=False).agg(
    total_customers=('Churn', 'count'),
    churned_customers=('Churn', 'sum')
).reset_index()

# Add churn percentage column
churn_analysis_3_months['churn_percentage'] = (
    churn_analysis_3_months['churned_customers'] / churn_analysis_3_months['total_customers']
) * 100

# Display churn analysis
print("Churn analysis for 3-month bins:")
print(churn_analysis_3_months)

# Visualize churn percentage for 3-month bins
fig_churn_3_months = px.bar(
    churn_analysis_3_months,
    x="tenure_bins_3_months",
    y="churn_percentage",
    title="Churn Percentage Across 3-Month Tenure Bins",
    labels={"tenure_bins_3_months": "Tenure Bins (3-Months)", "churn_percentage": "Churn Percentage (%)"},
    text="churn_percentage"  # Show churn percentage on bars
)
fig_churn_3_months.update_layout(xaxis=dict(tickangle=45))
fig_churn_3_months.show()



churn_three_month_percentage_list=list(churn_analysis_3_months['churn_percentage'])

tenure_bin_3_months_count_list=list(tenure_bin_counts_3_months['count'])


monthly_charges_summary_3_months['customer_count']=tenure_bin_3_months_count_list
monthly_charges_summary_3_months['churn_percentage']=churn_three_month_percentage_list
monthly_charges_summary_3_months.to_csv("monthly_charges_summary_3_months.csv", index=False)


''' histogramn plotting for mean, median, minimum and maximum monthly charges for 6 month intervals'''
import pandas as pd
import plotly.express as px

# Define custom bins for tenure (same as before)
bins = [0] + [x + 1 for x in range(6, cleaned_stage2_dataset['tenure'].max() + 6, 6)]  # Adjust bins for non-overlap
labels = [f"{bins[i]}-{bins[i+1]-1}" for i in range(len(bins) - 1)]  # Create proper labels

# Bin the tenure data
cleaned_stage2_dataset['tenure_bins'] = pd.cut(
    cleaned_stage2_dataset['tenure'],
    bins=bins,
    labels=labels,
    right=False,  # Left-inclusive bins, e.g., [0-6), [7-12)
    include_lowest=True
)

# Group MonthlyCharges by tenure bins and calculate summary stats
monthly_charges_summary_6_mnonths = cleaned_stage2_dataset.groupby('tenure_bins')['MonthlyCharges'].agg(['mean', 'median', 'min', 'max']).reset_index()
monthly_charges_summary_6_mnonths.columns = ['tenure_bins', 'mean_monthly_charges', 'median_monthly_charges', 'min_monthly_charges', 'max_monthly_charges']

# Display the summary statistics for MonthlyCharges per tenure bin
print(monthly_charges_summary_6_mnonths)

# Create an interactive bar chart for mean MonthlyCharges by tenure bins
fig_mean = px.bar(
    monthly_charges_summary_6_mnonths,
    x="tenure_bins",
    y="mean_monthly_charges",
    title="Mean Monthly Charges by Tenure Bins",
    labels={'tenure_bins': 'Tenure (Binned)', 'mean_monthly_charges': 'Mean Monthly Charges'},
    text="mean_monthly_charges"  # Show values on bars
)
fig_mean.update_layout(
    xaxis_title="Tenure (Binned)",
    yaxis_title="Mean Monthly Charges",
    xaxis=dict(tickangle=45)
)
fig_mean.show()



# Create 6-month tenure bins
bin_edges = range(0, int(cleaned_stage2_dataset['tenure'].max()) + 6, 6)
bin_labels = [f'{i}-{i+5}' for i in range(0, int(cleaned_stage2_dataset['tenure'].max()), 6)]

# Assign tenure bins to the dataset
cleaned_stage2_dataset['tenure_bins_6_months'] = pd.cut(
    cleaned_stage2_dataset['tenure'], bins=bin_edges, right=False, labels=bin_labels
)

# Calculate churn percentage for 6-month bins
churn_analysis_6_months = cleaned_stage2_dataset.groupby('tenure_bins_6_months', observed=False).agg(
    total_customers=('Churn', 'count'),
    churned_customers=('Churn', 'sum')
).reset_index()

# Add churn percentage column
churn_analysis_6_months['churn_percentage'] = (
    churn_analysis_6_months['churned_customers'] / churn_analysis_6_months['total_customers']
) * 100

# Display churn analysis for 6-month bins
print("Churn analysis for 6-month bins:")
print(churn_analysis_6_months)

# Visualize churn percentage for 6-month bins
fig_churn_6_months = px.bar(
    churn_analysis_6_months,
    x="tenure_bins_6_months",
    y="churn_percentage",
    title="Churn Percentage Across 6-Month Tenure Bins",
    labels={"tenure_bins_6_months": "Tenure Bins (6-Months)", "churn_percentage": "Churn Percentage (%)"},
    text="churn_percentage"  # Show churn percentage on bars
)
fig_churn_6_months.update_layout(xaxis=dict(tickangle=45))
fig_churn_6_months.show()



# Calculate churn percentage for each tenure bin
churn_analysis_6_months = cleaned_stage2_dataset.groupby('tenure_bins').agg(
    total_customers=('Churn', 'count'),
    churned_customers=('Churn', 'sum')
).reset_index()

# Add churn percentage column
churn_analysis_6_months['churn_percentage'] = (churn_analysis_6_months['churned_customers'] / churn_analysis_6_months['total_customers']) * 100

# Display churn analysis
print(churn_analysis_6_months)

# Visualize churn percentage
import plotly.express as px
fig = px.bar(
    churn_analysis_6_months,
    x="tenure_bins",
    y="churn_percentage",
    title="Churn Percentage Across Tenure Bins",
    labels={"tenure_bins": "Tenure Bins", "churn_percentage": "Churn Percentage (%)"},
    text="churn_percentage"  # Show churn percentage on bars
)
fig.update_layout(xaxis=dict(tickangle=45))
fig.show()


churn_rate_list_for_6_months=list(churn_analysis_6_months['churn_percentage'])
monthly_charges_summary_6_mnonths['customer_count']=customer_count_6_month_interval
monthly_charges_summary_6_mnonths['churn_percentage']=churn_rate_list_for_6_months

list_contract_6_months=list(cleaned_stage2_dataset['Contract'])
len(list_contract_6_months)

tenure_bin_counts_monthly_charges_summary=pd.merge(tenure_bin_counts,monthly_charges_summary_6_mnonths, on='tenure_bins')
tenure_bin_counts_monthly_charges_summary.drop(columns=['count'])


import matplotlib.pyplot as plt


# Plotting Churn vs Tenure
plt.figure(figsize=(5, 2))
plt.plot(tenure_bin_counts_monthly_charges_summary['tenure_bins'], tenure_bin_counts_monthly_charges_summary['churn_percentage'], marker='o', linestyle='-', color='b')
plt.title('Churn Percentage vs Tenure')
plt.xlabel('Tenure Bins')
plt.ylabel('Churn Percentage')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# Plotting Churn vs Monthly Charges
plt.figure(figsize=(5, 2))
plt.plot(tenure_bin_counts_monthly_charges_summary['mean_monthly_charges'], tenure_bin_counts_monthly_charges_summary['churn_percentage'], marker='o', linestyle='-', color='r')
plt.title('Churn Percentage vs Monthly Charges')
plt.xlabel('Mean Monthly Charges')
plt.ylabel('Churn Percentage')
plt.grid(True)
plt.tight_layout()
plt.show()


# Assuming we now have the 'tenure_bin_counts_monthly_charges_summary' dataframe, we will need to add a 'contract_type' column
# to simulate the relationship between contract type and churn. Since no contract data is provided, I'll assume it exists.

# Let's proceed with the assumption that there is a "contract_type" and we have to calculate the churn for each contract type
import pandas as pd
import matplotlib.pyplot as plt

# Assuming the dataframe looks like this with the 'contract_type' data
# Here we assume 'contract_type' as an additional feature (e.g., 'Month-to-Month', 'One-Year', 'Two-Year')
# Since the contract type data is not explicitly available, let's simulate it for demonstration purposes.

# Add 'contract_type' (Simulating the data for demonstration)
tenure_bin_counts_monthly_charges_summary['contract_type'] = ['Month-to-Month', 'One-Year', 'Two-Year', 'Month-to-Month', 'One-Year', 
                                                             'Two-Year', 'Month-to-Month', 'One-Year', 'Two-Year', 'One-Year', 
                                                             'Two-Year', 'Month-to-Month']

# Calculate churn for each contract type
churn_by_contract = tenure_bin_counts_monthly_charges_summary.groupby('contract_type')['churn_percentage'].mean()

# Plotting Churn vs Contract Type
plt.figure(figsize=(10, 6))
churn_by_contract.plot(kind='bar', color='g')
plt.title('Churn Percentage vs Contract Type')
plt.xlabel('Contract Type')
plt.ylabel('Churn Percentage')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


import pandas as pd
import matplotlib.pyplot as plt

# Sample data structure for contract type and churn (assuming you have a 'contract_type' and 'churn' column)
# Replace this with your actual data (if you already have the 'contract_type' column in your data)


# Calculate churn percentage for each contract type
churn_percentage_by_contract = df.groupby('contract_type')['churn'].mean() * 100

# Plotting Churn vs Contract Type
plt.figure(figsize=(8, 6))
churn_percentage_by_contract.plot(kind='bar', color='b')
plt.title('Churn Percentage vs Contract Type')
plt.xlabel('Contract Type')
plt.ylabel('Churn Percentage')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


tenure_bin_counts_monthly_charges_summary.to_csv("tenure_bin_counts_monthly_charges_summary.csv", index=False)

####################### correllation ###################### 

# Correlation between MonthlyCharges and Tenure for 6 months dataset
correlation = cleaned_stage2_dataset[['MonthlyCharges', 'tenure']].corr()
print("Correlation between MonthlyCharges and Tenure:")
print(correlation)

# Line Chart: Monthly Charges vs Tenure Bins
import plotly.express as px
fig = px.line(
    monthly_charges_summary_6_mnonths, 
    x='tenure_bins', 
    y='mean_monthly_charges',
    title="Mean Monthly Charges vs Tenure Bins (6 months)",
    labels={'tenure_bins': 'Tenure Bins', 'mean_monthly_charges': 'Mean Monthly Charges'},
    markers=True
)
fig.show()

fig = px.line(
    monthly_charges_summary_3_months, 
    x='tenure_bins_3_months', 
    y='mean_monthly_charges',
    title="Mean Monthly Charges vs Tenure Bins (6 months)",
    labels={'tenure_bins': 'Tenure Bins', 'mean_monthly_charges': 'Mean Monthly Charges'},
    markers=True
)
fig.show()

# Check for customers with 0 tenure but non-zero TotalCharges
anomalies = cleaned_stage2_dataset[(dataset_cleaned['tenure'] == 0) & (dataset_cleaned['TotalCharges'] > 0)]
print(f"Anomalies with 0 tenure but non-zero TotalCharges:\n{anomalies}")

monthly_charges_summary_6_mnonths.to_csv("explored_data_6_months_interval_stage_2_EDA.csv", index=False)
monthly_charges_summary_3_months.to_csv("explored_data_3_months_interval_stage_2_EDA.csv",index=False)


import plotly.express as px

# Graph 1: Churn Rate vs. Time Active
fig_churn = px.bar(
    monthly_charges_summary_3_months,
    x="tenure_bins_3_months",
    y="churn_percentage",
    title="Churn Rate vs. Time Active",
    labels={"tenure_bins_3_months": "Time Active (Months)", "churn_rate": "Churn Rate (%)"},
    text="churn_percentage"
)
fig_churn.update_layout(xaxis=dict(tickangle=45), yaxis_title="Churn Rate (%)")
fig_churn.show()

# Graph 2: Monthly Charges vs. Time Active
fig_charges = px.bar(
    monthly_charges_summary_3_months,
    x="tenure_bins_3_months",
    y="mean_monthly_charges",
    title="Monthly Charges vs. Time Active",
    labels={"tenure_bins_3_months": "Time Active (Months)", "mean_monthly_charges": "Mean Monthly Charges"},
    text="mean_monthly_charges"
)
fig_charges.update_layout(xaxis=dict(tickangle=45), yaxis_title="Mean Monthly Charges")
fig_charges.show()

import plotly.express as px
import plotly.graph_objects as go

# Heatmap: Churn Rate vs. Time Active
fig_heatmap = px.imshow(
    monthly_charges_summary_3_months[['churn_percentage']].T,  # Transpose to get the right format for a heatmap
    labels={
        'x': "Time Active (Months)",
        'y': "Churn Rate",
        'color': "Churn Rate (%)"
    },
    x=monthly_charges_summary_3_months["tenure_bins_3_months"],
    title="Heatmap of Churn Rate vs. Time Active",
    color_continuous_scale="Reds"
)
fig_heatmap.update_layout(xaxis=dict(tickangle=45))
fig_heatmap.show()

# Line Graph: Monthly Charges vs. Time Active
fig_line = go.Figure()

fig_line.add_trace(go.Scatter(
    x=monthly_charges_summary_3_months["tenure_bins_3_months"],
    y=monthly_charges_summary_3_months["mean_monthly_charges"],
    mode='lines+markers',
    name="Mean Monthly Charges",
    line=dict(color='blue', width=2),
    marker=dict(size=8)
))

fig_line.update_layout(
    title="Mean Monthly Charges vs. Time Active",
    xaxis_title="Time Active (Months)",
    yaxis_title="Mean Monthly Charges",
    xaxis=dict(tickangle=45)
)
fig_line.show()



################################## for 6 months #########################################


import plotly.express as px

# Graph 1: Churn Rate vs. Time Active
fig_churn = px.bar(
    monthly_charges_summary_6_mnonths,
    x="tenure_bins",
    y="churn_percentage",
    title="Churn Rate vs. Time Active",
    labels={"tenure_bins_3_months": "Time Active (Months)", "churn_rate": "Churn Rate (%)"},
    text="churn_percentage"
)
fig_churn.update_layout(xaxis=dict(tickangle=45), yaxis_title="Churn Rate (%)")
fig_churn.show()

# Graph 2: Monthly Charges vs. Time Active
fig_charges = px.bar(
    monthly_charges_summary_6_mnonths,
    x="tenure_bins",
    y="mean_monthly_charges",
    title="Monthly Charges vs. Time Active",
    labels={"tenure_bins_3_months": "Time Active (Months)", "mean_monthly_charges": "Mean Monthly Charges"},
    text="mean_monthly_charges"
)
fig_charges.update_layout(xaxis=dict(tickangle=45), yaxis_title="Mean Monthly Charges")
fig_charges.show()


import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Example dataset for 6-month bins
data_6_months = {
    "tenure_bins_6_months": [
        "0-6", "7-12", "13-18", "19-24", "25-30", "31-36", "37-42",
        "43-48", "49-54", "55-60", "61-66", "67-72"
    ],
    "mean_monthly_charges": [
        54.73, 58.95, 61.23, 61.49, 63.92, 67.34, 65.61,
        67.01, 70.03, 71.07, 75.33, 76.25
    ],
    "churn_rate": [
        52.93, 35.88, 32.29, 24.57, 21.80, 21.44, 21.89,
        16.18, 16.19, 12.62, 9.28, 5.29
    ],
    "customer_count": [
        1481, 705, 548, 476, 431, 401, 379, 383, 420, 412, 463, 944
    ]
}
df_6_months = pd.DataFrame(data_6_months)

# Heatmap: Churn Rate vs. Time Active (6-Month Bins)
fig_heatmap_6 = px.imshow(
    df_6_months[['churn_rate']].T,  # Transpose for heatmap structure
    labels={
        'x': "Time Active (Months)",
        'y': "Churn Rate",
        'color': "Churn Rate (%)"
    },
    x=df_6_months["tenure_bins_6_months"],
    title="Heatmap of Churn Rate vs. Time Active (6-Month Bins)",
    color_continuous_scale="Reds"
)
fig_heatmap_6.update_layout(xaxis=dict(tickangle=45))
fig_heatmap_6.show()

# Line Graph: Monthly Charges vs. Time Active (6-Month Bins)
fig_line_6 = go.Figure()

fig_line_6.add_trace(go.Scatter(
    x=df_6_months["tenure_bins_6_months"],
    y=df_6_months["mean_monthly_charges"],
    mode='lines+markers',
    name="Mean Monthly Charges",
    line=dict(color='blue', width=2),
    marker=dict(size=8)
))

fig_line_6.update_layout(
    title="Mean Monthly Charges vs. Time Active (6-Month Bins)",
    xaxis_title="Time Active (6-Month Bins)",
    yaxis_title="Mean Monthly Charges",
    xaxis=dict(tickangle=45)
)
fig_line_6.show()


##############################################################################################################################################################################3

import pandas as pd
import plotly.express as px

# Define function to calculate churn rate by a factor
def churn_rate_by_factor(dataset, factor):
    analysis = dataset.groupby(factor, observed=False).agg(
        total_customers=('Churn', 'count'),
        churned_customers=('Churn', 'sum')
    ).reset_index()
    analysis['churn_rate'] = (analysis['churned_customers'] / analysis['total_customers']) * 100
    return analysis

# Factors to analyze
factors = ['gender', 'SeniorCitizen', 'Contract', 'PaymentMethod', 'InternetService',
           'OnlineSecurity', 'TechSupport', 'StreamingTV', 'StreamingMovies']

# Loop through factors and calculate churn rate
for factor in factors:
    analysis = churn_rate_by_factor(cleaned_stage2_dataset, factor)
    print(f"Churn Analysis for {factor}:")
    print(analysis)
    
    # Visualize churn rate
    fig = px.bar(
        analysis,
        x=factor,
        y="churn_rate",
        title=f"Churn Rate by {factor}",
        labels={factor: factor, "churn_rate": "Churn Rate (%)"},
        text="churn_rate"
    )
    fig.update_layout(xaxis=dict(tickangle=45), yaxis_title="Churn Rate (%)")
    fig.show()


################## going to use this function when optimizing the readability of the code #############
''' this function takes the cleaned dataset and a sql filter condition and finds the churned percentage of every parameter in categories list for a specific demographic. for example, only senior citizens, 
senior citizens who are men etc. the results for this will be analyized in the next function analyze_dataset'''
def analyze_category(data,filter_condition=None):
    """
    Perform analysis for specified categories on a given dataset.
    
    Parameters:
    - data: DataFrame, the dataset to analyze.
    - filter_condition: Function or None, a condition to filter data (e.g., lambda df: df[df['SeniorCitizen'] == 1]).
    
    Returns:
    - DataFrame containing analysis for each category.
    """
    categories = [
    'PaymentMethod', 'Contract', 'InternetService','OnlineSecurity', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    if filter_condition:
        data = filter_condition(data)

    detailed_analysis = []

    for category in categories:
        analysis = data.groupby(category).agg(
            total_customers=('Churn', 'count'),
            churned_customers=('Churn', 'sum')
        ).reset_index()
        
        # Add churn rate column
        analysis['churn_rate'] = (analysis['churned_customers'] / analysis['total_customers']) * 100
        
        # Add a column to identify the category being analyzed
        analysis['category'] = category
        analysis.rename(columns={category: 'Value', 'total_customers': 'Count', 'churn_rate': 'Churn Rate (%)'}, inplace=True)
        detailed_analysis.append(analysis)

    return pd.concat(detailed_analysis, ignore_index=True)

senior_citizen_male_analysis = analyze_category(cleaned_stage2_dataset, filter_condition=lambda df: df[(df['SeniorCitizen'] == 1) & (df['gender'] == 'Male')])

senior_citizen_male_analysis=senior_citizen_male_analysis[['category','Value','Count', 'churned_customers', 'Churn Rate (%)']]
senior_citizen_male_analysis.to_csv("senior_citizen_male_analysis.csv",index=False)

senior_citizens_whole=analyze_category(cleaned_stage2_dataset, filter_condition=lambda df: df[(df['SeniorCitizen'] == 1)])
senior_citizens_whole=senior_citizens_whole[['category','Value','Count', 'churned_customers', 'Churn Rate (%)']]
senior_citizens_whole.to_csv("senior_citizens_whole.csv",index=False)

senior_citizen_female_analysis = analyze_category(cleaned_stage2_dataset, filter_condition=lambda df: df[(df['SeniorCitizen'] == 1) & (df['gender'] == 'Female')])

senior_citizen_female_analysis=senior_citizen_female_analysis[['category','Value','Count', 'churned_customers', 'Churn Rate (%)']]
senior_citizen_female_analysis.to_csv("senior_citizen_female_analysis.csv",index=False)

######################### now for non senior citizens ######################################

non_senior_citizens_whole=analyze_category(cleaned_stage2_dataset, filter_condition=lambda df: df[(df['SeniorCitizen'] == 0)])

non_senior_citizens_whole=non_senior_citizens_whole[['category','Value','Count', 'churned_customers', 'Churn Rate (%)']]
non_senior_citizens_whole.to_csv("non_senior_citizens_whole.csv",index=False)

non_senior_citizen_male_analysis = analyze_category(cleaned_stage2_dataset, filter_condition=lambda df: df[(df['SeniorCitizen'] == 0) & (df['gender'] == 'Male')])
non_senior_citizen_male_analysis=non_senior_citizen_male_analysis[['category','Value','Count', 'churned_customers', 'Churn Rate (%)']]
non_senior_citizen_male_analysis.to_csv("non_senior_citizen_male_analysis.csv",index=False)

non_senior_citizen_female_analysis = analyze_category(cleaned_stage2_dataset, filter_condition=lambda df: df[(df['SeniorCitizen'] == 0) & (df['gender'] == 'Female')])
non_senior_citizen_female_analysis=non_senior_citizen_female_analysis[['category','Value','Count', 'churned_customers', 'Churn Rate (%)']]
non_senior_citizen_female_analysis.to_csv("non_senior_citizen_female_analysis.csv",index=False)

import pandas as pd
import plotly.express as px

def analyze_dataset(dataset, title_prefix):
    # Ensure column names are cleaned
    dataset.columns = dataset.columns.str.strip()

    # Bar Chart for Customer Count Across Categories
    category_counts = dataset.groupby('category')['Count'].sum().reset_index()
    fig_bar_count = px.bar(
        category_counts,
        x="category",
        y="Count",
        title=f"{title_prefix} - Customer Count Across Categories",
        text="Count"
    )
    fig_bar_count.update_layout(xaxis_title="Category", yaxis_title="Customer Count", xaxis=dict(tickangle=45))
    fig_bar_count.show()

    # Bar Chart for Churn Rate Across Categories
    category_churn = dataset.groupby('category')['Churn Rate (%)'].mean().reset_index()
    fig_bar_churn = px.bar(
        category_churn,
        x="category",
        y="Churn Rate (%)",
        title=f"{title_prefix} - Churn Rate Across Categories",
        text="Churn Rate (%)"
    )
    fig_bar_churn.update_layout(xaxis_title="Category", yaxis_title="Churn Rate (%)", xaxis=dict(tickangle=45))
    fig_bar_churn.show()

    # Line Chart for Customer Count Across Values in Categories
    for category in dataset['category'].unique():
        subset = dataset[dataset['category'] == category]
        fig_line = px.line(
            subset,
            x="Value",
            y="Count",
            title=f"{title_prefix} - Customer Count for {category}",
            text="Count",
            markers=True
        )
        fig_line.update_layout(xaxis_title=f"{category} Options", yaxis_title="Customer Count", xaxis=dict(tickangle=45))
        fig_line.show()

    # Line Chart for Churn Rate Across Values in Categories
    for category in dataset['category'].unique():
        subset = dataset[dataset['category'] == category]
        fig_line_churn = px.line(
            subset,
            x="Value",
            y="Churn Rate (%)",
            title=f"{title_prefix} - Churn Rate for {category}",
            text="Churn Rate (%)",
            markers=True
        )
        fig_line_churn.update_layout(xaxis_title=f"{category} Options", yaxis_title="Churn Rate (%)", xaxis=dict(tickangle=45))
        fig_line_churn.show()

# Example Usage
senior_citizen = pd.read_csv("senior_citizens_whole.csv")
senior_citizen_male = pd.read_csv("senior_citizen_male_analysis.csv")
senior_citizen_female = pd.read_csv("senior_citizen_female_analysis.csv")
non_senior_citizen = pd.read_csv("non_senior_citizens_whole.csv")
non_senior_citizen_male = pd.read_csv("non_senior_citizen_male_analysis.csv")
non_senior_citizen_female = pd.read_csv("non_senior_citizen_female_analysis.csv")

dataframes_dict = {
    "Senior Citizens": senior_citizen,
    "Senior Citizen Males": senior_citizen_male,
    "Senior Citizen Females": senior_citizen_female,
    "Non-Senior Citizens": non_senior_citizen,
    "Non-Senior Citizen Males": non_senior_citizen_male,
    "Non-Senior Citizen Females": non_senior_citizen_female
}

# Iterate through DataFrames and analyze
count=0
for name, dataframe in dataframes_dict.items():
    count+=1
    dataframe.columns = dataframe.columns.str.strip()  # Clean column names
    analyze_dataset(dataframe, name)
    print("****************************************************************************************************************************************")
    print(f"finished {count}")


###############################################################3 Modelling ########################################

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load Dataset
dataset_modelling = pd.read_csv("cleaned_stage2.csv")
pd.set_option('display.max_columns', None)

# Define Features (X) and Target (y)
X = dataset_modelling.drop(["Churn", "customerID"], axis=1, errors="ignore")  # Drop target and ID columns
y = dataset_modelling["Churn"]  # Target column

# Identify Categorical and Numeric Columns
categorical_columns = X.select_dtypes(include=["object", "category"]).columns
numeric_columns = X.select_dtypes(include=["int64", "float64"]).columns

# Encode Categorical Columns
X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)

# Standardize Numeric Columns
scaler = StandardScaler()
X[numeric_columns] = scaler.fit_transform(X[numeric_columns])

# Verify Shape and Columns
print(f"Shape of Features: {X.shape}")
print(f"Columns in Features:\n{X.columns.tolist()}")


# trainig and testing dataset.. dividing in 80-20. Random state ensures we get same results by running again and again, 365 is just some arbitary number
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=365)
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


############################# logistic regression modelling starts ########################

from sklearn.linear_model import LogisticRegression

# Initialize the model
logistic_model = LogisticRegression(random_state=365, max_iter=2000,solver='liblinear')

# Train the model on the training data
logistic_model.fit(X_train, y_train)

'''
True Negatives (TN): These are the customers who did not churn and the model correctly predicted them as non-churners.

False Positives (FP):These are the customers who did not churn but the model incorrectly predicted them as churners.

False Negatives (FN): These are the customers who did churn but the model incorrectly predicted them as non-churners.

True Positives (TP):These are the customers who did churn and the model correctly predicted them as churners.

precision vs accuracy 

Accuracy=  TP+TN\TP+TN+FP+FN
           
​Precision= TP/TP+FP

​Recall= TP\TP+FN

​
F1=2× (Precision×Recall\Precision+Recall)
​
'''


''' finding the testing and training accuracy on both datasets. note that the custom threshold is by default 0.50'''
# step 1- testding and training on default threshold on both training and testing data 

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Predictions
y_pred_train = logistic_model.predict(X_train)
y_pred_test = logistic_model.predict(X_test)

# Accuracy
print("Training Accuracy:", accuracy_score(y_train, y_pred_train))
print("Testing Accuracy:", accuracy_score(y_test, y_pred_test))

# Classification report
print("\nClassification Report (Test Data):")
classification_report(y_test, y_pred_test)

# Confusion matrix
confusion_matrix(y_test, y_pred_test)

# step 2: corss-validation i put cv=5 to test the data 5 times and then calculated the mean
from sklearn.model_selection import cross_val_score
scores = cross_val_score(logistic_model, X_train, y_train, cv=5, scoring='accuracy')
print(f"Cross-Validation Scores: {scores}")
print(f"Mean CV Accuracy: {scores.mean()}")



# step 3  finding features contributing to churn 
feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Coefficient': logistic_model.coef_[0]
}).sort_values(by='Coefficient', ascending=False)
print(feature_importance)

# applying hyperparameters. choosing C from small values to large to check which value would be best fit for my data. using both lasso and ridge regularizations
#L1 (Lasso) or L2 (Ridge). Ridge regularization (L2) tends to perform better when the data has many features, and Lasso (L1) can be useful for feature selection by setting some feature weights to zero.
from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.01, 0.1, 1, 10, 100], 'penalty': ['l1', 'l2']}
grid_search = GridSearchCV(logistic_model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best CV Accuracy: {grid_search.best_score_}")
# the result means that ridge regularization is best fit for model and regulaization value is 100 as it would lead to the best fit for my data.


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

y_prob = logistic_model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='blue', label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.show()
### since value of ROC curve is 0.84, Logistic regression is a good model for this dataset since the blue line is upward of grey dashed line, it means that this model does a good 
### job in distinguishing churners from non-churners

''' we try to find the best threshold that can predict the most accuracte values'''

from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt

# Predicted probabilities for the positive class (Churn = 1)
y_prob_logistic = logistic_model.predict_proba(X_test)[:, 1]

# Compute Precision-Recall curve and AUC
precision, recall, thresholds = precision_recall_curve(y_test, y_prob_logistic)
pr_auc = auc(recall, precision)

# Compute F1 scores for all thresholds and find the best threshold
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
best_idx = f1_scores.argmax()
best_threshold = thresholds[best_idx]

# Plot Precision-Recall Curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color="blue", label=f"PR Curve (AUC = {pr_auc:.2f})")
plt.scatter(recall[best_idx], precision[best_idx], color="red", label=f"Best Threshold = {best_threshold:.2f}", zorder=5)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve - Logistic Regression")
plt.legend(loc="best")
plt.grid()
plt.show()

# Save PR AUC and best threshold for reference
pr_auc_logistic = pr_auc
best_threshold_logistic = best_threshold

print(f"Best Threshold: {best_threshold_logistic}")
print(f"Precision-Recall AUC: {pr_auc_logistic}")
### value of 0.28 means hat when the predicted probability of churn is greater than 28.3%, the model classifies the instance as churned. The PR Curve AUC 0.66 suggests it is better than
### rqandom guessing but can be improved which i am going to do in the next stage


class_report = classification_report(y_test, y_pred_test, output_dict=True)
print(class_report)

# Recompute predicted probabilities for the training set with the default (untuned) logistic regression model
y_prob_train_default = logistic_model.predict_proba(X_train)[:, 1]

# Compute ROC curve and AUC for the training set (default model)
fpr_train_default, tpr_train_default, thresholds_train_default = roc_curve(y_train, y_prob_train_default)
roc_auc_train_default = auc(fpr_train_default, tpr_train_default)

print(roc_auc_train_default)

logistic_model = LogisticRegression(random_state=365, max_iter=2000, solver='liblinear')

# Train the model
logistic_model.fit(X_train, y_train)

# Predict probabilities
y_prob_train = logistic_model.predict_proba(X_train)[:, 1]
y_prob_test = logistic_model.predict_proba(X_test)[:, 1]

# Apply custom threshold
threshold = best_threshold  # Replace with 0.28 or your calculated value
y_pred_train_custom = (y_prob_train >= threshold).astype(int)
y_pred_test_custom = (y_prob_test >= threshold).astype(int)

# Evaluate performance
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Accuracy
print("Training Accuracy with Custom Threshold:", accuracy_score(y_train, y_pred_train_custom))
print("Testing Accuracy with Custom Threshold:", accuracy_score(y_test, y_pred_test_custom))

# Classification report
print("\nClassification Report (Test Data with Custom Threshold):")
print(classification_report(y_test, y_pred_test_custom))

# Confusion matrix
print("\nConfusion Matrix (Test Data with Custom Threshold):")
print(confusion_matrix(y_test, y_pred_test_custom))

# Cross-validation (still uses default threshold for scoring)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(logistic_model, X_train, y_train, cv=5, scoring='accuracy')
print(f"Cross-Validation Scores: {scores}")
print(f"Mean CV Accuracy: {scores.mean()}")

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Coefficient': logistic_model.coef_[0]
}).sort_values(by='Coefficient', ascending=False)
print("\nFeature Importance:")
print(feature_importance)

# Hyperparameter tuning (still uses default threshold)
from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.01, 0.1, 1, 10, 100], 'penalty': ['l1', 'l2']}
grid_search = GridSearchCV(logistic_model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print(f"\nBest Parameters: {grid_search.best_params_}")
print(f"Best CV Accuracy: {grid_search.best_score_}")
class_report_tuned_logistic = classification_report(y_test,y_pred_test_custom, output_dict=True)

fpr_tuned, tpr_tuned, thresholds_tuned = roc_curve(y_test, y_prob_test)
roc_auc_tuned = auc(fpr_tuned, tpr_tuned)

print(f"AUC (Tuned on Test Set): {roc_auc_tuned}")


############################## End of logistic regression model##################################################

############################# start of decsion tree classifier ##################################################

from sklearn.tree import DecisionTreeClassifier

decision_tree_model = DecisionTreeClassifier(random_state=365)
decision_tree_model.fit(X_train, y_train)


# step 1 : checking the accuracy , f1 score,confusion matrix  
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Predictions
y_pred_train_dt = decision_tree_model.predict(X_train)
y_pred_test_dt = decision_tree_model.predict(X_test)

# Accuracy
train_accuracy_dt = accuracy_score(y_train, y_pred_train_dt)
test_accuracy_dt = accuracy_score(y_test, y_pred_test_dt)

print(f"Training Accuracy: {train_accuracy_dt}")
print(f"Testing Accuracy: {test_accuracy_dt}")

# Classification Report
print("Classification Report (Test Data):")
print(classification_report(y_test, y_pred_test_dt))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred_test_dt)
print("Confusion Matrix:")
print(conf_matrix)




# analysis: the output seems to be overfitting due to considerable difference between testing and traing dataset accuracies. so I am going to check for solution for this


# pruning the decision tree classifer model to try to have a better accuracy. the code will travel to 5 levels of the tree, run till there are atleast 10 samples left on the node, 5 on the leaf, with random state being same for entire dataset models

from sklearn.tree import DecisionTreeClassifier
pruned_tree = DecisionTreeClassifier(max_depth=5, min_samples_split=10, min_samples_leaf=5, random_state=365)
pruned_tree.fit(X_train, y_train)
# Predictions
y_pred_train_pruned = pruned_tree.predict(X_train)
y_pred_test_pruned = pruned_tree.predict(X_test)
print(f"Training Accuracy (Pruned Tree): {accuracy_score(y_train, y_pred_train_pruned)}")
print(f"Testing Accuracy (Pruned Tree): {accuracy_score(y_test, y_pred_test_pruned)}")
print("\nClassification Report (Pruned Tree):")
print(classification_report(y_test, y_pred_test_pruned))
print("\nConfusion Matrix (Pruned Tree):")
print(confusion_matrix(y_test, y_pred_test_pruned))


### checking the features which contribute the most for churning
import pandas as pd
import matplotlib.pyplot as plt

# Extract feature importance from the model
feature_importances = pruned_tree.feature_importances_

# Create a DataFrame for better readability
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

# Display the top 10 features
print("Top 10 Important Features:")
print(feature_importance_df.head(10))

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Feature Importance for Pruned Decision Tree')
plt.gca().invert_yaxis()  # Flip the chart for better readability
plt.show()


### doing partiial dependence plotting for features that cause the most for customers to churn from the company. Shap and lime will be done later.

from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt

# Features to visualize (as indices or column names)
features_to_plot = [3, 16, 14, 19]  # Indices for tenure, InternetService_Fiber optic, TotalCharges, Contract_Two year

# Create the Partial Dependence Plots
fig, ax = plt.subplots(figsize=(14, 10))
PartialDependenceDisplay.from_estimator(pruned_tree, X_train, features_to_plot, ax=ax)
plt.suptitle("Partial Dependence Plots for Top Features", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


# performing cross validation 5 times on the pruned tree
from sklearn.model_selection import cross_val_score

# Perform 5-fold cross-validation
cv_scores = cross_val_score(pruned_tree, X_train, y_train, cv=5, scoring='accuracy')

# Print Cross-Validation Scores
print(f"Cross-Validation Scores: {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean()}")


from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

# Define the parameter grid
param_grid = {
    'max_depth': [3, 5, 10, 15, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 5, 10]
}

# Initialize the model
decision_tree = DecisionTreeClassifier(random_state=365)

# Set up GridSearchCV
grid_search = GridSearchCV(
    estimator=decision_tree,
    param_grid=param_grid,
    scoring='accuracy',
    cv=5,
    n_jobs=-1
)

# Fit the model
grid_search.fit(X_train, y_train)

# Get the best parameters and best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print(f"Best Parameters: {best_params}")
print(f"Best CV Accuracy: {best_score}")

# Evaluate the model with the best parameters on the test set
best_tree = grid_search.best_estimator_
y_pred_test_best_tree = best_tree.predict(X_test)

print("\nClassification Report (Best Decision Tree):")
print(classification_report(y_test, y_pred_test_best_tree))
print("\nConfusion Matrix (Best Decision Tree):")
print(confusion_matrix(y_test, y_pred_test_best_tree))


### we found that the best results come when the tree depth is 3, with min sample size of 1 on the leaf with min samples left to be split being 2

### graphical representaion of features which contribute the most. 
import pandas as pd
import matplotlib.pyplot as plt

# Feature importance data from the tuned decision tree
feature_importances = {
    "Feature": [
        "tenure", "InternetService_Fiber optic", "TotalCharges", 
        "InternetService_No", "Contract_Two year", 
        "PaymentMethod_Electronic check", "Contract_One year", 
        "MonthlyCharges", "OnlineSecurity", "StreamingTV"
    ],
    "Importance": [0.458658, 0.344321, 0.049238, 0.040273, 0.024828, 
                   0.017986, 0.013288, 0.012366, 0.009501, 0.007009]
}

# Convert to DataFrame
feature_importances_df = pd.DataFrame(feature_importances)

# Plot
plt.figure(figsize=(10, 6))
plt.barh(feature_importances_df["Feature"], feature_importances_df["Importance"], color='skyblue')
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 10 Important Features - Tuned Decision Tree")
plt.gca().invert_yaxis()  # Invert y-axis to have the highest importance on top
plt.tight_layout()
plt.show()





# finding the best custom threshold on which our model works the best. checking the new improved accuracy and the confusion matrix
from sklearn.metrics import precision_recall_curve, classification_report, confusion_matrix, accuracy_score

# Obtain probabilities for the Decision Tree Classifier
y_prob_dt = pruned_tree.predict_proba(X_test)[:, 1]

# Find the best threshold using precision-recall curve
precision_dt, recall_dt, thresholds_dt = precision_recall_curve(y_test, y_prob_dt)
f1_scores_dt = 2 * (precision_dt * recall_dt) / (precision_dt + recall_dt)
best_idx_dt = f1_scores_dt.argmax()
best_threshold_dt = thresholds_dt[best_idx_dt]

# Apply the best threshold to make predictions
y_pred_custom_dt = (y_prob_dt >= best_threshold_dt).astype(int)

# Evaluate the results with the custom threshold
training_accuracy_dt_custom = accuracy_score(y_train, pruned_tree.predict(X_train))
testing_accuracy_dt_custom = accuracy_score(y_test, y_pred_custom_dt)
classification_report_dt_custom = classification_report(y_test, y_pred_custom_dt)
confusion_matrix_dt_custom = confusion_matrix(y_test, y_pred_custom_dt)

# Save the results
custom_dt_results = {
    "Best Threshold": best_threshold_dt,
    "Training Accuracy": training_accuracy_dt_custom,
    "Testing Accuracy": testing_accuracy_dt_custom,
    "Classification Report": classification_report_dt_custom,
    "Confusion Matrix": confusion_matrix_dt_custom,
}

print(custom_dt_results)

import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc

# Default Precision-Recall Curve
precision_default, recall_default, thresholds_default = precision_recall_curve(y_test, y_prob_dt)
auc_default = auc(recall_default, precision_default)

# Custom Threshold Precision-Recall Curve
f1_scores_custom = 2 * (precision_default * recall_default) / (precision_default + recall_default)
best_idx_custom = f1_scores_custom.argmax()
best_threshold_custom = thresholds_default[best_idx_custom]

# Plotting Precision-Recall Curves
plt.figure(figsize=(10, 6))
plt.plot(recall_default, precision_default, label=f"Default PR Curve (AUC = {auc_default:.2f})", color="blue")
plt.scatter(
    recall_default[best_idx_custom], 
    precision_default[best_idx_custom], 
    color="red", 
    label=f"Best Threshold = {best_threshold_custom:.2f} (F1-Score: {f1_scores_custom[best_idx_custom]:.2f})"
)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve for Decision Tree")
plt.legend(loc="lower left")
plt.grid()
plt.tight_layout()
plt.show()

# Save the best threshold and F1-score
print(f"Best Threshold: {best_threshold_custom}")
print(f"Best F1-Score: {f1_scores_custom[best_idx_custom]:.2f}")



from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Predicted probabilities for the positive class (Churn = 1) from the pruned decision tree
y_prob_pruned_tree = pruned_tree.predict_proba(X_test)[:, 1]

# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_prob_pruned_tree)
roc_auc_pruned_tree = auc(fpr, tpr)

# Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC Curve (AUC = {roc_auc_pruned_tree:.2f})")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1, label="Random Classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Pruned Decision Tree")
plt.legend(loc="lower right")
plt.grid()
plt.show()


roc_auc_pruned_tree

##################################### Random Forest ###############################



from sklearn.ensemble import RandomForestClassifier

# Initialize the model with default parameters
rf_model = RandomForestClassifier(random_state=365)
rf_model.fit(X_train, y_train)


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Predictions
y_pred_train_rf = rf_model.predict(X_train)
y_pred_test_rf = rf_model.predict(X_test)

# Training and Testing Accuracy
print(f"Training Accuracy (RF): {accuracy_score(y_train, y_pred_train_rf)}")
print(f"Testing Accuracy (RF): {accuracy_score(y_test, y_pred_test_rf)}")

# Classification Report
print("\nClassification Report (RF):")
print(classification_report(y_test, y_pred_test_rf))

# Confusion Matrix
print("\nConfusion Matrix (RF):")
print(confusion_matrix(y_test, y_pred_test_rf))



import pandas as pd
import matplotlib.pyplot as plt

# Extract feature importances
feature_importances_rf = rf_model.feature_importances_

# Create a DataFrame for better visualization
feature_importance_df_rf = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': feature_importances_rf
}).sort_values(by='Importance', ascending=False)

# Display the top 10 features
print("Top 10 Important Features (Random Forest):")
print(feature_importance_df_rf.head(10))

# Plot Feature Importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df_rf['Feature'], feature_importance_df_rf['Importance'], color='skyblue')
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Feature Importance - Random Forest")
plt.gca().invert_yaxis()  # Flip the chart for better readability
plt.tight_layout()
plt.show()


from sklearn.model_selection import cross_val_score

# Perform 5-fold cross-validation
cv_scores_rf = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='accuracy')

# Print the cross-validation results
print(f"Cross-Validation Scores (RF): {cv_scores_rf}")
print(f"Mean CV Accuracy (RF): {cv_scores_rf.mean()}")

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Define the parameter grid
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Initialize the model
rf_model_tuning = RandomForestClassifier(random_state=365)

# Set up GridSearchCV
grid_search_rf = GridSearchCV(
    estimator=rf_model_tuning,
    param_grid=param_grid_rf,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2
)

# Fit the model
grid_search_rf.fit(X_train, y_train)

# Best parameters and accuracy
best_params_rf = grid_search_rf.best_params_
best_score_rf = grid_search_rf.best_score_

print(f"Best Parameters (RF): {best_params_rf}")
print(f"Best CV Accuracy (RF): {best_score_rf}")

# Evaluate the best model on the test set
best_rf_model = grid_search_rf.best_estimator_
y_pred_test_rf_best = best_rf_model.predict(X_test)

# Classification report and confusion matrix for the best model
print("\nClassification Report (Best RF Model):")
print(classification_report(y_test, y_pred_test_rf_best))

print("\nConfusion Matrix (Best RF Model):")
print(confusion_matrix(y_test, y_pred_test_rf_best))



import matplotlib.pyplot as plt
import pandas as pd

# Extract feature importance
feature_importance_rf = pd.DataFrame({
    "Feature": X.columns,
    "Importance": best_rf_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

# Plot Feature Importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_rf["Feature"], feature_importance_rf["Importance"], color='skyblue')
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top Features - Tuned Random Forest")
plt.gca().invert_yaxis()  # Highest importance on top
plt.tight_layout()
plt.show()


from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt

# Predicted probabilities for the positive class (Churn = 1)
y_prob_rf = best_rf_model.predict_proba(X_test)[:, 1]

# Compute Precision-Recall Curve
precision_rf, recall_rf, thresholds_rf = precision_recall_curve(y_test, y_prob_rf)
pr_auc_rf = auc(recall_rf, precision_rf)

# Calculate F1 Scores and Identify the Best Threshold
f1_scores_rf = 2 * (precision_rf * recall_rf) / (precision_rf + recall_rf)
best_idx_rf = f1_scores_rf.argmax()
best_threshold_rf = thresholds_rf[best_idx_rf]

# Plot Precision-Recall Curve
plt.figure(figsize=(10, 6))
plt.plot(recall_rf, precision_rf, label=f"PR Curve (AUC = {pr_auc_rf:.2f})", color="green")
plt.scatter(recall_rf[best_idx_rf], precision_rf[best_idx_rf], color='red', label=f"Best Threshold = {best_threshold_rf:.2f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve - Tuned Random Forest")
plt.legend(loc="best")
plt.grid()
plt.show()

# Save the best threshold for reporting
print(best_threshold_rf)


from sklearn.metrics import roc_auc_score

auc_roc_rf = roc_auc_score(y_test, y_prob_rf)
print(auc_roc_rf)
precision_rf, recall_rf, thresholds_rf = precision_recall_curve(y_test, y_prob_rf)
pr_auc_rf = auc(recall_rf, precision_rf)
print(pr_auc_rf)


f1_scores_rf = 2 * (precision_rf * recall_rf) / (precision_rf + recall_rf)
best_idx_rf = f1_scores_rf.argmax()
best_threshold_rf = thresholds_rf[best_idx_rf]
print(best_threshold_rf)


from sklearn.metrics import roc_curve, auc

# Compute ROC Curve
fpr_rf, tpr_rf, thresholds_roc_rf = roc_curve(y_test, y_prob_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)

# Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr_rf, tpr_rf, label=f"ROC Curve (AUC = {roc_auc_rf:.2f})", color='purple')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Tuned Random Forest")
plt.legend(loc="lower right")
plt.grid()
plt.show()


# Apply the best threshold to make predictions
y_pred_custom_rf = (y_prob_rf >= best_threshold_rf).astype(int)

# Evaluate the model with the custom threshold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Training Accuracy with Custom Threshold:", accuracy_score(y_train, best_rf_model.predict(X_train)))
print("Testing Accuracy with Custom Threshold:", accuracy_score(y_test, y_pred_custom_rf))

print("\nClassification Report (Test Data with Custom Threshold):")
print(classification_report(y_test, y_pred_custom_rf))

print("\nConfusion Matrix (Test Data with Custom Threshold):")
print(confusion_matrix(y_test, y_pred_custom_rf))

######################################################################################################################################################

##################################################################Gradient boosting####################################################################################


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Initialize the model
gb_model = GradientBoostingClassifier(random_state=365)

# Train the model
gb_model.fit(X_train, y_train)

# Predictions
y_pred_train_gb = gb_model.predict(X_train)
y_pred_test_gb = gb_model.predict(X_test)

# Training and Testing Accuracy
print(f"Training Accuracy (GB): {accuracy_score(y_train, y_pred_train_gb)}")
print(f"Testing Accuracy (GB): {accuracy_score(y_test, y_pred_test_gb)}")

# Classification Report
print("\nClassification Report (GB):")
print(classification_report(y_test, y_pred_test_gb))

# Confusion Matrix
print("\nConfusion Matrix (GB):")
print(confusion_matrix(y_test, y_pred_test_gb))



import pandas as pd
import matplotlib.pyplot as plt

# Extract feature importances
feature_importance_gb = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': gb_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

# Display Top 10 Features
print("Top 10 Important Features (Gradient Boosting):")
print(feature_importance_gb.head(10))

# Plot Feature Importances
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_gb['Feature'][:10], feature_importance_gb['Importance'][:10], color='teal')
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Top 10 Feature Importances - Gradient Boosting")
plt.gca().invert_yaxis()
plt.show()



from sklearn.model_selection import cross_val_score

# Perform 5-fold cross-validation
cv_scores_gb = cross_val_score(gb_model, X_train, y_train, cv=5, scoring='accuracy')

# Print Cross-Validation Scores and Mean Accuracy
print(f"Cross-Validation Scores (Gradient Boosting): {cv_scores_gb}")
print(f"Mean CV Accuracy (Gradient Boosting): {cv_scores_gb.mean()}")




from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier

# Define the parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5]
}

# Initialize the model
gb_model = GradientBoostingClassifier(random_state=365)

# Set up GridSearchCV
grid_search_gb = GridSearchCV(
    estimator=gb_model,
    param_grid=param_grid,
    scoring='accuracy',
    cv=5,
    n_jobs=-1,
    verbose=1
)

# Fit the model
grid_search_gb.fit(X_train, y_train)

# Output best parameters and CV accuracy
print(f"Best Parameters (Gradient Boosting): {grid_search_gb.best_params_}")
print(f"Best CV Accuracy (Gradient Boosting): {grid_search_gb.best_score_}")

# Evaluate the best model on the test set
best_gb_model = grid_search_gb.best_estimator_
y_pred_test_best_gb = best_gb_model.predict(X_test)

print("\nClassification Report (Best Gradient Boosting Model):")
print(classification_report(y_test, y_pred_test_best_gb))
print("\nConfusion Matrix (Best Gradient Boosting Model):")
print(confusion_matrix(y_test, y_pred_test_best_gb))






# Predictions using the best Gradient Boosting model
y_pred_train_gb_tuned = best_gb_model.predict(X_train)
y_pred_test_gb_tuned = best_gb_model.predict(X_test)

# Training and Testing Accuracy
train_accuracy_gb_tuned = accuracy_score(y_train, y_pred_train_gb_tuned)
test_accuracy_gb_tuned = accuracy_score(y_test, y_pred_test_gb_tuned)

print(f"Training Accuracy (Tuned GB): {train_accuracy_gb_tuned}")
print(f"Testing Accuracy (Tuned GB): {test_accuracy_gb_tuned}")

# Classification Report
print("\nClassification Report (Tuned GB):")
print(classification_report(y_test, y_pred_test_gb_tuned))

# Confusion Matrix
conf_matrix_gb_tuned = confusion_matrix(y_test, y_pred_test_gb_tuned)
print("\nConfusion Matrix (Tuned GB):")
print(conf_matrix_gb_tuned)

import pandas as pd
import matplotlib.pyplot as plt

# Feature importance data from the tuned Gradient Boosting model
tuned_gb_importances = best_gb_model.feature_importances_

# Create a DataFrame for better readability
tuned_feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': tuned_gb_importances
}).sort_values(by='Importance', ascending=False)

# Display the top 10 features
print("Top 10 Important Features (Tuned Gradient Boosting):")
print(tuned_feature_importance_df.head(10))

# Plot
plt.figure(figsize=(10, 6))
plt.barh(tuned_feature_importance_df['Feature'], tuned_feature_importance_df['Importance'], color='skyblue')
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Top 10 Important Features - Tuned Gradient Boosting")
plt.gca().invert_yaxis()  # Invert y-axis to have the highest importance on top
plt.tight_layout()
plt.show()


from sklearn.metrics import precision_recall_curve

# Predicted probabilities for the positive class (Churn = 1)
y_prob_gb_tuned = best_gb_model.predict_proba(X_test)[:, 1]

# Compute Precision-Recall curve
precision_gb_tuned, recall_gb_tuned, thresholds_gb_tuned = precision_recall_curve(y_test, y_prob_gb_tuned)

# Compute F1 scores
f1_scores_gb_tuned = 2 * (precision_gb_tuned * recall_gb_tuned) / (precision_gb_tuned + recall_gb_tuned)
best_idx_gb_tuned = f1_scores_gb_tuned.argmax()
best_threshold_gb_tuned = thresholds_gb_tuned[best_idx_gb_tuned]

print(f"Best Threshold for Tuned Gradient Boosting: {best_threshold_gb_tuned}")

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Apply the best threshold to make predictions
y_pred_custom_gb_tuned = (y_prob_gb_tuned >= best_threshold_gb_tuned).astype(int)

# Evaluate the results with the custom threshold
training_accuracy_gb_tuned = accuracy_score(y_train, best_gb_model.predict(X_train))
testing_accuracy_gb_tuned = accuracy_score(y_test, y_pred_custom_gb_tuned)
classification_report_gb_tuned = classification_report(y_test, y_pred_custom_gb_tuned)
confusion_matrix_gb_tuned = confusion_matrix(y_test, y_pred_custom_gb_tuned)

# Print the evaluation metrics
print(f"Training Accuracy with Custom Threshold: {training_accuracy_gb_tuned}")
print(f"Testing Accuracy with Custom Threshold: {testing_accuracy_gb_tuned}")
print("\nClassification Report (Test Data with Custom Threshold):")
print(classification_report_gb_tuned)
print("\nConfusion Matrix (Test Data with Custom Threshold):")
print(confusion_matrix_gb_tuned)


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Obtain predicted probabilities for the positive class (churn = 1)
y_prob_tuned_gb = best_gb_model.predict_proba(X_test)[:, 1]

# Compute ROC Curve and AUC
fpr_tuned_gb, tpr_tuned_gb, thresholds_roc_tuned_gb = roc_curve(y_test, y_prob_tuned_gb)
roc_auc_tuned_gb = auc(fpr_tuned_gb, tpr_tuned_gb)

# Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr_tuned_gb, tpr_tuned_gb, label=f"ROC Curve (AUC = {roc_auc_tuned_gb:.2f})", color='orange')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Tuned Gradient Boosting")
plt.legend(loc="lower right")
plt.grid()
plt.show()

# Save ROC AUC
print(roc_auc_tuned_gb)



########################### End of gradient boosting ##############################333
####################################################### HYBRID MODEL ################################################################

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Base Models
logistic_model = LogisticRegression(random_state=365, max_iter=2000)
rf_model = RandomForestClassifier(random_state=365)
gb_model = GradientBoostingClassifier(random_state=365)
svm_model = SVC(probability=True, random_state=365)
knn_model = KNeighborsClassifier(n_neighbors=5)


from sklearn.model_selection import GridSearchCV
#Hyperparameter Tuning for Logistic Regression in Hybrid Model
# Define parameter grid for Logistic Regression
param_grid_logistic = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']
}

# Perform GridSearchCV
grid_search_logistic = GridSearchCV(
    estimator=LogisticRegression(random_state=365, max_iter=2000),
    param_grid=param_grid_logistic,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search_logistic.fit(X_train, y_train)

# Best Logistic Regression Model
logistic_model_tuned = grid_search_logistic.best_estimator_
print(f"Best Parameters (Logistic Regression): {grid_search_logistic.best_params_}")
print(f"Best CV Accuracy (Logistic Regression): {grid_search_logistic.best_score_}")



### Hyperparameter Tuning for Logistic Regression in radnom forest Model
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search_rf = GridSearchCV(
    estimator=RandomForestClassifier(random_state=365),
    param_grid=param_grid_rf,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search_rf.fit(X_train, y_train)

# Best Random Forest Model
rf_model_tuned = grid_search_rf.best_estimator_
print(f"Best Parameters (Random Forest): {grid_search_rf.best_params_}")
print(f"Best CV Accuracy (Random Forest): {grid_search_rf.best_score_}")




# hypertuning parameter for gradient boosting hybrid model
param_grid_gb = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5]
}

grid_search_gb = GridSearchCV(
    estimator=GradientBoostingClassifier(random_state=365),
    param_grid=param_grid_gb,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search_gb.fit(X_train, y_train)

# Best Gradient Boosting Model
gb_model_tuned = grid_search_gb.best_estimator_
print(f"Best Parameters (Gradient Boosting): {grid_search_gb.best_params_}")
print(f"Best CV Accuracy (Gradient Boosting): {grid_search_gb.best_score_}")



# hyperparameter tuning for SVM for hybrid model

param_grid_svm = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto']
}

grid_search_svm = GridSearchCV(
    estimator=SVC(probability=True, random_state=365),
    param_grid=param_grid_svm,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search_svm.fit(X_train, y_train)

# Best SVM Model
svm_model_tuned = grid_search_svm.best_estimator_
print(f"Best Parameters (SVM): {grid_search_svm.best_params_}")
print(f"Best CV Accuracy (SVM): {grid_search_svm.best_score_}")


### Hyperparameter tuning for KNN for hybrid model
param_grid_knn = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'p': [1, 2]  # Manhattan (p=1) or Euclidean (p=2)
}

grid_search_knn = GridSearchCV(
    estimator=KNeighborsClassifier(),
    param_grid=param_grid_knn,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search_knn.fit(X_train, y_train)

# Best KNN Model
knn_model_tuned = grid_search_knn.best_estimator_
print(f"Best Parameters (KNN): {grid_search_knn.best_params_}")
print(f"Best CV Accuracy (KNN): {grid_search_knn.best_score_}")





#created a Stacking Classifier using the previously tuned base models
from sklearn.ensemble import StackingClassifier

# Stacking Model with Tuned Base Models
stacking_model_tuned = StackingClassifier(
    estimators=[
        ('Logistic', logistic_model_tuned),
        ('RF', rf_model_tuned),
        ('GB', gb_model_tuned),
        ('SVM', svm_model_tuned),
        ('KNN', knn_model_tuned)
    ],
    final_estimator=GradientBoostingClassifier(random_state=365),
    cv=5
)

# Train the Stacking Model
stacking_model_tuned.fit(X_train, y_train)

# Evaluate Stacking Model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

y_pred_train_stack_tuned = stacking_model_tuned.predict(X_train)
y_pred_test_stack_tuned = stacking_model_tuned.predict(X_test)

print(f"Training Accuracy (Stacking Model): {accuracy_score(y_train, y_pred_train_stack_tuned)}")
print(f"Testing Accuracy (Stacking Model): {accuracy_score(y_test, y_pred_test_stack_tuned)}")

print("\nClassification Report (Stacking Model):")
print(classification_report(y_test, y_pred_test_stack_tuned))

print("\nConfusion Matrix (Stacking Model):")
print(confusion_matrix(y_test, y_pred_test_stack_tuned))


#Evaluating Stacked Model with Custom Threshold.
from sklearn.metrics import precision_recall_curve, f1_score, accuracy_score, classification_report, confusion_matrix

# Predicted probabilities for the positive class (Churn = 1)
y_prob_stack = stacking_model_tuned.predict_proba(X_test)[:, 1]

# Compute Precision-Recall Curve
precision_stack, recall_stack, thresholds_stack = precision_recall_curve(y_test, y_prob_stack)

# Compute F1 scores and find the best threshold
f1_scores_stack = 2 * (precision_stack * recall_stack) / (precision_stack + recall_stack + 1e-9)
best_idx_stack = f1_scores_stack.argmax()
best_threshold_stack = thresholds_stack[best_idx_stack]

print(f"Best Threshold for Stacking Model: {best_threshold_stack}")

# Apply the best threshold to make predictions
y_pred_custom_stack = (y_prob_stack >= best_threshold_stack).astype(int)

# Evaluate the model with the custom threshold
training_accuracy_stack = accuracy_score(y_train, stacking_model_tuned.predict(X_train))
testing_accuracy_stack = accuracy_score(y_test, y_pred_custom_stack)
classification_report_stack = classification_report(y_test, y_pred_custom_stack)
confusion_matrix_stack = confusion_matrix(y_test, y_pred_custom_stack)

print(f"Training Accuracy with Custom Threshold: {training_accuracy_stack}")
print(f"Testing Accuracy with Custom Threshold: {testing_accuracy_stack}")

print("\nClassification Report (Test Data with Custom Threshold):")
print(classification_report_stack)

print("\nConfusion Matrix (Test Data with Custom Threshold):")
print(confusion_matrix_stack)



#Precision-Recall Curve for Stacked Model.
import matplotlib.pyplot as plt
from sklearn.metrics import auc

# Plot Precision-Recall Curve
plt.figure(figsize=(8, 6))
plt.plot(recall_stack, precision_stack, label=f"PR Curve (AUC = {auc(recall_stack, precision_stack):.2f})", color="purple")
plt.scatter(recall_stack[best_idx_stack], precision_stack[best_idx_stack], color='red', 
            label=f"Best Threshold = {best_threshold_stack:.2f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve - Stacking Model")
plt.legend(loc="best")
plt.grid()
plt.show()




#ROC Curve for Stacked Model 
from sklearn.metrics import roc_curve, auc

# Compute ROC Curve
fpr_stack, tpr_stack, thresholds_roc_stack = roc_curve(y_test, y_prob_stack)
roc_auc_stack = auc(fpr_stack, tpr_stack)

# Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr_stack, tpr_stack, label=f"ROC Curve (AUC = {roc_auc_stack:.2f})", color='green')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Stacking Model")
plt.legend(loc="lower right")
plt.grid()
plt.show()




###################### Using Shap and lime interpretability technique in addidion to pdp ################

pip install lime


pip install shap



print(best_gb_model)

import shap

# SHAP for Gradient Boosting. shap uses an explainer to compute the Shapley values for the model which is TreeExplainer
explainer_gb = shap.TreeExplainer(best_gb_model)  # Replace 'best_gb_model' with your tuned Gradient Boosting model
#computes the Shapley values for the test set (X_test)
shap_values_gb = explainer_gb.shap_values(X_test)

# Global Interpretability: Summary Plot
shap.summary_plot(shap_values_gb, X_test, plot_type="bar")  # Feature importance as a bar plot
shap.summary_plot(shap_values_gb, X_test)  # Detailed feature importance

# Local Interpretability: Force Plot for a single prediction
instance_index = 0  # Change to test instance index to explain
shap.force_plot(
    explainer_gb.expected_value,
    shap_values_gb[instance_index],
    X_test.iloc[instance_index],
    matplotlib=True
)


'''reeExplainer is used to explain tree-based models.
SHAP values are computed for the test set to measure feature contributions.
The summary plot visualizes global feature importance.
Force plot explains individual predictions and their feature contributions.'''



import shap

# Compute SHAP values for Random Forest

explainer_rf = shap.TreeExplainer(best_rf_model)  # Replace 'best_rf_model' with your Random Forest model
shap_values_rf = explainer_rf.shap_values(X_test)

# If shap_values_rf is a list (multi-class), select the relevant class (e.g., Class 1 - Churn)
if isinstance(shap_values_rf, list):
    shap_values_rf = shap_values_rf[1]  # Use index 1 for Class 1 (Churn)

# Ensure X_test has no extra indices or mismatched columns
X_test = X_test.reset_index(drop=True)

# Summary plot to show global feature importance
shap.summary_plot(shap_values_rf, X_test, plot_type="bar")  # Bar plot for feature importance


print("SHAP Values Shape:", shap_values_rf.shape)
print("X_test Shape:", X_test.shape)

import shap
shap.initjs()
# Ensure SHAP values and explainer are computed
instance_index = 1  # Index of the test instance to explain

# Correct handling for multi-class models
shap_values_rf_single = shap_values_rf[..., 1]  # Use Class 1 (Churn) SHAP values
base_value_rf = explainer_rf.expected_value[1]  # Base value for Class 1 (Churn)

# Generate the force plot for a single instance
shap.plots.force(
    base_value=base_value_rf,  # Base value for Class 1
    shap_values=shap_values_rf_single[instance_index],  # SHAP values for the selected instance
    features=X_test.iloc[instance_index]  # Feature values for the selected instance
)



#################################### shap done ###########################


#################################### lime begins ##########################

pip install lime


from lime.lime_tabular import LimeTabularExplainer

# Initialize LIME Explainer
lime_explainer = LimeTabularExplainer(
    training_data=X_train.values,  # Use values for training
    feature_names=X_train.columns.tolist(),  # Pass feature names explicitly
    class_names=["No Churn", "Churn"],  # Class names for binary classification
    mode="classification"
)

# Explain a single prediction
instance_index = 0  # Test instance index to explain
lime_exp_gb = lime_explainer.explain_instance(
    data_row=X_test.iloc[instance_index].values,  # Ensure this is a NumPy array (values)
    predict_fn=lambda x: best_gb_model.predict_proba(pd.DataFrame(x, columns=X_train.columns))  # Convert back to DataFrame
)

# Visualize explanation
lime_exp_gb.show_in_notebook()
lime_exp_gb.save_to_file("lime_explanation_gradient_boosting.html")


############################ LIME Implementation for Random Forest############################################

from lime.lime_tabular import LimeTabularExplainer

# Initialize LIME Explainer
lime_explainer = LimeTabularExplainer(
    training_data=X_train.values,  # Use training data as NumPy array
    feature_names=X_train.columns.tolist(),  # Provide feature names explicitly
    class_names=["No Churn", "Churn"],  # Class names for binary classification
    mode="classification"
)

# Explain a single prediction
instance_index = 0  # Test instance index to explain
lime_exp_rf = lime_explainer.explain_instance(
    data_row=X_test.iloc[instance_index].values,  # Convert test instance to NumPy array
    predict_fn=lambda x: best_rf_model.predict_proba(pd.DataFrame(x, columns=X_train.columns))  # Convert back to DataFrame
)

# Visualize explanation
lime_exp_rf.show_in_notebook()
lime_exp_rf.save_to_file("lime_explanation_random_forest.html")


from collections import defaultdict
import numpy as np

# Initialize a dictionary to store feature importance
feature_importance_gb = defaultdict(float)

# Loop through multiple test instances
num_samples = 50  # Number of instances to analyze for global trends
for idx in range(num_samples):
    # Explain the instance using LIME
    lime_exp_gb = lime_explainer.explain_instance(
        data_row=X_test.iloc[idx].values,  # Test instance as NumPy array
        predict_fn=lambda x: best_gb_model.predict_proba(pd.DataFrame(x, columns=X_train.columns))
    )
    
    # Aggregate feature importance from the explanation
    for feature, importance in lime_exp_gb.as_list():
        feature_importance_gb[feature] += np.abs(importance)  # Aggregate absolute importance

# Convert aggregated feature importance to a sorted list
sorted_importance_gb = sorted(feature_importance_gb.items(), key=lambda x: x[1], reverse=True)

# Display global feature importance for Gradient Boosting
print("Global Feature Importance (Gradient Boosting):")
for feature, importance in sorted_importance_gb:
    print(f"{feature}: {importance:.4f}")


# Initialize a dictionary to store feature importance
feature_importance_rf = defaultdict(float)

# Loop through multiple test instances
num_samples = 50  # Number of instances to analyze for global trends
for idx in range(num_samples):
    # Explain the instance using LIME
    lime_exp_rf = lime_explainer.explain_instance(
        data_row=X_test.iloc[idx].values,  # Test instance as NumPy array
        predict_fn=lambda x: best_rf_model.predict_proba(pd.DataFrame(x, columns=X_train.columns))
    )
    
    # Aggregate feature importance from the explanation
    for feature, importance in lime_exp_rf.as_list():
        feature_importance_rf[feature] += np.abs(importance)  # Aggregate absolute importance

# Convert aggregated feature importance to a sorted list
sorted_importance_rf = sorted(feature_importance_rf.items(), key=lambda x: x[1], reverse=True)

# Display global feature importance for Random Forest
print("Global Feature Importance (Random Forest):")
for feature, importance in sorted_importance_rf:
    print(f"{feature}: {importance:.4f}")


import matplotlib.pyplot as plt

# Data from both Gradient Boosting and Random Forest for feature importance
features = [
    "Contract_Two year", "InternetService_Fiber optic", "tenure", 
    "TotalCharges", "InternetService_No", "PaymentMethod_Electronic check", 
    "Contract_One year", "OnlineSecurity", "TechSupport", "SeniorCitizen"
]

# Sample importance scores (You can replace these with actual values from your models)
importance_gb = [6.1627, 4.2734, 1.57, 1.77, 3.3890, 1.9044, 3.1801, 1.4327, 1.56, 0.9127]
importance_rf = [3.8055, 3.4434, 1.57, 1.77, 2.1334, 1.8590, 1.8021, 1.58, 1.56, 2.27]

# Combine the importance scores (we can average or just stack them for simplicity)
avg_importance = [(gb + rf) / 2 for gb, rf in zip(importance_gb, importance_rf)]

# Bar Chart for Feature Importance
plt.figure(figsize=(10, 6))
plt.barh(features, avg_importance, color='skyblue')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.title('Average Feature Importance for Churn Prediction (Global Insights)')
plt.gca().invert_yaxis()
plt.show()

# Pie Chart for Feature Importance Distribution
plt.figure(figsize=(8, 8))
plt.pie(avg_importance, labels=features, autopct='%1.1f%%', startangle=140)
plt.title('Feature Importance Distribution for Churn Prediction (Global Insights)')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()


































