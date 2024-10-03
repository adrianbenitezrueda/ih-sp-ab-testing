import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
import scipy.stats as stats

# Function to read and merge initial web data
def load_web_data(csv1_path, csv2_path):
    df_wd_1 = pd.read_csv(csv1_path)
    df_wd_2 = pd.read_csv(csv2_path)
    df_wd_merged = pd.concat([df_wd_1, df_wd_2])
    return df_wd_merged


# Function to read demo data
def load_demo_data(csv_path):
    return pd.read_csv(csv_path)


# Function to read experiment clients data
def load_exp_clients_data(csv_path):
    return pd.read_csv(csv_path)


# Function to check column types and nulls in dataframes
def check_data_info(df, df_name):
    print(f"Dataframe {df_name} info:")
    print(df.info())
    print(f"Null values in {df_name}:")
    print(df.isnull().sum())


# Function to check duplicates in a dataframe
def check_duplicates(df, df_name):
    print(f"Checking duplicates in {df_name}:")
    print(df.duplicated().sum())


# Function to clean the 'gendr' column and drop rows with missing 'Variation'
def clean_data(df_demo, df_exp_clients):
    df_demo['gendr'].fillna('Unknown', inplace=True)
    df_exp_clients.dropna(subset=["Variation"], inplace=True)
    return df_demo, df_exp_clients


# Function to limit 'bal' column values to two decimal places
def limit_bal_column(df):
    df['bal'] = df['bal'].round(2)
    return df


# Function to transform 'date_time' column to datetime type
def transform_datetime(df, column_name):
    df[column_name] = pd.to_datetime(df[column_name])
    return df


# Function to merge dataframes and filter based on conditions
def merge_dataframes(df1, df2, on_column):
    return pd.merge(df1, df2, on=on_column)


# Function to compare values between two sets (e.g., client_id sets from different DataFrames)
def compare_sets(set1, set2, set1_name="Set1", set2_name="Set2"):
    # Find missing values in each set
    missing_in_set2 = set1 - set2
    missing_in_set1 = set2 - set1
    
    # Show results for set1
    if missing_in_set2:
        print(f"Values in {set1_name} but not in {set2_name}: {list(missing_in_set2)}")
    else:
        print(f"All values in {set1_name} are present in {set2_name}.")
    
    # Show results for set2
    if missing_in_set1:
        print(f"Values in {set2_name} but not in {set1_name}: {list(missing_in_set1)}")
    else:
        print(f"All values in {set2_name} are present in {set1_name}.")


# Custom function to show the percentage and total number in matplotlib
def show_pct(pct, allvals):
    absolute = int(pct / 100. * sum(allvals))
    return f"{pct:.1f}%\n({absolute:d})"


# Function to perform Chi-Square Test for Independence
def chi_square_test(contingency_table):
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
    print(f"Chi2: {chi2}, p-value: {p}")
    return chi2, p


# Function to perform ANOVA test
def anova_test(df, group_col, value_col):
    model = stats.f_oneway(df[group_col], df[value_col])
    print(f"ANOVA results: {model}")
    return model


# Function to save dataframe to csv
def save_to_csv(df, path):
    df.to_csv(path, index=False)


# Function to compare Test vs Control Group
def compare_test_control_groups(df, variation_col, balance_col):
    # Import necessary libraries within the function
    import pandas as pd
    
    # Filter data for Control and Test groups
    df_control = df[df[variation_col] == "Control"]
    df_test = df[df[variation_col] == "Test"]
    
    # Calculate average balances for each group
    test_balance = round(df_test[balance_col].mean(), 2)
    control_balance = round(df_control[balance_col].mean(), 2)
    
    # Print the results
    print(f"Average balance for Experiment Group: {test_balance}")
    print(f"Average balance for Control Group: {control_balance}")
    
    return test_balance, control_balance


# Function to perform Chi-Square Test for Independence
def perform_chi_square_test(df, col1, col2, alpha=0.05):
    # Import necessary libraries within the function
    import pandas as pd
    import scipy.stats as stats

    # Create contingency table
    contingency_table = pd.crosstab(df[col1], df[col2])
    
    # Perform Chi-Square Test
    chi2, p, dof, ex = stats.chi2_contingency(contingency_table)

    # Print results
    print(f"Chi-square Statistic: {chi2}")
    print(f"P-value: {p}")
    
    # Interpretation
    if p < alpha:
        print("There is a significant association between the two variables.")
    else:
        print("No significant association between the two variables.")
    
    return chi2, p


# Function to perform ANOVA test for balance across age groups
def perform_anova_by_age_group(df, age_col, balance_col, bins, labels, alpha=0.05):
    import pandas as pd
    import scipy.stats as stats

    # Create age groups using pd.cut
    df['age_group'] = pd.cut(df[age_col], bins=bins, labels=labels)

    # Perform ANOVA test
    anova_result = stats.f_oneway(
        df[df['age_group'] == labels[0]][balance_col],
        df[df['age_group'] == labels[1]][balance_col],
        df[df['age_group'] == labels[2]][balance_col],
        df[df['age_group'] == labels[3]][balance_col],
        df[df['age_group'] == labels[4]][balance_col],
        df[df['age_group'] == labels[5]][balance_col]
    )
    
    # Print results
    print(f"ANOVA F-Statistic: {anova_result.statistic}")
    print(f"P-value: {anova_result.pvalue}")

    # Interpretation
    if anova_result.pvalue < alpha:
        print("There is a statistically significant difference in average balances across different age groups.")
    else:
        print("There is no statistically significant difference in average balances across different age groups.")
    
    return anova_result


# Function to assign try_session based on process logic
def assign_try_session(df):
    import pandas as pd

    # Define the correct process order with its associated numerical value
    process_order = {'start': 0, 'step_1': 1, 'step_2': 2, 'step_3': 3, 'confirm': 4}

    df = df.sort_values('date_time').reset_index(drop=True)  # Sort by date_time
    try_session = 1  # Initialize the try session
    current_process = None
    df['try_session'] = None
    
    for i in range(len(df)):
        step = df.loc[i, 'process_step']
        process_number = process_order[step]
        
        # If we find a 'start', increment the try_session
        if step == 'start' or current_process is None:
            df.loc[i, 'try_session'] = 'T{:02d}'.format(try_session)
            try_session += 1
            current_process = process_number
        else:
            # Verify that the step is consecutive in the process
            if abs(process_number - current_process) <= 1:
                df.loc[i, 'try_session'] = 'T{:02d}'.format(try_session - 1)
                current_process = process_number
            else:
                # If there's a big jump, it's a sign of a process failure, not considered the same try
                df.loc[i, 'try_session'] = 'T{:02d}'.format(try_session)
                try_session += 1
                current_process = process_number

    return df


# Function to add the 'next_step', 'step_time_seconds', and remove 'next_step' column
def add_next_step_and_step_time_seconds(df):
    # Convert the 'date_time' column to datetime format if it's not already
    df['date_time'] = pd.to_datetime(df['date_time'])

    # Sort by date_time
    df = df.sort_values('date_time').reset_index(drop=True)  
    
    # Add the 'next_step' column, which will be the next value of process_step within the same try_session
    df['next_step'] = df['process_step'].shift(-1)
    
    # Replace the value of next_step when the next value is not in the same try_session
    df.loc[df['try_session'] != df['try_session'].shift(-1), 'next_step'] = None

    # Calculate the time in seconds between the current process_step and the next one within the same try_session
    df['step_time_seconds'] = (df['date_time'].shift(-1) - df['date_time']).dt.total_seconds()
    
    # Replace the value of step_time when the next value is not in the same try_session
    df.loc[df['try_session'] != df['try_session'].shift(-1), 'step_time_seconds'] = None
    
    # Remove the 'next_step' column from the DataFrame
    df = df.drop(columns=['next_step'])
    
    return df


# Function to calculate metrics per client
def calculate_process_metrics(df_client):
    import pandas as pd

    # Define the correct process order with its associated numeric value
    process_order = {'start': 0, 'step_1': 1, 'step_2': 2, 'step_3': 3, 'confirm': 4}

    # Initialize metrics
    number_try = 0
    process_fails = 0
    process_complete = 0
    time_fails = []
    time_complete = []
    
    # Initialize counters for each step
    total_start = 0
    total_step_1 = 0
    total_step_2 = 0
    total_step_3 = 0
    total_confirm = 0
    
    # Initialize lists to calculate the average time per step
    time_start = []
    time_step_1 = []
    time_step_2 = []
    time_step_3 = []
    
    # Filter all the attempts for the client (each attempt starts with "start")
    starts = df_client[df_client['process_step'] == 'start']
    
    # Loop through each attempt (each "start" indicates a new process attempt)
    for _, start_row in starts.iterrows():
        # Get the subset of steps for this particular attempt
        df_visit = df_client[(df_client['visit_id'] == start_row['visit_id']) & 
                             (df_client['date_time'] >= start_row['date_time'])]

        # Sort steps by `date_time`
        df_visit_sorted = df_visit.sort_values('date_time')
        
        # Count the steps in this attempt
        total_start += df_visit_sorted['process_step'].tolist().count('start')
        total_step_1 += df_visit_sorted['process_step'].tolist().count('step_1')
        total_step_2 += df_visit_sorted['process_step'].tolist().count('step_2')
        total_step_3 += df_visit_sorted['process_step'].tolist().count('step_3')
        total_confirm += df_visit_sorted['process_step'].tolist().count('confirm')

        # Save times per step to calculate the average
        process_steps = df_visit_sorted['process_step'].tolist()
        process_numbers = [process_order[step] for step in process_steps]
        
        for i, step in enumerate(process_steps):
            if step == 'start' and i < len(df_visit_sorted)-1:
                time_start.append((df_visit_sorted['date_time'].iloc[i+1] - df_visit_sorted['date_time'].iloc[i]).total_seconds())
            elif step == 'step_1' and i < len(df_visit_sorted)-1:
                time_step_1.append((df_visit_sorted['date_time'].iloc[i+1] - df_visit_sorted['date_time'].iloc[i]).total_seconds())
            elif step == 'step_2' and i < len(df_visit_sorted)-1:
                time_step_2.append((df_visit_sorted['date_time'].iloc[i+1] - df_visit_sorted['date_time'].iloc[i]).total_seconds())
            elif step == 'step_3' and i < len(df_visit_sorted)-1:
                time_step_3.append((df_visit_sorted['date_time'].iloc[i+1] - df_visit_sorted['date_time'].iloc[i]).total_seconds())

        # Check if the steps follow the allowed order (can advance or go back by 1 unit)
        step_diff = [process_numbers[i+1] - process_numbers[i] for i in range(len(process_numbers)-1)]
        
        # Check if all step changes are within the allowed range (-1 or 1)
        valid_process = all(diff in [-1, 1] for diff in step_diff)
        
        # If the process is valid and ends with 'confirm', it's a complete process
        if valid_process and process_steps[-1] == 'confirm':
            process_complete += 1
            start_time = df_visit_sorted['date_time'].iloc[0]
            end_time = df_visit_sorted['date_time'].iloc[-1]
            time_complete.append((end_time - start_time).total_seconds())
        else:
            # It is a failed process
            process_fails += 1
            start_time = df_visit_sorted['date_time'].iloc[0]
            end_time = df_visit_sorted['date_time'].iloc[-1]
            time_fails.append((end_time - start_time).total_seconds())
        
        # Count as an attempt
        number_try += 1
    
    # Calculate total times
    time_fails = sum(time_fails) if time_fails else 0
    time_complete = sum(time_complete) if time_complete else 0
    
    # Calculate the averages based on the number of process_fails and process_complete
    avg_time_fails = round(time_fails / process_fails, 2) if process_fails > 0 else 0
    avg_time_complete = round(time_complete / process_complete, 2) if process_complete > 0 else 0
    
    # Calculate the average time per step
    avg_time_start = round(sum(time_start) / len(time_start), 2) if len(time_start) > 0 else 0
    avg_time_step_1 = round(sum(time_step_1) / len(time_step_1), 2) if len(time_step_1) > 0 else 0
    avg_time_step_2 = round(sum(time_step_2) / len(time_step_2), 2) if len(time_step_2) > 0 else 0
    avg_time_step_3 = round(sum(time_step_3) / len(time_step_3), 2) if len(time_step_3) > 0 else 0

    # Return metrics
    return pd.Series({
        'total_tries': number_try,
        'process_fail': process_fails,
        'process_complete': process_complete,
        'time_fails': time_fails,
        'time_completes': time_complete,
        'avg_time_failed': avg_time_fails,
        'avg_time_completed': avg_time_complete,
        'total_start': total_start,
        'total_step_1': total_step_1,
        'total_step_2': total_step_2,
        'total_step_3': total_step_3,
        'total_confirm': total_confirm,
        'avg_time_start': avg_time_start,
        'avg_time_step_1': avg_time_step_1,
        'avg_time_step_2': avg_time_step_2,
        'avg_time_step_3': avg_time_step_3
    })

