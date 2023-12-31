from utils.helpers import *

###########################
## Weak Form 
###########################

def plot_autocorrelation(index_returns, readable_names):
    """
    Plots the autocorrelation of given indices.

    Parameters:
    - index_returns (pd.DataFrame): DataFrame containing index returns.
    - readable_names (list): List of readable names corresponding to indices in `index_returns`.

    Returns:
    - None: This function only plots the data.
    """
    
    # Exclude 'Sentiment Score' from the list
    readable_names = [name for name in readable_names if name != 'Sentiment Score']
    
    num_rows = len(readable_names) // 2 + len(readable_names) % 2
    num_cols = 2

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(10, 3 * num_rows))

    # Plot autocorrelation for each index
    for idx, name in enumerate(readable_names):
        ax = axs[idx//2, idx%2]
        
        # Fetch the color for the asset from the asset_colors dictionary
        asset_color = asset_colors.get(name, '#000000')  # Default to black if the asset is not in the dictionary
        
        # Plotting autocorrelation
        sm.graphics.tsa.plot_acf(index_returns[name].dropna(), lags=30, title=f"{name} Autocorrelation of Daily Returns", ax=ax, color=asset_color)
        
        # Modifying colors for both line and markers to match
        line = ax.lines[0]  # This should get the line from the ACF plot
        line.set_color(asset_color)          # Set line color
        line.set_markerfacecolor(asset_color) # Set marker fill color
        line.set_markeredgecolor(asset_color) # Set marker edge color
        
        ax.set_xlabel("Lags")
        ax.set_ylabel("Autocorrelation")
        ax.grid(True)

    plt.tight_layout()
    plt.show()
    

# Function to get test dates
def get_test_dates(df, pre_days, post_days_list):
    test_dates_dict = {}
    for post_days in post_days_list:
        test_dates = []
        for date in df['Event Date']:
            start_date = date - timedelta(days=pre_days)
            end_date = date + timedelta(days=post_days)
            test_dates.extend(pd.date_range(start=start_date, end=end_date).tolist())
        test_dates_dict[post_days] = test_dates
    return test_dates_dict


###########################
## Semi-Strong Form
###########################

## xgboost

# Function to train the models and evaluate them
def train_initial_xgb_models(X_train_dict, y_train_dict, X_test_dict, y_test_dict, asset_class, day):
    print(f'Training {asset_class} {day} day XGBoost model...')
    results = {}
    
    for asset in X_train_dict:
        X_train = X_train_dict[asset]
        y_train = y_train_dict[asset]
        
        X_test = X_test_dict[asset]
        y_test = y_test_dict[asset]
        
         # Standardizing the data
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        
        model = XGBClassifier(objective='binary:logistic', tree_method='hist')
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        results[asset] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'F1 Score': f1_score(y_test, y_pred),
            'ROC AUC': roc_auc_score(y_test, y_pred_proba)
        }
    return results, model

# Function to loop through days and save information
def main(combined_data):
    print("Main function started")
    
    days_after_event = [5, 30, 60, 90]
    
    asset_columns = [name for name in readable_names if name != 'Sentiment Score']
    
    # Ensure the necessary directories exist
    if not os.path.exists('./models/'):
        os.makedirs('./models/')
    
    # Dictionary declarations
    test_dates_dict = get_test_dates(df_events, 0, days_after_event)
    all_evaluation_metrics = {asset: {} for asset in readable_names}
    X_train_dict = {}
    y_train_dict = {}
    X_test_dict = {}
    y_test_dict = {}
   
    for days in days_after_event:
        test_dates = test_dates_dict[days]
        
        for asset_class in readable_names:
            # Filter columns related to the current asset class
            relevant_columns = [col for col in combined_data.columns if col.startswith(asset_class)]
            
            asset_data = combined_data[relevant_columns]

            # Assuming the target variable for each asset class is simply its name
            if asset_class not in asset_data.columns:
                continue

            train = asset_data[~asset_data.index.isin(test_dates)]
            test = asset_data[asset_data.index.isin(test_dates)]

            X_train_dict[asset_class] = train.drop(asset_class, axis=1)
            y_train_dict[asset_class] = train[asset_class]
            
            X_test_dict[asset_class] = test.drop(asset_class, axis=1)
            y_test_dict[asset_class] = test[asset_class]

            # Train the model
            evaluation_metrics, model = train_initial_xgb_models(X_train_dict, y_train_dict, X_test_dict, y_test_dict, asset_class, days)
            all_evaluation_metrics[asset_class][f"{days} days"] = evaluation_metrics[asset_class]["F1 Score"]

            # Save the trained model
            model_save_path = f'./models/{asset_class}_XGBoost_model_{days}_days.pkl'
            if not os.path.exists(os.path.dirname(model_save_path)):
                os.makedirs(os.path.dirname(model_save_path))
            with open(model_save_path, 'wb') as f:
                pickle.dump(model, f)

    # Convert nested dictionary to DataFrame
    result_df = pd.DataFrame(all_evaluation_metrics).T
    result_df = result_df[['5 days', '30 days', '60 days', '90 days']]
    return result_df

def highlight_f1(row):
    return ['background-color: yellow' if col == 'F1 Score' else '' for col in row.index]

def xgb_display_styled_evaluation(combined_data, highlight_function):
    final_table = main(combined_data)
  
    # Calculate average F1 scores for each asset class
    final_table["Average F1"] = final_table.mean(axis=1)

    # Identify the best performing asset class
    best_asset_class = final_table["Average F1"].idxmax()
    print(f"The best performing asset class is: {best_asset_class}")
    
    xgb_styled_evaluation_df = (final_table.style
                            .apply(highlight_function)
                            .format("{:.2f}")
                            .set_caption("<b style='font-size: 16px'>F1 Metrics for XGBoost Across Different Time Intervals</b>")
                            .set_table_styles({
                                'F1 Score': [{'selector': '',
                                              'props': [('color', 'black'),
                                                        ('font-weight', 'bold')]}]
                            }))

    # Delete models for other asset classes
    model_folder = './models/'
    for model_file in os.listdir(model_folder):
        full_path = os.path.join(model_folder, model_file)

        # Skip directories or other unwanted files
        if os.path.isdir(full_path) or not model_file.endswith('.pkl'):
            continue

        if best_asset_class not in model_file:
            os.remove(full_path)
            # print(f"Deleted: {model_file}")

    print("Cleanup complete!")
    
    return final_table, xgb_styled_evaluation_df  # Return both DataFrames

    
def plot_f1_scores_over_time(final_table):
    """
    Plots F1 scores across different time intervals based on the given DataFrame.
    
    Parameters:
    - result_df: DataFrame containing F1 scores.
    """
    # Set a seaborn style for better aesthetics
    sns.set(style="whitegrid")

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))

    # Define colors for each category
    colors = sns.color_palette("husl", n_colors=len(result_df))
    # colors = asset_colors

    # Iterate through each row (category) in result_df
    for i, (index, row) in enumerate(final_table.iterrows()):
        # Plot a smooth line for each category with the corresponding color
        plt.plot(row.index, row.values, marker='o', label=index, color=colors[i], linewidth=2)

    # Customize the plot
    plt.title("F1 Score Across Different Time Intervals", fontsize=16)
    plt.xlabel("Time Intervals", fontsize=14)
    plt.ylabel("F1 Score", fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Display the plot
    plt.tight_layout()
    plt.show()

    
###########################
## Strong Form - Keras
###########################

# Function to train the models and evaluate them
def train_initial_keras_models(X_train_dict, y_train_dict, X_test_dict, y_test_dict, asset_class, day):
    print(f'Training {asset_class} {day} day Keras model...')
    results = {}
    
    for asset in X_train_dict:
        X_train = X_train_dict[asset]
        y_train = y_train_dict[asset]
        
        X_test = X_test_dict[asset]
        y_test = y_test_dict[asset]
        
         # Standardizing the data
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Define the Keras model
        model = Sequential([
            Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
            Dense(64, activation='relu'),
            # Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
        
        # Train the model
        model.fit(X_train, y_train, epochs=1, batch_size=32, verbose=0, validation_split=0.2)
        
        # Predict
        y_pred_proba = model.predict(X_test, verbose=0).flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        results[asset] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'F1 Score': f1_score(y_test, y_pred),
            'ROC AUC': roc_auc_score(y_test, y_pred_proba)
        }
    return results, model

# Function to loop through days and save information
def keras_main(combined_data):
    print("Main function started")
    
    days_after_event = [5, 30, 60, 90]
    
    asset_columns = [name for name in readable_names if name != 'Sentiment Score']
    
    # Ensure the necessary directories exist
    if not os.path.exists('./models/'):
        os.makedirs('./models/')
    
    # Dictionary declarations
    test_dates_dict = get_test_dates(df_events, 0, days_after_event)
    all_evaluation_metrics = {asset: {} for asset in readable_names}
    X_train_dict = {}
    y_train_dict = {}
    X_test_dict = {}
    y_test_dict = {}
   
    for days in days_after_event:
        test_dates = test_dates_dict[days]
        
        for asset_class in readable_names:
            # Filter columns related to the current asset class
            relevant_columns = [col for col in combined_data.columns if col.startswith(asset_class)]
            
            asset_data = combined_data[relevant_columns]

            # Assuming the target variable for each asset class is simply its name
            if asset_class not in asset_data.columns:
                continue

            train = asset_data[~asset_data.index.isin(test_dates)]
            test = asset_data[asset_data.index.isin(test_dates)]

            X_train_dict[asset_class] = train.drop(asset_class, axis=1)
            y_train_dict[asset_class] = train[asset_class]
            
            X_test_dict[asset_class] = test.drop(asset_class, axis=1)
            y_test_dict[asset_class] = test[asset_class]

            # Train the model
            evaluation_metrics, model = train_initial_keras_models(X_train_dict, y_train_dict, X_test_dict, y_test_dict, asset_class, days)
            all_evaluation_metrics[asset_class][f"{days} days"] = evaluation_metrics[asset_class]["F1 Score"]

            # Save the trained model
            model.save(f'./models/{asset_class}_Keras_model_{days}_days.keras')


    # Convert nested dictionary to DataFrame
    result_df = pd.DataFrame(all_evaluation_metrics).T
    result_df = result_df[['5 days', '30 days', '60 days', '90 days']]
    return result_df

def keras_display_styled_evaluation(combined_data, highlight_function):
    final_table = keras_main(combined_data)
   
    # Calculate average F1 scores for each asset class
    final_table["Average F1"] = final_table.mean(axis=1)

    # Identify the best performing asset class
    best_asset_class = final_table["Average F1"].idxmax()
    print(f"The best performing asset class is: {best_asset_class}")
    
    keras_styled_evaluation_df = (final_table.style
                            .apply(highlight_function)
                            .format("{:.2f}")
                            .set_caption("<b style='font-size: 16px'>F1 Metrics for Keras Across Different Time Intervals</b>")
                            .set_table_styles({
                                'F1 Score': [{'selector': '',
                                              'props': [('color', 'black'),
                                                        ('font-weight', 'bold')]}]
                            }))

    # display(keras_styled_evaluation_df)

    # Delete models for other asset classes
    model_folder = './models/'
    for model_file in os.listdir(model_folder):
        full_path = os.path.join(model_folder, model_file)

        # Skip directories or other unwanted files
        if os.path.isdir(full_path) or not model_file.endswith('.keras'):
            continue

        if best_asset_class not in model_file:
            os.remove(full_path)

    print("Cleanup complete!")
    
    return final_table, keras_styled_evaluation_df  # Return both DataFrames

################################
## transfer learning functions
###################################

def score_transfer_keras_models(X_train_dict, y_train_dict, X_test_dict, y_test_dict, asset_class, day):
    # print(f'Scoring {asset_class} {day} day with Bank Loan Keras {day} day model...')
    results = {}
    
    for asset in X_train_dict:
        X_train = X_train_dict[asset]
        y_train = y_train_dict[asset]
        
        X_test = X_test_dict[asset]
        y_test = y_test_dict[asset]
        
        
         # Standardizing the data
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        
        model=load_model(f'./models/US Bank Loans_Keras_model_{day}_days.keras')
    
        # Predict
        y_pred_proba = model.predict(X_test, verbose=0).flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        results[asset] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'F1 Score': f1_score(y_test, y_pred),
            'ROC AUC': roc_auc_score(y_test, y_pred_proba)
        }
    return results

# Function to loop through days and save information
def keras_scoring_main(combined_data):
    print("Main function started")
    
    days_after_event = [5, 30, 60, 90]
    
    asset_columns = [name for name in readable_names if name != 'Sentiment Score']
    
    # Ensure the necessary directories exist
    if not os.path.exists('./models/'):
        os.makedirs('./models/')
    
    # Dictionary declarations
    test_dates_dict = get_test_dates(df_events, 0, days_after_event)
    all_evaluation_metrics = {asset: {} for asset in readable_names}
    X_train_dict = {}
    y_train_dict = {}
    X_test_dict = {}
    y_test_dict = {}
   
    for days in days_after_event:
        test_dates = test_dates_dict[days]
        
        for asset_class in readable_names:
            # Filter columns related to the current asset class
            relevant_columns = [col for col in combined_data.columns if col.startswith(asset_class)]
            
            asset_data = combined_data[relevant_columns]

            # Assuming the target variable for each asset class is simply its name
            if asset_class not in asset_data.columns:
                continue

            train = asset_data[~asset_data.index.isin(test_dates)]
            test = asset_data[asset_data.index.isin(test_dates)]

            X_train_dict[asset_class] = train.drop(asset_class, axis=1)
            y_train_dict[asset_class] = train[asset_class]
            
            X_test_dict[asset_class] = test.drop(asset_class, axis=1)
            y_test_dict[asset_class] = test[asset_class]
            

            # Train the model
            evaluation_metrics = score_transfer_keras_models(X_train_dict, y_train_dict, X_test_dict, y_test_dict, asset_class, days)
            all_evaluation_metrics[asset_class][f"{days} days"] = evaluation_metrics[asset_class]["F1 Score"]



    # Convert nested dictionary to DataFrame
    result_df = pd.DataFrame(all_evaluation_metrics).T
    result_df = result_df[['5 days', '30 days', '60 days', '90 days']]
    return result_df

def transfer_display_styled_evaluation(combined_data, highlight_function):
    transfer_final_table = keras_scoring_main(combined_data)

    transfer_styled_evaluation_df = (transfer_final_table.style
                            .apply(highlight_function)
                            .format("{:.2f}")
                            .set_caption("<b style='font-size: 16px'>Transfer Learning F1 Metrics for Keras Across Different Time Intervals</b>")
                            .set_table_styles({
                                'F1 Score': [{'selector': '',
                                              'props': [('color', 'black'),
                                                        ('font-weight', 'bold')]}]
                            }))
    

    # display(transfer_styled_evaluation_df)
    return transfer_final_table,transfer_styled_evaluation_df