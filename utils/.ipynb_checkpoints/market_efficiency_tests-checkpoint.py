import matplotlib.pyplot as plt
import statsmodels.api as sm
from utils.helpers import *
import os
import pickle
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score
import ta
from utils.helpers import *
from utils.helpers import asset_columns
print(asset_columns)


## Weak Form 
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
    
## Semi-Strong Form

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

# Function to train the models and evaluate them
def train_initial_models(X_train_dict, y_train_dict, X_test_dict, y_test_dict):
    # [Add the body of the function from your original code]
    print("Training initial models...")
    
    # DataFrame to store feature importances for all assets
    feature_importances_df = pd.DataFrame()
    
    results = {}
    for asset in X_train_dict:
        X_train = X_train_dict[asset]
        y_train = y_train_dict[asset]
        
        X_test = X_test_dict[asset]
        y_test = y_test_dict[asset]
        
        model = XGBClassifier(objective='binary:logistic', tree_method='gpu_hist')
        model.fit(X_train, y_train)
        
        # Extract feature importances
        feature_importances = model.feature_importances_
        temp_df = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': feature_importances,
            'Asset': asset  # Adding the asset name to differentiate
        })
        # feature_importances_df = feature_importances_df.append(temp_df)
        feature_importances_df = pd.concat([feature_importances_df, temp_df], ignore_index=True)
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        results[asset] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'F1 Score': f1_score(y_test, y_pred),
            'ROC AUC': roc_auc_score(y_test, y_pred_proba)
        }

def main():
    print("Main function started")
    
    # Ensure the necessary directories exist
    if not os.path.exists('./models/'):
        os.makedirs('./models/')
    
    # Dictionary declarations
   
    test_dates_dict = get_test_dates(df_events, 0, days_after_event)
    all_evaluation_metrics = {}
    X_train_dict = {}
    y_train_dict = {}
    X_test_dict = {}
    y_test_dict = {} 

    # Loop through each day interval
    for days in days_after_event:
        test_dates = test_dates_dict[days]
        for asset_class in asset_classes:
            
            # Filter columns related to the current asset class
            relevant_columns = [col for col in combined_data.columns if col.startswith(asset_class)]
            
            asset_data = combined_data[relevant_columns]
            
            # Assuming the target variable for each asset class is simply its name (e.g., 'US Large Cap Equities')
            if asset_class not in asset_data.columns:
                continue
            
            train = asset_data[~asset_data.index.isin(test_dates)]
            test = asset_data[asset_data.index.isin(test_dates)]

            X_train_dict[asset_class] = train.drop(asset_class, axis=1)
            y_train_dict[asset_class] = train[asset_class]
            
            X_test_dict[asset_class] = test.drop(asset_class, axis=1)
            y_test_dict[asset_class] = test[asset_class]

            # Step 2: Train the models using the dictionaries
            evaluation_metrics, model = train_initial_models(X_train_dict, y_train_dict, X_test_dict, y_test_dict)
            all_evaluation_metrics[f"{asset_class}_{days} days"] = evaluation_metrics

            # Save the trained model
            with open(f'./models/{asset_class}_model_{days}_days.pkl', 'wb') as f:
                pickle.dump(model, f)

    # Aggregate all evaluation metrics into a single dataframe
    staging_evaluation_df = pd.concat({k: pd.DataFrame(v).T for k, v in all_evaluation_metrics.items()}, axis=0)
    pivot_df = staging_evaluation_df.unstack(level=0)
    pivot_df.columns = pivot_df.columns.get_level_values(1)
    ordered_columns = ['5 days', '30 days', '60 days', '90 days']
    final_evaluation_df = pivot_df[ordered_columns]
    
    # Optionally: Print or save the final dataframe
    print(final_evaluation_df)






