import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import skew, kurtosis
import plotly.graph_objects as go
from IPython.display import display, HTML
from utils.helpers import *


def calculate_skewness_kurtosis(data):
    """
    Calculate skewness and kurtosis for given numeric columns.

    Parameters:
    - data (DataFrame): DataFrame containing the data.
    - numeric_columns (list): List of column names to calculate skewness and kurtosis for.

    Returns:
    - DataFrame: A DataFrame containing the skewness and kurtosis values.
    """
    numeric_columns = data.select_dtypes(include='number').columns
    
    skewness_values = data[numeric_columns].skew()
    kurtosis_values = data[numeric_columns].kurt()

    skew_kurt_df = pd.DataFrame({
        'Skewness': skewness_values,
        'Kurtosis': kurtosis_values
    })
    
    print(skew_kurt_df)
    
    return skew_kurt_df

def plot_distribution(data, skew_kurt_df, start_date, end_date, num_rows=4, num_cols=2):
    """
    Plot the distribution of data, highlighting skewness and kurtosis.

    Parameters:
    - data (DataFrame): Data to plot.
    - skew_kurt_df (DataFrame): DataFrame containing skewness and kurtosis values.
    - start_date (datetime.datetime): The start date.
    - end_date (datetime.datetime): The end date.
    - num_rows (int, optional): Number of rows for subplots.
    - num_cols (int, optional): Number of columns for subplots.

    Returns:
    - None
    """

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(10, 3 * num_rows))

    for idx, (name, row) in enumerate(skew_kurt_df.iterrows()):
        row_idx = idx // num_cols
        col_idx = idx % num_cols

        ax = axs[row_idx, col_idx]
        asset_color = asset_colors.get(name, '#000000')  # Default to black if the asset is not in the dictionary
        sns.histplot(data[name], bins=50, kde=True, ax=ax, color=asset_color)

        # Calculate the 25th and 75th percentiles for the overlay rectangle
        q25, q75 = np.percentile(data[name], [25, 75])
        ax.fill_between([q25, q75], ax.get_ylim()[0], ax.get_ylim()[1], color=asset_color, alpha=0.3)

        # Format the start_date and end_date for the title
        formatted_start_date = start_date.strftime('%m/%d/%Y')
        formatted_end_date = end_date.strftime('%m/%d/%Y')
        
        # Create a multi-line title
        title = f"{name}\nReturns Distribution\n({formatted_start_date} to {formatted_end_date})"
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Returns")
        ax.set_ylabel("Density")
        ax.grid(True)

        # Add Skewness and Kurtosis values in upper right corner
        skewness = row['Skewness']
        kurtosis = row['Kurtosis']
        ax.text(0.95, 0.95, f"Skewness: {skewness:.2f}\nKurtosis: {kurtosis:.2f}",
                transform=ax.transAxes, va='top', ha='right', fontsize=10, color='black')

    # Adjust layout and display the plots
    plt.tight_layout()
    plt.show()
    
    
def plot_drawdowns(index_returns, start_date, end_date):
    # Compute the running cumulative returns
    cum_returns = (1 + index_returns).cumprod()

    # Compute the running maximum
    running_max = cum_returns.cummax()

    # Compute the drawdowns
    drawdowns = (cum_returns - running_max) / running_max

    # Find the largest drawdown for each asset class
    largest_drawdowns = drawdowns.min()

    # Get the minimum y-axis value
    min_y_value = largest_drawdowns.min() - 0.1
  
    # Visualization
    plt.figure(figsize=(9, 6))

    # Define the color gradient based on the values
    colors = plt.cm.Reds(np.linspace(0.1, 1, len(largest_drawdowns)))

    sorted_indices = largest_drawdowns.argsort()[::-1]  # Sorting values for better visualization
    sorted_drawdowns = largest_drawdowns.iloc[sorted_indices]

    bars = plt.bar(sorted_drawdowns.index, sorted_drawdowns.values, color=colors)

    # Set title and labels with increased font size
    title = f"Largest Drawdowns by Asset Class ({start_date.strftime('%m/%d/%Y')} to {end_date.strftime('%m/%d/%Y')})"
    plt.title(title, fontsize=14)  
    plt.ylabel('Drawdown Magnitude', fontsize=14)
    plt.xlabel('Asset Class', fontsize=10)
    plt.grid(axis='y')
    plt.xticks(rotation=90, fontsize=12)
    plt.yticks(fontsize=12)

    # Set y-axis limits
    plt.ylim(min_y_value, None)

    # Format the y-axis labels as percentages
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:.0%}".format(x)))

    # Highlighting the largest drawdown value
    for i, v in enumerate(sorted_drawdowns):
        if v < -0.15:
            plt.text(i, v - 0.03, f"{v:.2%}", ha='center', va='top', fontweight='regular', fontsize=10, color='black')
        else:
            plt.text(i, v + 0.02, f"{v:.2%}", ha='center', va='bottom', fontweight='regular', fontsize=10, color='black')

    plt.tight_layout()
    plt.show()

def plot_correlation(index_returns, start_date, end_date):

    corr= index_returns.corr()

    # Getting the Upper Triangle of the co-relation matrix
    matrix = np.triu(corr)

    plt.figure(figsize=(8,4))

    # using the upper triangle matrix as mask 
    sns.heatmap(corr, annot=True, mask=matrix, cmap='coolwarm', vmin=-1, vmax=1)
    title = f"Correlation Matrix between Asset Classes ({start_date.strftime('%m/%d/%Y')} to {end_date.strftime('%m/%d/%Y')})"
    plt.title(title, fontsize=14)
    plt.show()

##Would not render interactive graph on nbviewer   
# from IPython.display import display, HTML
# import plotly.graph_objects as go

# def plot_cumulative_returns_with_events(index_returns):
#     cumulative_returns = (1 + index_returns).cumprod()

#     # Extracting the dates and labels directly from df_events in Constants.py
#     events_dates = df_events['Event Date']
#     events_labels = df_events['Event'].tolist()

#     # Define test periods based on events
#     test_periods = [(date, date + pd.Timedelta(days=30)) for date in events_dates]

#     # Create a new figure
#     fig = go.Figure()

#     # Plot cumulative returns for each asset
#     for asset in asset_columns:
#         fig.add_trace(go.Scatter(x=cumulative_returns.index, y=cumulative_returns[asset], mode='lines', name=asset))

#         # Highlight events with shaded regions and label them
#     for date, label in zip(events_dates, events_labels):
#         # Shading 30 days post-event
#         fig.add_vrect(x0=date, x1=date + pd.Timedelta(days=30), fillcolor="gray", opacity=0.2, line_width=0)
#         # Annotating the event
#         fig.add_annotation(x=date, y=max(cumulative_returns.loc[date]), text=label, showarrow=True, arrowhead=4, ax=0, ay=-40)

#     # Annotate Test Periods
#     for start, end in test_periods:
#         mid_date = start + (end - start) / 2
#         closest_date = cumulative_returns.index[np.argmin(np.abs(cumulative_returns.index - mid_date))]
#         fig.add_annotation(
#             x=closest_date, 
#             y=cumulative_returns.min().min(),  # Set to minimum cumulative return for bottom
#             text="Test Period", 
#             showarrow=False, 
#             font=dict(color="blue", size=10),
#             textangle=90
#         )

#     # Annotate Train Periods
#     train_start = cumulative_returns.index[0]
#     for start, _ in test_periods:
#         mid_date = train_start + (start - train_start) / 2
#         closest_date = cumulative_returns.index[np.argmin(np.abs(cumulative_returns.index - mid_date))]
#         fig.add_annotation(
#             x=closest_date, 
#             y=cumulative_returns.min().min(),  # Set to minimum cumulative return for bottom
#             text="Train Period", 
#             showarrow=False, 
#             font=dict(color="red", size=10),
#             textangle=90
#         )
#         train_start = start + pd.Timedelta(days=30)

#     # Annotate last Train Period if necessary
#     if train_start < cumulative_returns.index[-1]:
#         mid_date = train_start + (cumulative_returns.index[-1] - train_start) / 2
#         closest_date = cumulative_returns.index[np.argmin(np.abs(cumulative_returns.index - mid_date))]
#         fig.add_annotation(
#             x=closest_date, 
#             y=cumulative_returns.min().min(),  # Set to minimum cumulative return for bottom
#             text="Train Period", 
#             showarrow=False, 
#             font=dict(color="red", size=10),
#             textangle=90
#         )

#     fig.update_layout(title='Cumulative Returns and Model Test/Train Periods', yaxis_title='Cumulative Return', height=600)
#     fig.show()

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd

def plot_cumulative_returns_with_events(index_returns):
    cumulative_returns = (1 + index_returns).cumprod()

    # Extracting the dates and labels directly from df_events in Constants.py
    events_dates = df_events['Event Date']
    events_labels = df_events['Event'].tolist()

    # Define test periods based on events
    test_periods = [(date, date + pd.Timedelta(days=30)) for date in events_dates]

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot cumulative returns for each asset
    for asset in asset_columns:
        ax.plot(cumulative_returns.index, cumulative_returns[asset], label=asset)

    # Annotate Test Periods
    for start, end in test_periods:
        ax.axvspan(start, end, color="gray", alpha=0.2)
        mid_date = start + (end - start) / 2
        ax.annotate("Test Period", (mid_date, cumulative_returns.min().min()), color="blue", rotation=90)

    # Annotate Train Periods
    train_start = cumulative_returns.index[0]
    for start, _ in test_periods:
        mid_date = train_start + (start - train_start) / 2
        ax.annotate("Train Period", (mid_date, cumulative_returns.min().min()), color="red", rotation=90)
        train_start = start + pd.Timedelta(days=30)

    # Annotate last Train Period if necessary
    if train_start < cumulative_returns.index[-1]:
        mid_date = train_start + (cumulative_returns.index[-1] - train_start) / 2
        ax.annotate("Train Period", (mid_date, cumulative_returns.min().min()), color="red", rotation=90)
        
    # Annotate events
    for date, label in zip(events_dates, events_labels):
        ax.annotate(label, (date, max(cumulative_returns.loc[date])), arrowprops=dict(arrowstyle="->"), xytext=(0,40), textcoords="offset points")

    ax.set_title('Cumulative Returns and Model Test/Train Periods')
    ax.set_ylabel('Cumulative Return')
    ax.legend()
    
    plt.tight_layout()
    plt.show()

def display_event_table():
    # Creating the centered table directly using df_events from Constants.py
    centered_table = f"<center>{df_events.to_html(index=False)}</center>"

    # Display the centered HTML table
    display(HTML(centered_table))
