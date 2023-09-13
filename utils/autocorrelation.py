import matplotlib.pyplot as plt
import statsmodels.api as sm
from utils.constants import asset_colors


##Weak form 
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
