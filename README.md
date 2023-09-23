# Quantitative Analysis of Market Efficiency

Dive deep into the world of financial markets through a data-driven lens. Using Python and state-of-the-art Machine Learning models, this repository explores different levels of the Efficient Market Hypothesis across various asset classes.

## Table of Contents
1. [Overview](#overview)
2. [Installation & Usage](#installation--usage)
3. [Project Structure](#project-structure)
4. [Data Sources](#data-sources)
5. [Future Work](#future-work)
6. [Contributing](#contributing)
7. [License](#license)

## Overview
This project tests the weak, semi-strong, and strong forms of market efficiency across different asset classes. By leveraging techniques such as AutoCorrelation, XGBoost, and Transfer Learning (Keras), we gain insights into the behavior of various assets in response to significant market events.

## Installation & Usage
1. Clone this repository: `git clone https://github.com/[YourUsername]/MarketEfficiencyAnalysis.git`
2. Navigate to the repository: `cd MarketEfficiencyAnalysis`
3. Install dependencies: `pip install -r utils/requirements.txt`
4. (Optional) If using Docker: Build the Docker image using the provided `Dockerfile` within the `utils` directory.
5. Run Jupyter Notebook or Jupyter Lab to view and execute the analysis.

## Project Structure
- `MarketEfficiencyComparison_AssetClass.ipynb` - Main Jupyter notebook containing the analysis.
- `utils/` - Contains utility scripts, `requirements.txt`, and the Dockerfile.
- `data/` - Directory for data storage (if applicable).
- `models/` - Directory for saved models (if applicable).

## Data Sources
The data used for this analysis was sourced from [Your Data Source Here]. It encompasses various asset classes and their reactions to significant market events between [Start Date] and [End Date].

## Future Work
- In-depth analysis of individual companies like Apple, Microsoft, and AT&T.
- Exploring variable importance for model predictions.
- Integrating generative AI, especially tools like AutoGPT, for dynamic stress-testing and strategy ideation.

## Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.

## License
This project is licensed under the MIT License - see the `LICENSE.md` file for details.
