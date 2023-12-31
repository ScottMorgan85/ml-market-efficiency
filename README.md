# Shedding Light on Market Efficiency with ML and Python

Dive deep into the world of financial markets through a data-driven lens. Using Python and machine learning models, this repository explores different levels of the Efficient Market Hypothesis (EMF) across various asset classes.

See [MarketEfficiencyLightShed.ipynb](https://github.com/ScottMorgan85/ml-market-efficiency/blob/main/MarketEfficiencyLightShed.ipynb) for analysis notebook and [Shedding Light on Market Efficiency with ML and Python](https://medium.com/@smm253/shedding-light-on-market-efficiency-with-ml-and-python-d136c4872a98?source=friends_link&sk=508b422fa01f9175b618ae3ed65e2bf7) for Medium article.

## Table of Contents
1. [Overview](#overview)
2. [Installation & Usage](#installation--usage)
3. [Project Structure](#project-structure)
4. [Data Sources](#data-sources)
5. [Future Work](#future-work)
6. [Contributing](#contributing)
7. [License](#license)

## Overview
This project tests the weak, semi-strong, and strong forms of market efficiency across different asset classes. By leveraging techniques such as AutoCorrelation, XGBoost, and Transfer Learning (Keras), we gain insights into the predictive characteristics of various asset classes.

## Installation & Usage
1. Clone this repository: `git clone https://github.com/ScottMorgan85/MarketEfficiencyLightShed.git`
3. Install dependencies: `pip install -r requirements.txt`
4. (Optional) If using Docker: Build the Docker image using the provided `Dockerfile` within the root directory.
5. Run Jupyter Notebook or Jupyter Lab to view and execute the analysis.

## Project Structure
- `MarketEfficiencyLightShed.ipynb` - Main Jupyter notebook containing the analysis.
- `Dockerfile` and `requirements.txt` - Contains environment requirements.
- `utils/` - Contains utility scripts.
- `data/` - Directory for data storage .
- `models/` - Directory for saved models.

## Data Sources
The data used for this analysis was sourced from Yahoo!Finance and Bloomberg. It encompasses the index prices and returns of various market indices between 4/2/2007 and 8/25/2023.

## Future Work
- Exploring variable importance for model predictions.
- In-depth analysis of individual companies like Apple, Microsoft, and AT&T.
- Integrating generative AI, such as AutoGPT, for dynamic stress-testing and strategy ideation.

## Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.
