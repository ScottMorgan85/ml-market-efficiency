@echo off

:: Create a new conda environment
conda create --name MarketEfficiencyEnv  python=3.8 -y

:: Activate the environment
conda activate MarketEfficiencyEnv 

:: Install the packages from the requirements.txt file
pip install -r "C:\Users\Scott Morgan\Documents\GitHub\ml-market-efficiency\requirements.txt"

:: Install pip in the new environment (to use it for packages not available in conda)
conda install pip -y

:: Install the requirements
pip install -r requirements.txt

::Install xbbg and blpapi (Bloomberg specific packages)
pip install xbbg
pip install blpapi --index-url=https://bcms.bloomberg.com/pip/simple/
pip install ipykernel

::Install compatible CUDA and CuDNN versions
conda install cudatoolkit=11.2 cudnn=8.1 -y

:: Add the environment as a kernel to Jupyter
python -m ipykernel install --user --name MarketEfficiencyEnv --display-name "Market Efficiency Kernel"

::Verify xgboost is using GPU
python -c "import xgboost; print(xgboost.__version__); print(xgboost.runtime_capabilities())"

::Verify TensorFlow's GPU usage
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

@echo.
@echo Setup completed successfully. Your new kernel "Market Efficiency Kernel" is now available in Jupyter!

