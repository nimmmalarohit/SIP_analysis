# Open Command Prompt as Administrator
# Navigate to your project directory
cd path\to\your\project

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
venv\Scripts\activate

# Install required packages
pip install pandas numpy matplotlib seaborn scikit-learn ta jinja2
pip install xgboost catboost prophet tensorflow statsmodels

# Run the enhanced script
python optimal_investment_day_calendar.py --input data --output results