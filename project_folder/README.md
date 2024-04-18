# Salary Prediction App

## Project Description

The Salary Prediction App is a tool designed to predict salaries for data science jobs. It leverages several datasets gathered from Kaggle, which were merged, cleaned, and preprocessed to prepare them for machine learning model training. The datasets include information about data science job salaries, and various features such as job titles, locations, and experience levels. The data was processed to handle outliers, map job titles, and categorize countries into regions like "North America," "Europe," or "Other."

The machine learning model was built using a pipeline consisting of a OneHotEncoder for categorical features and a RandomForestRegressor for predictions. After training and evaluating the model's performance, the pipeline was saved to a pickle file. This pickle file was then imported into a user-friendly Streamlit app, which allows users to input their job details and receive salary predictions.

## Installation

To run the Salary Prediction App, you'll need the following dependencies:

- Python
- Jupyter
- pip-tools
- ipykernel
- Pandas
- Scikit-learn
- NumPy
- Streamlit
- Pickle
- Plotly
- Scipy

## Usage

1. **Load the dataset**: Access the provided dataset, perform exploratory data analysis (EDA) to understand the data, and select appropriate features for modeling.
2. **Create a pipeline**: Construct a pipeline with a OneHotEncoder for categorical features and a RandomForestRegressor for predictions.
3. **Evaluate model performance**: Train the model, evaluate its performance, and fine-tune parameters if necessary.
4. **Save Pipeline**: Save the trained pipeline to a pickle file for future use.
5. **Import pickle to Streamlit app**: Incorporate the pickle file containing the pipeline into the Streamlit app code.
6. **Deploy app**: Deploy the Streamlit app to allow users to predict salaries based on their input.

## File Structure

- **Data**: Contains both clean and raw datasets used in the project.
- **Notebooks**: Includes Jupyter notebooks where exploratory data analysis (EDA) was conducted, and the machine learning model was developed, fitted, and evaluated.
- **Pickles**: Stores the saved pipeline pickle file.
- **Streamlit**: Holds the Streamlit app code for the salary prediction application.
- **Pictures**: Contains plots and visualizations created during EDA.

## Credits

This project acknowledges the original Kaggle datasets used for gathering salary and job-related information.
- dataset1 = https://www.kaggle.com/datasets/arnabchaki/data-science-salaries-2023/data
- dataset2 = https://www.kaggle.com/datasets/ruchi798/data-science-job-salaries

## License

This project does not utilize any known licenses.

## Contact

For inquiries or support, please contact: sergio.pizzolante95@gmail.com

## Version History

- Version 1.0: Initial release
