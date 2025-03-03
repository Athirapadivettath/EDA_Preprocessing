# EDA and Preprocessing

## Objective
The primary objective of this project is to design and implement a robust data preprocessing system that addresses common challenges such as missing values, outliers, inconsistent formatting, and noise. By performing effective data preprocessing, this project aims to enhance the quality, reliability, and usefulness of the data for machine learning.



## Key Components

### 1. Data Exploration
- Loaded the dataset and explored unique values in each feature.
- Performed statistical analysis.
- Renamed columns for consistency.

### 2. Data Cleaning
- Identified and treated missing values.
- Replaced `0` in the `age` column with `NaN`.
- Removed duplicate rows.
- Detected and treated outliers using the IQR method.
- Filled null values using mean/median/mode.

### 3. Data Analysis
- Filtered data with `age > 40` and `salary < 5000`.
- Plotted a scatter plot of `age` vs `salary`.
- Counted and visualized the number of people from each place.

### 4. Data Encoding
- Converted categorical variables into numerical representations using Label Encoding.

### 5. Feature Scaling
- Applied `StandardScaler` and `MinMaxScaler` to normalize numerical features.

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn


   ```

## Results and Insights
- The dataset was successfully cleaned and preprocessed for machine learning.
- Missing values were handled appropriately.
- Outliers were detected and treated.
- Data encoding and scaling were applied for better model performance.


