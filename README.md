# IBM HR Attrition Prediction

## Overview
The **IBM HR Attrition Prediction** project aims to develop a predictive model to assess employee attrition in the workplace. Using machine learning techniques, we can identify patterns and key factors that influence employee turnover, enabling organizations to implement strategies for retaining talent.

## Features
- Predict employee attrition using a trained Random Forest Classifier.
- User-friendly interface built with Streamlit for easy interaction.
- Scalable data preprocessing with StandardScaler for normalization.
- Insights into the importance of various features affecting attrition.

## Table of Contents
1. [Technologies Used](#technologies-used)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Model Training](#model-training)
5. [Results](#results)
6. [Contributing](#contributing)
7. [License](#license)

## Technologies Used
- Python
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- Pickle

## Installation
To run this project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/IBM_HR_Attrition.git
   cd IBM_HR_Attrition
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows use `env\Scripts\activate`
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
To run the Streamlit application:
1. Ensure you have the model and scaler files in the specified directory.
2. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```
3. Open your browser and go to `http://localhost:8501` to access the application.

## Model Training
The model was trained using a dataset from IBM containing various employee attributes. The target variable is whether an employee will leave the company (attrition). The model's performance metrics include accuracy, precision, and F1 score.

## Results
- **Accuracy**: 98.38%
- **Confusion Matrix**: 
  ```
  [[240   6]
   [  2 246]]
  ```
- **F1 Score**: 98.40%
- **Precision**: 97.62%

## Contributing
Contributions are welcome! Please fork the repository and create a pull request for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.