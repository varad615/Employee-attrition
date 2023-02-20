import seaborn as sn
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow import keras
import tensorflow as tf
import time
import streamlit as st
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


csv_file = 'data.csv'
sheet_name = 'DATA'
df = pd.read_csv('data.csv')
ndf = pd.read_csv('data.csv')

df.drop('EmployeeNumber', axis='columns', inplace=True)


def print_unique_col_values(df):
    for column in df:
        if df[column].dtypes == 'object':
            print(f'{column}: {df[column].unique()}')


print_unique_col_values(df)

yes_no_columns = ['Attrition', 'OverTime']
for col in yes_no_columns:
    df[col].replace({'Yes': 1, 'No': 0}, inplace=True)

for col in df:
    print(f'{col}: {df[col].unique()}')

df['Gender'].replace({'Female': 1, 'Male': 0}, inplace=True)
df.Gender.unique()

df2 = pd.get_dummies(
    data=df,
    columns=['Department', 'EducationField', 'JobRole', 'MaritalStatus'])

cols_to_scale = [
    'Age', 'DistanceFromHome', 'Education', 'EnvironmentSatisfaction',
    'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'MonthlyRate',
    'NumCompaniesWorked', 'PercentSalaryHike', 'PerformanceRating',
    'Work Environment', 'Work Accident', 'TotalWorkingYears',
    'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany',
    'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager'
]
scaler = MinMaxScaler()
df2[cols_to_scale] = scaler.fit_transform(df2[cols_to_scale])
for col in df2:
    print(f'{col}: {df2[col].unique()}')

X = df2.drop('Attrition', axis='columns')
y = df2['Attrition']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=5)
# X_train.shape
# X_test.shape
# X_train[:1]
print(len(X_train.columns))


model = keras.Sequential([
    keras.layers.Dense(43, input_shape=(43,), activation='relu'),
    keras.layers.Dense(25, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=50)

model.evaluate(X_test, y_test)

yp = model.predict(X_test)
# yp[:5]

y_pred = []
for element in yp:
    if element > 0.5:
        y_pred.append(1)
    else:
        y_pred.append(0)

# y_pred[:10]

# y_test[:]


print(classification_report(y_test, y_pred))

cm = tf.math.confusion_matrix(labels=y_test, predictions=y_pred)

plt.figure(figsize=(10, 7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')

st.sidebar.header("Employee Attrition")
attrition = st.sidebar.multiselect(
    "Employee Number",
    options=ndf["EmployeeNumber"].unique()
)
num_yrs = st.sidebar.slider('Select number values', min_value=1, max_value=50)
st.sidebar.write('Values:', num_yrs)
ds = ndf.query(
    "EmployeeNumber == @attrition"
)
ds.drop(columns=["Education", "Age", "Department", "JobLevel", "JobRole", "JobSatisfaction", "MaritalStatus", "MonthlyRate", "NumCompaniesWorked", "OverTime",
        "PercentSalaryHike", "PerformanceRating", "Work Environment", "Work Accident", "TotalWorkingYears", "TrainingTimesLastYear", "WorkLifeBalance", "YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager", "Gender", "EducationField", "EnvironmentSatisfaction", "DistanceFromHome", "JobInvolvement"], axis=1, inplace=True)
def color_coding(row):
    return ['background-color:green'] * len(
        row) if row.Attrition == 'No' else ['background-color:red'] * len(row)
st.dataframe(ds.style.apply(color_coding, axis=1))
