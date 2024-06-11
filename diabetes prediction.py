from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import pandas as pd
import gradio as gr
import warnings
warnings.filterwarnings('ignore')
data = pd.read_csv('diabetes.csv')
data
data.head()
print(data.columns)
data.shape
data.isnull().sum()
data.hist(figsize=(20, 20))
p = msno.bar(data)
plt.subplot(121), sns.distplot(data['Insulin'])
plt.subplot(122), data['Insulin'].plot.box(figsize=(16, 5))
plt.show()
p = sns.heatmap(data.corr(), annot=True, cmap='RdYlGn')
print(data.columns)
X = data.drop('Outcome', axis=1)
y = data['Outcome']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=7)
# Perform feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Train the model
model = LogisticRegression()
model.fit(X_scaled, y)
# Calculate accuracy using cross-validation
accuracy = np.mean(cross_val_score(model, X_scaled, y, cv=5,
                   scoring='accuracy'))  # 5 partitions in parameter
print(f"Model Accuracy: {accuracy:.2f}")
# Define the prediction function


def predict_diabetes(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, pedigree_function, age):
    # Prepare the input data as a numpy array
    input_data = np.array([[pregnancies, glucose, blood_pressure,
                          skin_thickness, insulin, bmi, pedigree_function, age]])
    # Scale the input data
    input_data_scaled = scaler.transform(input_data)
    # Make predictions
    predicted_class = model.predict(input_data_scaled)
    # Return the prediction
    if predicted_class[0] == 0:
        return 'No Diabetes'
    else:
        return 'Diabetes'


# Create input and output interfaces
inputs = ['text'] * 8
outputs = 'text'
# Create the web interface
interface = gr.Interface(fn=predict_diabetes, inputs=inputs,
                         outputs=outputs, title='Diabetes Predictor')
# Launch the web interface
interface.launch()
