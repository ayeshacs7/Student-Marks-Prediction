import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Load Dataset
df = pd.read_csv("student_marks.csv")

print(df.head())

# Features
X = df[['Hours_Studied', 'Attendance', 'Previous_Marks']]

# Target
y = df['Final_Marks']

# Train Model
model = LinearRegression()

model.fit(X, y)

# Save Model
pickle.dump(model, open("model.pkl", "wb"))

print("Model Trained Successfully!")
