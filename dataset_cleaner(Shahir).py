import pandas as pd

df = pd.read_csv("heart_2020_cleaned.csv")

df['HeartDisease'] = df['HeartDisease'].map({'Yes': 1, 'No': 0})
df = df.rename(columns={"HeartDisease": "Cardiovascular Disease"}) 

df['BMI'] = df['BMI'].astype(int)

df['Smoking'] = df['Smoking'].map({'Yes': 1, 'No': 0})
df = df.rename(columns={"Smoking": "Smoking Status"})

df['AlcoholDrinking'] = df['AlcoholDrinking'].map({'Yes': 1, 'No': 0})
df = df.rename(columns={"AlcoholDrinking": "Alcohol Intake"})

df['PhysicalActivity'] = df['PhysicalActivity'].map({'Yes': 1, 'No': 0})
df = df.rename(columns={"PhysicalActivity": "Physical Activity"}) 

age_midpoints = {
    "18-24": 21,
    "25-29": 27,
    "30-34": 32,
    "35-39": 37,
    "40-44": 42,
    "45-49": 47,
    "50-54": 52,
    "55-59": 57,
    "60-64": 62,
    "65-69": 67,
    "70-74": 72,
    "75-79": 77,
    "80 or older": 80
}
df['AgeCategory'] = df['AgeCategory'].map(age_midpoints)
df = df.rename(columns={"AgeCategory": "Age"})

df['Sex'] = df['Sex'].map({'Female': 1, 'Male': 2})
df = df.rename(columns={"Sex": "Gender"}) 

df.drop(df.columns[[4, 5, 6, 7, 10, 11, 13, 14, 15, 16, 17]], axis=1, inplace=True)

df.to_csv("Dataset_A.csv", index = False)