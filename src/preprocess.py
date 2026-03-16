import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess(path="data/data.csv"):
    df = pd.read_csv(path)
    df = df.drop(columns=['Unnamed: 7','Unnamed: 8'], errors='ignore')
    df.columns = df.columns.str.strip()

    # Scale numeric features
    scaler = StandardScaler()
    cols = ['Ia','Ib','Ic','Va','Vb','Vc']
    for col in cols:
        df[col] = scaler.fit_transform(df[[col]])

    X = df.drop('Output (S)', axis=1)
    y = df['Output (S)']

    return train_test_split(X, y, test_size=0.3, random_state=42)