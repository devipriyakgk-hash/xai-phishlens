"""
Data loading and preprocessing for XAI-PhishGuard.
Dataset: 10,000 URLs (5,000 phishing / 5,000 legitimate), 16 engineered
URL/domain/HTML features derived from the UCI Phishing Websites feature set.
"""
import pandas as pd
from sklearn.model_selection import train_test_split

FEATURE_COLS = [
    'Have_IP', 'Have_At', 'URL_Length', 'URL_Depth', 'Redirection',
    'https_Domain', 'TinyURL', 'Prefix/Suffix', 'DNS_Record', 'Web_Traffic',
    'Domain_Age', 'Domain_End', 'iFrame', 'Mouse_Over', 'Right_Click',
    'Web_Forwards'
]
TARGET_COL = 'Label'


def load_data(path='data/urldata.csv'):
    df = pd.read_csv(path)
    return df


def get_train_test(path='data/urldata.csv', test_size=0.2, random_state=42):
    df = load_data(path)
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = get_train_test()
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Train label balance:\n{y_train.value_counts(normalize=True)}")
