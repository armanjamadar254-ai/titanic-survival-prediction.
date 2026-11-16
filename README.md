
# Titanic Survival Prediction

Predicting which passengers survived using machine learning.

## What I did
- Cleaned data (filled missing ages, converted text to numbers)
- Created features: family size, is_alone
- Compared Decision Tree, KNN, and Random Forest using cross-validation
- Tuned Random Forest with GridSearchCV

## Results
- Baseline Random Forest: 80.4%
- Tuned Random Forest: 82.6%

## What I learned
- Scaling data fixed KNN (69% to 82%)
- Limiting model complexity reduced overfitting
- Not every new feature helps

## Tools
Python, pandas, scikit-learn, Google Colab

## Code
```python
import seaborn as sns
import pandas as pd
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

df = sns.load_dataset("titanic")
df = df[["survived","pclass","sex","age","sibsp","parch","fare","embarked","deck"]].copy()

df["has_deck"] = df["deck"].notna().astype(int)
df = df.drop("deck", axis=1)
df["age"] = df["age"].fillna(df["age"].median())
df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])
df["sex"] = df["sex"].map({"male": 0, "female": 1})
df["family_size"] = df["sibsp"] + df["parch"] + 1
df["is_alone"] = (df["family_size"] == 1).astype(int)
df = pd.get_dummies(df, columns=["embarked"], dtype=int)

X = df.drop("survived", axis=1)
y = df["survived"]

print("New features:", cross_val_score(RandomForestClassifier(random_state=42), X, y, cv=5).mean())

params = {
    "n_estimators": [100, 300],
    "max_depth": [4, 6, 8, None],
    "min_samples_leaf": [1, 3, 5],
}
grid = GridSearchCV(RandomForestClassifier(random_state=42), params, cv=5)
grid.fit(X, y)
print("Best settings:", grid.best_params_)
print("Tuned score:", grid.best_score_)
```