# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer

# 1. Import and preprocess data
# Load Titanic dataset from seaborn (no file path needed)
df = sns.load_dataset('titanic')

# Handle missing values
num_imputer = SimpleImputer(strategy='median')
df[['age', 'fare']] = num_imputer.fit_transform(df[['age', 'fare']])

cat_imputer = SimpleImputer(strategy='most_frequent')
df[['embarked']] = cat_imputer.fit_transform(df[['embarked']])

# Encode categorical features
df['sex'] = df['sex'].map({'male': 0, 'female': 1})
df = pd.get_dummies(df, columns=['embarked'], drop_first=True)

# 2. Generate summary statistics
print("\nSummary Statistics:")
print(df[['survived', 'pclass', 'age', 'sibsp', 'parch', 'fare']].describe())

# 3. Create histograms and boxplots for numeric features
numeric_features = ['age', 'fare', 'pclass', 'sibsp', 'parch']

plt.figure(figsize=(15, 10))
for i, feature in enumerate(numeric_features, 1):
    # Histogram
    plt.subplot(5, 2, 2*i-1)
    sns.histplot(df[feature], kde=True)
    plt.title(f'Histogram of {feature}')
    
    # Boxplot
    plt.subplot(5, 2, 2*i)
    sns.boxplot(x=df[feature])
    plt.title(f'Boxplot of {feature}')

plt.tight_layout()
plt.show()

# 4. Pairplot and correlation matrix
# Pairplot (may take time to render)
sns.pairplot(df[['survived', 'pclass', 'age', 'fare', 'sibsp', 'parch']])
plt.suptitle("Pairplot of Numerical Features", y=1.02)
plt.show()

# Correlation matrix
plt.figure(figsize=(10, 6))
corr = df[['survived', 'pclass', 'age', 'sibsp', 'parch', 'fare']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Correlation Matrix")
plt.show()

# 5. Print key observations
print("\nKey Observations from Visuals:")
print("- Survival rate is ~38% (mean of 'survived' = 0.38)")
print("- Strong negative correlation (-0.55) between fare and pclass")
print("- Age distribution is right-skewed with most passengers 20-40 years old")
print("- Fare has extreme outliers (some tickets cost > £500)")
print("- Pclass 3 had the most passengers but lowest survival rate")
