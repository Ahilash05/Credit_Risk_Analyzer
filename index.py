import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import cross_val_score


df = pd.read_csv("synthetic_german_credit.csv")

df.head()
print(df.shape)
print(df.info())
df.describe()
print(df.columns)
df=pd.get_dummies(df,drop_first=True)


X = df.drop("default", axis=1)  
y = df["default"]


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4,random_state=42,stratify=y)
model=LogisticRegression(max_iter=5000,class_weight='balanced')
model.fit(X_train,y_train)
y_pred=model.predict(X_test)

print("Accuracy:",accuracy_score(y_test,y_pred))
print("Classification Report:\n",classification_report(y_test,y_pred))
sns.heatmap(confusion_matrix(y_test,y_pred),annot=True)
plt.title("Confusion Matrix")
plt.show()

rf_model=RandomForestClassifier(random_state=42)
rf_model.fit(X_train,y_train)
y_pred_rf=rf_model.predict(X_test)
print("RandomForestClassification Accuracy:",accuracy_score(y_test,y_pred_rf))


importances = rf_model.feature_importances_
feature_names = X.columns
sorted_indices = np.argsort(importances)[::-1]

print("\nSorted Feature Importances (High to Low):")
for idx in sorted_indices:
    print(f"{feature_names[idx]}: {importances[idx]:.4f}",flush=True)


param_grid={

    'n_estimators':[50,100],
    'max_depth':[None,5,10]
}
grid_search=GridSearchCV(rf_model,param_grid,cv=3)
grid_search.fit(X_train,y_train)
print("Best parameters:",grid_search.best_params_)


scores=cross_val_score(rf_model,X,y,cv=5)
print("Cross-validation scores:",scores)
print("Mean CV accuracy:",scores.mean())






st.title("Credit Risk Analyzer")

age = st.number_input("Age", min_value=18, max_value=100, value=30)
credit_amount = st.number_input("Credit Amount", min_value=100, max_value=10000, value=2000)

if st.button("Predict Credit Risk"):
    st.write(f"Age: {age}, Credit Amount: {credit_amount}")
    
    
    input_data = pd.DataFrame(np.zeros((1, X.shape[1])), columns=X.columns)
    
   
    if 'age' in input_data.columns:
        input_data.at[0, 'age'] = age
    if 'credit_amount' in input_data.columns:
        input_data.at[0, 'credit_amount'] = credit_amount
    
    

    # Prediction with LOGISTIC REGRESSION
    pred_logreg = model.predict(input_data)[0]
    prob_logreg = model.predict_proba(input_data)[0][1] 
    
    # Prediction with RANDOM FOREST
    pred_rf = rf_model.predict(input_data)[0]
    prob_rf = rf_model.predict_proba(input_data)[0][1]
    
    
    def risk_label(pred):
        return "High credit risk" if pred == 1 else "Good credit risk"
    
    st.write(f"Logistic Regression Prediction: {risk_label(pred_logreg)} (Probability of default: {prob_logreg:.2f})")
    st.write(f"Random Forest Prediction: {risk_label(pred_rf)} (Probability of default: {prob_rf:.2f})")
