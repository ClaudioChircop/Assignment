import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import random
import shap
import joblib
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from mlxtend.plotting import plot_confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
# Function to load the dataset  

file_path = 'micro_world_139countries.csv'
df = pd.read_csv(file_path, encoding='ISO-8859-1')

sample_df = df[['remittances', 'educ', 'age', 'female', 'mobileowner','internetaccess', 'pay_utilities', 'receive_transfers','receive_pension', 'economy', 'regionwb','account']].sample(n=5000, random_state=42)
sample_df = sample_df.dropna(subset=['account','remittances', 'educ', 'age', 'female', 'mobileowner','internetaccess', 'pay_utilities', 'receive_transfers','receive_pension', 'economy', 'regionwb']) 
print(sample_df['regionwb'].unique)

le_country_economy = LabelEncoder()
sample_df['economy'] = le_country_economy.fit_transform(sample_df['economy'])#Giving unique int values to economies
le_region = LabelEncoder()
sample_df['regionwb'] = le_region.fit_transform(sample_df['regionwb'])#Unique int values to regions

X = sample_df.drop('account', axis=1)
y = sample_df['account']
labelencoder_y = LabelEncoder()
y= labelencoder_y.fit_transform(y)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 21)#Creating Test and Training samples, test sample = 20% of the dataset


#Creating SML Model
model = LogisticRegression()#multi_class="auto" could also work
# Fit the model to your training data
model.fit(X_train, y_train)
model.score(X_train, y_train)

true_accounts = labelencoder_y.inverse_transform(y_train)

predicted_accounts = labelencoder_y.inverse_transform(model.predict(X_train))

df = pd.DataFrame({'true_accounts': true_accounts, 'predicted_accounts': predicted_accounts})

pd.crosstab(df.true_accounts, df.predicted_accounts)
#print(classification_report(true_accounts,predicted_accounts, labels=labelencoder_y.classes_))

#print(model.score(X_test, y_test))#Final Evaluation
true_accounts = labelencoder_y.inverse_transform(y_test)
predicted_accounts = labelencoder_y.inverse_transform(model.predict(X_test))
#print(classification_report(true_accounts,predicted_accounts, labels=labelencoder_y.classes_))

model = LogisticRegression()
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')  # 5-fold cross-validation
#print("Cross-validation scores: ", scores)
#print("Average cross-validation score: ", scores.mean())
#Cross-Val Score: 0.775

#Using XGBClassifier Model
model = XGBClassifier()
model.fit(X_train, y_train)
true_accounts = labelencoder_y.inverse_transform(y_train)

predicted_accounts = labelencoder_y.inverse_transform(model.predict(X_train))

df = pd.DataFrame({'true_accounts': true_accounts, 'predicted_accounts': predicted_accounts})

pd.crosstab(df.true_accounts, df.predicted_accounts)
#print(classification_report(true_accounts,predicted_accounts, labels=labelencoder_y.classes_))
#We see using training dataset XGBoost performs better with an accuracy of 97% compared to 78% of LogisticRegression.


#print(model.score(X_test, y_test))#Final Evaluation
true_accounts = labelencoder_y.inverse_transform(y_test)
predicted_accounts = labelencoder_y.inverse_transform(model.predict(X_test))
#print(classification_report(true_accounts,predicted_accounts, labels=labelencoder_y.classes_))

model = XGBClassifier()
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')  # 5-fold cross-validation
#print("Cross-validation scores: ", scores)
#print("Average cross-validation score: ", scores.mean())
#Cross Val Score = 0.824
#Using Test dataset XBoost = 83% accuracy, LogisticRegression = 79%

#Hyperparameter tuning
model_xgb = XGBClassifier()
model_xgb.fit(X_train, y_train)
#print('Model LG' + ' ' + str(model_lg.score(X_test, y_test)))
#print('Model XGB' + ' ' + str(model_xgb.score(X_test, y_test)))
scorer = make_scorer(mean_squared_error)

#Define the parameter 
parameters_xgb = {'n_estimators': [100, 200, 300],'max_depth': [3, 5, 7],'learning_rate': [0.01, 0.1, 0.3]}
# Perform grid search on the classifier using 'scorer' as the scoring method.
grid_obj = GridSearchCV(model_xgb, parameters_xgb, scoring=scorer)
grid_fit = grid_obj.fit(X, y)
# Get the estimator.
best_reg = grid_fit.best_estimator_

# Fit the new model.
best_reg.fit(X_train, y_train)
best_reg.score(X_test, y_test)
#print(best_reg.score(X_test, y_test))
#After Hyperameter tuning we find the XGBoost had a score of 0.786

#Evaluating  Model
# Generate predictions for the test set
y_pred = best_reg.predict(X_test)

# If this is a binary classification problem, you'll need the predicted probabilities for ROC-AUC
y_pred_proba = best_reg.predict_proba(X_test)[:, 1]

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
# Precision
precision = precision_score(y_test, y_pred)
# Recall
recall = recall_score(y_test, y_pred)
# F1 Score
f1 = f1_score(y_test, y_pred)
# ROC-AUC Score (for binary classification)
roc_auc = roc_auc_score(y_test, y_pred_proba)
# Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
# Print the results
#print(f"Accuracy: {accuracy:.4f}")
#print(f"Precision: {precision:.4f}")
#print(f"Recall: {recall:.4f}")
#print(f"F1 Score: {f1:.4f}")
#print(f"ROC-AUC Score: {roc_auc:.4f}")
#print(f"Mean Squared Error: {mse:.4f}")

#Plotting Confusion Matrix
# Generate predictions
y_pred = best_reg.predict(X_test)

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=labelencoder_y.classes_, yticklabels=labelencoder_y.classes_, annot_kws={"size": 10})
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.xticks(rotation=45, fontsize=12)  # Rotate x-axis labels
plt.yticks(rotation=0, fontsize=12)  # Rotate y-axis labels
plt.tight_layout()
#plt.show()
#Our model is 90% accurate at predicting when True label for account = true, but inaccurate when True Label for account = false.


# Define the SHAP explainer
explainer_shap = shap.Explainer(model_xgb)

# Calculate SHAP values for test and train sets
shap_values_test = explainer_shap(X_test)
shap_values_train = explainer_shap(X_train)

# Convert SHAP values to DataFrame
df_shap_test = pd.DataFrame(shap_values_test.values, columns=sample_df.columns.drop('account'))
df_shap_train = pd.DataFrame(shap_values_train.values, columns=sample_df.columns.drop('account'))

# Display the first 10 rows of SHAP values for the test set
#print(df_shap_test.head(10))

# Identify categorical features based on the number of unique values
categorical_features = np.argwhere(np.array([len(set(X_train[:, x])) for x in range(X_train.shape[1])]) <= 10).flatten()

# Create a summary plot for SHAP values of the training set
shap.summary_plot(shap_values_train.values, X_train, feature_names=sample_df.columns.drop('account'))

joblib.dump(model_xgb, 'xgb_clf.joblib')
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(labelencoder_y, 'encoder.joblib')
joblib.dump(le_country_economy, 'country_encoder.joblib')
joblib.dump(le_region, 'regionwb_encoder.joblib')

