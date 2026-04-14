
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import RocCurveDisplay
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
import optuna
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import StackingClassifier




#---- Loading Data ----- 
df = pd.read_csv("csv/cs-training.csv")
df.head(5)

# Drop the default index column as it provides no predictive value
df = df.drop(columns=["Unnamed: 0"])

df.shape
df.columns

# Analyze the count and percentage of null values per column.
df.isnull().sum()

#We analyze how much those Nan represent.
(df.isnull().sum() / len(df)) * 100


#--- Missing Values and Data Imputation ----
df["MonthlyIncome"] = df["MonthlyIncome"].fillna(df["MonthlyIncome"].median())
df["NumberOfDependents"] = df["NumberOfDependents"].fillna(0)

df.isnull().sum()

#target variable 
df["SeriousDlqin2yrs"].value_counts()
df["SeriousDlqin2yrs"].value_counts().plot(kind="bar")

#separate variables and target
X = df.drop("SeriousDlqin2yrs", axis=1)
y = df["SeriousDlqin2yrs"]

# 80/20 split. The 'stratify=y' parameter is CRUCIAL here to ensure the same proportion of defaulters (class 1) is maintained in Train and Test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

y_train.value_counts(normalize=True)
y_test.value_counts(normalize=True)


# --- Baseline Model ---
baseline_model = LogisticRegression(max_iter=1000, random_state=42)

baseline_model.fit(X_train, y_train)

# This makes predictions and calculates the ROC-AUC score
y_proba_baseline = baseline_model.predict_proba(X_test)[:, 1]

print("=== BASELINE MODEL ===")
print(classification_report(y_test, y_pred_baseline))
print("ROC-AUC:", roc_auc_score(y_test, y_proba_baseline))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred_baseline)
plt.title("Baseline Model - Confusion Matrix")
plt.show()




# --- Class Weighting --- 
# A second logistic regression model was trained using class_weight="balanced" to penalize errors in the minority class more strongly. This helps the model pay more attention to default cases.
balanced_model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    class_weight="balanced"
)

balanced_model.fit(X_train, y_train)

y_pred_balanced = balanced_model.predict(X_test)
y_proba_balanced = balanced_model.predict_proba(X_test)[:, 1]

print("=== CLASS WEIGHT = BALANCED ===")
print(classification_report(y_test, y_pred_balanced))
print("ROC-AUC:", roc_auc_score(y_test, y_proba_balanced))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred_balanced)
plt.title("Balanced Logistic Regression - Confusion Matrix")
plt.show()




#--- SMOTE --- 
#Apply SMOTE ONLY to training data
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print("Original distribution in y_train:")
print(y_train.value_counts())

print("\nDistribution after SMOTE:")
print(y_train_smote.value_counts())


# Smote + Logistic Regression
smote_model = LogisticRegression(max_iter=1000, random_state=42)

smote_model.fit(X_train_smote, y_train_smote)

y_pred_smote = smote_model.predict(X_test)
y_proba_smote = smote_model.predict_proba(X_test)[:, 1]

print("=== LOGISTIC REGRESSION + SMOTE ===")
print(classification_report(y_test, y_pred_smote))
print("ROC-AUC:", roc_auc_score(y_test, y_proba_smote))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred_smote)
plt.title("Logistic Regression with SMOTE - Confusion Matrix")
plt.show()


#--- Comparison of Models ---
results = pd.DataFrame({
    "Model": ["Baseline", "Class Weight Balanced", "SMOTE"],
    "Precision (class 1)": [
        precision_score(y_test, y_pred_baseline),
        precision_score(y_test, y_pred_balanced),
        precision_score(y_test, y_pred_smote)
    ],
    "Recall (class 1)": [
        recall_score(y_test, y_pred_baseline),
        recall_score(y_test, y_pred_balanced),
        recall_score(y_test, y_pred_smote)
    ],
    "F1-score (class 1)": [
        f1_score(y_test, y_pred_baseline),
        f1_score(y_test, y_pred_balanced),
        f1_score(y_test, y_pred_smote)
    ],
    "ROC-AUC": [
        roc_auc_score(y_test, y_proba_baseline),
        roc_auc_score(y_test, y_proba_balanced),
        roc_auc_score(y_test, y_proba_smote)
    ]
})

print(results)


RocCurveDisplay.from_predictions(y_test, y_proba_baseline, name="Baseline")
RocCurveDisplay.from_predictions(y_test, y_proba_balanced, name="Class weight")
RocCurveDisplay.from_predictions(y_test, y_proba_smote, name="SMOTE")

plt.plot([0,1],[0,1],'k--')
plt.title("ROC Curve Comparison")
plt.show()


#--- Threshold Tuning for the Best Performing Model --- 
thresholds = np.arange(0.10, 0.91, 0.05)

results = []

for t in thresholds:
    y_pred_thr = (y_proba_balanced >= t).astype(int) #Turning probabilities into predictions

    results.append({
        "threshold": t,
        "accuracy": accuracy_score(y_test, y_pred_thr),
        "precision_class1": precision_score(y_test, y_pred_thr, zero_division=0), #Of all those you predict as default, how many actually are?
        "recall_class1": recall_score(y_test, y_pred_thr, zero_division=0), #Of all the actual defaults, how many does the model detect?
        "f1_class1": f1_score(y_test, y_pred_thr, zero_division=0)
    })

threshold_df = pd.DataFrame(results)
threshold_df


# --- SMOTE + ENN Combined Resampling --- 
# Apply SMOTE + ENN to the training data

smote_enn = SMOTEENN(random_state=42)
X_train_smoteenn, y_train_smoteenn = smote_enn.fit_resample(X_train, y_train)

print("Original training distribution:")
print(y_train.value_counts())

print("\nDistribution after SMOTE + ENN:")
print(y_train_smoteenn.value_counts())


# Train logistic regression on the resampled data
smoteenn_model = LogisticRegression(max_iter=1000, random_state=42)

smoteenn_model.fit(X_train_smoteenn, y_train_smoteenn)

# Predictions on the test set
y_pred_smoteenn = smoteenn_model.predict(X_test)
y_proba_smoteenn = smoteenn_model.predict_proba(X_test)[:, 1]
print("=== LOGISTIC REGRESSION + SMOTE + ENN ===")
print(classification_report(y_test, y_pred_smoteenn))
print("ROC-AUC:", roc_auc_score(y_test, y_proba_smoteenn))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred_smoteenn)
plt.title("Logistic Regression with SMOTE + ENN - Confusion Matrix")
plt.show()

print("Precision class 1:", precision_score(y_test, y_pred_smoteenn))
print("Recall class 1:", recall_score(y_test, y_pred_smoteenn))
print("F1 class 1:", f1_score(y_test, y_pred_smoteenn))



#--- Random Under Sampling ---
rus = RandomUnderSampler(random_state=42)
X_train_under, y_train_under = rus.fit_resample(X_train, y_train)

print("Distribution after undersampling:")
print(y_train_under.value_counts())

under_model = LogisticRegression(max_iter=1000, random_state=42)
under_model.fit(X_train_under, y_train_under)

y_pred_under = under_model.predict(X_test)
print(classification_report(y_test, y_pred_under))

balanced_accuracy_score(y_test, y_pred_under)



# --- Train/Set Split and Missing Values Imputation ---


# 1. Separate predictor variables (X) from target variable (y)
X = df.drop('SeriousDlqin2yrs', axis=1) 
y = df['SeriousDlqin2yrs']              

# 2. Train/Test Split (80/20) with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Configure KNN Imputer
print("Starting KNN imputation (this might take a few minutes due to the dataset size)...")
imputer = KNNImputer(n_neighbors=5)
columnas = X_train.columns 

# 4. Fit on Train and transform Train (avoiding data leakage)
X_train_imputed = pd.DataFrame(
    imputer.fit_transform(X_train), 
    columns=columnas, 
    index=X_train.index
)

# 5. Transform Test using the fitted imputer
X_test_imputed = pd.DataFrame(
    imputer.transform(X_test), 
    columns=columnas, 
    index=X_test.index
)

print("\nImputation completed successfully!")
print("Remaining nulls in X_train:\n", X_train_imputed[['MonthlyIncome', 'NumberOfDependents']].isnull().sum())
print("Remaining nulls in X_test:\n", X_test_imputed[['MonthlyIncome', 'NumberOfDependents']].isnull().sum())


# --- Outlier Handling --- 
# Select columns that are known to have extreme outliers based on the EDA
outlier_cols = ['RevolvingUtilizationOfUnsecuredLines', 'age', 'DebtRatio', 'MonthlyIncome']

print("Applying Winsorization (capping at 1st and 99th percentiles)...")

for col in outlier_cols:
    # 1. Calculate limits ONLY on the Training set to avoid data leakage
    lower_limit = X_train_imputed[col].quantile(0.01)
    upper_limit = X_train_imputed[col].quantile(0.99)
    
    # 2. Clip (cap) the values in both Train and Test sets
    X_train_imputed[col] = X_train_imputed[col].clip(lower=lower_limit, upper=upper_limit)
    X_test_imputed[col] = X_test_imputed[col].clip(lower=lower_limit, upper=upper_limit)

print("Winsorization completed!")

# Show the new minimum and maximum values to verify
print("\nNew Min & Max values in Train:")
for col in outlier_cols:
    print(f"{col}: Min = {X_train_imputed[col].min():.2f}, Max = {X_train_imputed[col].max():.2f}")



#--- Ensemble Learning --- 
bagging_model = BaggingClassifier(
    estimator=DecisionTreeClassifier(max_depth=5, random_state=42), #Each model within the Bagging will be a decision tree
    n_estimators=200, #200 different trees
    max_samples=1.0,
    bootstrap=True,    #sampling with replacement
    random_state=42,
    n_jobs=-1 #uses all cores
)


bagging_model.fit(X_train_imputed, y_train)   #new line: we train the model with the imputed and outlier-treated data, because the model cannot handle missing values or extreme outliers.

y_pred_bag = bagging_model.predict(X_test_imputed) #new line: we make predictions with the trained model using the test data that has also been imputed and treated for outliers.
y_proba_bag = bagging_model.predict_proba(X_test_imputed)[:, 1] #new line: we get the probabilities of the positive class (default) to calculate the ROC-AUC.

print("=== BAGGING CLASSIFIER ===")
print(classification_report(y_test, y_pred_bag))
print("ROC-AUC:", roc_auc_score(y_test, y_proba_bag))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred_bag)
plt.title("Bagging Classifier - Confusion Matrix")
plt.show()



adaboost_model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1, random_state=42), #simple tree
    n_estimators=200, #200 trees
    learning_rate=0.5, #Control how much influence each model has.
    random_state=42
)

adaboost_model.fit(X_train_imputed, y_train)   #new line: we train the model with the imputed and outlier-treated data, because the model cannot handle missing values or extreme outliers.

y_pred_ada = adaboost_model.predict(X_test_imputed)    #new line: we make predictions with the trained model using the test data that has also been imputed and treated for outliers.
y_proba_ada = adaboost_model.predict_proba(X_test_imputed)[:, 1] #new line: we get the probabilities of the positive class (default) to calculate the ROC-AUC.

print("=== ADABOOST CLASSIFIER ===")
print(classification_report(y_test, y_pred_ada))
print("ROC-AUC:", roc_auc_score(y_test, y_proba_ada))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred_ada)
plt.title("AdaBoost Classifier - Confusion Matrix")
plt.show()



comparison_ensemble = pd.DataFrame([
    {
        "Model": "Bagging",
        "Accuracy": accuracy_score(y_test, y_pred_bag),
        "Precision class 1": precision_score(y_test, y_pred_bag, zero_division=0),
        "Recall class 1": recall_score(y_test, y_pred_bag, zero_division=0),
        "F1 class 1": f1_score(y_test, y_pred_bag, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_proba_bag)
    },
    {
        "Model": "AdaBoost",
        "Accuracy": accuracy_score(y_test, y_pred_ada),
        "Precision class 1": precision_score(y_test, y_pred_ada, zero_division=0),
        "Recall class 1": recall_score(y_test, y_pred_ada, zero_division=0),
        "F1 class 1": f1_score(y_test, y_pred_ada, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_proba_ada)
    }
])

comparison_ensemble



# --- Optimization of Hyperparameters with Optuna ---

def objective(trial):
    #define the hyperparameters to optimize
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.5, log=True)
    max_depth = trial.suggest_int('max_depth', 1, 5)
    
    model = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=max_depth, random_state=42),
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=42
    )
    
    #Optimize using cross-validation on the training set to get a more robust estimate of performance
    score = cross_val_score(model, X_train_imputed, y_train, cv=3, scoring='roc_auc').mean()
    return score

print("\Inizialitation Optimization with Optuna...")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20) 

print(f"\Best parameters founded: {study.best_params}")

# We train a final AdaBoost model using the best hyperparameters found by Optuna
best_p = study.best_params
final_ada_model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=best_p['max_depth'], random_state=42),
    n_estimators=best_p['n_estimators'],
    learning_rate=best_p['learning_rate'],
    random_state=42
)
final_ada_model.fit(X_train_imputed, y_train)

# Evaluate the optimized model on the test set
y_proba_opt = final_ada_model.predict_proba(X_test_imputed)[:, 1]
print(f"\nROC-AUC Final with Optuna: {roc_auc_score(y_test, y_proba_opt):.4f}")



#--- Cost Sensitive Learning ---

# We penalize missing a default (1) ten times more than a false alarm (0)
custom_weights = {0: 1, 1: 10} 

cost_model = LogisticRegression(
    max_iter=1000, 
    random_state=42, 
    class_weight=custom_weights
)

# Training with the imputed and cleaned data
cost_model.fit(X_train_imputed, y_train)

# Evaluation
y_pred_cost = cost_model.predict(X_test_imputed)
y_proba_cost = cost_model.predict_proba(X_test_imputed)[:, 1]

print("=== COST-SENSITIVE LOGISTIC REGRESSION (Ratio 1:10) ===")
print(classification_report(y_test, y_pred_cost))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba_cost):.4f}")


#--- Stacking Classifier ---


# 1. Defining the Base Learners
base_learners = [
    ('ada_optimized', final_ada_model),
    ('log_balanced', balanced_model)
]

# 2. Defining the Meta-Learner (the final judge)
stacking_clf = StackingClassifier(
    estimators=base_learners,
    final_estimator=LogisticRegression(),
    cv=5 
)

print("Training Stacking Classifier...")
stacking_clf.fit(X_train_imputed, y_train)

# 3. Evaluation
y_pred_stack = stacking_clf.predict(X_test_imputed)
y_proba_stack = stacking_clf.predict_proba(X_test_imputed)[:, 1]

print("=== STACKING CLASSIFIER PERFORMANCE ===")
print(classification_report(y_test, y_pred_stack))
print(f"Stacking ROC-AUC: {roc_auc_score(y_test, y_proba_stack):.4f}")