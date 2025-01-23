#!/usr/bin/env python
# coding: utf-8

# # Predictive Credit Risk Analysis and Loan Portfolio Optimization

# In[ ]:





# In[1]:


import pandas as pd
import requests
from io import StringIO
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_score


# In[2]:


dataset_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"


# In[3]:


# Fetch the dataset
response = requests.get(dataset_url)
response.raise_for_status()  # Ensure we notice bad responses

# Load the dataset into a pandas DataFrame
column_names = [
    'Status_of_existing_checking_account', 'Duration_in_month', 'Credit_history', 'Purpose',
    'Credit_amount', 'Savings_account_bonds', 'Present_employment_since', 'Installment_rate_in_percentage_of_disposable_income',
    'Personal_status_and_sex', 'Other_debtors_guarantors', 'Present_residence_since', 'Property',
    'Age_in_years', 'Other_installment_plans', 'Housing', 'Number_of_existing_credits_at_this_bank',
    'Job', 'Number_of_people_being_liable_to_provide_maintenance_for', 'Telephone', 'Foreign_worker', 'Target'
]

# Read the dataset into a DataFrame
data = pd.read_csv(StringIO(response.text), sep=' ', header=None, names=column_names)


# In[4]:


# Display the first few rows
print(data.head())

# Display basic information
print(data.info())

# Check for missing values
print(data.isnull().sum())


# In[5]:


# Map target variable to descriptive labels
data['Target'] = data['Target'].map({1: 'Good', 2: 'Bad'})


# In[6]:


df = data


# ## Now we do some EDA (Exploratory Data Analysis) y'all! <3 

# In[7]:


# Shape of the DataFrame
print(f"Dataset Shape: {df.shape}")

# Basic info about data types and non-null counts
print("\nDataset Info:")
df.info()

# Descriptive statistics for numerical features
print("\nSummary Statistics:")
print(df.describe())

# Check unique values in each column
print("\nUnique Values per Column:")
for column in df.columns:
    print(f"{column}: {df[column].nunique()} unique values")

# And finally, the head
df.head()


# In[8]:


# Check for missing values
missing_values = df.isnull().sum()
print("\nMissing Values in Each Column:")
print(missing_values)

# Check for duplicates
duplicates = df.duplicated().sum()
print(f"\nNumber of Duplicate Rows: {duplicates}")


# In[9]:


import matplotlib.pyplot as plt
import seaborn as sns

def plot_all_columns(df):
    """
    Generates appropriate plots for all columns in the DataFrame.
    - Histograms for numerical columns.
    - Count plots for categorical columns.
    
    Parameters:
        df (DataFrame): The DataFrame to plot.
    """
    for column in df.columns:
        plt.figure(figsize=(8, 6))
        
        # Check column type
        if df[column].dtype in ['int64', 'float64']:
            # Numerical column: plot histogram
            sns.histplot(df[column].dropna(), kde=True, bins=20)
            plt.title(f'Distribution of {column}')
            plt.xlabel(column)
            plt.ylabel('Frequency')
        else:
            # Categorical column: plot count plot
            sns.countplot(x=column, data=df)
            plt.title(f'Count Plot of {column}')
            plt.xlabel(column)
            plt.ylabel('Count')
        
        # Show the plot
        plt.tight_layout()
        plt.show()


# In[10]:


plot_all_columns(df)


# In[11]:


# Correlation heatmap (for numerical features only)
numerical_features = df.select_dtypes(include=[np.number]).columns
correlation_matrix = df[numerical_features].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Correlation Heatmap of Numerical Features")
plt.show()


# ## Data processing

# ### We don't have missing values as far as I can see, but I still leave the code here to deal with missing values. Change this from markdown to code, and comment out the text if needed:
# 
# Handle missing values
# For numerical features: Fill with the median
# numerical_features = df.select_dtypes(include=[np.number]).columns
# df[numerical_features] = df[numerical_features].fillna(df[numerical_features].median())
# 
# For categorical features: Fill with the mode
# categorical_features = df.select_dtypes(include=['object']).columns
# df[categorical_features] = df[categorical_features].fillna(df[categorical_features].mode().iloc[0])
# 
# Confirm no missing values remain
# print("Missing values after imputation:")
# print(df.isnull().sum().sum())  # Should output 0
# 

# Now let's encode the whole df so that the modells will know what to do with it:

# In[12]:


# Identify categorical features
categorical_features = df.select_dtypes(include=['object']).columns

# One-hot encoding for categorical variables
df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)

print("Shape after encoding:", df_encoded.shape)


# In[13]:


from sklearn.preprocessing import StandardScaler

# Standardize numerical features
scaler = StandardScaler()
df_encoded[numerical_features] = scaler.fit_transform(df_encoded[numerical_features])

print("Sample of scaled data:")
print(df_encoded.head())


# In[14]:


from sklearn.model_selection import train_test_split

# Define features (X) and target (y)
X = df_encoded.drop(columns=['Target_Good']) 
y = df_encoded['Target_Good']

# Split the data
df_X_train, df_X_test, df_y_train, df_y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {df_X_train.shape}")
print(f"Test set size: {df_X_test.shape}")


# In[15]:


df = df_encoded


# ### Here is the method I use to try and find the best method with tests:

# In[16]:


def find_best_model_classification(df_X_train, df_X_test, df_y_train, df_y_test):
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression, RidgeClassifier, PassiveAggressiveClassifier, LinearRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                                  AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier)
    from sklearn.svm import SVC, LinearSVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB, BernoulliNB
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score, 
                                 confusion_matrix, classification_report)
    from sklearn.exceptions import ConvergenceWarning
    import warnings
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    
    # Try importing additional models; if not installed, they will be skipped.
    try:
        from xgboost import XGBClassifier
    except ImportError:
        XGBClassifier = None
    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        LGBMClassifier = None
    try:
        from catboost import CatBoostClassifier
    except ImportError:
        CatBoostClassifier = None

    # Define a list of diverse classification models and configurations:
    models = [
        ("Logistic Regression (L2, C=1)", LogisticRegression(penalty='l2', C=1, solver='lbfgs', 
                                                              max_iter=1000, random_state=42)),
        ("Logistic Regression (L2, C=0.1)", LogisticRegression(penalty='l2', C=0.1, solver='lbfgs', 
                                                                max_iter=1000, random_state=42)),
        ("Logistic Regression (L2, C=10)", LogisticRegression(penalty='l2', C=10, solver='lbfgs', 
                                                               max_iter=1000, random_state=42)),
        ("Logistic Regression (L1, C=1)", LogisticRegression(penalty='l1', C=1, solver='liblinear', 
                                                              max_iter=1000, random_state=42)),
        ("Decision Tree Classifier", DecisionTreeClassifier(random_state=42)),
        ("Random Forest Classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
        ("Extra Trees Classifier", ExtraTreesClassifier(random_state=42)),
        ("Gradient Boosting Classifier", GradientBoostingClassifier(random_state=42)),
        ("AdaBoost Classifier", AdaBoostClassifier(random_state=42)),
        ("Bagging Classifier", BaggingClassifier(random_state=42)),
        ("SVC", SVC(probability=True, random_state=42)),
        ("Linear SVC", LinearSVC(max_iter=1000, random_state=42)),
        ("K-Nearest Neighbors Classifier", KNeighborsClassifier()),
        ("Gaussian NB", GaussianNB()),
        ("Bernoulli NB", BernoulliNB()),
        ("Ridge Classifier", RidgeClassifier()),
        ("Passive Aggressive Classifier", PassiveAggressiveClassifier(max_iter=1000, random_state=42)),
        ("Quadratic Discriminant Analysis", QuadraticDiscriminantAnalysis()),
        ("MLP Classifier", MLPClassifier(max_iter=1000, random_state=42))
    ]
    
    if XGBClassifier is not None:
        models.append(("XGBoost Classifier", 
                       XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)))
    if LGBMClassifier is not None:
        models.append(("LightGBM Classifier", LGBMClassifier(random_state=42)))
    if CatBoostClassifier is not None:
        models.append(("CatBoost Classifier", CatBoostClassifier(verbose=0, random_state=42)))

    best_model = None
    best_auc = -np.inf
    best_model_name = None
    best_model_instance = None

    for name, model in models:
        # Train the model
        model.fit(df_X_train, df_y_train)
        # Make predictions
        y_pred = model.predict(df_X_test)
        try:
            # Some models might not have predict_proba; if so, skip ROC AUC
            y_pred_proba = model.predict_proba(df_X_test)[:, 1]
            roc_auc = roc_auc_score(df_y_test, y_pred_proba)
        except AttributeError:
            roc_auc = None

        acc = accuracy_score(df_y_test, y_pred)
        f1 = f1_score(df_y_test, y_pred)
        
        print(f"{name} Model:")
        print("Accuracy:", acc)
        if roc_auc is not None:
            print("ROC AUC Score:", roc_auc)
        else:
            print("ROC AUC Score: Not available")
        print("F1 Score:", f1)
        print("Confusion Matrix:")
        print(confusion_matrix(df_y_test, y_pred))
        print("Classification Report:")
        print(classification_report(df_y_test, y_pred))
        print("\n")

        # Update best model based on ROC AUC if available; otherwise, use F1-score as backup
        if roc_auc is not None:
            if roc_auc > best_auc:
                best_auc = roc_auc
                best_model = name
                best_model_instance = model
        else:
            if f1 > best_auc:
                best_auc = f1
                best_model = name
                best_model_instance = model

    print("The best model is:", best_model, "with a ROC AUC Score of", best_auc)
    # Print the metrics for the best model
    y_pred_best = best_model_instance.predict(df_X_test)
    best_acc = accuracy_score(df_y_test, y_pred_best)
    best_f1 = f1_score(df_y_test, y_pred_best)
    print("Metrics of the best model:")
    print("Accuracy:", best_acc)
    print("F1 Score:", best_f1)
    print("ROC AUC Score:", best_auc)


# In[17]:


find_best_model_classification(df_X_train, df_X_test, df_y_train, df_y_test)


# ## THE FIRST RESULT:
# 
# The best model is: Ridge Classifier with a ROC AUC Score of 0.87248322147651
# Metrics of the best model:
# Accuracy: 0.81
# F1 Score: 0.87248322147651
# ROC AUC Score: 0.87248322147651

# In[18]:


def forward_feature_selection(df, target_column, test_size=0.3, auc_threshold=0.01, max_features=None):
    """
    Perform forward feature selection for a classification task.
    
    This function iteratively evaluates which explanatory variable improves the ROC AUC
    the most when added to the current feature set, using a Logistic Regression classifier.
    
    Parameters:
        df (pd.DataFrame): The full DataFrame containing all features and the target.
        target_column (str): The name of the target column.
        test_size (float): Fraction of the data to use for testing in each iteration.
        auc_threshold (float): Minimum improvement in ROC AUC required to continue adding features.
        max_features (int or None): Maximum number of features to select. If None, all features are considered.
        
    Returns:
        selected_features (list): List of selected feature names.
        metrics_history (dict): History of the selection process with iteration index as key.
    """
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    # Separate out target and features
    y = df[target_column]
    X = df.drop(columns=[target_column])
    
    # Initialize lists and metrics storage
    remaining_features = list(X.columns)
    selected_features = []
    best_auc_global = 0  # Starting from 0
    metrics_history = {}

    # Set maximum number of features to select if not provided
    if max_features is None:
        max_features = len(remaining_features)

    for i in range(max_features):
        best_auc_this_round = 0
        best_feature = None
        
        # Test each remaining feature to see which gives the best improvement
        for feature in remaining_features:
            candidate_features = selected_features + [feature]
            X_subset = X[candidate_features]
            X_train, X_test, y_train, y_test = train_test_split(X_subset, y, test_size=test_size, random_state=42)
            model = LogisticRegression(max_iter=1000, random_state=42)
            model.fit(X_train, y_train)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            current_auc = roc_auc_score(y_test, y_pred_proba)
            if current_auc > best_auc_this_round:
                best_auc_this_round = current_auc
                best_feature = feature
        
        # Check if the best improvement in this round is significant enough; if not, stop selecting further features.
        if best_auc_this_round - best_auc_global < auc_threshold:
            print(f"Stopping: Improvement in ROC AUC below threshold of {auc_threshold}.")
            break
        
        # Update selected features and metrics
        selected_features.append(best_feature)
        remaining_features.remove(best_feature)
        best_auc_global = best_auc_this_round
        metrics_history[len(selected_features)] = {"Feature": best_feature, "ROC_AUC": best_auc_global}
        print(f"Iteration {len(selected_features)}: Selected '{best_feature}' with ROC AUC = {best_auc_global:.4f}")
    
    return selected_features, metrics_history


# In[19]:


selected_features, metrics_history = forward_feature_selection(df, target_column='Target_Good', 
                                                                test_size=0.3, auc_threshold=0.01, max_features=None)


# ### Explanation
# In the forward feature selection process, the algorithm started with an empty set of features and then iteratively added the variable that provided the highest improvement in ROC AUC when used in a simple Logistic Regression model. The process resulted in the following selected features (in order):
# 
# Status_of_existing_checking_account_A14 – with a ROC AUC of 0.6979
# Duration_in_month – improved the ROC AUC to 0.7746
# Purpose_A41 – further raised ROC AUC to 0.7895
# Credit_history_A31 – increased ROC AUC to 0.8020
# Present_employment_since_A74 – achieved a final ROC AUC of 0.8121
# After the fifth iteration, the improvement in ROC AUC was below the threshold (0.01), so the selection process was stopped.

# In[20]:


# df_1 creation
selected_features = ['Status_of_existing_checking_account_A14', 
                     'Duration_in_month', 
                     'Purpose_A41', 
                     'Credit_history_A31', 
                     'Present_employment_since_A74']


df_1 = df[selected_features + ['Target_Good']]
print("df_1 shape:", df_1.shape)
print("df_1 head:")
print(df_1.head())


# In[21]:


# Split df_1 into training and test sets (80% train, 20% test)
from sklearn.model_selection import train_test_split

X = df_1.drop(columns=['Target_Good'])
y = df_1['Target_Good']

df_X_train, df_X_test, df_y_train, df_y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training set size:", df_X_train.shape)
print("Test set size:", df_X_test.shape)


# ## THE SECOND RESULT:

# In[22]:


find_best_model_classification(df_X_train, df_X_test, df_y_train, df_y_test)


# In[ ]:





# ### Comparison:
# 
# The first result:
# 
# The best model is: Ridge Classifier with a ROC AUC Score of 0.87248322147651
# Metrics of the best model:
# Accuracy: 0.81
# F1 Score: 0.87248322147651
# ROC AUC Score: 0.87248322147651
# 
# The sexond result:
# The best model is: Logistic Regression (L2, C=10) with a ROC AUC Score of 0.8508234162759948
# Metrics of the best model:
# Accuracy: 0.78
# F1 Score: 0.8580645161290322
# ROC AUC Score: 0.8508234162759948

# Since all the scores are lower, I will try the feature selection with 0,001 threshold. This is because the data consists of only 1000 lines. If the data would be of the millions, I would use the 5 chosen explanatory variables, but here, we can increase the amount of data, so that we don't loose redicting quality. 

# In[23]:


selected_features, metrics_history = forward_feature_selection(df, target_column='Target_Good', 
                                                                test_size=0.3, auc_threshold=0.001, max_features=None)


# In[24]:


# Selected features from forward feature selection
selected_features_2 = [
    'Status_of_existing_checking_account_A14', 
    'Duration_in_month', 
    'Purpose_A41', 
    'Credit_history_A31', 
    'Present_employment_since_A74', 
    'Savings_account_bonds_A65', 
    'Other_debtors_guarantors_A103', 
    'Other_installment_plans_A143', 
    'Personal_status_and_sex_A92', 
    'Status_of_existing_checking_account_A13', 
    'Savings_account_bonds_A64', 
    'Credit_amount', 
    'Personal_status_and_sex_A93'
]

# Ensure the target variable is included
df_2 = df[selected_features_2 + ['Target_Good']]
print("df_2 shape:", df_2.shape)
print("df_2 head:")
print(df_2.head())


# In[25]:


from sklearn.model_selection import train_test_split

# Define features and target
X = df_2.drop(columns=['Target_Good'])
y = df_2['Target_Good']

# Split into training and testing sets (80-20 split)
df_X_train, df_X_test, df_y_train, df_y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Output the sizes of the splits
print("Training set size:", df_X_train.shape)
print("Test set size:", df_X_test.shape)


# ## THE THIRD RESULT:

# In[26]:


find_best_model_classification(df_X_train, df_X_test, df_y_train, df_y_test)


# ### Comparison:

# The first result:
# 
# The best model is: Ridge Classifier with a ROC AUC Score of 0.87248322147651 Metrics of the best model: Accuracy: 0.81 F1 Score: 0.87248322147651 ROC AUC Score: 0.87248322147651
# 
# The second result: The best model is: Logistic Regression (L2, C=10) with a ROC AUC Score of 0.8508234162759948 Metrics of the best model: Accuracy: 0.78 F1 Score: 0.8580645161290322 ROC AUC Score: 0.8508234162759948
# 
# The third result:
# The best model is: Gradient Boosting Classifier with a ROC AUC Score of 0.870176703930761
# Metrics of the best model:
# Accuracy: 0.8
# F1 Score: 0.8666666666666667
# ROC AUC Score: 0.870176703930761

# ### Okey dokey, we will go with the original df, with all the explanatory variables, as the data frame is of a small size, and none of the variables seem to deminish the results. 

# # Note to self: WRITE A FUNCTION THAT DOES THIS AUTOMATICALLY!! WHY DO THIS MANUALLY 3 TIMES? GET A METRIC THAT CAN SHOW HOW FAST THE COMPUTING IS COMPARED TO HOW BIG THE ACCURACY LOSS IS!

# In[27]:


def split_data(df, target_column, test_size=0.2, random_state=1):
    """
    Splits the input DataFrame into training and testing sets.
    
    Parameters:
        df (pd.DataFrame): The DataFrame to split.
        target_column (str): The name of the target column.
        test_size (float): The fraction of the dataset to include in the test split (default is 0.2).
        random_state (int): Random seed for reproducibility (default is 42).
    
    Returns:
        X_train (pd.DataFrame): Training set features.
        X_test (pd.DataFrame): Test set features.
        y_train (pd.Series): Training set target.
        y_test (pd.Series): Test set target.
    """
    from sklearn.model_selection import train_test_split
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    df_X_train, df_X_test, df_y_train, df_y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return df_X_train, df_X_test, df_y_train, df_y_test


# In[28]:


split_data(df, 'Target_Good')


# ## Check for overfitting:

# In[29]:


# Initialize Ridge Classifier
ridge_clf = RidgeClassifier()

# Train the model
ridge_clf.fit(df_X_train, df_y_train)

# Predictions
y_train_pred = ridge_clf.predict(df_X_train)
y_test_pred = ridge_clf.predict(df_X_test)

# Evaluate on training and test data
train_accuracy = accuracy_score(df_y_train, y_train_pred)
test_accuracy = accuracy_score(df_y_test, y_test_pred)
train_f1 = f1_score(df_y_train, y_train_pred)
test_f1 = f1_score(df_y_test, y_test_pred)

print("Training Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)
print("Training F1 Score:", train_f1)
print("Test F1 Score:", test_f1)

# Cross-validation
cv_scores = cross_val_score(ridge_clf, df_X_train, df_y_train, cv=5, scoring='accuracy')
print("\nCross-Validation Scores:", cv_scores)
print("Mean CV Accuracy:", cv_scores.mean())


# ### Okey dokey, let's now look into the overfitting for all the data frame structures

# In[30]:


def Over_Fitting_Inquiry(*dfs, target_column, test_size=0.2, random_state=1, names=None, overfitting_threshold=0.1):
    """
    This function evaluates overfitting and performance-related metrics for multiple DataFrames using the Ridge Classifier.
    It computes key metrics for each dataset, generates a detailed summary, and provides a recommendation for the best model,
    considering both performance and generalization capabilities.

    Key Features:
    - **Model Evaluation**:
      Computes metrics including Training/Test Accuracy, F1 Score, Precision, Recall, ROC AUC (if applicable), and Cross-Validation Accuracy.
    - **Overfitting Analysis**:
      Detects overfitting based on the gap between training and test performance (Performance Gap). Models with significant overfitting are penalized.
    - **Aggregated Score**:
      Combines Test Accuracy, Test F1 Score, Mean CV Accuracy, and Test ROC AUC (if available) into a single score. Penalizes overfitted models.
    - **Analytical Summary**:
      Generates a table summarizing all metrics for easy comparison across datasets.
    - **Best Model Recommendation**:
      Identifies the best-performing model based on the Aggregated Score while ensuring it is not overfitted.

    Parameters:
        *dfs: One or more pandas DataFrames, each containing explanatory variables and the target column.
        target_column (str): The name of the target column (binary classification is required).
        test_size (float): Fraction of the data to allocate for the test split (default is 0.2).
        random_state (int): Random seed for reproducibility (default is 1).
        names (list or None): Optional list of custom names for the DataFrames. If None, default names (e.g., df_1, df_2, ...) are assigned.
        overfitting_threshold (float): Threshold for the acceptable gap between training and test performance 
                                       before penalizing the model's Aggregated Score (default is 0.1).

    Returns:
        results (dict): A dictionary where each key corresponds to a DataFrame name, and each value is a dictionary of metrics:
            - **Training Accuracy**: Accuracy on the training set.
            - **Test Accuracy**: Accuracy on the test set.
            - **Training F1 Score**: F1 Score on the training set.
            - **Test F1 Score**: F1 Score on the test set.
            - **Training Precision**: Precision on the training set.
            - **Test Precision**: Precision on the test set.
            - **Training Recall**: Recall on the training set.
            - **Test Recall**: Recall on the test set.
            - **Training ROC AUC**: ROC AUC on the training set (if applicable).
            - **Test ROC AUC**: ROC AUC on the test set (if applicable).
            - **Cross-Validation Scores**: Array of accuracy scores from 5-fold cross-validation on the training set.
            - **Mean CV Accuracy**: Mean accuracy from cross-validation scores.
            - **Performance Gap**: Difference between Training and Test Accuracy.
            - **Aggregated Score**: Overall score combining key metrics, penalized for overfitting.
            - **Analysis**: A brief interpretation of the model's performance, highlighting generalization or overfitting issues.

    Output:
        - **Detailed Metrics for Each Dataset**:
          Displays all metrics and their values for each DataFrame.
        - **Analytical Summary Table**:
          Provides a tabular comparison of key metrics (Test Accuracy, Test F1 Score, Mean CV Accuracy, Test ROC AUC, Aggregated Score, and Analysis).
        - **Best Model Suggestion**:
          Recommends the model with the highest Aggregated Score while ensuring it generalizes well and is not overfitted.

    Notes:
    - Models flagged as overfitting based on the `overfitting_threshold` will be penalized and deprioritized.
    - If all models exhibit overfitting, the function will recommend revisiting the model complexity or dataset quality.
    - Default scoring for cross-validation is Accuracy; this can be adjusted to other metrics if needed.
    """

    import numpy as np
    import pandas as pd
    from sklearn.linear_model import RidgeClassifier
    from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                                 precision_score, recall_score)
    from sklearn.model_selection import train_test_split, cross_val_score
    
    results = {}
    
    # Handle naming: auto-generate names if none or too few are provided.
    if names is None:
        names = [f"df_{i+1}" for i in range(len(dfs))]
    else:
        names = list(names)
        if len(names) < len(dfs):
            for i in range(len(dfs) - len(names)):
                names.append(f"df_{len(names) + i + 1}")
        elif len(names) > len(dfs):
            print("Warning: More names provided than DataFrames. Extra names will be ignored.")
            names = names[:len(dfs)]
    
    # Process each DataFrame
    for idx, df in enumerate(dfs):
        current_name = names[idx]
        
        # Ensure target exists
        if target_column not in df.columns:
            print(f"Error: Target column '{target_column}' not found in '{current_name}'. Skipping this DataFrame.")
            continue
        
        # Extract features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Check if target is binary
        if y.nunique() != 2:
            print(f"Error: Target column '{target_column}' in '{current_name}' is not binary. Skipping this DataFrame.")
            continue
        
        # Split the data
        df_X_train, df_X_test, df_y_train, df_y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state)
        
        # Initialize and train the model
        model = RidgeClassifier()
        model.fit(df_X_train, df_y_train)
        
        # Predictions
        y_train_pred = model.predict(df_X_train)
        y_test_pred = model.predict(df_X_test)
        
        # Basic metrics
        train_accuracy = accuracy_score(df_y_train, y_train_pred)
        test_accuracy = accuracy_score(df_y_test, y_test_pred)
        train_f1 = f1_score(df_y_train, y_train_pred)
        test_f1 = f1_score(df_y_test, y_test_pred)
        train_precision = precision_score(df_y_train, y_train_pred, zero_division=0)
        test_precision = precision_score(df_y_test, y_test_pred, zero_division=0)
        train_recall = recall_score(df_y_train, y_train_pred, zero_division=0)
        test_recall = recall_score(df_y_test, y_test_pred, zero_division=0)
        
        # ROC AUC using decision_function if possible
        try:
            y_train_proba = model.decision_function(df_X_train)
            y_test_proba = model.decision_function(df_X_test)
            train_roc_auc = roc_auc_score(df_y_train, y_train_proba)
            test_roc_auc = roc_auc_score(df_y_test, y_test_proba)
        except Exception as e:
            train_roc_auc = None
            test_roc_auc = None
        
        # Cross-validation on training set
        cv_scores = cross_val_score(model, df_X_train, df_y_train, cv=5, scoring='accuracy')
        cv_mean = np.mean(cv_scores)
        
        # Compute Performance Gap
        performance_gap = train_accuracy - test_accuracy
        
        # -------------------------------
        # Compute Aggregated Score with Penalties
        # -------------------------------
        # Base score is an average of key metrics
        metrics_to_average = [test_accuracy, test_f1, cv_mean]
        if test_roc_auc is not None:
            metrics_to_average.append(test_roc_auc)
        
        # Calculate base score
        base_score = sum(metrics_to_average) / len(metrics_to_average)
        
        # Apply penalty for overfitting
        penalty = 0
        if performance_gap > overfitting_threshold:
            penalty = performance_gap * 0.5  # Adjust the multiplier as needed
            analysis_gap = performance_gap
        else:
            analysis_gap = 0
        
        # Final Aggregated Score
        aggregated_score = base_score - penalty
        
        # Overfitting Analysis
        if performance_gap > overfitting_threshold:
            analysis = (f"Overfitting detected (Performance Gap: {round(performance_gap, 4)}). "
                        "Aggregated score penalized.")
        elif (test_accuracy - train_accuracy) > overfitting_threshold:
            analysis = (f"Possible underfitting (Performance Gap: {round(performance_gap, 4)}). "
                        "Consider model complexity or data quality.")
        else:
            analysis = "Good generalization."
        
        # Store everything in results
        results[current_name] = {
            "Training Accuracy": round(train_accuracy, 4),
            "Test Accuracy": round(test_accuracy, 4),
            "Training F1 Score": round(train_f1, 4),
            "Test F1 Score": round(test_f1, 4),
            "Training Precision": round(train_precision, 4),
            "Test Precision": round(test_precision, 4),
            "Training Recall": round(train_recall, 4),
            "Test Recall": round(test_recall, 4),
            "Training ROC AUC": round(train_roc_auc, 4) if train_roc_auc is not None else "Not Available",
            "Test ROC AUC": round(test_roc_auc, 4) if test_roc_auc is not None else "Not Available",
            "Cross-Validation Scores": [round(score, 4) for score in cv_scores],
            "Mean CV Accuracy": round(cv_mean, 4),
            "Performance Gap": round(performance_gap, 4),
            "Aggregated Score": round(aggregated_score, 4),
            "Analysis": analysis
        }
    
    # -------------------------------
    # Print out individual results
    # -------------------------------
    for name, metrics in results.items():
        print(f"Results for {name}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")
        print("\n")
    
    # -------------------------------
    # Build an analytical summary table
    # -------------------------------
    print("=== Analytical Summary ===")
    summary = []
    for name, metrics in results.items():
        summary.append({
            "Name": name,
            "Test Accuracy": metrics["Test Accuracy"],
            "Test F1 Score": metrics["Test F1 Score"],
            "Mean CV Accuracy": metrics["Mean CV Accuracy"],
            "Test ROC AUC": metrics["Test ROC AUC"],
            "Aggregated Score": metrics["Aggregated Score"],
            "Analysis": metrics["Analysis"]
        })
    
    # Create DataFrame for better formatting
    summary_df = pd.DataFrame(summary)
    
    # Display the summary table
    print(summary_df.to_string(index=False))
    
    # -------------------------------
    # Choose a winner based on Aggregated Score
    # -------------------------------
    if len(summary) > 0:
        # Exclude models that are overfitting
        non_overfitting_models = [model for model in summary if "Overfitting detected" not in model["Analysis"]]
        
        if non_overfitting_models:
            # Select the model with the highest Aggregated Score among non-overfitting models
            best_model = max(non_overfitting_models, key=lambda x: x["Aggregated Score"])
            print("\n=== Best Model Suggestion ===")
            print(
                f"The best model is '{best_model['Name']}' "
                f"with an Aggregated Score of {best_model['Aggregated Score']}. \n"
                "This indicates strong overall performance across Test Accuracy, F1 Score, Mean CV Accuracy, "
                "and (if available) ROC AUC, without signs of overfitting."
            )
        else:
            print("\n=== Best Model Suggestion ===")
            print("All models exhibit signs of overfitting based on the defined threshold. Consider revisiting model complexity or data quality.")
    else:
        print("\nNo valid DataFrames were processed. No winner can be selected.")
    
    return results


# In[31]:


results = Over_Fitting_Inquiry(df, df_1, df_2, target_column="Target_Good", test_size=0.2, random_state=1, names=["df", "df_1", "df_2"])


# ## So df2 wins the aggregated score! 

# ## Quick analysis
# 
# 
# Training and Test Performance:
# 
# For all DataFrames (df, df_1, and df_2), the training and test accuracies and F1 scores are close to each other. This indicates the model generalizes well across both training and unseen test data.
# No large discrepancy between training and test metrics is observed, which is a key indicator that the model is not overfitting.
# Cross-Validation (CV) Scores:
# 
# Cross-validation accuracy scores for each DataFrame are slightly lower than the test accuracy, but still reasonably close. This is expected because CV provides a more robust estimate of model performance by splitting the data multiple times, making it less optimistic compared to a single test set evaluation.
# A higher variance in CV scores could suggest instability or overfitting, but the scores here are relatively consistent.
# Why This Is Not Overfitting
# Consistency Between Metrics:
# Overfitting occurs when the model learns the training data too well (memorizing patterns, including noise) but performs poorly on unseen data. This manifests as a large gap between training and test performance. Here, training and test accuracies are within a few percentage points, which suggests good generalization.
# Cross-Validation Agreement:
# The mean CV accuracy for each DataFrame is close to the test accuracy. This further indicates that the model is not overfitting and its performance is stable across different subsets of the data.
# When Would It Be Overfitting?
# Large Training-Test Discrepancy:
# 
# If training accuracy is much higher than test accuracy (e.g., training = 0.95, test = 0.65), it suggests overfitting, as the model is performing well on training data but poorly on unseen data.
# High Cross-Validation Variance:
# 
# If CV scores vary significantly (e.g., [0.85, 0.60, 0.90, 0.55, 0.80]), it indicates that the model’s performance depends heavily on the specific subset of data, which is another sign of overfitting.
# Low Test Scores:
# 
# Despite high training accuracy, if test accuracy and F1 scores are consistently low, the model has likely memorized the training data rather than learning generalizable patterns.
# 
# 
# 
# ### Long story short, this is good! :)

# ## Here is a quick explanation on what they do:
# 
# What Does Each Metric Do?
# Accuracy:
# 
# How It Works: Measures the proportion of correct predictions out of total predictions.
# Limitations: Can be misleading if the dataset is imbalanced (e.g., 95% accuracy might just mean the model predicts the majority class).
# F1 Score:
# 
# How It Works: Combines precision and recall into a single metric using the harmonic mean.
# Why It’s Useful: Balances the trade-off between precision and recall, especially in imbalanced datasets.
# Cross-Validation Scores:
# 
# How It Works: Evaluates model performance on multiple splits of the training data.
# Why It’s Useful: Provides a robust estimate of model performance, reducing reliance on a single train-test split.
# ROC AUC:
# 
# How It Works: Summarizes the model's performance at distinguishing between classes across all classification thresholds.
# Why It’s Useful: Provides a single score that considers both sensitivity and specificity.
# Precision/Recall:
# 
# How It Works: Measures the model’s ability to correctly identify positive cases.
# Why It’s Useful: Especially important in cases where false positives/negatives have high costs.

# In[32]:


split_data(df_2, "Target_Good")


# ## Let's do the parameter tuning

# In[33]:


def tune_ridge_classifier(df, target_column,
                          test_size=0.2,
                          random_state=52,
                          param_grid=None,
                          scoring='accuracy',
                          cv=5,
                          n_jobs=-1,
                          overfitting_threshold=0.1,
                          complexity=5):
    """
    Fine-tunes a RidgeClassifier using GridSearchCV on the provided DataFrame and evaluates it with multiple metrics.
    
    The evaluation uses:
      - Test Accuracy
      - Test F1 Score
      - Mean CV Accuracy (via stratified cross-validation)
      - Test ROC AUC (if available; computed via decision_function)
      
    An aggregated score is computed as the average of these metrics. If the training-test
    performance gap exceeds 'overfitting_threshold', a penalty is applied.

    The 'complexity' parameter (1 to 10) controls how large the parameter grid is:
      - Lower = smaller alpha grid
      - Higher = more extensive alpha grid
      - We always include alpha=1.0 to match the default parameter if it is indeed best.
    
    Parameters:
        df (pd.DataFrame): The DataFrame with features + a binary target column.
        target_column (str): Name of the target column in df (binary classification).
        test_size (float): Fraction for test split (default 0.2).
        random_state (int): Seed for reproducibility (default 52).
        param_grid (dict or None): Custom grid. If None, dynamically generated based on 'complexity'.
        scoring (str): Scoring metric for GridSearchCV (default 'accuracy').
        cv (int): Number of folds for cross-validation (default 5).
        n_jobs (int): Jobs to run in parallel (-1 uses all processors).
        overfitting_threshold (float): Gap threshold for penalizing overfitting (default 0.1).
        complexity (int): 1 to 10, controlling the range/granularity of alphas in the grid (default 5).
        
    Returns:
        best_estimator: The best RidgeClassifier model found.
        final_aggregated_score: The aggregated score (after overfitting penalty).
        best_params: The best hyperparameters as a dictionary.
    """
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import RidgeClassifier
    from sklearn.model_selection import (train_test_split, GridSearchCV,
                                         StratifiedKFold, cross_val_score)
    from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                                 precision_score, recall_score, classification_report)
    
    # 1. Dynamically build a parameter grid if none is provided
    if param_grid is None:
        # Create alpha ranges based on complexity
        if complexity <= 3:
            alpha_grid = [0.01, 0.1, 1.0, 10.0]
        elif complexity <= 7:
            alpha_grid = np.logspace(-3, 2, num=complexity*2).tolist()
            alpha_grid.append(1.0)  # Ensure alpha=1.0 is included
        else:  # complexity 8-10
            alpha_grid = np.logspace(-4, 3, num=complexity*2).tolist()
            alpha_grid.append(1.0)  # Ensure alpha=1.0 is present

        alpha_grid = sorted(set(alpha_grid))  # Remove duplicates, keep ascending order
        
        # We'll always test these solvers
        solver_options = ['sag', 'saga', 'auto']
        
        # For class_weight, add 'balanced' only if complexity >= 5
        if complexity >= 5:
            class_weights = [None, 'balanced']
        else:
            class_weights = [None]

        param_grid = {
            'alpha': alpha_grid,
            'solver': solver_options,
            'class_weight': class_weights
        }
    
    # 2. Prepare data + stratified split
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    # 3. Define StratifiedKFold
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    
    # 4. Grid Search on RidgeClassifier
    base_model = RidgeClassifier()
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=skf,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=1,
        refit=True
    )
    grid_search.fit(X_train, y_train)
    
    # Retrieve best estimator and details
    best_estimator = grid_search.best_estimator_
    best_cv_score = grid_search.best_score_
    best_params = grid_search.best_params_
    
    # 5. Evaluate best estimator on test set
    y_test_pred = best_estimator.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, zero_division=0)
    test_recall = recall_score(y_test, y_test_pred, zero_division=0)
    
    # Overfitting analysis
    y_train_pred = best_estimator.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    performance_gap = train_accuracy - test_accuracy
    
    # 6. Compute Test ROC AUC if possible
    try:
        y_test_proba = best_estimator.decision_function(X_test)
        test_roc_auc = roc_auc_score(y_test, y_test_proba)
    except Exception:
        test_roc_auc = None
    
    # Cross-validation with best_estimator
    cv_scores = cross_val_score(best_estimator, X_train, y_train, cv=skf, scoring=scoring)
    mean_cv_accuracy = np.mean(cv_scores)
    
    # 7. Compute aggregated score
    metrics_list = [test_accuracy, test_f1, mean_cv_accuracy]
    if test_roc_auc is not None:
        metrics_list.append(test_roc_auc)
    base_score = sum(metrics_list)/len(metrics_list)
    
    # Overfitting penalty
    penalty = 0
    if performance_gap > overfitting_threshold:
        penalty = 0.5 * performance_gap
    final_aggregated_score = base_score - penalty
    
    # 8. Create analysis message
    if performance_gap > overfitting_threshold:
        analysis_message = (f"Overfitting detected (Perf Gap: {performance_gap:.4f}). "
                            f"Penalty = {penalty:.4f}.")
    elif (test_accuracy - train_accuracy) > overfitting_threshold:
        analysis_message = (f"Possible underfitting (Perf Gap: {performance_gap:.4f}). "
                            "Test performance is higher than training performance.")
    else:
        analysis_message = "Good generalization."
    
    # 9. Printing the results
    print("\n=== Grid Search Results ===")
    print("Best Parameters:", best_params)
    print(f"Best Cross-Validation Score ({scoring}): {best_cv_score:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")
    print(f"Mean CV Accuracy: {mean_cv_accuracy:.4f}")
    if test_roc_auc is not None:
        print(f"Test ROC AUC: {test_roc_auc:.4f}")
    else:
        print("Test ROC AUC: Not Available")
    print(f"Performance Gap (Train - Test Accuracy): {performance_gap:.4f}")
    print(f"Aggregated Score (before penalty): {base_score:.4f}")
    print(f"Overfitting Penalty: {penalty:.4f}")
    print(f"Final Aggregated Score: {final_aggregated_score:.4f}")
    print(f"Analysis: {analysis_message}")
    
    # Detailed classification report
    from sklearn.metrics import classification_report
    print("\nDetailed Classification Report on Test Set:")
    print(classification_report(y_test, y_test_pred))
    
    # Grid search CV results
    results_df = pd.DataFrame(grid_search.cv_results_)
    try:
        from IPython.display import display
        display(results_df.sort_values(by='mean_test_score', ascending=False))
    except ImportError:
        print("\nTop 10 CV Results (sorted by mean_test_score):")
        print(results_df.sort_values(by='mean_test_score', ascending=False).head(10))
    
    print("\n=== Winner Summary ===")
    print(f"The best RidgeClassifier is obtained with parameters: {best_params}")
    print(f"Final Aggregated Score: {final_aggregated_score:.4f} ("
          "combining Test Accuracy, Test F1, Mean CV Accuracy, and Test ROC AUC if available, "
          f"with an overfitting penalty if Perf Gap > {overfitting_threshold}).")
    
    return best_estimator, final_aggregated_score, best_params


# In[34]:


best_model, final_score, best_params = tune_ridge_classifier(
    df_2,
    target_column="Target_Good",
    test_size=0.2,
    random_state=52,
    complexity=10  
)


# Name  Test Accuracy  Test F1 Score  Mean CV Accuracy  Test ROC AUC  Aggregated Score             Analysis
# df_2          0.775         0.8544            0.7262        0.8030            0.7897 Good generalization.
# 
# 
# === Grid Search Results ===
# Best Parameters: {'alpha': 1.1288378916846884, 'class_weight': None, 'solver': 'sag'}
# Best Cross-Validation Score (Accuracy): 0.7425
# Test Accuracy: 0.77
# Test F1 Score: 0.8497
# Mean CV Accuracy: 0.7425
# Test ROC AUC: 0.7994
# Performance Gap (Train Accuracy - Test Accuracy): -0.025
# Aggregated Score (before penalty): 0.7904
# Overfitting Penalty Applied: 0
# Final Aggregated Score: 0.7904
# Analysis: Good generalization.
# 
# 
# 

# ### FIXED: Okay, so apparently the hyperparameter tuning is making things worse, so we will create a new test, to see if it creates and actually worse model as before:

# In[35]:


def compare_ridge_models(df, target_column, tuned_params,
                                    test_size=0.2, random_state=1, cv=5, scoring='accuracy',
                                    overfitting_threshold=0.1):
    """
    Compares the baseline RidgeClassifier (default parameters) with a tuned RidgeClassifier.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing features and the target column.
        target_column (str): The name of the target column (must be binary).
        tuned_params (dict): A dictionary of hyperparameters for the tuned model.
        test_size (float): Fraction of the data used for testing (default is 0.2).
        random_state (int): Random seed for reproducibility (default is 1).
        cv (int): Number of folds for cross-validation (default is 5).
        scoring (str): Scoring metric for cross-validation (default 'accuracy').
        overfitting_threshold (float): Threshold for the training-test gap to trigger an overfitting penalty (default is 0.1).
    
    Returns:
        A tuple (baseline_results, tuned_results) where each is a dictionary of computed metrics.
    """
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import RidgeClassifier
    from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
    from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                                 precision_score, recall_score, classification_report)
    
    # -------------------------------
    # Data Preparation: Separate features and target.
    # -------------------------------
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Split the data with stratification to preserve class distribution.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    # -------------------------------
    # Define Stratified K-Fold
    # -------------------------------
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    
    # -------------------------------
    # Baseline Model (Default Parameters)
    # -------------------------------
    baseline_model = RidgeClassifier()
    baseline_model.fit(X_train, y_train)
    y_train_pred_baseline = baseline_model.predict(X_train)
    y_test_pred_baseline = baseline_model.predict(X_test)
    
    # Compute Baseline Metrics
    baseline_train_accuracy = accuracy_score(y_train, y_train_pred_baseline)
    baseline_test_accuracy = accuracy_score(y_test, y_test_pred_baseline)
    baseline_test_f1 = f1_score(y_test, y_test_pred_baseline)
    baseline_test_precision = precision_score(y_test, y_test_pred_baseline, zero_division=0)
    baseline_test_recall = recall_score(y_test, y_test_pred_baseline, zero_division=0)
    
    # ROC AUC for Baseline
    try:
        y_test_proba_baseline = baseline_model.decision_function(X_test)
        baseline_test_roc_auc = roc_auc_score(y_test, y_test_proba_baseline)
    except Exception:
        baseline_test_roc_auc = None
    
    # Cross-Validation for Baseline
    baseline_cv_scores = cross_val_score(baseline_model, X_train, y_train, cv=skf, scoring=scoring)
    baseline_mean_cv_accuracy = np.mean(baseline_cv_scores)
    
    # Performance Gap for Baseline
    baseline_performance_gap = baseline_train_accuracy - baseline_test_accuracy
    
    # Aggregated Score for Baseline
    baseline_metrics = [baseline_test_accuracy, baseline_test_f1, baseline_mean_cv_accuracy]
    if baseline_test_roc_auc is not None:
        baseline_metrics.append(baseline_test_roc_auc)
    baseline_base_score = sum(baseline_metrics) / len(baseline_metrics)
    
    baseline_penalty = 0
    if baseline_performance_gap > overfitting_threshold:
        baseline_penalty = baseline_performance_gap * 0.5
    baseline_final_aggregated_score = baseline_base_score - baseline_penalty
    
    # Analysis for Baseline
    if baseline_performance_gap > overfitting_threshold:
        baseline_analysis = (f"Overfitting detected (Performance Gap: {round(baseline_performance_gap, 4)}). "
                             f"Aggregated score penalized by {round(baseline_penalty, 4)}.")
    elif (baseline_test_accuracy - baseline_train_accuracy) > overfitting_threshold:
        baseline_analysis = (f"Possible underfitting (Performance Gap: {round(baseline_performance_gap, 4)}). "
                             "Test performance is higher than training performance.")
    else:
        baseline_analysis = "Good generalization."
    
    # -------------------------------
    # Tuned Model (Provided Parameters)
    # -------------------------------
    tuned_model = RidgeClassifier(**tuned_params)
    tuned_model.fit(X_train, y_train)
    y_train_pred_tuned = tuned_model.predict(X_train)
    y_test_pred_tuned = tuned_model.predict(X_test)
    
    # Compute Tuned Metrics
    tuned_train_accuracy = accuracy_score(y_train, y_train_pred_tuned)
    tuned_test_accuracy = accuracy_score(y_test, y_test_pred_tuned)
    tuned_test_f1 = f1_score(y_test, y_test_pred_tuned)
    tuned_test_precision = precision_score(y_test, y_test_pred_tuned, zero_division=0)
    tuned_test_recall = recall_score(y_test, y_test_pred_tuned, zero_division=0)
    
    # ROC AUC for Tuned Model
    try:
        y_test_proba_tuned = tuned_model.decision_function(X_test)
        tuned_test_roc_auc = roc_auc_score(y_test, y_test_proba_tuned)
    except Exception:
        tuned_test_roc_auc = None
    
    # Cross-Validation for Tuned Model
    tuned_cv_scores = cross_val_score(tuned_model, X_train, y_train, cv=skf, scoring=scoring)
    tuned_mean_cv_accuracy = np.mean(tuned_cv_scores)
    
    # Performance Gap for Tuned Model
    tuned_performance_gap = tuned_train_accuracy - tuned_test_accuracy
    
    # Aggregated Score for Tuned Model
    tuned_metrics = [tuned_test_accuracy, tuned_test_f1, tuned_mean_cv_accuracy]
    if tuned_test_roc_auc is not None:
        tuned_metrics.append(tuned_test_roc_auc)
    tuned_base_score = sum(tuned_metrics) / len(tuned_metrics)
    
    tuned_penalty = 0
    if tuned_performance_gap > overfitting_threshold:
        tuned_penalty = tuned_performance_gap * 0.5
    tuned_final_aggregated_score = tuned_base_score - tuned_penalty
    
    # Analysis for Tuned Model
    if tuned_performance_gap > overfitting_threshold:
        tuned_analysis = (f"Overfitting detected (Performance Gap: {round(tuned_performance_gap, 4)}). "
                          f"Aggregated score penalized by {round(tuned_penalty, 4)}.")
    elif (tuned_test_accuracy - tuned_train_accuracy) > overfitting_threshold:
        tuned_analysis = (f"Possible underfitting (Performance Gap: {round(tuned_performance_gap, 4)}). "
                          "Test performance is higher than training performance.")
    else:
        tuned_analysis = "Good generalization."
    
    # -------------------------------
    # Compile Results
    # -------------------------------
    baseline_results = {
        "Model": "Baseline",
        "Training Accuracy": round(baseline_train_accuracy, 4),
        "Test Accuracy": round(baseline_test_accuracy, 4),
        "Test F1 Score": round(baseline_test_f1, 4),
        "Mean CV Accuracy": round(baseline_mean_cv_accuracy, 4),
        "Test ROC AUC": round(baseline_test_roc_auc, 4) if baseline_test_roc_auc is not None else "Not Available",
        "Performance Gap": round(baseline_performance_gap, 4),
        "Aggregated Score": round(baseline_base_score, 4),
        "Overfitting Penalty Applied": round(baseline_penalty, 4),
        "Final Aggregated Score": round(baseline_final_aggregated_score, 4),
        "Analysis": baseline_analysis
    }
    
    tuned_results = {
        "Model": "Tuned",
        "Training Accuracy": round(tuned_train_accuracy, 4),
        "Test Accuracy": round(tuned_test_accuracy, 4),
        "Test F1 Score": round(tuned_test_f1, 4),
        "Mean CV Accuracy": round(tuned_mean_cv_accuracy, 4),
        "Test ROC AUC": round(tuned_test_roc_auc, 4) if tuned_test_roc_auc is not None else "Not Available",
        "Performance Gap": round(tuned_performance_gap, 4),
        "Aggregated Score": round(tuned_base_score, 4),
        "Overfitting Penalty Applied": round(tuned_penalty, 4),
        "Final Aggregated Score": round(tuned_final_aggregated_score, 4),
        "Analysis": tuned_analysis
    }
    
    # -------------------------------
    # Create Analytical Summary Table
    # -------------------------------
    summary_df = pd.DataFrame([baseline_results, tuned_results])
    print("\n=== Analytical Summary ===")
    print(summary_df[[
        "Model", "Test Accuracy", "Test F1 Score", "Mean CV Accuracy", "Test ROC AUC",
        "Aggregated Score", "Analysis"
    ]].to_string(index=False))
    
    # -------------------------------
    # Winner Recommendation
    # -------------------------------
    print("\n=== Winner Summary ===")
    if tuned_final_aggregated_score > baseline_final_aggregated_score:
        winner = "Tuned"
        winner_details = tuned_results
    else:
        winner = "Baseline"
        winner_details = baseline_results
    
    print(f"The best model is the {winner} model with an Aggregated Score of {winner_details['Final Aggregated Score']}.")
    print(f"Detailed Results: {winner_details}")
    
    return baseline_results, tuned_results


# In[36]:


chosen_params = {'alpha': 6.1584821106602545, 'class_weight': None, 'solver': 'sag'}
baseline_res, tuned_res = compare_ridge_models(df_2, target_column="Target_Good",
                                               tuned_params=chosen_params,
                                               test_size=0.2,
                                               random_state=1,
                                               cv=5,
                                               scoring='accuracy',
                                               overfitting_threshold=0.1)


# In[37]:


chosen_params ={'alpha': 1.0, 'class_weight': None, 'solver': 'sag'}
baseline_res, tuned_res = compare_ridge_models(df_2, target_column="Target_Good",
                                               tuned_params=chosen_params,
                                               test_size=0.2,
                                               random_state=1,
                                               cv=5,
                                               scoring='accuracy',
                                               overfitting_threshold=0.1)


# ### Fine, untuned it is, but this is weird. I will need to find a different model later for educational purposes.

# # Finalizing the model:

# In[38]:


from sklearn.model_selection import train_test_split

# Example: Splitting df_2 into training+validation and holdout sets
X = df_2.drop(columns=["Target_Good"])
y = df_2["Target_Good"]

# First split: 80% for training+validation, 20% holdout
X_temp, X_holdout, y_temp, y_holdout = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Second split: from the 80%, split 75% training and 25% validation (~60% train, 20% validation overall)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)

print("Training set shape:", X_train.shape)
print("Validation set shape:", X_val.shape)
print("Holdout set shape:", X_holdout.shape)


# In[39]:


best_model = RidgeClassifier(alpha=1.0, solver='sag', class_weight=None)
best_model.fit(X, y)


# ## Now some details on the model:

# In[40]:


import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

def plot_learning_curve(model, X, y, title='Learning Curve', cv=5, n_jobs=-1):
    train_sizes, train_scores, val_scores = learning_curve(model, X, y, cv=cv,
                                                           n_jobs=n_jobs,
                                                           train_sizes=np.linspace(0.1, 1.0, 5),
                                                           scoring='accuracy')
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_mean, 'o-', color='r', label='Training Score')
    plt.plot(train_sizes, val_mean, 'o-', color='g', label='Validation Score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='r')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='g')
    plt.title(title)
    plt.xlabel("Training Size")
    plt.ylabel("Score")
    plt.legend(loc="best")
    plt.grid(True)
    plt.show()
    
# Example usage:
plot_learning_curve(best_model, X_train, y_train, title="Learning Curve for RidgeClassifier", cv=5)


# In[41]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Retrieve feature names and coefficients
coefficients = best_model.coef_[0]
features = X.columns  # Assuming X is the feature DataFrame after preprocessing

feature_importance = pd.DataFrame({"Feature": features, "Coefficient": coefficients})
feature_importance = feature_importance.sort_values(by="Coefficient", ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(x="Coefficient", y="Feature", data=feature_importance)
plt.title("Feature Importance from RidgeClassifier")
plt.show()

print(feature_importance)


# ## Saving the modell
# 

# In[42]:


import joblib

# Save the best model to disk.
joblib.dump(best_model, "ridge_classifier_best_model.pkl")
print("Model saved as 'ridge_classifier_best_model.pkl'.")

# Later, load the model using:
# loaded_model = joblib.load("ridge_classifier_best_model.pkl")


# ## And Evaluating it:

# In[43]:


def evaluate_model_on_holdout(model, X_holdout, y_holdout, plot_roc=True):
    """
    Evaluates a trained model on a holdout set and prints key performance metrics.
    
    Computes:
      - Accuracy
      - F1 Score
      - Confusion Matrix
      - Detailed Classification Report
      - ROC AUC (if available; computed via decision_function)
      
    Optionally plots the ROC curve.
    
    Parameters:
        model: A trained scikit-learn model that supports decision_function (or predict_proba as a fallback).
        X_holdout (pd.DataFrame): Features of the holdout set.
        y_holdout (pd.Series): Target values of the holdout set.
        plot_roc (bool): If True, plots the ROC curve (default is True).
    
    Returns:
        A dictionary containing the computed metrics.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.metrics import (classification_report, confusion_matrix, 
                                 accuracy_score, f1_score, roc_auc_score, roc_curve)
    
    # Make predictions on the holdout set
    y_pred = model.predict(X_holdout)
    
    # Compute basic metrics
    accuracy = accuracy_score(y_holdout, y_pred)
    f1 = f1_score(y_holdout, y_pred)
    cm = confusion_matrix(y_holdout, y_pred)
    report = classification_report(y_holdout, y_pred)
    
    # Print metrics
    print("=== Holdout Set Evaluation ===")
    print("Classification Report:")
    print(report)
    print("Confusion Matrix:")
    print(cm)
    print(f"Holdout Accuracy: {accuracy:.4f}")
    print(f"Holdout F1 Score: {f1:.4f}")
    
    # Compute ROC AUC and plot ROC curve if possible
    roc_auc = None
    try:
        # RidgeClassifier has no predict_proba, so we use decision_function
        y_proba = model.decision_function(X_holdout)
        roc_auc = roc_auc_score(y_holdout, y_proba)
        print(f"Holdout ROC AUC: {roc_auc:.4f}")
        
        if plot_roc:
            fpr, tpr, thresholds = roc_curve(y_holdout, y_proba)
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve on Holdout Set")
            plt.legend(loc="lower right")
            plt.grid(True)
            plt.show()
    except Exception as e:
        print("ROC AUC could not be computed:", e)
    
    # Compile results into a dictionary
    results = {
        "Accuracy": round(accuracy, 4),
        "F1 Score": round(f1, 4),
        "Confusion Matrix": cm,
        "Classification Report": report,
        "ROC AUC": round(roc_auc, 4) if roc_auc is not None else "Not Available"
    }
    
    return results


# In[44]:


from sklearn.model_selection import train_test_split

# Assuming df_2 is your final dataset with a binary target "Target_Good"
target_column = "Target_Good"
X = df_2.drop(columns=[target_column])
y = df_2[target_column]

# Split data: 80% training/validation and 20% holdout
X_temp, X_holdout, y_temp, y_holdout = train_test_split(X, y, test_size=0.2, random_state=52, stratify=y)


# In[45]:


holdout_results = evaluate_model_on_holdout(best_model, X_holdout, y_holdout)


# ### okay, this result is not good at all. SO!! I will try out every single model in a brutal way. I can do this because of the low number of rows.

# In[46]:


def all_modell_tester(df, target_column, test_size=0.2, random_state=52, cv=5, 
                      scoring='accuracy', overfitting_threshold=0.1, use_feature_subsets=True):
    """
    Evaluates a suite of classification models on the given DataFrame and returns an analytical summary
    along with a recommendation for the overall best model.
    
    The evaluation is based on:
      - Test Accuracy
      - Test F1 Score
      - Mean CV Accuracy (via stratified cross-validation)
      - Test ROC AUC (if available)
      - Performance Gap (Train Accuracy - Test Accuracy)
      - A base aggregated score (the average of the available metrics)
      - A Confusion Point computed from the confusion matrix:
            CP = (TN + TP) - 2*(FP + FN)
      - The Confusion Point is normalized:
            CP_norm = (CP + 2*N)/(3*N), where N = number of test samples.
      
    A new final score is computed as:
           final_score_new = 0.5 * CP_norm + 0.3 * (Mean CV Accuracy) + 0.1 * (Test Accuracy) + 0.1 * (Test F1 Score)
    
    Models with a training-test gap greater than 'overfitting_threshold' are disqualified.
    
    If use_feature_subsets is True (default), forward feature selection (using Logistic Regression)
    is performed to determine an ordering of features, and candidate models are evaluated on cumulative feature subsets.
    
    Parameters:
        df (pd.DataFrame): A fully preprocessed DataFrame (encoded, scaled) with features and a binary target column.
        target_column (str): The name of the target column.
        test_size (float): Fraction of the data to use for testing (default 0.2).
        random_state (int): Random seed for reproducibility (default 52).
        cv (int): Number of folds for stratified cross-validation (default 5).
        scoring (str): Scoring metric for cross-validation (default 'accuracy').
        overfitting_threshold (float): Threshold for the training-test gap to trigger disqualification (default 0.1).
        use_feature_subsets (bool): If True, perform forward feature selection to order features (default True).
    
    Returns:
        summary_df (pd.DataFrame): A summary table with key metrics for each candidate model.
    """
    import numpy as np
    import pandas as pd
    import warnings
    warnings.filterwarnings("ignore")
    
    from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report
    from sklearn.base import clone
    from sklearn.linear_model import LogisticRegression

    # ----- Forward Feature Selection (Optional) -----
    def forward_feature_selection(df, target_column, test_size=0.3, auc_threshold=0.001, max_features=None):
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import train_test_split
        y_fs = df[target_column]
        X_fs = df.drop(columns=[target_column])
        remaining = list(X_fs.columns)
        selected = []
        best_auc_global = 0
        fs_history = {}
        if max_features is None:
            max_features = len(remaining)
        for i in range(max_features):
            best_auc_this = 0
            best_feat = None
            for feat in remaining:
                candidate = selected + [feat]
                X_sub = X_fs[candidate]
                X_train_sub, X_test_sub, y_train_sub, y_test_sub = train_test_split(X_sub, y_fs, test_size=test_size, random_state=42)
                model_sub = LogisticRegression(max_iter=1000, random_state=42)
                model_sub.fit(X_train_sub, y_train_sub)
                try:
                    y_pred_proba = model_sub.predict_proba(X_test_sub)[:, 1]
                    auc_val = roc_auc_score(y_test_sub, y_pred_proba)
                except Exception:
                    auc_val = 0
                if auc_val > best_auc_this:
                    best_auc_this = auc_val
                    best_feat = feat
            if best_auc_this - best_auc_global < auc_threshold:
                print(f"Stopping feature selection: improvement below {auc_threshold}.\n")
                break
            selected.append(best_feat)
            remaining.remove(best_feat)
            best_auc_global = best_auc_this
            fs_history[len(selected)] = {"Feature": best_feat, "ROC_AUC": best_auc_global}
            print(f"Iteration {len(selected)}: Selected '{best_feat}' with ROC AUC = {best_auc_global:.4f}\n")
        return selected, fs_history
    
    # ----- Evaluation Function for a Given Feature Subset -----
    def evaluate_model(model, X_train_subset, X_test_subset, y_train, y_test, skf):
        m = clone(model)
        m.fit(X_train_subset, y_train)
        y_train_pred = m.predict(X_train_subset)
        y_test_pred = m.predict(X_test_subset)
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred)
        try:
            if hasattr(m, "decision_function"):
                y_test_scores = m.decision_function(X_test_subset)
            elif hasattr(m, "predict_proba"):
                y_test_scores = m.predict_proba(X_test_subset)[:, 1]
            else:
                y_test_scores = None
            roc_auc = roc_auc_score(y_test, y_test_scores) if y_test_scores is not None else None
        except Exception:
            roc_auc = None
        cv_scores = cross_val_score(m, X_train_subset, y_train, cv=skf, scoring=scoring)
        mean_cv_acc = np.mean(cv_scores)
        
        # Base aggregated score from available metrics
        metrics_list = [test_acc, test_f1, mean_cv_acc]
        if roc_auc is not None:
            metrics_list.append(roc_auc)
        base_score = np.mean(metrics_list)
        
        performance_gap = train_acc - test_acc
        penalty = 0.5 * performance_gap if performance_gap > overfitting_threshold else 0
        final_score = base_score - penalty
        
        # Compute confusion matrix and confusion point
        cm = confusion_matrix(y_test, y_test_pred)
        if cm.shape == (2, 2):
            TN, FP, FN, TP = cm.ravel()
            # Confusion Point: (TN + TP) - 2*(FP + FN)
            confusion_point = TN + TP - 2 * (FP + FN)
            N = len(y_test)
            # Normalize confusion point to range [0,1]:
            cp_norm = (confusion_point + 2 * N) / (3 * N)
        else:
            confusion_point = None
            cp_norm = None
        
        # New final aggregated score: weighted average of final_score and cp_norm
        # We give more weight to cp_norm (0.5) and mean CV accuracy (0.3) and lower weight to test_acc and test_f1 (0.1 each)
        if cp_norm is not None:
            weighted_score = 0.5 * cp_norm + 0.3 * mean_cv_acc + 0.1 * test_acc + 0.1 * test_f1
        else:
            weighted_score = final_score
        
        if performance_gap > overfitting_threshold:
            analysis = f"Overfitting detected (gap: {performance_gap:.4f}). Penalty = {penalty:.4f}."
        elif (test_acc - train_acc) > overfitting_threshold:
            analysis = f"Possible underfitting (gap: {performance_gap:.4f})."
        else:
            analysis = "Good generalization."
            
        return {
            "Training Accuracy": round(train_acc, 4),
            "Test Accuracy": round(test_acc, 4),
            "Test F1 Score": round(test_f1, 4),
            "Mean CV Accuracy": round(mean_cv_acc, 4),
            "Test ROC AUC": round(roc_auc, 4) if roc_auc is not None else "Not Available",
            "Confusion Matrix": cm,
            "Confusion Point": round(confusion_point, 4) if confusion_point is not None else "Not Available",
            "Normalized Confusion Point": round(cp_norm, 4) if cp_norm is not None else "Not Available",
            "Performance Gap": round(performance_gap, 4),
            "Aggregated Score": round(base_score, 4),
            "Overfitting Penalty": round(penalty, 4),
            "Final Aggregated Score": round(weighted_score, 4),
            "Analysis": analysis,
            "Classification Report": classification_report(y_test, y_test_pred)
        }
    
    # ----- Prepare Data and Split (with stratification) -----
    X_full = df.drop(columns=[target_column])
    y_full = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, test_size=test_size, random_state=random_state, stratify=y_full)
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    
    # ----- Determine feature order via forward feature selection (if enabled) -----
    if use_feature_subsets:
        print("Running forward feature selection to determine feature order...\n")
        selected_order, fs_history = forward_feature_selection(df, target_column, test_size=0.3, auc_threshold=0.001, max_features=None)
    else:
        selected_order = list(X_full.columns)
    
    # ----- Define Candidate Models -----
    from sklearn.linear_model import LogisticRegression, RidgeClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neural_network import MLPClassifier
    
    models_to_test = [
        ("Logistic Regression (L2, C=1)", LogisticRegression(penalty='l2', C=1, solver='lbfgs', max_iter=1000, random_state=random_state)),
        ("Logistic Regression (L2, C=10)", LogisticRegression(penalty='l2', C=10, solver='lbfgs', max_iter=1000, random_state=random_state)),
        ("Ridge Classifier (Default)", RidgeClassifier()),
        ("Decision Tree", DecisionTreeClassifier(random_state=random_state)),
        ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=random_state)),
        ("Gradient Boosting", GradientBoostingClassifier(random_state=random_state)),
        ("AdaBoost", AdaBoostClassifier(random_state=random_state)),
        ("Extra Trees", ExtraTreesClassifier(random_state=random_state)),
        ("Bagging", BaggingClassifier(random_state=random_state)),
        ("SVC", SVC(probability=True, random_state=random_state)),
        ("K-Nearest Neighbors", KNeighborsClassifier()),
        ("Gaussian NB", GaussianNB()),
        #("MLP", MLPClassifier(max_iter=1000, random_state=random_state))
    ]
    
    results_list = []
    
    # ----- Evaluate Each Candidate Model on Cumulative Feature Subsets -----
    for model_name, model in models_to_test:
        best_model_score = -np.inf
        best_subset = None
        best_metrics = None
        print(f"\nEvaluating model: {model_name}\n{'-'*50}\n")
        for i in range(1, len(selected_order) + 1):
            subset = selected_order[:i]
            X_train_subset = X_train[subset]
            X_test_subset = X_test[subset]
            metrics = evaluate_model(clone(model), X_train_subset, X_test_subset, y_train, y_test, skf)
            print(f"Using {i} features: {subset}")
            print(f"Test Accuracy: {metrics['Test Accuracy']:.4f}, Test F1 Score: {metrics['Test F1 Score']:.4f}, Mean CV Accuracy: {metrics['Mean CV Accuracy']:.4f}, Test ROC AUC: {metrics['Test ROC AUC']}")
            print("Confusion Matrix:")
            print(metrics["Confusion Matrix"])
            print("Confusion Point:", metrics["Confusion Point"])
            print("Normalized Confusion Point:", metrics["Normalized Confusion Point"])
            print("\n")
            if metrics["Final Aggregated Score"] > best_model_score:
                best_model_score = metrics["Final Aggregated Score"]
                best_subset = subset
                best_metrics = metrics
        # Mark best result for this candidate model
        best_metrics["Model"] = model_name + f" (Best subset: {best_subset})"
        # Disqualify if performance gap is over threshold
        best_metrics["Disqualified"] = "Yes" if best_metrics["Performance Gap"] > overfitting_threshold else "No"
        results_list.append(best_metrics)
        print("\n" + "="*50 + "\n")
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(results_list)
    display_cols = ["Model", "Test Accuracy", "Test F1 Score", "Mean CV Accuracy", "Test ROC AUC", 
                    "Confusion Point", "Aggregated Score", "Final Aggregated Score", "Disqualified", "Analysis"]
    print("\n=== Analytical Summary ===\n")
    print(summary_df[display_cols].to_string(index=False))
    
    # ----- Winner Selection -----
    # Only consider non-disqualified models and then consider the top 5% confusion point values.
    non_disqualified = summary_df[summary_df["Disqualified"]=="No"]
    if not non_disqualified.empty:
        valid_cp = non_disqualified["Confusion Point"].astype(float)
        cp_threshold = np.percentile(valid_cp, 95)
        top_models = non_disqualified[valid_cp >= cp_threshold]
        if not top_models.empty:
            best_model_row = top_models.loc[top_models["Final Aggregated Score"].idxmax()]
        else:
            best_model_row = non_disqualified.loc[non_disqualified["Final Aggregated Score"].idxmax()]
    else:
        best_model_row = summary_df.loc[summary_df["Final Aggregated Score"].idxmax()]
    
    print("\n=== Winner Summary ===\n")
    winner_df = pd.DataFrame([best_model_row])
    print(winner_df[display_cols].to_string(index=False))
    print("\nFull Winner Details:")
    print(best_model_row.to_dict())
    
    return summary_df




# In[47]:


summary_df = all_modell_tester(df_2, target_column="Target_Good", test_size=0.2, random_state=52, 
                               cv=5, scoring='accuracy', overfitting_threshold=0.1, use_feature_subsets=True)


# ###  Okay, this is better. Let's try this out

# In[48]:


import joblib
from sklearn.ensemble import GradientBoostingClassifier
import ast

# Extract the winning model's details (from your output)
winning_model_name = "Gradient Boosting"
winning_feature_subset = ['Status_of_existing_checking_account_A14', 'Duration_in_month', 'Purpose_A41', 
                          'Credit_history_A31', 'Present_employment_since_A74', 'Savings_account_bonds_A65', 
                          'Other_debtors_guarantors_A103', 'Other_installment_plans_A143', 'Personal_status_and_sex_A92', 
                          'Status_of_existing_checking_account_A13', 'Savings_account_bonds_A64', 'Credit_amount', 
                          'Personal_status_and_sex_A93']

# Prepare the data
X = df_2[winning_feature_subset]  # Use the selected features
y = df_2["Target_Good"]

# Retrain the Gradient Boosting model on the full dataset
winning_model = GradientBoostingClassifier(random_state=52)
winning_model.fit(X, y)

# Save the model to disk
model_filename = "gradient_boosting_winning_model.pkl"
joblib.dump(winning_model, model_filename)
print(f"Winning model saved as '{model_filename}'.")


# ### Okay, we saved the model, and now we can publish it with Flask

# In[49]:


# app.py
from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load the pre-trained model
model = joblib.load("gradient_boosting_winning_model.pkl")

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Expects JSON input in the format:
    {
        "features": [feature1_value, feature2_value, ...]
    }
    and returns the prediction.
    """
    data = request.get_json(force=True)
    
    try:
        # Ensure the features are in the correct shape
        features = np.array(data["features"]).reshape(1, -1)
        prediction = model.predict(features)
        # You may also want to include probabilities if your model supports it
        return jsonify({"prediction": int(prediction[0])})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)


# ### Lets upload the model to github (via the terminal)

# In[51]:


# This is how we use it:

import joblib
import requests
from io import BytesIO

# URL to the raw file on GitHub
url = "https://raw.githubusercontent.com/OMGITSLACKO/Predictive-Credit-Risk-Analysis/main/gradient_boosting_winning_model.pkl"

response = requests.get(url)
response.raise_for_status()  # Ensure the request was successful

# Load the model directly from the response content
model = joblib.load(BytesIO(response.content))

# Example test cases with real-world features (based on German credit dataset)
test_cases = [
    [1, 18, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1169, 0],  # Example 1
    [0, 36, 0, 0, 0, 1, 1, 0, 0, 1, 0, 3000, 1],  # Example 2
]

# Predict using the loaded model
for i, features in enumerate(test_cases, 1):
    prediction = model.predict([features])
    print(f"Prediction for Example {i}: {prediction[0]} (0=Default, 1=Non-default)")


# 0 → Default.
# 1 → Non-default.

# In[1]:





# In[ ]:





# In[ ]:





# In[ ]:




