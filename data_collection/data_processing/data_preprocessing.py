from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix


def one_hot_encode(df, categorical_columns):
    return pd.get_dummies(df, columns=categorical_columns)


# Create a new function or adjust your existing scale_features function here
def scale_features_post_split(x_train, x_val, x_test, feature_columns):
    scaler = StandardScaler()
    x_train[feature_columns] = scaler.fit_transform(x_train[feature_columns])
    x_val[feature_columns] = scaler.transform(x_val[feature_columns])
    x_test[feature_columns] = scaler.transform(x_test[feature_columns])
    return x_train, x_val, x_test


def split_data(df, target_column, test_size=0.2, validation_size=0.2, random_state=None):
    """
    Splits data into training, validation, and test sets, ensuring that the validation and test sets each are 20% of the total data.
    
    Parameters:
    - df: DataFrame containing your data.
    - target_column: The name of the target variable column.
    - test_size: Proportion of the dataset to include in the test split (20% of the total data).
    - validation_size: Proportion of the dataset to include in the validation split (20% of the total data).
    - random_state: Controls the shuffling applied to the data before applying the split. Pass an int for reproducible output across multiple function calls.
    
    Returns:
    - X_train, X_val, X_test, y_train, y_val, y_test: Split datasets.
    """
    
    # Split the data into training+temp and test sets
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    X_train_temp, X_test, y_train_temp, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Calculate the adjusted size for the validation split from the remaining data (to make sure validation is 20% of the total data)
    validation_size_adj = validation_size / (1 - test_size)
    
    # Split the remaining data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train_temp, y_train_temp, test_size=validation_size_adj, random_state=random_state)
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_and_evaluate_model(x_train, y_train):
    # Initialize and train the RandomForestClassifier
    model = RandomForestClassifier()
    model.fit(x_train, y_train)
    
    # Feature importances
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Plot the feature importances
    plt.figure(figsize=(10, 6))
    plt.title('Feature Importances')
    plt.bar(range(x_train.shape[1]), importances[indices], color="r", align="center")
    plt.xticks(range(x_train.shape[1]), x_train.columns[indices], rotation=90)
    plt.xlim([-1, x_train.shape[1]])
    plt.tight_layout()
    plt.show()
    
    # Predict on the training data
    y_pred_train = model.predict(x_train)
    
    # Generate the confusion matrix
    cm_train = confusion_matrix(y_train, y_pred_train)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_train, annot=True, fmt="d")
    plt.title('Confusion Matrix - Training Data')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
    
    # Calculate and print the accuracy
    accuracy = accuracy_score(y_train, y_pred_train)
    print(f"Model Accuracy on Training Data: {accuracy:.4f}")
    
    # Print the classification report
    print(classification_report(y_train, y_pred_train))
    

def entropy_based_feature_importance(X_train, y_train, feature_names, top_n=20):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    # Assuming feature_names is aligned with columns in X_train
    top_features = [feature_names[i] for i in indices[:top_n]]
    
    # Select data for the top N features
    X_train_top_n = X_train[top_features]
    
    # Create a DataFrame for feature importances
    feature_importance_df = pd.DataFrame({
        'Feature': np.array(feature_names)[indices][:top_n],  # Adjusted to use only top N features
        'Importance': importances[indices][:top_n]
    })
    
    # Plot the feature importances for the top N features
    plt.figure(figsize=(10, 6))
    plt.title('Feature Importances - Training Data')
    plt.bar(range(top_n), feature_importance_df['Importance'], align='center')
    plt.xticks(range(top_n), feature_importance_df['Feature'], rotation=90)
    plt.xlim([-1, top_n])
    plt.tight_layout()
    plt.show()

    # Retrain and evaluate the model using only the selected top N features
    model_selected = RandomForestClassifier()
    model_selected.fit(X_train_top_n, y_train)
    
    y_pred_train = model_selected.predict(X_train_top_n)
    cm_train = confusion_matrix(y_train, y_pred_train)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_train, annot=True, fmt="d")
    plt.title('Confusion Matrix - Training Data')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

    # Calculate and print the accuracy
    accuracy = np.trace(cm_train) / np.sum(cm_train)
    print(f"Model Accuracy on Training Data: {accuracy:.4f}")

    # Print the top N features and their importances
    print("Top N features based on importance:")
    print(feature_importance_df.head(top_n))

    # Print classification report
    print(classification_report(y_train, y_pred_train))

    # Return the names of the top_n features based on importance
    return top_features


def recursive_feature_elimination(x_train, y_train, n_features_to_select=20):
    model = RandomForestClassifier()
    rfe = RFE(estimator=model, n_features_to_select=n_features_to_select)
    rfe.fit(x_train, y_train)
    selected_features = x_train.columns[rfe.support_]
    
    # Print the selected features
    print("Top {} selected features:".format(n_features_to_select))
    for feature in selected_features:
        print(feature)
    
    # Optionally, retrain and evaluate the model using only the selected features...
    model_selected = RandomForestClassifier()
    model_selected.fit(x_train[selected_features], y_train)
    
    y_pred_train = model_selected.predict(x_train[selected_features])
    cm_train = confusion_matrix(y_train, y_pred_train)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_train, annot=True, fmt="d")
    plt.title('Confusion Matrix - Training Data')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
    
    # Calculate and print the accuracy
    accuracy = np.trace(cm_train) / np.sum(cm_train)
    print(f"Model Accuracy on Training Data: {accuracy:.4f}")

    # Print classification report
    print(classification_report(y_train, y_pred_train))

    # Visualize the ranking of features
    plt.figure(figsize=(10, 6))
    ranking = rfe.ranking_
    sns.barplot(x=np.arange(len(ranking)), y=ranking, order=np.argsort(ranking))
    plt.xlabel("Feature Index")
    plt.ylabel("Ranking (1=selected)")
    plt.tight_layout()
    plt.show()
    
    return selected_features


def apply_pca(x_train, y_train, n_components=20):
    # Apply PCA
    pca = PCA(n_components=n_components)
    x_train_pca = pca.fit_transform(x_train)
    
    print(f"Explained variance by {n_components} components:", np.sum(pca.explained_variance_ratio_))
    
    # Train the model using PCA-transformed training data
    model_pca = RandomForestClassifier()
    model_pca.fit(x_train_pca, y_train)
    
    # Feature importance and indices
    importances_pca = model_pca.feature_importances_
    indices_pca = np.argsort(importances_pca)[::-1]
    
    # Plot feature importances for PCA components
    plt.figure(figsize=(10, 6))
    plt.title('Feature Importances (PCA Components) - Training Data')
    plt.bar(range(n_components), importances_pca[indices_pca], align='center')
    plt.xticks(range(n_components), [f"PC {i+1}" for i in indices_pca], rotation=90)
    plt.xlabel('Principal Component Index')
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.show()
    
    # Predict and evaluate on the training set itself
    y_pred_train_pca = model_pca.predict(x_train_pca)
    
    # Confusion matrix for training data predictions
    cm_train_pca = confusion_matrix(y_train, y_pred_train_pca)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_train_pca, annot=True, fmt="d", cmap='Blues')
    plt.title('Confusion Matrix (PCA) - Training Data')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    # Accuracy and classification report for training data
    accuracy_train_pca = accuracy_score(y_train, y_pred_train_pca)
    print(f"Model Accuracy on Training Data (PCA): {accuracy_train_pca:.4f}")
    print(classification_report(y_train, y_pred_train_pca))
    
    return x_train_pca