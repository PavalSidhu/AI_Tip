from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

# Load the trained model
model_entropy = joblib.load('/home/pavalsidhu/AI_TIP/ai_model/models/class_random_forest_model_nsl_kdd_entropy.pkl')

# x_train_selected_entropy_nsl = joblib.load('/home/pavalsidhu/AI_TIP/ai_model/train_data/class_nsl_kdd_x_train_selected_entropy.pkl')
# y_train_nsl = joblib.load('/home/pavalsidhu/AI_TIP/ai_model/train_data/class_nsl_kdd_y_train.pkl')

# x_val_selected_entropy_nsl = joblib.load('/home/pavalsidhu/AI_TIP/ai_model/validation_data/class_nsl_kdd_x_val_selected_entropy.pkl')
# y_val_nsl = joblib.load('/home/pavalsidhu/AI_TIP/ai_model/validation_data/class_nsl_kdd_y_val.pkl')

# x_test_selected_entropy_nsl = joblib.load('/home/pavalsidhu/AI_TIP/ai_model/test_data/class_nsl_kdd_x_test_selected_entropy.pkl')
# y_test_nsl = joblib.load('/home/pavalsidhu/AI_TIP/ai_model/test_data/class_nsl_kdd_y_test.pkl')

# # Initialize classifiers
# rf_nsl = RandomForestClassifier()
# svm_nsl = SVC()
# knn_nsl = KNeighborsClassifier()

# # Define parameter grids
# param_grid_rf_nsl = {
#     'n_estimators': [100, 200],
#     'max_depth': [None, 10, 20],
#     'min_samples_split': [2, 5]
# }

# param_grid_svm_nsl = {
#     'C': [0.1, 1],
#     'kernel': ['linear', 'rbf']
# }

# param_grid_knn_nsl = {
#     'n_neighbors': [3, 5],
#     'weights': ['uniform', 'distance']
# }

# # Perform GridSearchCV
# def perform_grid_search(clf, param_grid, x_train, y_train, x_val, y_val):
#     grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, scoring='f1_macro', verbose=2, n_jobs=-1)
#     grid_search.fit(x_val, y_val)
#     best_params = grid_search.best_params_
#     print(f"Best parameters found: {best_params}")
#     clf.set_params(**best_params)
#     clf.fit(x_train, y_train)
#     y_pred_val = clf.predict(x_val)
#     return clf, y_pred_val


# # Evaluate models
# def evaluate_model(y_true, y_pred):
#     accuracy = accuracy_score(y_true, y_pred)
#     f1_micro = f1_score(y_true, y_pred, average='micro')
#     f1_macro = f1_score(y_true, y_pred, average='macro')
#     return accuracy, f1_micro, f1_macro


# # RandomForest
# rf_model_nsl, y_pred_val_rf_nsl = perform_grid_search(rf_nsl, param_grid_rf_nsl, x_train_selected_entropy_nsl, y_train_nsl, x_val_selected_entropy_nsl, y_val_nsl)
# accuracy_rf_nsl, f1_micro_rf_nsl, f1_macro_rf_nsl = evaluate_model(y_val_nsl, y_pred_val_rf_nsl)

# # SVM
# svm_model_nsl, y_pred_val_svm_nsl = perform_grid_search(svm_nsl, param_grid_svm_nsl, x_train_selected_entropy_nsl, y_train_nsl, x_val_selected_entropy_nsl, y_val_nsl)
# accuracy_svm_nsl, f1_micro_svm_nsl, f1_macro_svm_nsl = evaluate_model(y_val_nsl, y_pred_val_svm_nsl)

# # KNN
# knn_model_nsl, y_pred_val_knn_nsl = perform_grid_search(knn_nsl, param_grid_knn_nsl, x_train_selected_entropy_nsl, y_train_nsl, x_val_selected_entropy_nsl, y_val_nsl)
# accuracy_knn_nsl, f1_micro_knn_nsl, f1_macro_knn_nsl = evaluate_model(y_val_nsl, y_pred_val_knn_nsl)

# # Print results
# print(f"RandomForest - Accuracy: {accuracy_rf_nsl}, F1 Micro: {f1_micro_rf_nsl}, F1 Macro: {f1_macro_rf_nsl}")
# print(classification_report(y_val_nsl, y_pred_val_rf_nsl, digits=4))
# print(f"SVM - Accuracy: {accuracy_svm_nsl}, F1 Micro: {f1_micro_svm_nsl}, F1 Macro: {f1_macro_svm_nsl}")
# print(classification_report(y_val_nsl, y_pred_val_svm_nsl, digits=4))
# print(f"KNN - Accuracy: {accuracy_knn_nsl}, F1 Micro: {f1_micro_knn_nsl}, F1 Macro: {f1_macro_knn_nsl}")
# print(classification_report(y_val_nsl, y_pred_val_knn_nsl, digits=4))

# # Choose the best model based on F1 Macro (or any preferred metric)
# best_model = max([(rf_model_nsl, f1_macro_rf_nsl), (svm_model_nsl, f1_macro_svm_nsl), (knn_model_nsl, f1_macro_knn_nsl)], key=lambda item: item[1])
# print(f"Best model based on F1 Macro: {best_model[0].__class__.__name__} with F1 Macro: {best_model[1]:.4f}")

# # Save the final model
# joblib.dump(best_model[0], f'/home/pavalsidhu/AI_TIP/ai_model/models/label_{best_model[0].__class__.__name__}_model_nsl_kdd_final.pkl')
# joblib.dump(rf_model_nsl, f'/home/pavalsidhu/AI_TIP/ai_model/models/label_RandomForestClassifier_model_nsl_kdd.pkl')
# joblib.dump(svm_model_nsl, f'/home/pavalsidhu/AI_TIP/ai_model/models/label_SVC_model_nsl_kdd.pkl')
# joblib.dump(knn_model_nsl, f'/home/pavalsidhu/AI_TIP/ai_model/models/label_KNeighborsClassifier_model_nsl_kdd.pkl')


x_train_selected_entropy_nb = joblib.load('/home/pavalsidhu/AI_TIP/ai_model/train_data/attack_cat_nb_x_train_selected_entropy.pkl')
y_train_nb = joblib.load('/home/pavalsidhu/AI_TIP/ai_model/train_data/attack_cat_nb_y_train.pkl')

x_val_selected_entropy_nb = joblib.load('/home/pavalsidhu/AI_TIP/ai_model/validation_data/attack_cat_nb_x_val_selected_entropy.pkl')
y_val_nb = joblib.load('/home/pavalsidhu/AI_TIP/ai_model/validation_data/attack_cat_nb_y_val.pkl')

x_test_selected_entropy_nb = joblib.load('/home/pavalsidhu/AI_TIP/ai_model/test_data/attack_cat_nb_x_test_selected_entropy.pkl')
y_test_nb = joblib.load('/home/pavalsidhu/AI_TIP/ai_model/test_data/attack_cat_nb_y_test.pkl')

# # Initialize classifiers
rf_nb = RandomForestClassifier()
# svm_nb = SVC()
# knn_nb = KNeighborsClassifier()

# Define parameter grids
param_grid_rf_nb = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

# param_grid_svm_nb = {
#     'C': [0.1, 1],
#     'kernel': ['linear', 'rbf']
# }

# param_grid_knn_nb = {
#     'n_neighbors': [3, 5],
#     'weights': ['uniform', 'distance']
# }

# Perform GridSearchCV
def perform_grid_search(clf, param_grid, x_train, y_train, x_val, y_val):
    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, scoring='f1_macro', verbose=2, n_jobs=-1)
    grid_search.fit(x_val, y_val)
    best_params = grid_search.best_params_
    print(f"Best parameters found: {best_params}")
    clf.set_params(**best_params)
    clf.fit(x_train, y_train)
    y_pred_val = clf.predict(x_val)
    return clf, y_pred_val


# Evaluate models
def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    f1_micro = f1_score(y_true, y_pred, average='micro')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    return accuracy, f1_micro, f1_macro


# RandomForest
rf_model_nb, y_pred_val_rf_nb = perform_grid_search(rf_nb, param_grid_rf_nb, x_train_selected_entropy_nb, y_train_nb, x_val_selected_entropy_nb, y_val_nb)
accuracy_rf_nb, f1_micro_rf_nb, f1_macro_rf_nb = evaluate_model(y_val_nb, y_pred_val_rf_nb)

# # # SVM
# # svm_model_nb, y_pred_val_svm_nb = perform_grid_search(svm_nb, param_grid_svm_nb, x_train_selected_entropy_nb, y_train_nb, x_val_selected_entropy_nb, y_val_nb)
# # accuracy_svm_nb, f1_micro_svm_nb, f1_macro_svm_nb = evaluate_model(y_val_nb, y_pred_val_svm_nb)

# # KNN
# knn_model_nb, y_pred_val_knn_nb = perform_grid_search(knn_nb, param_grid_knn_nb, x_train_selected_entropy_nb, y_train_nb, x_val_selected_entropy_nb, y_val_nb)
# accuracy_knn_nb, f1_micro_knn_nb, f1_macro_knn_nb = evaluate_model(y_val_nb, y_pred_val_knn_nb)

# Print results
print(f"RandomForest - Accuracy: {accuracy_rf_nb}, F1 Micro: {f1_micro_rf_nb}, F1 Macro: {f1_macro_rf_nb}")
print(classification_report(y_val_nb, y_pred_val_rf_nb, digits=4))
# print(f"SVM - Accuracy: {accuracy_svm_nb}, F1 Micro: {f1_micro_svm_nb}, F1 Macro: {f1_macro_svm_nb}")
# print(classification_report(y_val_nb, y_pred_val_svm_nb, digits=4))
# print(f"KNN - Accuracy: {accuracy_knn_nb}, F1 Micro: {f1_micro_knn_nb}, F1 Macro: {f1_macro_knn_nb}")
# print(classification_report(y_val_nb, y_pred_val_knn_nb, digits=4))

# # # Choose the best model based on F1 Macro (or any preferred metric)
# # nb_best_model = max([(rf_model_nb, f1_macro_rf_nb), (svm_model_nb, f1_macro_svm_nb), (knn_model_nb, f1_macro_knn_nb)], key=lambda item: item[1])
# # print(f"Best model based on F1 Macro: {nb_best_model[0].__class__.__name__} with F1 Macro: {nb_best_model[1]:.4f}")
# # Choose the best model based on F1 Macro (or any preferred metric)
# nb_best_model = max([(rf_model_nb, f1_macro_rf_nb), (knn_model_nb, f1_macro_knn_nb)], key=lambda item: item[1])
# print(f"Best model based on F1 Macro: {nb_best_model[0].__class__.__name__} with F1 Macro: {nb_best_model[1]:.4f}")
# # Choose the best model based on F1 Macro (or any preferred metric)
nb_best_model = max([(rf_model_nb, f1_macro_rf_nb)], key=lambda item: item[1])
print(f"Best model based on F1 Macro: {nb_best_model[0].__class__.__name__} with F1 Macro: {nb_best_model[1]:.4f}")

# # Save the final model
# joblib.dump(nb_best_model[0], f'/home/pavalsidhu/AI_TIP/ai_model/models/attack_cat_{nb_best_model[0].__class__.__name__}_model_nb_final.pkl')
# joblib.dump(rf_model_nb, f'/home/pavalsidhu/AI_TIP/ai_model/models/attack_cat_RandomForestClassifier_model_nb.pkl')
# # joblib.dump(svm_model_nb, f'/home/pavalsidhu/AI_TIP/ai_model/models/attack_cat_SVC_model_nb.pkl')
# joblib.dump(knn_model_nb, f'/home/pavalsidhu/AI_TIP/ai_model/models/attack_cat_KNeighborsClassifier_model_nb.pkl')



