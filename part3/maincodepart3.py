import pandas as pd
import numpy as np


from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, roc_auc_score


import joblib
import warnings
warnings.filterwarnings('ignore')
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def main():
    print("- loading data -")
    df = pd.read_csv('../part1/cleaned_data.csv')
    df = df.drop(columns=['deck', 'alive']) 
    df = df.dropna()


    y_clf = df['survived']
    X = df.drop(columns=['fare', 'survived'])


    X['class'] = X['class'].map({'Third': 0, 'Second': 1, 'First': 2})
    X = pd.get_dummies(X, columns=['sex', 'embarked', 'embark_town', 'who', 'alone'], drop_first=True)
    

    X_train, X_test, y_clf_train, y_clf_test = train_test_split(
        X, y_clf, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)


    print("\n- Task 1: DT Baseline -")
    dt1 = DecisionTreeClassifier()
    dt1.fit(X_train_scaled, y_clf_train)
    print("train acc:", accuracy_score(y_clf_train, dt1.predict(X_train_scaled)))
    print("test acc:", accuracy_score(y_clf_test, dt1.predict(X_test_scaled)))
    

    print("\n- Task 2: Controlled DT -")
    dt2 = DecisionTreeClassifier(max_depth=5, min_samples_split=20)
    dt2.fit(X_train_scaled, y_clf_train)
    print("train acc:", accuracy_score(y_clf_train, dt2.predict(X_train_scaled)))
    print("test acc:", accuracy_score(y_clf_test, dt2.predict(X_test_scaled)))


    print("\n- Task 3: Gini vs Entropy -")
    dt_gini = DecisionTreeClassifier(max_depth=5, criterion='gini')
    dt_gini.fit(X_train_scaled, y_clf_train)
    print("gini test acc:", accuracy_score(y_clf_test, dt_gini.predict(X_test_scaled)))
    
    dt_entropy = DecisionTreeClassifier(max_depth=5, criterion='entropy')
    dt_entropy.fit(X_train_scaled, y_clf_train)
    print("entropy test acc:", accuracy_score(y_clf_test, dt_entropy.predict(X_test_scaled)))


    print("\n- Task 4: Random Forest -")
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train_scaled, y_clf_train)
    
    print("rf train acc:", accuracy_score(y_clf_train, rf.predict(X_train_scaled)))
    print("rf test acc:", accuracy_score(y_clf_test, rf.predict(X_test_scaled)))
    print("rf auc:", roc_auc_score(y_clf_test, rf.predict_proba(X_test_scaled)[:, 1]))
    
    importances = pd.Series(rf.feature_importances_, index=X.columns)
    print("top 5 features:")
    print(importances.nlargest(5))
    

    print("\n- Task 4a: Gradient Boosting -")
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    gb.fit(X_train_scaled, y_clf_train)
    print("gb train acc:", accuracy_score(y_clf_train, gb.predict(X_train_scaled)))
    print("gb test acc:", accuracy_score(y_clf_test, gb.predict(X_test_scaled)))
    print("gb auc:", roc_auc_score(y_clf_test, gb.predict_proba(X_test_scaled)[:, 1]))


    print("\n- Task 4b: Feature ablation -")
    worst_5 = importances.nsmallest(5).index.tolist()
    
    X_train_reduced = pd.DataFrame(X_train_scaled, columns=X.columns).drop(columns=worst_5)
    X_test_reduced = pd.DataFrame(X_test_scaled, columns=X.columns).drop(columns=worst_5)
    
    rf_reduced = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf_reduced.fit(X_train_reduced, y_clf_train)
    
    rf_reduced_auc = roc_auc_score(y_clf_test, rf_reduced.predict_proba(X_test_reduced)[:, 1])
    print("full rf auc:", roc_auc_score(y_clf_test, rf.predict_proba(X_test_scaled)[:, 1]))
    print("reduced rf auc:", rf_reduced_auc)


    print("\n- Task 5: CV Comparison -")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    models = {
        'Logistic': LogisticRegression(max_iter=1000, class_weight='balanced'),
        'DT Controlled': dt2,
        'Random Forest': rf,
        'Gradient Boosting': gb
    }
    
    for name, model in models.items():
        scores = cross_val_score(model, X_train_scaled, y_clf_train, cv=cv, scoring='roc_auc')
        print(f"{name} CV Mean: {scores.mean():.4f}, Std: {scores.std():.4f}")


    print("\n- Task 6: GridSearchCV Pipeline -")
    pipe = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), RandomForestClassifier(random_state=42))
    
    param_grid = {
        'randomforestclassifier__n_estimators': [50, 100, 200],
        'randomforestclassifier__max_depth': [5, 10, None],
        'randomforestclassifier__min_samples_leaf': [1, 5]
    }
    
    grid = GridSearchCV(pipe, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1)
    grid.fit(X_train, y_clf_train)
    
    print("best params:", grid.best_params_)
    print("best score:", grid.best_score_)


    print("\n- Task 7: Manual learning curve -")
    print("Training fraction | Training AUC | Test AUC")
    fractions = [0.2, 0.4, 0.6, 0.8, 1.0]
    best_pipe = grid.best_estimator_
    
    for f in fractions:
        n = int(f * len(X_train))
        X_sub = X_train.iloc[:n]
        y_sub = y_clf_train.iloc[:n]
        
        best_pipe.fit(X_sub, y_sub)
        
        train_auc = roc_auc_score(y_sub, best_pipe.predict_proba(X_sub)[:, 1])
        test_auc = roc_auc_score(y_clf_test, best_pipe.predict_proba(X_test)[:, 1])
        
        print(f"{f:.1f} | {train_auc:.4f} | {test_auc:.4f}")


    print("\n- Task 8: Serialize -")
    joblib.dump(best_pipe, 'best_model.pkl')
    
    loaded_model = joblib.load('best_model.pkl')
    
    test_rows = X_test.iloc[:2]
    preds = loaded_model.predict(test_rows)
    print("loaded model predictions on 2 rows:", preds)


if __name__ == "__main__":
    main()
