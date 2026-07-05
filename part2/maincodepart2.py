import pandas as pd
import numpy as np


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_curve, roc_auc_score


import matplotlib.pyplot as plt
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def main():
    print("loading data..")
    df = pd.read_csv('../part1/cleaned_data.csv')
    df = df.drop(columns=['deck', 'alive']) 
    df = df.dropna()


    y_reg = df['fare']
    y_clf = df['survived']
    X = df.drop(columns=['fare', 'survived'])


    X['class'] = X['class'].map({'Third': 0, 'Second': 1, 'First': 2})
    X = pd.get_dummies(X, columns=['sex', 'embarked', 'embark_town', 'who', 'alone'], drop_first=True)
    

    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        X, y_reg, y_clf, test_size=0.2, random_state=42
    )
    

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)


    print("\n--- Regression ---\n")
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_reg_train)
    preds_reg = lr.predict(X_test_scaled)
    

    mse_lr = mean_squared_error(y_reg_test, preds_reg)
    r2_lr = r2_score(y_reg_test, preds_reg)
    print(f"Linear Reg - MSE: {mse_lr:.2f}, R2: {r2_lr:.2f}")


    coefs = pd.DataFrame({'feature': X.columns, 'coef': lr.coef_})
    print("\nLR Coefs:\n", coefs)
    top_3 = coefs.assign(abs_coef=coefs['coef'].abs()).nlargest(3, 'abs_coef')
    print("\ntop 3 coefs:\n", top_3)


    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_scaled, y_reg_train)
    preds_ridge = ridge.predict(X_test_scaled)
    mse_ridge = mean_squared_error(y_reg_test, preds_ridge)
    r2_ridge = r2_score(y_reg_test, preds_ridge)
    print(f"\nRidge - MSE: {mse_ridge:.2f}, R2: {r2_ridge:.2f}")


    print("\n--- Classification ---\n")
    print("Imbalance check:")
    print(y_clf_train.value_counts(normalize=True))


    clf = LogisticRegression(max_iter=1000, class_weight='balanced')
    clf.fit(X_train_scaled, y_clf_train)
    preds_clf = clf.predict(X_test_scaled)
    probs_clf = clf.predict_proba(X_test_scaled)[:, 1]


    print("\nConfusion Matrix:\n", confusion_matrix(y_clf_test, preds_clf))
    print("\nReport:\n", classification_report(y_clf_test, preds_clf))


    fpr, tpr, _ = roc_curve(y_clf_test, probs_clf)
    auc_base = roc_auc_score(y_clf_test, probs_clf)
    print(f"AUC: {auc_base:.4f}")
    
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {auc_base:.2f}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig('roc_curve.png')
    plt.close()


    print("\nThreshold Sensitivity:")
    print("Threshold | Precision | Recall | F1")
    for t in [0.3, 0.4, 0.5, 0.6, 0.7]:
        t_preds = (probs_clf >= t).astype(int)
        p = precision_score(y_clf_test, t_preds)
        r = recall_score(y_clf_test, t_preds)
        f = f1_score(y_clf_test, t_preds)
        print(f"{t:.2f} | {p:.4f} | {r:.4f} | {f:.4f}")


    clf_strong = LogisticRegression(C=0.01, max_iter=1000, class_weight='balanced')
    clf_strong.fit(X_train_scaled, y_clf_train)
    preds_strong = clf_strong.predict(X_test_scaled)
    probs_strong = clf_strong.predict_proba(X_test_scaled)[:, 1]
    
    auc_strong = roc_auc_score(y_clf_test, probs_strong)
    print(f"\nC=0.01 Model - Precision: {precision_score(y_clf_test, preds_strong):.4f}, Recall: {recall_score(y_clf_test, preds_strong):.4f}, AUC: {auc_strong:.4f}")


    print("\nBootstrap for AUC...")
    diffs = []
    y_test_array = np.array(y_clf_test)
    for i in range(500):
        idx = np.random.choice(len(y_test_array), size=len(y_test_array), replace=True)
        sample_y = y_test_array[idx]
        sample_p_base = probs_clf[idx]
        sample_p_strong = probs_strong[idx]
        
        if len(np.unique(sample_y)) > 1: 
            a_base = roc_auc_score(sample_y, sample_p_base)
            a_strong = roc_auc_score(sample_y, sample_p_strong)
            diffs.append(a_base - a_strong)


    mean_diff = np.mean(diffs)
    p2_5 = np.percentile(diffs, 2.5)
    p97_5 = np.percentile(diffs, 97.5)
    print(f"AUC Diff Mean: {mean_diff:.4f}")
    print(f"2.5th: {p2_5:.4f}, 97.5th: {p97_5:.4f}")

if __name__ == "__main__":
    main()
