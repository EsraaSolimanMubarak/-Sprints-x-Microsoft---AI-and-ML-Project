from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import joblib


def train_classifiers(X, y, preproc, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

    models = {
        'logreg': LogisticRegression(max_iter=1000),
        'dt': DecisionTreeClassifier(random_state=random_state),
        'rf': RandomForestClassifier(random_state=random_state),
        'svm': SVC(probability=True, random_state=random_state)
    }

    pipelines = {}
    results = {}

    for name, clf in models.items():
        pipe = Pipeline([('preproc', preproc), ('clf', clf)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:,1] if hasattr(pipe.named_steps['clf'], 'predict_proba') else None
        results[name] = {
            'model': pipe,
            'report': classification_report(y_test, y_pred, output_dict=True),
            'roc_auc': roc_auc_score(y_test, y_proba) if y_proba is not None else None
        }
        pipelines[name] = pipe
    return pipelines, results, (X_test, y_test)


def save_model(pipe, path='models/final_pipeline.pkl'):
    joblib.dump(pipe, path)
    print(f'Saved pipeline to {path}')