import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score
from tabpfn import TabPFNClassifier
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import warnings

# Ignorar warnings para una salida más limpia
warnings.filterwarnings('ignore', category=UserWarning)


def find_optimal_k(input_csv: str, seed: int = 42):
    """
    Prueba diferentes valores de k para SelectKBest y encuentra el número óptimo de características.
    """
    df = pd.read_csv(input_csv)
    y = df['BRS3'].astype(int)
    X_raw = df.drop(columns=['BRS3', 'patient_id'])

    # Preprocesamiento básico
    X = X_raw.apply(pd.to_numeric, errors="coerce").fillna(0)
    X[X < 0] = 0

    max_features = X.shape[1]
    k_values = range(1, max_features + 1)

    results = []

    print(f"Probando k desde 1 hasta {max_features}...")

    for k in k_values:
        print(f"  - Probando k={k}")
        # Usamos menos splits (ej. 5) para que la ejecución sea más rápida
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)

        fold_f1 = []
        fold_balanced_accuracy = []

        for train_idx, test_idx in kf.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            selector = SelectKBest(chi2, k=k)
            X_train_sel = selector.fit_transform(X_train, y_train)
            X_test_sel = selector.transform(X_test)

            smote = SMOTE(random_state=seed)
            X_train_res, y_train_res = smote.fit_resample(X_train_sel, y_train)

            # --- LÍNEA CORREGIDA ---
            # Se eliminó el argumento 'N_ensemble_configurations'
            classifier = TabPFNClassifier(device='cpu')

            classifier.fit(X_train_res, y_train_res)
            y_pred = classifier.predict(X_test_sel)

            fold_f1.append(f1_score(y_test, y_pred, average='binary', zero_division=0))
            fold_balanced_accuracy.append(balanced_accuracy_score(y_test, y_pred))

        results.append({
            'k': k,
            'mean_f1_score': np.mean(fold_f1),
            'mean_balanced_accuracy': np.mean(fold_balanced_accuracy)
        })

    results_df = pd.DataFrame(results)

    plt.figure(figsize=(12, 6))
    plt.plot(results_df['k'], results_df['mean_f1_score'], marker='s', label='Mean F1-Score (Clase 1)')
    plt.plot(results_df['k'], results_df['mean_balanced_accuracy'], marker='^', label='Mean Balanced Accuracy')

    best_k_f1 = results_df.loc[results_df['mean_f1_score'].idxmax()]
    plt.axvline(x=best_k_f1['k'], color='crimson', linestyle='--',
                label=f"Mejor k = {int(best_k_f1['k'])} (F1-Score: {best_k_f1['mean_f1_score']:.3f})")

    plt.title('Rendimiento del Modelo vs. Número de Características (k)')
    plt.xlabel('Número de Características (k)')
    plt.ylabel('Puntuación de la Métrica')
    plt.xticks(k_values)
    plt.grid(True)
    plt.legend()
    plt.savefig('k_features_performance.png')

    print("\n--- Resultados de la Optimización ---")
    print(results_df.round(4).to_string())
    print(f"\n🏆 Mejor rendimiento se obtuvo con k = {int(best_k_f1['k'])}")
    print("Gráfico guardado en 'k_features_performance.png'")


if __name__ == "__main__":
    # Asegúrate de que la ruta al archivo sea correcta para tu sistema
    try:
        find_optimal_k('dataset/clinical_data_BRS_binary.csv')
    except FileNotFoundError:
        print("\nError: No se encontró el archivo 'clinical_data_BRS_binary.csv'.")
        print("Por favor, asegúrate de que el script se ejecute desde el directorio correcto o ajusta la ruta.")

