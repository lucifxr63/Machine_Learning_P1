import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ast import literal_eval
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

# Crear carpeta para gráficos
os.makedirs("graficos_no_neuronales", exist_ok=True)

# Cargar dataset
df = pd.read_csv("DataSets/DataSetTokenizados.csv")
for col in ['Diag_Principal_Token', 'Diag_Secundario_Token', 'Proced_Principal_Token', 'Proced_Secundario_Token']:
    df[col] = df[col].apply(literal_eval)

def combinar_tokens(row, max_len=60):
    tokens = row['Diag_Principal_Token'] + row['Diag_Secundario_Token'] + \
             row['Proced_Principal_Token'] + row['Proced_Secundario_Token']
    return tokens[:max_len] + [0]*(max_len - len(tokens)) if len(tokens) < max_len else tokens[:max_len]

X_tokens = df.apply(combinar_tokens, axis=1)
X = np.stack(X_tokens.to_numpy())
X = np.hstack([X, df[['Edad', 'Sexo_bin']].to_numpy()])

targets = {
    "CDM": df["CDM"],
    "Tipo": df["Tipo_GRD"],
    "GRD": df["GRD_"]
}

modelos = {
    "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1),
    "Logistic Regression": LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='multinomial'),
    "Extra Trees": ExtraTreesClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
}

# Lista para almacenar resultados
resultados = []

for nombre_variable, y in targets.items():
    print(f"\n🔵 Variable objetivo: {nombre_variable}")
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    for nombre_modelo, modelo in modelos.items():
        print(f"\n✅ Entrenando {nombre_modelo}...")
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        p_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
        r_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
        p_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        r_weighted = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        print(f"• Accuracy: {acc:.2f}")
        print(f"• Precisión Ponderado: {p_weighted:.2f}")
        print(f"• Recall Ponderado: {r_weighted:.2f}")
        print(f"• F1-Score Ponderado: {f1_weighted:.2f}")
        print(f"• Precisión Macro: {p_macro:.2f}")
        print(f"• Recall Macro: {r_macro:.2f}")
        print(f"• F1-Score Macro: {f1_macro:.2f}")

        resultados.append({
            "Modelo": nombre_modelo,
            "Variable": nombre_variable,
            "Accuracy": acc,
            "Precision (macro)": p_macro,
            "Recall (macro)": r_macro,
            "F1 (macro)": f1_macro,
            "Precision (weighted)": p_weighted,
            "Recall (weighted)": r_weighted,
            "F1 (weighted)": f1_weighted
        })

# Guardar resultados en CSV
df_resultados = pd.DataFrame(resultados)
df_resultados.to_csv("resultados_no_neuronales.csv", index=False)
print("✅ Resultados guardados en 'resultados_no_neuronales.csv'")

# Crear gráficos de métricas
metricas = ["Accuracy", "F1 (macro)", "Precision (macro)", "Recall (macro)"]
for metrica in metricas:
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_resultados, x="Variable", y=metrica, hue="Modelo")
    plt.title(f"{metrica} por Variable y Modelo (No Neuronales)")
    plt.ylim(0, 1)
    plt.ylabel(metrica)
    plt.xlabel("Variable")
    plt.legend(title="Modelo")
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(f"graficos_no_neuronales/{metrica.replace(' ', '_')}.png")
    plt.close()

# Guardar tabla resumen de modelos no neuronales
pd.DataFrame([
    {
        "Modelo": "Random Forest",
        "Tipo": "Árbol de decisión en conjunto",
        "Uso": "Multiclase",
        "Ventajas": "Robusto a outliers, bueno con datos mixtos"
    },
    {
        "Modelo": "Logistic Regression",
        "Tipo": "Lineal (multinomial)",
        "Uso": "Multiclase",
        "Ventajas": "Rápido, interpretable"
    },
    {
        "Modelo": "Extra Trees",
        "Tipo": "Ensemble de árboles aleatorios",
        "Uso": "Multiclase",
        "Ventajas": "Reduce varianza, rápido entrenamiento"
    }
]).to_csv("graficos_no_neuronales/modelos_utilizados.csv", index=False)

print("✅ Gráficos y descripción guardados en 'graficos_no_neuronales/'")
