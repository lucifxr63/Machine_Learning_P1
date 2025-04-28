import os
import numpy as np
import pandas as pd
from ast import literal_eval
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# --- Configuración ---
max_len = 20
base_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(base_dir, "..", "DataSets", "DataSetTokenizados.csv")
model_dir = os.path.join(base_dir, "modelo_secuencial")

# --- Cargar modelos entrenados ---
modelo_CDM = load_model(os.path.join(model_dir, "modelo_CDM_secuencial.h5"))
modelo_Tipo = load_model(os.path.join(model_dir, "modelo_Tipo_secuencial.h5"))
modelo_GRD = load_model(os.path.join(model_dir, "modelo_GRD_secuencial.h5"))

# --- Cargar Label Encoders (opcional si quieres decodificar después)
classes_CDM = np.load(os.path.join(model_dir, "label_encoder_CDM_classes.npy"), allow_pickle=True)
classes_Tipo = np.load(os.path.join(model_dir, "label_encoder_Tipo_classes.npy"), allow_pickle=True)
classes_GRD = np.load(os.path.join(model_dir, "label_encoder_GRD_classes.npy"), allow_pickle=True)

# --- Cargar dataset ---
df = pd.read_csv(dataset_path)
for col in ['Diag_Principal_Token', 'Diag_Secundario_Token', 'Proced_Principal_Token', 'Proced_Secundario_Token']:
    df[col] = df[col].apply(literal_eval)

def preparar_tokens(columna):
    return np.array(df[columna].apply(lambda x: x[:max_len] + [0]*(max_len - len(x)) if len(x) < max_len else x[:max_len]).to_list())

diag_princ = preparar_tokens('Diag_Principal_Token')
diag_sec = preparar_tokens('Diag_Secundario_Token')
proc_princ = preparar_tokens('Proced_Principal_Token')
proc_sec = preparar_tokens('Proced_Secundario_Token')
X_extra = df[['Edad', 'Sexo_bin']].to_numpy()

# --- Predicciones en cascada ---
X_tokens = [diag_princ, diag_sec, proc_princ, proc_sec, X_extra]

# 1. Predecir CDM
pred_CDM = modelo_CDM.predict(X_tokens, verbose=0)
pred_CDM_classes = np.argmax(pred_CDM, axis=1).reshape(-1, 1)

# 2. Predecir Tipo usando Edad, Sexo y CDM predicho
X_tipo_input = np.hstack([X_extra, pred_CDM_classes])
pred_Tipo = modelo_Tipo.predict(X_tipo_input, verbose=0)
pred_Tipo_classes = np.argmax(pred_Tipo, axis=1).reshape(-1, 1)

# 3. Predecir GRD usando Edad, Sexo, CDM predicho y Tipo predicho
X_grd_input = np.hstack([X_extra, pred_CDM_classes, pred_Tipo_classes])
pred_GRD = modelo_GRD.predict(X_grd_input, verbose=0)
pred_GRD_classes = np.argmax(pred_GRD, axis=1)

# --- Crear DataFrame de comparación ---
comparacion = pd.DataFrame({
    "ID": df.index,
    "Real_CDM": df["CDM"],
    "Prediccion_CDM": classes_CDM[pred_CDM_classes.flatten()],
    "Real_Tipo": df["Tipo_GRD"],
    "Prediccion_Tipo": classes_Tipo[pred_Tipo_classes.flatten()],
    "Real_GRD": df["GRD_"],
    "Prediccion_GRD": classes_GRD[pred_GRD_classes]
})

# --- Guardar archivo CSV ---
output_csv = os.path.join(base_dir, "predicciones_finales.csv")
comparacion.to_csv(output_csv, index=False)
print(f"✅ Predicciones finales guardadas en: {output_csv}")
