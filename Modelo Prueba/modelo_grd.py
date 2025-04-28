import os
import pandas as pd
import numpy as np
from ast import literal_eval
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# --- Configuración general ---
max_len = 20
vocab_size = 500
embedding_dim = 16
dense_units = 64
epochs = 50
batch_size = 128

# Crear carpetas necesarias
base_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(base_dir, "modelo_secuencial"), exist_ok=True)

# --- Cargar dataset ---
dataset_path = os.path.join(base_dir, "..", "DataSets", "DataSetTokenizados.csv")
df = pd.read_csv(dataset_path)

# Procesar tokens
for col in ['Diag_Principal_Token', 'Diag_Secundario_Token', 'Proced_Principal_Token', 'Proced_Secundario_Token']:
    df[col] = df[col].apply(literal_eval)

def preparar_tokens(columna):
    return np.array(df[columna].apply(lambda x: x[:max_len] + [0]*(max_len - len(x)) if len(x) < max_len else x[:max_len]).to_list())

diag_princ = preparar_tokens('Diag_Principal_Token')
diag_sec = preparar_tokens('Diag_Secundario_Token')
proc_princ = preparar_tokens('Proced_Principal_Token')
proc_sec = preparar_tokens('Proced_Secundario_Token')
X_extra = df[['Edad', 'Sexo_bin']].to_numpy()

# --- Cargar modelos de CDM y Tipo ---
modelo_CDM = load_model(os.path.join(base_dir, "modelo_secuencial", "modelo_CDM_secuencial.h5"))
modelo_Tipo = load_model(os.path.join(base_dir, "modelo_secuencial", "modelo_Tipo_secuencial.h5"))

# --- Predicciones CDM y Tipo ---
X_tokens = [diag_princ, diag_sec, proc_princ, proc_sec, X_extra]
pred_CDM = modelo_CDM.predict(X_tokens, verbose=0)
pred_CDM_classes = np.argmax(pred_CDM, axis=1).reshape(-1, 1)

X_tipo_input = np.hstack([X_extra, pred_CDM_classes])
pred_Tipo = modelo_Tipo.predict(X_tipo_input, verbose=0)
pred_Tipo_classes = np.argmax(pred_Tipo, axis=1).reshape(-1, 1)

# --- Nueva entrada para GRD: Edad, Sexo, CDM predicho, Tipo predicho ---
X_full = np.hstack([X_extra, pred_CDM_classes, pred_Tipo_classes])

# --- Preparar Target GRD ---
y_GRD = df['GRD_']
le_GRD = LabelEncoder()
y_encoded = le_GRD.fit_transform(y_GRD)
num_clases_GRD = len(le_GRD.classes_)
y_cat = to_categorical(y_encoded, num_classes=num_clases_GRD)

# --- División train/test ---
X_train, X_test, y_train, y_test = train_test_split(X_full, y_cat, test_size=0.2, random_state=42)

# --- Construcción del modelo GRD ---
input_features = Input(shape=(4,))  # Edad, Sexo, CDM predicho, Tipo predicho
x = Dense(dense_units, activation='relu')(input_features)
x = Dense(dense_units, activation='relu')(x)
x = Dense(dense_units, activation='relu')(x)
output = Dense(num_clases_GRD, activation='softmax')(x)

modelo_GRD = Model(inputs=input_features, outputs=output)
modelo_GRD.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# --- Entrenar modelo ---
history = modelo_GRD.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=epochs,
    batch_size=batch_size,
    verbose=2
)

# --- Guardar modelo y label encoder ---
modelo_GRD.save(os.path.join(base_dir, "modelo_secuencial", "modelo_GRD_secuencial.h5"))
np.save(os.path.join(base_dir, "modelo_secuencial", "label_encoder_GRD_classes.npy"), le_GRD.classes_)
print("✅ Modelo de GRD guardado.")

# --- Graficar pérdida ---
plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label="Training Loss")
plt.plot(history.history['val_loss'], label="Validation Loss")
plt.title("Loss vs Epochs - Modelo GRD")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(base_dir, "modelo_secuencial", "loss_vs_epochs_GRD.png"))
plt.close()
print("✅ Gráfico de pérdida guardado.")
