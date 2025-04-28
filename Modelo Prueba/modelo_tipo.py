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

# --- Cargar modelo de CDM y hacer predicciones ---
modelo_CDM_path = os.path.join(base_dir, "modelo_secuencial", "modelo_CDM_secuencial.h5")
modelo_CDM = load_model(modelo_CDM_path)

X_tokens = [diag_princ, diag_sec, proc_princ, proc_sec, X_extra]
pred_CDM = modelo_CDM.predict(X_tokens, verbose=0)
pred_CDM_classes = np.argmax(pred_CDM, axis=1).reshape(-1, 1)

# Añadir predicción de CDM como nueva feature
X_full = np.hstack([X_extra, pred_CDM_classes])

# --- Preparar Target Tipo ---
y_Tipo = df['Tipo_GRD']
le_Tipo = LabelEncoder()
y_encoded = le_Tipo.fit_transform(y_Tipo)
num_clases_Tipo = len(le_Tipo.classes_)
y_cat = to_categorical(y_encoded, num_classes=num_clases_Tipo)

# --- División train/test ---
X_train, X_test, y_train, y_test = train_test_split(X_full, y_cat, test_size=0.2, random_state=42)

# --- Construcción de modelo para Tipo ---
input_features = Input(shape=(3,))  # Edad, Sexo, CDM predicho
x = Dense(dense_units, activation='relu')(input_features)
x = Dense(dense_units, activation='relu')(x)
x = Dense(dense_units, activation='relu')(x)
output = Dense(num_clases_Tipo, activation='softmax')(x)

modelo_Tipo = Model(inputs=input_features, outputs=output)
modelo_Tipo.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# --- Entrenar modelo ---
history = modelo_Tipo.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=epochs,
    batch_size=batch_size,
    verbose=2
)

# --- Guardar modelo y label encoder ---
modelo_Tipo.save(os.path.join(base_dir, "modelo_secuencial", "modelo_Tipo_secuencial.h5"))
np.save(os.path.join(base_dir, "modelo_secuencial", "label_encoder_Tipo_classes.npy"), le_Tipo.classes_)
print("✅ Modelo y label encoder de Tipo guardados.")

# --- Graficar pérdida ---
plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label="Training Loss")
plt.plot(history.history['val_loss'], label="Validation Loss")
plt.title("Loss vs Epochs - Modelo Tipo")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(base_dir, "modelo_secuencial", "loss_vs_epochs_Tipo.png"))
plt.close()
print("✅ Gráfico de pérdida guardado.")
