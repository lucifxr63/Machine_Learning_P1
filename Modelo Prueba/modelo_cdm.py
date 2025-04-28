import os
import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Concatenate, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# --- Configuración general ---
max_len = 20
vocab_size = 500
embedding_dim = 16
dense_units = 64
epochs = 50
batch_size = 128

# Crear carpeta para guardar modelo
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

# --- Preparar Target CDM ---
y_CDM = df['CDM']
le_CDM = LabelEncoder()
y_encoded = le_CDM.fit_transform(y_CDM)
num_classes_CDM = len(le_CDM.classes_)
y_cat = to_categorical(y_encoded, num_classes=num_classes_CDM)

# --- Preparar entradas para el modelo ---
X_inputs = [diag_princ, diag_sec, proc_princ, proc_sec, X_extra]

# División sincronizada
indices = np.arange(len(df))
train_idx, test_idx, y_train, y_test = train_test_split(
    indices,
    y_cat,
    test_size=0.2,
    random_state=42
)

X_train_split = [x[train_idx] for x in X_inputs]
X_test_split = [x[test_idx] for x in X_inputs]

# --- Construcción del modelo MLP ---
def construir_modelo(num_clases):
    inp1 = Input(shape=(max_len,))
    inp2 = Input(shape=(max_len,))
    inp3 = Input(shape=(max_len,))
    inp4 = Input(shape=(max_len,))
    extra = Input(shape=(2,))

    emb1 = Flatten()(Embedding(vocab_size, embedding_dim)(inp1))
    emb2 = Flatten()(Embedding(vocab_size, embedding_dim)(inp2))
    emb3 = Flatten()(Embedding(vocab_size, embedding_dim)(inp3))
    emb4 = Flatten()(Embedding(vocab_size, embedding_dim)(inp4))

    concat = Concatenate()([emb1, emb2, emb3, emb4, extra])
    d1 = Dense(dense_units, activation='relu')(concat)
    d2 = Dense(dense_units, activation='relu')(d1)
    d3 = Dense(dense_units, activation='relu')(d2)
    out = Dense(num_clases, activation='softmax')(d3)

    model = Model(inputs=[inp1, inp2, inp3, inp4, extra], outputs=out)
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model_CDM = construir_modelo(num_classes_CDM)

# --- Entrenar modelo ---
history = model_CDM.fit(
    X_train_split, y_train,
    validation_data=(X_test_split, y_test),
    epochs=epochs,
    batch_size=batch_size,
    verbose=2
)

# --- Guardar modelo y label encoder ---
model_CDM.save(os.path.join(base_dir, "modelo_secuencial", "modelo_CDM_secuencial.h5"))
np.save(os.path.join(base_dir, "modelo_secuencial", "label_encoder_CDM_classes.npy"), le_CDM.classes_)
print("✅ Modelo y label encoder de CDM guardados.")

# --- Graficar pérdida ---
plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label="Training Loss")
plt.plot(history.history['val_loss'], label="Validation Loss")
plt.title("Loss vs Epochs - Modelo CDM")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(base_dir, "modelo_secuencial", "loss_vs_epochs_CDM.png"))
plt.close()
print("✅ Gráfico de pérdida guardado.")
