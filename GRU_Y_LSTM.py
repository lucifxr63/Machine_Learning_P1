import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from ast import literal_eval
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, GRU, LSTM, Dense, Concatenate
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

# Configuración
max_len = 20
vocab_size = 500
embedding_dim = 16
units = 64
epochs = 10
batch_size = 128
os.makedirs("graficos_rnn", exist_ok=True)

# Cargar y preparar datos
df = pd.read_csv("DataSets/DataSetTokenizados.csv")
for col in ['Diag_Principal_Token', 'Diag_Secundario_Token', 'Proced_Principal_Token', 'Proced_Secundario_Token']:
    df[col] = df[col].apply(literal_eval)

def preparar_entrada(col):
    return np.array(df[col].apply(lambda x: x[:max_len] + [0]*(max_len - len(x)) if len(x) < max_len else x[:max_len]).to_list())

diag_princ = preparar_entrada('Diag_Principal_Token')
diag_sec = preparar_entrada('Diag_Secundario_Token')
proc_princ = preparar_entrada('Proced_Principal_Token')
proc_sec = preparar_entrada('Proced_Secundario_Token')
X_extra = df[['Edad', 'Sexo_bin']].to_numpy()

# Construcción de modelos GRU/LSTM
def construir_modelo(num_clases, tipo='gru'):
    RNN = GRU if tipo == 'gru' else LSTM
    inp1 = Input(shape=(max_len,))
    inp2 = Input(shape=(max_len,))
    inp3 = Input(shape=(max_len,))
    inp4 = Input(shape=(max_len,))
    extra = Input(shape=(2,))
    emb1 = Embedding(vocab_size, embedding_dim)(inp1)
    emb2 = Embedding(vocab_size, embedding_dim)(inp2)
    emb3 = Embedding(vocab_size, embedding_dim)(inp3)
    emb4 = Embedding(vocab_size, embedding_dim)(inp4)
    rnn1 = RNN(units)(emb1)
    rnn2 = RNN(units)(emb2)
    rnn3 = RNN(units)(emb3)
    rnn4 = RNN(units)(emb4)
    concat = Concatenate()([rnn1, rnn2, rnn3, rnn4, extra])
    d1 = Dense(units, activation='relu')(concat)
    d2 = Dense(units, activation='relu')(d1)
    d3 = Dense(units, activation='relu')(d2)
    out = Dense(num_clases, activation='softmax')(d3)
    model = Model(inputs=[inp1, inp2, inp3, inp4, extra], outputs=out)
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Variables objetivo
targets = {
    "CDM": df["CDM"],
    "Tipo": df["Tipo_GRD"],
    "GRD": df["GRD_"]
}

resultados = []
loss_histories = {}
val_loss_histories = {}

for tipo_rnn in ['gru', 'lstm']:
    for nombre_var, y in targets.items():
        print(f"\n🔵 Entrenando {tipo_rnn.upper()} para {nombre_var}")

        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        num_classes = len(le.classes_)
        y_cat = to_categorical(y_encoded, num_classes)

        diag1_train, diag1_test, diag2_train, diag2_test, proc1_train, proc1_test, \
        proc2_train, proc2_test, extra_train, extra_test, y_train, y_test = train_test_split(
            diag_princ, diag_sec, proc_princ, proc_sec, X_extra, y_cat,
            test_size=0.2, random_state=42
        )

        X_train = [diag1_train, diag2_train, proc1_train, proc2_train, extra_train]
        X_test = [diag1_test, diag2_test, proc1_test, proc2_test, extra_test]

        model = construir_modelo(num_classes, tipo=tipo_rnn)
        history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                            epochs=epochs, batch_size=batch_size, verbose=2)

        loss_histories[(tipo_rnn, nombre_var)] = history.history['loss']
        val_loss_histories[(tipo_rnn, nombre_var)] = history.history['val_loss']

        y_pred_probs = model.predict(X_test)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = np.argmax(y_test, axis=1)

        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        resultados.append({
            "Modelo": tipo_rnn.upper(),
            "Variable": nombre_var,
            "Accuracy": report["accuracy"],
            "Precision (macro)": report["macro avg"]["precision"],
            "Recall (macro)": report["macro avg"]["recall"],
            "F1 (macro)": report["macro avg"]["f1-score"],
            "Precision (weighted)": report["weighted avg"]["precision"],
            "Recall (weighted)": report["weighted avg"]["recall"],
            "F1 (weighted)": report["weighted avg"]["f1-score"]
        })

        # Matriz de confusión
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='Blues', values_format='d')
        plt.title(f'Matriz de Confusión - {nombre_var} ({tipo_rnn.upper()})')
        plt.savefig(f"graficos_rnn/confusion_matrix_{nombre_var}_{tipo_rnn}.png")
        plt.close()

        # Curva ROC
        y_test_bin = label_binarize(y_true, classes=list(range(num_classes)))
        fpr, tpr, roc_auc = {}, {}, {}
        plt.figure(figsize=(8, 6))
        for i in range(num_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            plt.plot(fpr[i], tpr[i], lw=2, label=f'Clase {i} (AUC = {roc_auc[i]:.2f})')

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Tasa de Falsos Positivos')
        plt.ylabel('Tasa de Verdaderos Positivos')
        plt.title(f'Curva ROC - {nombre_var} ({tipo_rnn.upper()})')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig(f"graficos_rnn/curva_roc_{nombre_var}_{tipo_rnn}.png")
        plt.close()

# Guardar CSV de resultados
pd.DataFrame(resultados).to_csv("resultados_rnn.csv", index=False)

# Guardar gráficos de pérdida
for key in loss_histories.keys():
    tipo, var = key
    plt.figure(figsize=(8, 5))
    plt.plot(loss_histories[key], label='Training Loss')
    plt.plot(val_loss_histories[key], label='Validation Loss')
    plt.title(f'Loss vs Epochs - {var} ({tipo.upper()})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"graficos_rnn/loss_vs_epochs_{tipo}_{var}.png")
    plt.close()

# Guardar estructura de modelos
pd.DataFrame([
    {
        "Modelo": "GRU",
        "Estructura": "4 embeddings → GRU → concat Edad/Sexo → Dense x3 → softmax",
        "Entradas": "Tokens + Edad + Sexo",
        "Salidas": "CDM, Tipo, GRD"
    },
    {
        "Modelo": "LSTM",
        "Estructura": "4 embeddings → LSTM → concat Edad/Sexo → Dense x3 → softmax",
        "Entradas": "Tokens + Edad + Sexo",
        "Salidas": "CDM, Tipo, GRD"
    }
]).to_csv("graficos_rnn/modelos_utilizados.csv", index=False)
