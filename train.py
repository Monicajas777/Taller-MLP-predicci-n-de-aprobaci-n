"""Servicio de entrenamiento: entrena el MLP y guarda modelo_mlp.pkl"""
import pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

FEATURES = ['Tareas', 'Examenes', 'Asistencia', 'Estudio_Horas', 'Sueno_Horas', 'Extracurriculares']

def entrenar(ruta='dataset_estudiantes_balanceado.csv'):
    df = pd.read_csv(ruta)
    X, y = df[FEATURES], df['Aprobado']
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    modelo = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(hidden_layer_sizes=(16, 8), activation='relu', solver='adam',
                              alpha=0.01, max_iter=2000, random_state=42)),
    ])
    modelo.fit(X_tr, y_tr)
    y_pred = modelo.predict(X_te)
    joblib.dump(modelo, 'modelo_mlp.pkl')
    return modelo, X_te, y_te, y_pred, accuracy_score(y_te, y_pred)

if __name__ == '__main__':
    _, _, _, _, acc = entrenar()
    print(f'Modelo entrenado y guardado. Accuracy en prueba: {acc:.4f}')
