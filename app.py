"""App web Streamlit: entrena (si hace falta) y predice si un estudiante aprueba"""
import os
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from train import entrenar, FEATURES

st.set_page_config(page_title='Predicción de aprobación – MLP', page_icon='🎓')
st.title('🎓 ¿El estudiante aprobará?')
st.caption('Perceptrón Multicapa (MLP) entrenado con 600 estudiantes')

@st.cache_resource
def cargar_modelo():
    modelo, X_te, y_te, y_pred, acc = entrenar()
    return modelo, y_te, y_pred, acc

modelo, y_te, y_pred, acc = cargar_modelo()

st.subheader('Datos del estudiante')
c1, c2 = st.columns(2)
tareas = c1.slider('Tareas (0-100)', 0.0, 100.0, 70.0)
examenes = c1.slider('Exámenes (0-100)', 0.0, 100.0, 60.0)
asistencia = c1.slider('Asistencia (%)', 0.0, 100.0, 80.0)
estudio = c2.number_input('Horas de estudio por semana', 0.0, 40.0, 10.0)
sueno = c2.number_input('Horas de sueño por día', 0.0, 12.0, 7.0)
extra = c2.radio('Actividades extracurriculares', ['Sí', 'No'], horizontal=True)

if st.button('Predecir', type='primary'):
    entrada = pd.DataFrame([[tareas, examenes, asistencia, estudio, sueno, 1 if extra == 'Sí' else 0]],
                           columns=FEATURES)
    pred = modelo.predict(entrada)[0]
    prob = modelo.predict_proba(entrada)[0][1]
    if pred == 1:
        st.success(f'✅ APROBADO — probabilidad de aprobar: {prob:.1%}')
    else:
        st.error(f'❌ REPROBADO — probabilidad de aprobar: {prob:.1%}')

with st.expander('📊 Desempeño del modelo en el set de prueba'):
    st.metric('Accuracy', f'{acc:.2%}')
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(confusion_matrix(y_te, y_pred),
                           display_labels=['Reprobado', 'Aprobado']).plot(ax=ax, cmap='Greens')
    st.pyplot(fig)
    rep = classification_report(y_te, y_pred, target_names=['Reprobado', 'Aprobado'], output_dict=True)
    st.dataframe(pd.DataFrame(rep).T.round(2))
