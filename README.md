# Taller MLP – Predicción de aprobación de estudiantes

## Archivos
- `Taller_MLP.ipynb` – notebook con todo el taller (carga, imputación, escalado, MLP, matriz de confusión, conclusiones)
- `train.py` – servicio que entrena el modelo y lo guarda en `modelo_mlp.pkl`
- `app.py` – app web en Streamlit (formulario de features → predicción)
- `requirements.txt` – librerías necesarias
- `dataset_estudiantes_balanceado.csv` – datos

## Correr en tu computador
```
pip install -r requirements.txt
python train.py
streamlit run app.py
```
Se abre en http://localhost:8501

## Publicar y obtener el link del endpoint (gratis)
1. Crea un repositorio en GitHub y sube estos archivos (los 5 de arriba).
2. Entra a https://share.streamlit.io e inicia sesión con tu cuenta de GitHub.
3. Clic en **Create app** → elige el repositorio, rama `main` y archivo `app.py` → **Deploy**.
4. En 2–3 minutos te da un link tipo `https://tu-app.streamlit.app`. Ese es el link que adjuntas en el taller.
