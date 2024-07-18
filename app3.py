import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier

# Cargar los datos
cs = pd.read_csv('Anexo ET_demo_round_traces.csv', delimiter=';', low_memory=False)

# Convertir TimeAlive y TravelledDistance a numérico
cs['TimeAlive'] = pd.to_numeric(cs['TimeAlive'].str.replace('.', '', regex=False).str.replace(',', '', regex=False), errors='coerce')
cs['TravelledDistance'] = pd.to_numeric(cs['TravelledDistance'].str.replace('.', '', regex=False).str.replace(',', '', regex=False), errors='coerce')

# Filtrar datos no nulos en las columnas relevantes
cs = cs.dropna(subset=['TimeAlive', 'TravelledDistance', 'MatchWinner'])

# Limpiar la columna MatchWinner
cs['MatchWinner'] = cs['MatchWinner'].astype(str).str.strip().replace({'True': True, 'False': False, 'TRUE': True, 'FALSE': False})
cs['MatchWinner'] = cs['MatchWinner'].map({False: 0, True: 1})

# Filtrar nuevamente los datos no nulos
cs = cs.dropna(subset=['MatchWinner'])

# Seleccionar características y objetivo, excluyendo las columnas no numéricas
X = cs.select_dtypes(include=[np.number]).drop(columns=['MatchWinner'])
y = cs['MatchWinner']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar los datos
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Crear y entrenar el modelo Gradient Boosting con los mejores parámetros
best_gbc_model = GradientBoostingClassifier(
    learning_rate=0.5, 
    max_depth=9, 
    max_features=0.85, 
    min_samples_leaf=7, 
    min_samples_split=13, 
    n_estimators=100, 
    subsample=0.85,
    random_state=42
)
best_gbc_model.fit(X_train_scaled, y_train)

# Realizar predicciones con el modelo Gradient Boosting
gbc_preds = best_gbc_model.predict(X_test_scaled)
gbc_probs = best_gbc_model.predict_proba(X_test_scaled)[:, 1]

# Evaluar el modelo Gradient Boosting
accuracy_gbc = accuracy_score(y_test, gbc_preds)
precision_gbc = precision_score(y_test, gbc_preds)
recall_gbc = recall_score(y_test, gbc_preds)
f1_gbc = f1_score(y_test, gbc_preds)
roc_auc_gbc = roc_auc_score(y_test, gbc_probs)
conf_matrix_gbc = confusion_matrix(y_test, gbc_preds)
class_report_gbc = classification_report(y_test, gbc_preds)

# Interfaz de usuario de Streamlit
st.title('Predicción de Victoria de la partida utilizando Gradient Boosting Classifier')

# Precisión del modelo 
st.write(f"Precisión del modelo: {accuracy_gbc * 100:.2f}%")

# Inicializar la variable de estado de sesión
if 'show_chart1' not in st.session_state:
    st.session_state.show_chart1 = False

if 'show_chart2' not in st.session_state:
    st.session_state.show_chart2 = False

def toggle_chart1():
    st.session_state.show_chart1 = not st.session_state.show_chart1

def toggle_chart2():
    st.session_state.show_chart2 = not st.session_state.show_chart2

# Botones para mostrar/ocultar gráficos
col1, col2 = st.columns(2)

with col1:
    if st.button('Mostrar/Ocultar Matriz de Confusión'):
        toggle_chart1()

    if st.session_state.show_chart1:
        # Graficar la matriz de confusión
        fig, ax = plt.subplots(figsize=(8, 6), facecolor='#d9d9d9')
        sns.heatmap(conf_matrix_gbc, annot=True, fmt='d', cmap='Blues', xticklabels=['False', 'True'], yticklabels=['False', 'True'], ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix - Gradient Boosting')
        st.pyplot(fig)

with col2:
    if st.button('Mostrar/Ocultar Clasificación'):
        toggle_chart2()

    if st.session_state.show_chart2:
        # Aplicar PCA para reducir la dimensionalidad
        pca = PCA(n_components=2)
        X_test_pca = pca.fit_transform(X_test_scaled)

        # Graficar los resultados de la clasificación
        fig, ax = plt.subplots(figsize=(10, 6), facecolor='#d9d9d9')
        scatter = ax.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=gbc_preds, cmap='viridis', alpha=0.6)
        ax.set_xlabel('Componentes Principales')
        ax.set_ylabel('MatchWinner')
        ax.set_title('Resultados de la Clasificación Gradient Boosting')
        legend1 = ax.legend(*scatter.legend_elements(), title="MatchWinner")
        ax.add_artist(legend1)
        st.pyplot(fig)

# Sección para cargar un archivo y realizar predicciones
st.header('Realizar Nuevas Predicciones')

# Crear dos columnas: una para cargar archivos y otra para mostrar los resultados
upload_col, result_col = st.columns(2)

with upload_col:
    uploaded_file = st.file_uploader("Cargar archivo CSV", type="csv")

if uploaded_file is not None:
    new_data = pd.read_csv(uploaded_file, delimiter=';', low_memory=False)
    
    # Verificar si las columnas existen antes de procesarlas
    if 'TimeAlive' in new_data.columns:
        new_data['TimeAlive'] = pd.to_numeric(new_data['TimeAlive'].str.replace('.', '', regex=False).str.replace(',', '', regex=False), errors='coerce')
    if 'TravelledDistance' in new_data.columns:
        new_data['TravelledDistance'] = pd.to_numeric(new_data['TravelledDistance'].str.replace('.', '', regex=False).str.replace(',', '', regex=False), errors='coerce')
    
    # Verificar si las columnas requeridas existen
    required_columns = ['TimeAlive', 'TravelledDistance']
    missing_columns = [col for col in required_columns if col not in new_data.columns]
    
    if missing_columns:
        st.error(f"Las siguientes columnas faltan en el archivo cargado: {', '.join(missing_columns)}")
    else:
        # Filtrar filas con valores nulos en las columnas necesarias
        new_data = new_data.dropna(subset=required_columns)
        
        # Seleccionar las mismas columnas que se usaron para entrenar el modelo
        new_data_X = new_data[X_train.columns]
        
        # Normalizar los datos
        new_data_scaled = scaler.transform(new_data_X)
        
        # Realizar predicciones
        predictions = best_gbc_model.predict(new_data_scaled)
        prediction_probs = best_gbc_model.predict_proba(new_data_scaled)[:, 1]
        
        # Convertir predicciones a "Ganará" o "Perderá"
        new_data['Ganará o Perderá'] = predictions
        new_data['Ganará o Perderá'] = new_data['Ganará o Perderá'].map({0: 'Perderá', 1: 'Ganará'})
        new_data['Probabilidad'] = prediction_probs
        
        # Mostrar los resultados en la columna de resultados
        with result_col:
            st.write("Resultados de las predicciones:")
            st.write(new_data[['Ganará o Perderá', 'Probabilidad']])

st.markdown(
"""
    <style>
    .stButton button {
        background-color: #e38718;
        color: white;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.2);
    }
    .stButton button:hover, .stButton button:focus {
        background-color: #d97706;
        color: white;  /* Color del texto cuando el cursor está sobre el botón */
    }
    </style>
    """,
    unsafe_allow_html=True
) 

# Dejar todos los componentes para ocupar todo el espacio de la página
st.markdown(
    """
    <style>
    .st-emotion-cache-13ln4jf {
        max-width: none;
        background-color: #C3C3C3;
    }
    </style>
    """,
    unsafe_allow_html=True
)
