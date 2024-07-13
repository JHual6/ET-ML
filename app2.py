import streamlit as st
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# Definir funciones para cargar el modelo y el escalador, y para preprocesar y predecir
def load_model_and_scaler(model_path, scaler_path):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

def preprocess_and_predict(new_data, model, scaler, categorical_columns, X_columns):
    # Codificar nuevas características
    new_data_encoded = pd.get_dummies(new_data, columns=categorical_columns)
    
    # Asegurarse de que las nuevas características tengan las mismas columnas que las usadas en el entrenamiento
    missing_cols = set(X_columns) - set(new_data_encoded.columns)
    for col in missing_cols:
        new_data_encoded[col] = 0
    new_data_encoded = new_data_encoded[X_columns]

    # Escalar características
    new_data_scaled = scaler.transform(new_data_encoded)
    
    # Realizar predicciones
    predictions = model.predict(new_data_scaled)
    return predictions

# Ruta del archivo CSV
cs = pd.read_csv('Anexo ET_demo_round_traces.csv', delimiter=';')

# Convertir TimeAlive y TravelledDistance a numérico, manejando errores y eliminando puntos decimales incorrectos
cs['TimeAlive'] = pd.to_numeric(cs['TimeAlive'].str.replace('.', '', regex=False).str.replace(',', '', regex=False), errors='coerce')
cs['TravelledDistance'] = pd.to_numeric(cs['TravelledDistance'].str.replace('.', '', regex=False).str.replace(',', '', regex=False), errors='coerce')

# Imputar valores nulos con la mediana de cada columna
cs['TimeAlive'].fillna(cs['TimeAlive'].median(), inplace=True)
cs['TravelledDistance'].fillna(cs['TravelledDistance'].median(), inplace=True)

# Verificar que no queden valores nulos
null_values_final = cs.isnull().sum()

# Definir el rango deseado para la normalización
min_segundos = 0  # Valor mínimo en segundos
max_segundos = 155  # Valor máximo en segundos

# Aplicar Min-Max Scaling para normalizar los valores de TimeAlive
scaler = MinMaxScaler(feature_range=(min_segundos, max_segundos))
cs['TimeAlive_normalized_seconds'] = scaler.fit_transform(cs[['TimeAlive']])

# Eliminar filas con valores vacíos en la columna MatchWinner
cs.dropna(subset=['MatchWinner'], inplace=True)

# Verificar que no queden valores nulos en la columna MatchWinner
null_values_final = cs['MatchWinner'].isnull().sum()

# Seleccionar características y variable objetivo
features = cs.drop(columns=['MatchKills', 'TimeAlive', 'Unnamed: 0', 'Team', 'MatchId', 'RoundId', 'AbnormalMatch', 'RLethalGrenadesThrown', 'RNonLethalGrenadesThrown', 'FirstKillTime', 'TeamStartingEquipmentValue'])
target = cs['MatchKills']

# Convertir todas las columnas booleanas y categóricas a cadenas
for col in features.columns:
    if features[col].dtype == 'bool' or features[col].dtype == 'object':
        features[col] = features[col].astype(str)

# Seleccionar características categóricas
categorical_cols = features.select_dtypes(include=['object']).columns

# Reducir la cardinalidad seleccionando las categorías más frecuentes
for col in categorical_cols:
    top_categories = features[col].value_counts().index[:10]
    features[col] = features[col].apply(lambda x: x if x in top_categories else 'Other')

# Convertir características categóricas a valores numéricos utilizando LabelEncoder
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    features[col] = le.fit_transform(features[col])
    label_encoders[col] = le

# Dividir el conjunto de datos en entrenamiento y prueba (80% entrenamiento, 20% prueba) de forma aleatoria
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Normalizar las características numéricas
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Crear y entrenar el modelo de Random Forest Regressor con menos estimadores
rf_model = RandomForestRegressor(n_estimators=50, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Guardar el modelo entrenado y el escalador
joblib.dump(rf_model, 'best_rf_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Realizar predicciones con el modelo Random Forest Regressor
rf_preds = rf_model.predict(X_test_scaled)

# Evaluar el desempeño del modelo Random Forest Regressor
rf_mse = mean_squared_error(y_test, rf_preds)
rf_rmse = mean_squared_error(y_test, rf_preds, squared=False)
rf_mae = mean_absolute_error(y_test, rf_preds)
rf_r2 = r2_score(y_test, rf_preds)

# Interfaz de usuario de Streamlit
st.title('Predicción de Bajas en la partida utilizando Random Forest Regressor')

# Mostrar la precisión del modelo
st.write(f'Precisión del modelo: {rf_r2 * 100:.2f}%')


# Inicializar la variable de estado de sesión
if 'show_chart' not in st.session_state:
    st.session_state.show_chart = False

def toggle_chart():
    st.session_state.show_chart = not st.session_state.show_chart

# GRAFICO DEL MODELO RANDOM FOREST REGRESSOR
# Crear el gráfico de dispersión
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(y_test, rf_preds, color='green', alpha=0.5)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
ax.set_xlabel('Valores Reales')
ax.set_ylabel('Predicciones')
ax.set_title('Gráfico de Dispersión: Valores Reales vs. Predicciones (Random Forest Regressor)', fontsize=18)
fig.patch.set_facecolor('#D9D9D9')  # Fondo del gráfico

# Botón para mostrar/ocultar el gráfico
if st.button('Mostrar gráfico' if not st.session_state.show_chart else 'Ocultar gráfico'):
    toggle_chart()

# Mostrar u ocultar el gráfico basado en el estado de sesión
if st.session_state.show_chart:
    st.pyplot(fig)

# Sección para ingresar nuevos datos y realizar predicciones
st.header('Realizar Nuevas Predicciones')
new_data_input = {}

# Agregar campos de entrada para cada característica en filas de dos columnas
cols = st.columns(5)
with cols[0]:
    
    #MAPA
    # Diccionario para mapear los valores codificados a los nombres originales
    map_label_mapping = {0: 'de_dust2', 1: 'de_inferno', 2: 'de_mirage', 3: 'de_nuke'}
    # Invertir el diccionario para obtener el mapeo de nombres a valores codificados
    map_name_to_value = {v: k for k, v in map_label_mapping.items()}
    # Actualizar el selectbox para mostrar los nombres de los mapas
    selected_map_name = st.selectbox('Mapa', options=list(map_label_mapping.values()))
    # Obtener el valor codificado correspondiente al nombre seleccionado
    new_data_input['Map'] = map_name_to_value[selected_map_name]

    #EQUIPO
    # Crear un diccionario de mapeo para InternalTeamId
    team_mapping = {1: 'Terrorist', 2: 'Counter-Strike'}
    # Obtener los valores únicos y mapearlos a sus etiquetas
    unique_teams = [team_mapping[val] for val in features['InternalTeamId'].unique()]
    # Crear el selectbox usando las etiquetas mapeadas
    selected_team = st.selectbox('Equipo', options=unique_teams)
    # Obtener el valor original a partir de la etiqueta seleccionada
    original_value = {v: k for k, v in team_mapping.items()}[selected_team]
    new_data_input['InternalTeamId'] = original_value

    new_data_input['TimeAlive_normalized_seconds'] = st.number_input('Tiempo vivo', value=float(features['TimeAlive_normalized_seconds'].median()))
    new_data_input['TravelledDistance'] = st.number_input('Distancia Recorrida', value=float(features['TravelledDistance'].median()), step=1.0)

with cols[1]:

    #ARMA OCUPADA
    # Lista de todas las opciones posibles
    weapon_options = {
        'Ocupó rifle de asalto': 'PrimaryAssaultRifle',
        'Ocupó rifle francotirador': 'PrimarySniperRifle',
        'Ocupó una Heavy Machine Gun': 'PrimaryHeavy',
        'Ocupó una SMG': 'PrimarySMG',
        'Ocupó pistola': 'PrimaryPistol'
    }

    # Crear el selectbox con todas las opciones
    selected_weapon = st.selectbox('Seleccione el arma usada', options=list(weapon_options.keys()))

    # Asignar valores de 1 o 0 a las variables según la selección
    for weapon, column in weapon_options.items():
        if selected_weapon == weapon:
            new_data_input[column] = 1
        else:
            new_data_input[column] = 0

    #SOBREVIVIÓ
    survived_mapping = {0: 'No', 1: 'Si'}
    # Obtener los valores únicos y mapearlos a sus etiquetas
    unique_survived = [survived_mapping[val] for val in features['Survived'].unique()]
    # Crear el selectbox usando las etiquetas mapeadas
    selected_survived = st.selectbox('Sobrevivió', options=unique_survived)
    # Obtener el valor original a partir de la etiqueta seleccionada
    original_value_s = {v: k for k, v in survived_mapping.items()}[selected_survived]
    new_data_input['Survived'] = original_value_s

    new_data_input['RoundStartingEquipmentValue'] = st.number_input('Valor del equipamiento en la ronda', value=float(features['RoundStartingEquipmentValue'].median()), step=100.0)

with cols[2]:    

    new_data_input['RoundKills'] = st.number_input('Muertes en la ronda', value=float(features['RoundKills'].median()), step=1.0)

    #GANÓ LA RONDA
    # Cambiar los valores de 1 a 0 en la columna RoundWinner
    features['RoundWinner'] = features['RoundWinner'].replace(1, 0)

    # GANÓ LA RONDA
    roundwinner_mapping = {0: 'No', 2: 'Sí'} 
    # Obtener los valores únicos y mapearlos a sus etiquetas
    unique_roundwinner = [roundwinner_mapping[val] for val in features['RoundWinner'].unique()]
    # Crear el selectbox usando las etiquetas mapeadas
    selected_roundwinner = st.selectbox('Ganó la ronda', options=unique_roundwinner)
    # Obtener el valor original a partir de la etiqueta seleccionada
    original_value_rw = {v: k for k, v in roundwinner_mapping.items()}[selected_roundwinner]
    # Asignar el valor original al diccionario de entrada de nuevos datos
    new_data_input['RoundWinner'] = original_value_rw

    # GANÓ LA PARTIDA
    matchwinner_mapping = {0: 'No', 1: 'Sí'} 
    # Obtener los valores únicos y mapearlos a sus etiquetas
    unique_matchwinner = [matchwinner_mapping[val] for val in features['MatchWinner'].unique()]
    # Crear el selectbox usando las etiquetas mapeadas
    selected_matchwinner = st.selectbox('Ganó la partida', options=unique_matchwinner)
    # Obtener el valor original a partir de la etiqueta seleccionada
    original_value_mw = {v: k for k, v in matchwinner_mapping.items()}[selected_matchwinner]
    # Asignar el valor original al diccionario de entrada de nuevos datos
    new_data_input['MatchWinner'] = original_value_mw


with cols[3]:    
    new_data_input['RoundFlankKills'] = st.number_input('Muertes de cerca en la ronda', value=float(features['RoundFlankKills'].median()), step=1.0)
    new_data_input['MatchFlankKills'] = st.number_input('Muertes de cercanía en la partida', value=float(features['MatchFlankKills'].median()), step=1.0)
    new_data_input['RoundAssists'] = st.number_input('Asistencias en la ronda', value=float(features['RoundAssists'].median()), step=1.0)
with cols[4]:
    new_data_input['MatchAssists'] = st.number_input('Asistencias en la partida', value=float(features['MatchAssists'].median()), step=1.0)
    new_data_input['RoundHeadshots'] = st.number_input('Disparos en la cabeza en la ronda', value=float(features['RoundHeadshots'].median()), step=1.0)
    new_data_input['MatchHeadshots'] = st.number_input('Disparos en la cabeza en la partida', value=float(features['MatchHeadshots'].median()), step=1.0)

# Convertir la entrada de datos en un DataFrame
new_data_df = pd.DataFrame(new_data_input, index=[0])

if st.button('Predecir'):
    model, scaler = load_model_and_scaler('best_rf_model.pkl', 'scaler.pkl')
    predictions = preprocess_and_predict(new_data_df, model, scaler, categorical_cols, X_train.columns)
    st.write(f'Predicción de MatchKills: {predictions[0]:.2f}')

# Aplicar el estilo CSS específico para selectbox y number_input
st.markdown(
    """
    <style>
    .stSelectbox div[data-baseweb="select"], .stNumberInput div[data-baseweb="input"] {
        width: 180px;  /* Ajusta el ancho según sea necesario */
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
    }
    </style>
    """,
    unsafe_allow_html=True
)
