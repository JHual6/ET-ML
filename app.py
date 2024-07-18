import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Ruta del archivo CSV
cs = pd.read_csv('Anexo ET_demo_round_traces.csv', delimiter=';')

# Convertir TimeAlive y TravelledDistance a numérico, manejando errores y eliminando puntos decimales incorrectos
cs['TimeAlive'] = pd.to_numeric(cs['TimeAlive'].str.replace(
    '.', '', regex=False).str.replace(',', '', regex=False), errors='coerce')
cs['TravelledDistance'] = pd.to_numeric(cs['TravelledDistance'].str.replace(
    '.', '', regex=False).str.replace(',', '', regex=False), errors='coerce')

# Definir el rango deseado para la normalización
min_segundos = 0  # Valor mínimo en segundos
max_segundos = 155  # Valor máximo en segundos

# Aplicar Min-Max Scaling para normalizar los valores de TimeAlive
scaler = MinMaxScaler(feature_range=(min_segundos, max_segundos))
cs['TimeAlive_normalized_seconds'] = scaler.fit_transform(cs[['TimeAlive']])

# Calcular la media
media = cs['TimeAlive'].mean()

# Calcular la mediana
mediana = cs['TimeAlive'].median()

# Calcular el rango
# (diferencia entre el valor máximo y el valor mínimo)
rango = cs['TimeAlive_normalized_seconds'].max(
) - cs['TimeAlive_normalized_seconds'].min()

# Calcular la desviación estándar
# (Variación de los datos con respecto a la media)
desviacion_estandar = cs['TimeAlive_normalized_seconds'].std()

# Calcular la varianza
# (medida de dispersión que se utiliza para representar la variabilidad de un conjunto de datos respecto de la media aritmética)
varianza = cs['TimeAlive_normalized_seconds'].var()

# Calcular el coeficiente de variación
coeficiente_variacion = desviacion_estandar / media

# Sumar el TimeAlive_normalized_seconds por Team, MatchId y Mapa
time_alive_sum_by_team_match_map = cs.groupby(['Team', 'MatchId', 'Map'])[
    'TimeAlive_normalized_seconds'].sum()

# Calcular el promedio de las sumas por Team y Mapa
time_alive_avg_by_team_map = time_alive_sum_by_team_match_map.groupby([
                                                                      'Map', 'Team']).mean()

# Filtrar el DataFrame para incluir solo las filas donde MatchWinner no es nulo y es True
filtered_data = cs[cs['MatchWinner'].notnull() & cs['MatchWinner']]

# Eliminar duplicados basados en MatchId
filtered_data_unique = filtered_data.drop_duplicates(subset=['MatchId'])

# Seleccionar solo las columnas relevantes
relevant_columns = ['Team', 'MatchWinner', 'Survived',
                    'TimeAlive_normalized_seconds', 'TravelledDistance', 'RoundWinner', 'AbnormalMatch']

# Mostrar solo las columnas relevantes del DataFrame
relevant_data = cs[relevant_columns]
relevant_data.head()

# Eliminar filas con valores nulos
relevant_data = relevant_data.dropna()

# Agrupar por Mapa y Team y contar cuántas veces aparece MatchWinner = True
grouped_data = filtered_data_unique.groupby(['Map', 'Team']).size()

# Filtrar el DataFrame para incluir solo las filas donde MatchWinner no es nulo y es True
filtered_data = cs[cs['MatchWinner'].notnull() & cs['MatchWinner']]

# Eliminar duplicados basados en MatchId
filtered_data_unique = filtered_data.drop_duplicates(subset=['MatchId'])

# Agrupar por Mapa y Team y contar cuántas veces aparece MatchWinner = True
grouped_data = filtered_data_unique.groupby(['Map', 'Team']).size()

# Convertir el resultado en un DataFrame para facilitar la manipulación
grouped_data_df = grouped_data.reset_index(name='Count')

# GRAFICO DE PARTIDAS GANADAS POR EQUIPO Y AGRUPADO POR MAPA
fig_bar, ax_bar = plt.subplots(figsize=(14, 8))
pivot_table = grouped_data_df.pivot_table(index='Map', columns='Team', values='Count', fill_value=0)
pivot_table.plot(kind='bar', ax=ax_bar, color=['#dd8452', '#4682B4'])
ax_bar.set_title('Número de partidos ganadas por equipo y mapa', fontsize=24, pad=20, fontweight='bold')
ax_bar.set_xlabel('Mapa', fontsize=14, fontweight='bold')
ax_bar.set_ylabel('Número de partidos ganadas', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, fontsize=12)
plt.legend(title='Equipo', title_fontsize='13', fontsize='12')
plt.grid(axis='y', linestyle='--', alpha=0.7)
fig_bar.patch.set_facecolor('#D9D9D9')
ax_bar.set_facecolor('#D9D9D9')
plt.tight_layout()

# Crear las dos columnas, una más ancha que la otra
col1, col2 = st.columns([2, 1])

# Mostrar el gráfico de barras en la primera columna
with col1:
    st.pyplot(fig_bar)

# Crear un selector de opciones para el mapa con una opción vacía por defecto en la segunda columna
with col2:
    mapa_seleccionado = st.selectbox(
        'Seleccione un mapa para ver la distribución de victorias:', [''] + list(pivot_table.index)
    )

    # Cambiar el color de fondo del selectbox
    st.markdown(
        """
        <style>
        .stSelectbox div[data-baseweb="select"] > div {
            background-color: #D9D9D9;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Mostrar el gráfico circular solo si se ha seleccionado un mapa
    if mapa_seleccionado:
        # Crear gráfico circular según el mapa seleccionado
        filtered_data_mapa = filtered_data_unique[filtered_data_unique['Map'] == mapa_seleccionado]
        grouped_data = filtered_data_mapa.groupby(['Map', 'Team']).size()

        fig_pie, ax_pie = plt.subplots(figsize=(10, 6))
        ax_pie.pie(
            grouped_data,
            labels=grouped_data.index.get_level_values('Team'),
            autopct='%1.1f%%',
            startangle=140,
            colors=['#dd8452', '#4682B4']
        )
        ax_pie.axis('equal')
        ax_pie.set_title(
            f'Distribución de victorias en {mapa_seleccionado} por equipo',
            fontsize=24,
            pad=20,
            fontweight='bold'
        )
        fig_pie.patch.set_facecolor('#D9D9D9')
        ax_pie.set_facecolor('#D9D9D9')
        plt.tight_layout()
        st.pyplot(fig_pie)
        
st.markdown(
    """
    <style>
    .stSelectbox div[data-baseweb="select"] > div {
        background-color: #e38718;
        color: white;
        border-radius: 5px;
    }
    .stSelectbox div[data-baseweb="select"] > div > div {
        padding: 10px;
    }
    .stSelectbox div[data-baseweb="select"] > div > div > div {
        border-radius: 5px;
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