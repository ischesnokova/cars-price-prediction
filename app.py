import streamlit as st
import pandas as pd
import plotly.express as px
import pickle
from pathlib import Path
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Предсказание цен на машины",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MEDIAN_PATH = MODEL_DIR / "medians.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
MODEL_PATH = MODEL_DIR / "churn_model.pkl"
FEATURE_NAMES_PATH = MODEL_DIR / "feature_names.pkl"


@st.cache_resource
def load_model():
    """Загружаем модель через pickle"""

    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    with open(MEDIAN_PATH, 'rb') as f:
        medians = pickle.load(f)
    with open(FEATURE_NAMES_PATH, 'rb') as f:
        feature_names = pickle.load(f)
    return model, scaler, medians, feature_names

# --- Основной интерфейс ---
st.title("🚗 Предсказание цен на машины")


model, scaler, medians, feature_names = load_model()

#Нарисуем графики по train
#Загрузим датасет train
@st.cache_data
def load_train():
    url = 'https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_train.csv'
    df_train = pd.read_csv(url)
    return df_train

@st.cache_data
def obrabotka_train():
    url = 'https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_train.csv'
    df_train_new = pd.read_csv(url)
   #Нужно убрать все лишние единицы измерения
    processed_train = df_train_new.copy()
    
    #Убираем ' kmpl' из mileage
    if 'mileage' in processed_train.columns:
        if processed_train['mileage'].dtype == 'object':
            processed_train['mileage'] = processed_train['mileage'].str.replace(' kmpl', '', regex=False)
        processed_train['mileage'] = pd.to_numeric(processed_train['mileage'], errors='coerce')
    
    #Убираем ' CC' из engine
    if 'engine' in processed_train.columns:
        if processed_train['engine'].dtype == 'object':
            processed_train['engine'] = processed_train['engine'].str.replace(' CC', '', regex=False)
        processed_train['engine'] = pd.to_numeric(processed_train['engine'], errors='coerce')
    
    #Убираем ' bhp' для max_power
    if 'max_power' in processed_train.columns:
        if processed_train['max_power'].dtype == 'object':
            processed_train['max_power'] = processed_train['max_power'].str.replace(' bhp', '', regex=False)
        processed_train['max_power'] = pd.to_numeric(processed_train['max_power'], errors='coerce')
    
    return processed_train

st.subheader("📉 Визуализация датасета train, на котором обучалась модель")
#Визуализируем на графиках датасет

#Карточки с медианами по кчисловым признакам
def median_metrics():
    df_train = obrabotka_train()
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        median_year = int(df_train['year'].median())
        st.metric(
            label="Медиана года выпуска",
            value=f"{median_year}"
        )

    with col2:
        median_km = int(df_train['km_driven'].median())
        st.metric(
            label="Медиана пробега",
            value=f"{median_km} км"
        )

    with col3:
        median_mileage = df_train['mileage'].median()
        st.metric(
            label="Медиана расхода топлива",
            value=f"{median_mileage}",

        )
    with col4:
        median_engine = int(df_train['engine'].median())
        st.metric(
            label="Медиана объема двигателя",
            value=f"{median_engine}"
        )
    with col5:
        median_power = int(df_train['max_power'].median())
        st.metric(
            label="Медиана мощности",
            value=f"{median_power} л.с."
        )
    with col6:
        median_seats = int(df_train['seats'].median())
        st.metric(
            label="Медиана кол-ва мест в машине",
            value=f"{median_seats}"
        )


median_metrics()

def visualize_train():
    df_train = load_train()

    fig = px.histogram(
        df_train,
        x='selling_price',
        nbins=50,
        title='Распределение цен на машины'
    )
    st.plotly_chart(fig)

    #Корреляционная матрица
    cor_matrix = df_train.select_dtypes(include=[np.number]).corr()
    corr_fig = px.imshow(
        cor_matrix,
        text_auto='.2f',
        color_continuous_scale='RdBu_r',
        title='Корреляции между числовыми признаками в train'
    )
    st.plotly_chart(corr_fig)
    #Стоимость машин от вида топлива и корбки передач
    fig_fuel=  px.bar(
        df_train,
        x='fuel',
        y='selling_price',
        color='transmission',
        title='Стоимость машин от вида топлива и типа коробки',
        barmode='group'
    )
    st.plotly_chart(fig_fuel)

visualize_train()

st.subheader("Визуализация весов модели")
#Визуализируем веса модели
def visual_weights(model, feature_names):
    coefficients = model.coef_
    # Идею создания такого датафрейма из весов взяла в комментариях на степике
    weights_df = pd.DataFrame({
        'feature': feature_names,
        'weight': coefficients,
        'abs_weight': np.abs(coefficients)
    }).sort_values('weight', ascending=False)

    #Таблица с весами
    st.subheader("📋 Таблица весов модели")
    st.dataframe(
        weights_df[['feature', 'weight']].round(4),
        use_container_width=True
    )
    #График весов
    st.subheader("📊 График весов модели")
    fig = px.bar(weights_df, 
                 x='weight', 
                 y='feature',
                 orientation='h',
                 color=weights_df['weight'] > 0,
                 color_discrete_map={True: 'royalblue', False: 'lightsteelblue'},
                 title='График весов модели')

    st.plotly_chart(fig, use_container_width=True)

    return weights_df

visual_weights(model, feature_names)

st.subheader("📈Предсказание цен на машины по признакам из csv файла")

# Загрузка CSV файла
uploaded_file = st.file_uploader("Загрузите CSV файл", type=["csv"])

if uploaded_file is None:
    st.info("👈 Загрузите CSV файл для начала работы")
    st.stop()

# Загружаем данные и делаем предсказания
df = pd.read_csv(uploaded_file)


#Необходимо обработать полученный файл.
#1.Нужно убрать все единицы измерения из колонок и сделать их числовыми
#2.Нужно удалть все не вещественные признаки, так как делаем простую модель
#3.Нужно заполнить мединами, которые мы сохранили по тренировочному датасету, если есть пропуски
#4.Нужно стандартизировать данные

def preprocess_data(df, medians, feature_names):

    processed_df = df.copy()
    
    #Убираем ' kmpl' из mileage
    if 'mileage' in processed_df.columns:
        if processed_df['mileage'].dtype == 'object':
            processed_df['mileage'] = processed_df['mileage'].str.replace(' kmpl', '', regex=False)
        processed_df['mileage'] = pd.to_numeric(processed_df['mileage'], errors='coerce')
    
    #Убираем ' CC' из engine
    if 'engine' in processed_df.columns:
        if processed_df['engine'].dtype == 'object':
            processed_df['engine'] = processed_df['engine'].str.replace(' CC', '', regex=False)
        processed_df['engine'] = pd.to_numeric(processed_df['engine'], errors='coerce')
    
    #Убираем ' bhp' для max_power
    if 'max_power' in processed_df.columns:
        if processed_df['max_power'].dtype == 'object':
            processed_df['max_power'] = processed_df['max_power'].str.replace(' bhp', '', regex=False)
        processed_df['max_power'] = pd.to_numeric(processed_df['max_power'], errors='coerce')
    
    #Убираем все признаки, которых нету в pickle feature
    have_features = [col for col in feature_names if col in processed_df.columns]
    propush_features = [col for col in feature_names if col not in processed_df.columns]
    
    if propush_features:
        st.error(f"В данных нет признаков: {propush_features}")
        return None
    
    processed_df = processed_df[have_features]
    
    #Заполним пропуски мединами из pickle файла
    for feature in have_features:
        if processed_df[feature].isnull().any():
            processed_df[feature] = processed_df[feature].fillna(medians[feature])
    
    return processed_df
    
#Показываем что загрузилось из csv
st.subheader("Загруженные данные")
st.dataframe(df.head())
    
#Стандартизуем данные и сделаем предсказания
processed_features = preprocess_data(df, medians, feature_names)

st.header("Результат работы линейной модели") 
st.subheader("Только на вещественных признаках")    
st.caption("Линейная модель на вещественных признаках показывает плохие метрики качества и сильно ошибается")  

if processed_features is not None:
    #scaler загрузили из pickle, так как на тестовых данных мы не можем обучать, только transform
    features_scaled = scaler.transform(processed_features)
    #Предсказания
    predictions = model.predict(features_scaled)
    #Сделаем таблицу с признаками и предсказанной ценой
    result_df = processed_features.copy()
    result_df['predicted_price'] = predictions.round(2)
    
    #Выведем таблицу с предсказанием цены
    st.dataframe(result_df, use_container_width=True)


  


