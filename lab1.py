# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Diplomatura en Ciencias de Datos, Aprendizaje Automático y sus Aplicaciones - Introducción al aprendizaje supervisado
# # Laboratorio 1: Regresión en Boston
# Autores: Matías Oria, Antonela Sambuceti, Pamela Pairo, Benjamín Ocampo

# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_boston
# %% [markdown]
# ## Carga del conjunto de datos
#
# Cargamos el conjunto de datos y vemos su contenido.

# %%
boston = load_boston()
boston.keys()

# %% [markdown]
# ### Nombre de columnas y significados:

# %%
print(boston['DESCR'])

# %%
boston['data'].shape, boston['target'].shape

# %% [markdown]
# ## Ejercicio 1: Descripción de los Datos y la Tarea

# %% [markdown]
# ### 1. ¿De qué se trata el conjunto de datos?

# %% [markdown]
# La base de datos refiere a un conjunto de viviendas ubicadas en Boston y una
# serie de atributos relacionados con su estructura edilicia, condiciones
# ambientales, cuestiones sociales/étnicas, económicas, accesibilidad a
# servicios, educación, entre otras.

# %% [markdown]
# ### 2. ¿Cuál es la variable objetivo que hay que predecir? ¿Qué significado tiene?

# %% [markdown]
# La variable objetivo es el valor central de las viviendas ocupadas por sus
# dueños expresados en miles de dólares (MEDV).

# %% [markdown]
# ### 3. ¿Qué información (atributos) hay disponibles para hacer la predicción?

# %% [markdown]
# - CRIM: tasa de delincuencia per cápita por ciudad.
# - ZN: proporción de terreno residencial dividido en zonas para lotes de más de
#   25,000 pies cuadrados.
# - INDUS: proporción de industrias (comercios no *retails*) en la ciudad.
# - CHAS: variable binaria que indica si la vivienda está cerca del Charles
#   River (si limita con el rio asume valor 1).
# - NOX: concentración de óxidos nítricos en la zona (partes por 10 millones).
# - RM: número promedio de habitaciones por viviendas.
# - AGE: proporción de unidades ocupadas por dueños construidas antes del 1940.
# - DIS: distancias ponderadas a cinco centros de empleo en Boston.
# - RAD: índice de accesibilidad a carreteras radiales.
# - TAX: tasa de impuesto a la propiedad por USD 10,000
# - PTRATIO: proporción alumno-maestro por ciudad.
# - B: es la proporción de personas afroamericanas por ciudad.
# - LSTAT: nivel de "status" poblacional de la zona, medida en cantidad de
#   personas con nivel de estudio no finalizado y con trabajos de mano de obra
#   pesada.
# - MEDV: valor medio de las viviendas ocupadas por sus propietarios en USD
#   1000.

# %% [markdown]
# ### 4. ¿Qué atributos imagina ud. que serán los más determinantes para la predicción?

# %% [markdown]
# Los atributos más determinantes para nosotros son cantidad de habitaciones,
# antigüedad de la propiedad, proporción del terreno  y la tasa de delincuencia.

# %% [markdown]
# ### 5. ¿Qué problemas observa a priori en el conjunto de datos? ¿Observa posibles sesgos, riesgos, dilemas éticos, etc? Piense que los datos pueden ser utilizados para hacer predicciones futuras.

# %% [markdown]
# Creemos que algunos atributos como `B` y `CRIM` requieren ser evaluados de
# manera conjunta, y podrían introducir sesgos en caso de no contarse con
# algunas de ellas. Por ejemplo, si este *Dataset* tuviese solo la variable `B`,
# podría tenderse a interpretar que cuando esta variable asume valores altos, la
# tasa de criminalidad también lo es, sin embargo al tener la variable `CRIM`
# esta teoria se puede corroborar.

# %% [markdown]
# ## Ejercicio 2: Visualización de los Datos
# %%
boston['data'].shape, boston['target'].shape
# %%
X = pd.DataFrame(data=boston['data'], columns=boston['feature_names'])
y = pd.DataFrame(data=boston['target'], columns=["target"])
# %% [markdown]
# ### 1. Para cada atributo de entrada, haga una gráfica que muestre su relación con la variable objetivo.

# %%
feature_names = X.columns
nof_features = len(feature_names)
fig, axes = plt.subplots(nof_features, figsize=(15, 45))

for ax, feature in zip(axes, feature_names):
    ax.scatter(X[feature],
               y,
               facecolor="dodgerblue",
               edgecolor="k",
               label="datos")
    ax.set_title(feature)
    ax.tick_params(labelsize=12)

# %% [markdown]
# ### 2. Estudie las gráficas, identificando **a ojo** los atributos que a su criterio sean los más informativos para la predicción.

# %% [markdown]
# - **CRIM**
#
# Se observa que los valores analizados de la variable `CRIM` se encuentran
# concentrados en zonas de bajo crimen, es decir tenemos baja frecuencia de
# viviendas ubicadas en lugares con alta delincuencia. Sin embargo, vemos que en
# los lugares donde la delincuencia es baja los precios de las residencias
# también son bajos.
#
# - **ZN**
#
# Si bien, observamos que en las viviendas de menor área construida, los precios
# son variados, no existen valores bajos para casas con mas de 15 sq.ft
# aproxidamente.
#
# - **INDUS**
#
# Se observa que las viviendas con precios más altos se encuentran en lugares
# poco industriales.
#
# - **CHAS**
#
# En primer lugar, observamos que en este *dataset* tenemos mayor concentración
# de viviendas en zonas lejanas a Charles River. Sin embargo, en el grupo de
# casas cercanos a este rio el rango de precios arranca arriba de los 15 mil.
#
# - **NOX**
#
# La mayor concentración de viviendas se encuentra a niveles bajos de oxido
# nitrógeno. No se observa una relación concluyente. Se podría realizar un
# análisis conjunto con la variable INDUS.
#
# - **RM**
#
# Observamos una fuerte relación positiva entre el número de habitaciones y el
# precio de las viviendas.
#
# - **AGE**
#
# Podemos observar que los valores más bajos de viviendas corresponden a las
# propiedades mas antiguas.
#
# - **DIS**
#
# Podemos observar que los valores más bajos de viviendas corresponden a las
# propiedades más cercanas a los conglomerados laborales.
#
# - **RAD**
#
# A simple vista no se observa un patrón concluyente.
#
# - **TAX**
#
# Se observa una concentración de casas en lugares donde el costo de los
# servicios públicos es más barato.
#
# - **PTRATIO**
#
# A simple vista no se observa un patrón concluyente.
#
# - **B**
#
# La mayoría de las casas de este *dataset* están en regiones donde el índice
# `B` es alto. También vemos que los precios son más altos en esta zona.
#
# - **LSTAT**
#
# Se observa una relación negativa entre los sectores en donde residen las
# personas con baja educación y los precios de las casas. Es decir a mayor
# porcentaje de status bajo social, los precios son mas bajos.

# %% [markdown]
# ### 3. Para ud., ¿cuáles son esos atributos? Lístelos en orden de importancia.
# %% [markdown]
# #### Correlación de todas las variables con `target`.
# %%
corr = X.join(y).corr()
corr = corr[['target']]
corr.loc[:, 'abs_corr'] = np.abs(corr['target'])
corr.sort_values(by='abs_corr', ascending=False)
# %% [markdown]
# El orden de importancia esta asociado a la correlación lineal de todos los
# atributos con `target`.
# %% [markdown]
# ## Ejercicio 3: Regresión Lineal
# %% [markdown]
# #### División en Entrenamiento y Evaluación
# %% [markdown]
# Dividimos aleatoriamente los datos en 80% para entrenamiento y 20% para evaluación:
# %%
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    train_size=0.8,
                                                    random_state=0)
X_train.shape, X_test.shape
# %% [markdown]
# ### 1. Seleccione **un solo atributo** que considere puede ser el más apropiado.
# %%
feature_name = "RM" # selecciono el atributo "RM"
feature_train = X_train[[feature_name]]
feature_test = X_test[[feature_name]]
# %% [markdown]
# ### 2. Instancie una regresión lineal de **scikit-learn**, y entrénela usando sólo el atributo seleccionado.
# %%
reg = LinearRegression()
reg.fit(feature_train, y_train)

(reg.coef_, reg.intercept_)
# %% [markdown]
# ### 3. Evalúe, calculando error cuadrático medio para los conjuntos de entrenamiento y evaluación.
# %%
y_predict = reg.predict(feature_test)
print(f'Train error: {reg.score(feature_train, y_train):f}')
print(f'Test error: {reg.score(feature_test, y_test):f}')
# %% [markdown]
# ### 4. Grafique el modelo resultante, junto con los puntos de entrenamiento y evaluación.
# %%
plt.scatter(feature_train,
            y_train,
            color="dodgerblue",
            edgecolor="k",
            label="train")
plt.scatter(feature_test, y_test, color="white", edgecolor="k", label="test")
plt.plot(feature_test, y_predict, color='red', linewidth=3)
plt.legend()
plt.show()
# %% [markdown]
# ### 5. Interprete el resultado, haciendo algún comentario sobre las cualidades del modelo obtenido.

# %% [markdown]
# Al realizar una regresión lineal únicamente con el atributo `RM`, nuestro
# modelo muestra que a medida que aumenta el número de habitaciones, también
# aumenta el precio de las viviendas. Se obtuvo un valor más alto de error para
# el conjunto de testeo respecto al calculado para el conjunto de entrenamiento.
# %% [markdown]
# ## Ejercicio 4: Regresión Polinomial
# %% [markdown]
# ### 1. Entrenamiento y evaluación para varios grados de polinomio
# %%
feature_name = "RM"
feature_train = X_train[[feature_name]]
feature_test = X_test[[feature_name]]

train_errors = []
test_errors = []
degrees = np.arange(1, 12)
for degree in degrees:
    pf = PolynomialFeatures(degree)
    lr = LinearRegression(fit_intercept=False)
    model = make_pipeline(pf, lr)
    model.fit(feature_train, y_train)

    y_train_pred = model.predict(feature_train)
    y_test_pred = model.predict(feature_test)

    train_error = mean_squared_error(y_train, y_train_pred)
    test_error = mean_squared_error(y_test, y_test_pred)
    train_errors.append(train_error)
    test_errors.append(test_error)

    print(f"Grado {degree} \tTrain error {train_error.round(3)} \tTest error {test_error.round(3)}")
# %% [markdown]
# ### 2. Grafique las curvas de error en términos del grado del polinomio.

# %%
plt.plot(degrees, train_errors, color="blue", label="train")
plt.plot(degrees, test_errors, color="red", label="test")
plt.legend()
plt.xlabel("degree")
plt.ylabel("error")
plt.show()

# %% [markdown]
# ### 3. Interprete la curva, identificando el punto en que comienza a haber sobreajuste, si lo hay.

# %% [markdown]
# El mejor grado del polinomio es el grado 2, en donde el nivel de error en test
# alcanza su punto mínimo. Observamos que a partir de allí comienza a subir
# ligeramente, a pesar de que el error en train cae, esto quiero decir que
# nuestro modelo está adaptandose más a nuestros datos de entrenamiento, y
# alejandose de una mejor predicción en test.

# %% [markdown]
# ### 4. Seleccione el modelo que mejor funcione, y grafique el modelo conjuntamente con los puntos.
# %%
degree = 2
model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
model.fit(feature_train, y_train)

plt.scatter(feature_train,
            y_train,
            color="dodgerblue",
            edgecolor="k",
            label="train")
plt.scatter(feature_test, y_test, color="white", edgecolor="k", label="test")

x = np.arange(3, 9, step=0.1)
plt.plot(x, model.predict(x.reshape(-1, 1)), color="red", label="model")
plt.legend()
plt.show()

y_train_pred = model.predict(feature_train)
y_test_pred = model.predict(feature_test)
train_error = mean_squared_error(y_train, y_train_pred)
test_error = mean_squared_error(y_test, y_test_pred)

print(f'Train error: {train_error:f}')
print(f'Test error: {test_error:f}')
# %% [markdown]
# ### 5. Interprete el resultado, haciendo algún comentario sobre las cualidades del modelo obtenido.

# %% [markdown]
# Podemos observar que el modelo polinomial de grado 2 se adapta mejor al
# comportamiento de nuestros datos, en comparación al modelo de regresión
# lineal. Si bien, los errores en train y test disminuyen, el modelo se
# complejiza un poco más.

# %% [markdown]
# ## Ejercicio 5: Regresión con más de un Atributo

# %% [markdown]
# ### 1. Seleccione dos o tres atributos entre los más relevantes encontrados en el ejercicio 2.

# %%
selected_features = ["RM", "LSTAT", "PTRATIO"]
X_train_fs = X_train[selected_features]
X_test_fs = X_test[selected_features]
X_train_fs.shape, X_test_fs.shape
# %% [markdown]
# ### 2. Repita el ejercicio anterior, pero usando los atributos seleccionados. No hace falta graficar el modelo final.
# %%
train_errors = []
test_errors = []
degrees = np.arange(1, 12)
for degree in degrees:
    pf = PolynomialFeatures(degree)
    lr = LinearRegression(fit_intercept=False)
    model = make_pipeline(pf, lr)
    model.fit(X_train_fs, y_train)

    y_train_pred = model.predict(X_train_fs)
    y_test_pred = model.predict(X_test_fs)

    train_error = mean_squared_error(y_train, y_train_pred)
    test_error = mean_squared_error(y_test, y_test_pred)
    train_errors.append(train_error)
    test_errors.append(test_error)
    print(
        f"Grado {degree} \tTrain error {train_error.round(3)} \tTest error {test_error.round(3)}"
    )
# %%
plt.plot(degrees, train_errors, color="blue", label="train")
plt.plot(degrees, test_errors, color="red", label="test")
plt.legend()
plt.xlabel("degree")
plt.ylabel("error")
plt.show()

degree_p3=2
print('Grado: ' , degree_p3)
print(f'Train error: {train_errors[degree_p3 -1]:f}')
print(f'Test error: {test_errors[degree_p3 -1]:f}')
# %% [markdown]
# ### 3. Interprete el resultado y compare con los ejercicios anteriores. ¿Se obtuvieron mejores modelos? ¿Porqué?

# %% [markdown]
# Podemos observar que con la inclusión de dos features más se disminuyen los
# errores, es decir nuestro modelo predice mejor la variable objetivo.

# %% [markdown]
# ## Ejercicio 6: A Todo Feature

# %%
train_errors = []
test_errors = []
degrees = np.arange(1, 8)
for degree in degrees:
    pf = PolynomialFeatures(degree)
    lr = LinearRegression(fit_intercept=False)
    model = make_pipeline(pf, lr)
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_error = mean_squared_error(y_train, y_train_pred)
    test_error = mean_squared_error(y_test, y_test_pred)
    train_errors.append(train_error)
    test_errors.append(test_error)
    print(
        f"Grado {degree} \tTrain error {train_error.round(3)} \tTest error {test_error.round(3)}"
    )
# %%
plt.plot(degrees, train_errors, color="blue", label="train")
plt.plot(degrees, test_errors, color="red", label="test")
plt.legend()
plt.xlabel("degree")
plt.ylabel("error")
plt.show()
# %%
degree_all=1
print('Grado: ' , degree_all)
print(f'Train error: {train_errors[degree_all -1]:f}')
print(f'Test error: {test_errors[degree_all -1]:f}')
