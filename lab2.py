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
# # Laboratorio 2: Armado de un esquema de aprendizaje automático
# Autores: Matías Oria, Antonela Sambuceti, Pamela Pairo, Benjamín Ocampo

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree

# %% [markdown]
# ## Carga de datos y división en entrenamiento y evaluación
#
# La celda siguiente se encarga de la carga de datos (haciendo uso de pandas).

# %%
dataset = pd.read_csv("./data/loan_data.csv", comment="#")
seed = 0

# División entre instancias y etiquetas
X, y = dataset.iloc[:, 1:], dataset.TARGET

scaler = StandardScaler()
X = scaler.fit_transform(X)

# división entre entrenamiento y evaluación
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=seed)

# %%
(X_train.shape, X_test.shape)

# %%
dataset

# %% [markdown]
#
# Documentación:
#
# - https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

# %% [markdown]
# ## Ejercicio 1: Descripción de los Datos y la Tarea

# %% [markdown]
# ### 1. ¿De qué se trata el conjunto de datos?

# %% [markdown]
# El *dataset* contiene un conjunto de datos que describen el comportamiento
# crediticio histórico de los clientes de un banco, que hayan solicitado
# préstamos recientemente.

# %% [markdown]
# ### 2. ¿Cuál es la variable objetivo que hay que predecir? ¿Qué significado tiene?

# %% [markdown]
# La variable objetivo `TARGET` es una variable binaria que asume
# el valor 1, si el cliente no pagó el credito solicitado, y 0 para el caso
# contrario.

# %% [markdown]
# ### 3. ¿Qué información (atributos) hay disponible para hacer la predicción?

# %% [markdown]
# - LOAN: Monto del préstamo requerido.
# - MORTDUE: Saldo del crédito hipotecario existente.
# - VALUE: Valor actual de la propiedad.
# - YOJ: Años en el trabajo actual.
# - DEROG: Número de informes despectivos.
# - DELINQ: Número de créditos en estado moroso.
# - CLAGE: Linea de crédito más antigua medida en meses.
# - NINQ: Número reciente de lineas de crédito.
# - CLNO: Número de lineas de crédito.
# - DEBTINC: Cociente entre deuda e ingresos.

# %% [markdown]
# ### 4. ¿Qué atributos imagina ud. que son los más determinantes para la predicción?

# %% [markdown]
# Creemos que el monto del préstamo `LOAN`, el comportamiento del cliente,
# medido tanto en el número de informes despectivos `DEROG`como en su morosidad
# `DELINQ` pueden influir en esta predicción. También el estado actual de su
# deuda comparado con sus ingresos podría importar en esta predicción `DEBTINC`.

# %% [markdown]
# ### Análisis de la variable `TARGET`

# %%
dataset["TARGET"].value_counts()
nof_targets_train = len(y_train)
nof_ones_train = np.sum(y_train == 1)
nof_zeros_train = np.sum(y_train == 0)

nof_targets_test = len(y_test)
nof_ones_test = np.sum(y_test == 1)
nof_zeros_test = np.sum(y_test == 0)

(nof_ones_train / nof_targets_train, nof_ones_test / nof_targets_test)

# %% [markdown]
# Observamos que tanto en el conjunto de entrenamiento como en el conjunto que
# usaremos para validar nuestra predicción, la variable Target se encuentra
# desbalanceada. Tan solo un 16% de los casos pertenecen a la clase 1 que
# significa que el cliente no pagó el préstamo.

# %% [markdown]
# Antes de continuar creemos conveniente dejar acentadas algunas definiciones
# que nos servirán a la hora de evaluar nuestros modelos predictivos:
#
# - 0 (Clase negativa): Pagó el prestamo.
# - 1 (Clase positiva): No pagó el prestamo.
# - TP: Casos predichos que no pagarón y no pagaron efectivamente.
# - TF: Casos predichos que pagaron y pagaron efectivamente.
# - FP: Casos predichos que P (No pagó el prestamo) y pasó N (Pagó el prestamo).
# - FN: Casos predichos que N (Pagó el prestamo) y pasó P (No pagó el prestamo).
#
# - Recall: TP / (TP + FN)
# - Precisión: TP / (TP + FP)
# - F1-score : 2*(Precision*Recall/Precision + Recall)
#
# Teniendo en cuenta lo anteriormente mencionado, preferimos aquellos modelos
# con mayor Recall que predigan de la mejor forma la clase positiva, es decir
# aquellos casos que no pagarán el préstamo. Es decir, preferimos tener menos
# falsos negativos (casos predichos como que pagarán el préstamo, pero en
# realidad no pagaron el préstamo), evitando así pérdidas para el banco, a costa
# de perder algunas ventas teniendo más falsos positivos (casos predichos como
# que no iban a pagar el préstamo y en realidad sí lo hicieron).

# %% [markdown]
# ## Ejercicio 2: Predicción con Modelos Lineales
#
# En este ejercicio se entrenarán modelos lineales de clasificación para
# predecir la variable objetivo.
#
# Para ello, se utilizará la clase SGDClassifier de scikit-learn.
#
# Documentación:
# - https://scikit-learn.org/stable/modules/sgd.html
# - https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html
#

# %% [markdown]
# ### Ejercicio 2.1: SGDClassifier con hiperparámetros por defecto
#
# En primer lugar, entrenaremos y evaluaremos el clasificador SGDClassifier
# usando los valores por omisión de scikit-learn para todos los parámetros.

# %% [markdown]
# #### Entrenamiento

# %%
clf = SGDClassifier(random_state=seed)
clf.fit(X_train, y_train)
y_pred_1 = clf.predict(X_test)

# %%
y_pred_1

# %% [markdown]
# #### Evaluación y matríz de confusión

# %%
print(classification_report(y_test, y_pred_1))

# %% [markdown]
# Observamos que se logran valores superiores al 0.86 es todas las
# medidas con este modelo (utilizaremos el weighted avg ya que en este caso
# quermeos darle importancia a la clase minoritaria, es decir la clase 1 que
# responde a los casos que no pagaron el préstamo)

# %%
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_1).ravel()
(tn, fp, fn, tp)

# %%
plot_confusion_matrix(clf, X_test, y_test)

# %%
clf.coef_.shape

# %% [markdown]
# Existen solo 35 casos predichos como que iban a pagar el
# préstamo (clase 0) pero en realidad no lo hicieron.

# %% [markdown]
# ### Ejercicio 2.2: Ajuste de Hiperparámetros
#
# A continuación seleccionaremos valores para los hiperparámetros principales
# del SGDClassifier, y usaremos grid-search y 5-fold cross-validation sobre el
# conjunto de entrenamiento para explorar muchas combinaciones posibles de
# valores.
#
# Documentación:
# - https://scikit-learn.org/stable/modules/grid_search.html
# - https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html

# %% [markdown]
# #### Entrenamiento

# %%
param_grid = {
    'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'],
    'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1],
    'penalty': ['l2', 'l1'],
    'eta0': [1e-3, 1e-4, 1e-5, 10],
    'learning_rate': ['optimal', 'constant', 'adaptive']
    #'max_iter': [1000, 2000, 5000]
}

model = SGDClassifier(random_state=seed)
cv = GridSearchCV(model,
                  param_grid,
                  scoring=["recall", "accuracy", "precision", "f1"],
                  cv=5,
                  refit=False)
cv.fit(X_train, y_train)

results = cv.cv_results_
params = results['params']

# %%
results_df = pd.DataFrame(results)
results_df.columns

# %%
relevant_metrics = [
    "mean_test_recall", "std_test_recall", "mean_test_accuracy",
    "std_test_accuracy", "mean_test_precision", "std_test_precision",
    'mean_test_f1', "rank_test_recall", 'rank_test_precision',
    'rank_test_accuracy', 'rank_test_f1'
]

params = ["param_alpha", "param_loss", "param_penalty", 'param_eta0']

results_df = results_df[relevant_metrics + params]
results_df

# %%
highest_rank = 1
results_df[results_df['rank_test_recall'] == highest_rank]

# %%
results_df[results_df['rank_test_precision'] == highest_rank]

# %%
results_df[results_df['rank_test_f1'] == highest_rank]

# %%
results_df.loc[results_df['rank_test_recall'] == highest_rank, params]

# %% [markdown]
# #### Evaluación y matríz de confusión

# %%
clf_best = SGDClassifier(random_state=seed,
                         alpha=1,
                         loss="perceptron",
                         eta0=10,
                         penalty="l2")
clf_best.fit(X_train, y_train)
y_pred_best = clf_best.predict(X_test)

# %%
print(classification_report(y_test, y_pred_best))

# %%
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_1).ravel()
print("Anterior modelo", (tn, fp, fn, tp))

# %%
plot_confusion_matrix(clf_best, X_test, y_test)

# %% [markdown]
# Observamos que nuestros resultados son muy parecidos a los del
# modelo entrenado en el apartado anterior, sin embargo mejoran todas las
# predicciones.

# %% [markdown]
# ## Ejercicio 3: Árboles de Decisión
#
# En este ejercicio se entrenarán árboles de decisión para predecir la variable
# objetivo.
#
# Para ello, utilizaremos la clase DecisionTreeClassifier de scikit-learn.
#
# Documentación:
# - https://scikit-learn.org/stable/modules/tree.html
# - https://scikit-learn.org/stable/modules/tree.html#tips-on-practical-use
# - https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
# - https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html

# %% [markdown]
# ### Ejercicio 3.1: DecisionTreeClassifier con hiperparámetros por defecto
#
# Entrenaremos y evaluaremos el clasificador DecisionTreeClassifier usando los
# valores por omisión de scikit-learn para todos los parámetros.

# %%
# División entre instancias y etiquetas
X, y = dataset.iloc[:, 1:], dataset.TARGET

# división entre entrenamiento y evaluación
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=seed)

# %% [markdown]
# #### Entrenamiento

# %%
clf_tree = DecisionTreeClassifier(random_state=0)
clf_tree.fit(X_train, y_train)

# %% [markdown]
# #### Evaluación y gráfica del árbol

# %%
y_train_pred = clf_tree.predict(X_train)
y_test_pred = clf_tree.predict(X_test)

# %%
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)
print(f'Train accuracy: {train_acc:0.2}')
print(f'Test accuracy: {test_acc:0.2}')

# %%
print(classification_report(y_test, y_test_pred))

# %%
plt.figure(figsize=(100, 100))
plot_tree(clf, fontsize=20, feature_names=dataset.columns)
plt.show()
# %% [markdown]
# Si bien los valores de las métricas son altos (todos los
# promedios cercanos al 0.90) el árbol tiene una profundidad muy alta, perdiendo
# interpretabilidad. A continuación probaremos con otros hiperparametros.

# %% [markdown]
# ### Ejercicio 3.2: Ajuste de Hiperparámetros
#
# Seleccionaremos algunos valores para los hiperparámetros del
# DecisionTreeClassifier. Usaremos grid-search y 5-fold cross-validation sobre
# el conjunto de entrenamiento para explorar muchas combinaciones posibles de
# valores.
#
# Documentación:
# - https://scikit-learn.org/stable/modules/grid_search.html
# - https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html

# %%
param_grid = {
    'max_depth': [6, 7, 8, 9, 10],
    'criterion': ['gini', 'entropy'],
    'min_samples_split': [5, 10, 20, 30, 40, 50, 100],
    'min_samples_leaf': [5, 10, 20, 30, 40, 50]
}

model_tree = DecisionTreeClassifier(random_state=seed)
cv = GridSearchCV(model_tree,
                  param_grid,
                  scoring=["recall", "accuracy", "precision", "f1"],
                  cv=5,
                  refit=False)
cv.fit(X_train, y_train)

results_tree = cv.cv_results_
params_tree = results_tree['params']

# %%
results_df_tree = pd.DataFrame(results_tree)
results_df_tree.columns

# %%
relevant_metrics = [
    "mean_test_recall", "std_test_recall", "mean_test_accuracy",
    "std_test_accuracy", "mean_test_precision", "std_test_precision",
    'mean_test_f1', "rank_test_recall", 'rank_test_precision',
    'rank_test_accuracy', 'rank_test_f1'
]

params_tree = [
    'param_criterion', 'param_max_depth', 'param_min_samples_leaf',
    'param_min_samples_split'
]

results_df_tree = results_df_tree[relevant_metrics + params_tree]
results_df_tree

# %%
highest_rank = 1
results_df_tree[results_df_tree['rank_test_recall'] == highest_rank]

# %%
results_df_tree[results_df_tree['rank_test_precision'] == highest_rank]

# %%
results_df_tree[results_df_tree['rank_test_f1'] == highest_rank]

# %%
clf_best_tree = DecisionTreeClassifier(random_state=seed,
                                       criterion="gini",
                                       max_depth=9,
                                       min_samples_split=20,
                                       min_samples_leaf=10)
clf_best_tree.fit(X_train, y_train)
y_pred_best_tree = clf_best_tree.predict(X_test)

# %%
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_pred_best_tree)
print(f'Train accuracy: {train_acc:0.2}')
print(f'Test accuracy: {test_acc:0.2}')

# %%
print(classification_report(y_test, y_pred_best_tree))

# %%
plt.figure(figsize=(100, 100))
plot_tree(clf_best_tree, fontsize=20, feature_names=dataset.columns)
plt.show()

# %% [markdown]
# Observamos que las medidas de nuestro predictor continuan
# cercanas a al 0.90, mejorando notablemente la interpretabilidad del árbol, ya
# que ahora tiene una menor profundidad.
