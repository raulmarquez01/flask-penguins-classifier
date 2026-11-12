# Clasificador de especies de pingüinos con Flask

**Aplicación en vivo:** https://flask-penguins-classifier.onrender.com

El servicio usa el plan gratuito de Render: si lleva un rato sin visitas se "duerme" y el primer acceso puede tardar cerca de un minuto.

Aplicación web que estima si un pingüino es Adelie, Chinstrap o Gentoo a partir de la isla, el sexo y cuatro medidas (largo y ancho del pico, largo de la aleta y masa corporal). El modelo es una regresión logística entrenada en `notebook/eda.ipynb` y servida con Flask y gunicorn.

## Datos

Palmer Penguins (Gorman, Williams y Fraser, 2014): observaciones de la Estación Palmer, Antártida. Se usa la copia del repositorio `seaborn-data`, guardada en `data/raw/penguins.csv` (344 filas, 342 tras eliminar 2 sin ninguna medida). Especies: Adelie 44,2 %, Gentoo 36,0 %, Chinstrap 19,8 %.

## Método

Todo está en `notebook/eda.ipynb`, por bloques:

1. Carga e inspección. Limpieza: se eliminan las 2 filas sin medidas y el sexo desconocido pasa a una categoría `unknown`.
2. Partición estratificada 80/20 con semilla 42: 273 filas de entrenamiento y 69 de prueba.
3. Análisis exploratorio solo con entrenamiento.
4. Comparación de modelos con validación cruzada de 5 pliegues estratificada, solo sobre entrenamiento.
5. Ajuste de `C` y comparación de variantes (con o sin isla y sexo).
6. Evaluación final en la prueba, usada una sola vez.

Resultados de la validación cruzada (exactitud media ± desviación):

| Modelo | Exactitud |
|---|---|
| Baseline (siempre Adelie) | 0,443 ± 0,007 |
| Regresión logística | 0,993 ± 0,009 |
| Bosque aleatorio | 0,989 ± 0,009 |
| KNN | 0,985 ± 0,014 |
| Árbol de decisión | 0,963 ± 0,016 |

Se eligió la regresión logística (`C` = 1) por la regla fijada de antemano: mejor exactitud y, además, la más simple. En la prueba acertó los 69 pingüinos (baseline: 0,435).

## Limitaciones

- La prueba son solo 69 filas (Chinstrap, unas 14). Un 100 % con tan pocos datos indica que el error es bajo, no que el modelo sea perfecto; la estimación más informativa es la validación cruzada (0,993 ± 0,009).
- Los datos son de tres islas concretas. En ellos Gentoo solo aparece en Biscoe, Chinstrap solo en Dream y Torgersen solo tiene Adelie. La aplicación avisa si se introduce una combinación isla-especie que no existe en los datos o medidas fuera del rango de entrenamiento.
- No sirve para otras especies ni para pingüinos de otros lugares.
- Quitar la isla y el sexo (solo medidas) daba 0,982 en validación cruzada, algo menos que 0,993. Se mantuvieron por la regla fijada de antemano, en un caso cercano al límite.

## Estructura

- `data/raw/penguins.csv`: datos.
- `notebook/eda.ipynb`: análisis y modelado.
- `models/logistic_regression_seed42.pkl`: modelo entrenado.
- `src/app.py`, `src/templates/`, `src/static/`: aplicación Flask.
- `src/model/`: copia del modelo que usa la aplicación.
- `src/requirements.txt`: dependencias con versiones fijadas de la aplicación.
- `requirements.txt`: dependencias de la plantilla para el notebook.

## Cómo ejecutarlo en local

    pip install -r src/requirements.txt
    cd src
    gunicorn app:app

Después abre http://127.0.0.1:8000.

## Despliegue en Render

Web Service desde este repositorio público, rama `main`. Root Directory `src`, Build Command `pip install -r requirements.txt`, Start Command `gunicorn app:app`, plan gratuito y variable de entorno `PYTHON_VERSION=3.13.18`.

## Recursos externos

- Gorman, K. B., Williams, T. D. y Fraser, W. R. (2014). *Ecological sexual dimorphism and environmental variability within a community of Antarctic penguins (genus Pygoscelis)*. PLoS ONE 9(3): e90081. https://doi.org/10.1371/journal.pone.0090081
- Copia de los datos: https://github.com/mwaskom/seaborn-data
- Documentación de scikit-learn: https://scikit-learn.org
- Documentación de Flask: https://flask.palletsprojects.com
- Render, versión de Python: https://render.com/docs/python-version
- Lecciones del módulo de Flask de 4Geeks Academy.
- Trabajo desarrollado con la asistencia de Claude (Anthropic).
