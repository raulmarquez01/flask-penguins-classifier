from pathlib import Path
import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)
modelo = joblib.load(Path(__file__).parent / "model" / "logistic_regression_seed42.pkl")

ISLAS = ["Biscoe", "Dream", "Torgersen"]
SEXOS = {"FEMALE": "Hembra", "MALE": "Macho", "unknown": "No sé"}
CAMPOS = {  # nombre: (etiqueta, mínimo, máximo) aceptados por el formulario
    "bill_length_mm": ("Largo del pico (mm)", 30, 62),
    "bill_depth_mm": ("Ancho del pico (mm)", 12, 23),
    "flipper_length_mm": ("Largo de la aleta (mm)", 165, 240),
    "body_mass_g": ("Masa corporal (g)", 2500, 6500),
}
# Rango observado en los datos de entrenamiento, para avisar al salirse de él.
RANGO_ENTRENAMIENTO = {
    "bill_length_mm": (33.1, 59.6),
    "bill_depth_mm": (13.1, 21.5),
    "flipper_length_mm": (172, 231),
    "body_mass_g": (2700, 6300),
}
# Combinaciones isla-especie que aparecen en los datos.
COMBINACIONES_VISTAS = {("Biscoe", "Adelie"), ("Biscoe", "Gentoo"), ("Dream", "Adelie"),
                        ("Dream", "Chinstrap"), ("Torgersen", "Adelie")}


@app.route("/", methods=["GET", "POST"])
def index():
    valores, errores, avisos, resultado = {}, [], [], None
    if request.method == "POST":
        valores = request.form.to_dict()
        fila = {"island": valores.get("island"), "sex": valores.get("sex")}
        if fila["island"] not in ISLAS or fila["sex"] not in SEXOS:
            errores.append("Elige una isla y un sexo de la lista.")
        for campo, (etiqueta, minimo, maximo) in CAMPOS.items():
            try:
                x = float(valores.get(campo, "").replace(",", "."))
                if not minimo <= x <= maximo:
                    raise ValueError
                fila[campo] = x
            except ValueError:
                errores.append(f"{etiqueta}: introduce un número entre {minimo} y {maximo}.")
        if not errores:
            probas = modelo.predict_proba(pd.DataFrame([fila]))[0]
            ranking = sorted(zip(modelo.classes_, probas), key=lambda p: -p[1])
            especie = ranking[0][0]
            resultado = {"especie": especie, "probabilidad": round(100 * ranking[0][1], 1),
                         "todas": [(e, round(100 * p, 1)) for e, p in ranking]}
            for campo, (bajo, alto) in RANGO_ENTRENAMIENTO.items():
                if not bajo <= fila[campo] <= alto:
                    avisos.append(f"{CAMPOS[campo][0]} queda fuera del rango visto en el entrenamiento "
                                  f"({bajo}-{alto}); la predicción puede no ser fiable.")
            if (fila["island"], especie) not in COMBINACIONES_VISTAS:
                avisos.append(f"En los datos de entrenamiento no hay ningún {especie} en {fila['island']}; "
                              "la predicción puede no ser fiable.")
    return render_template("index.html", islas=ISLAS, sexos=SEXOS, campos=CAMPOS,
                           valores=valores, errores=errores, avisos=avisos, resultado=resultado)
