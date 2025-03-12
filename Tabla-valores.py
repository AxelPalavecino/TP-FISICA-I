import csv 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

columnas = [
    "Largo (tupla) [mm]",
    "Insertidumbre Largo [mm]",
    "Area [mm^2]",
    "Insertidumbre Area [mm^2]",
    "Diametro1 (tupla) [mm]",
    "Diametro2 (tupla) [mm]",
    "Diametro3 (tupla) [mm]",
    "Promedio Diametro [mm]",
    "Insertidumbre Diametro [mm]",
    "Volumen [mm^3]",
    "Insertidumbre Volumen [mm^3]",
    "Masa [g]",
    "Insertidumbre Masa [g]",
]

# AREA

def calcular_area(largo1, largo2):
    return largo1 * largo2


def calcular_insertidumbre_area(largo1, largo2, insertidumbre_largo1, insertidumbre_largo2):
    if (largo1 == largo2): 
        return 2 * largo1 * insertidumbre_largo1
    return np.sqrt((largo2 * insertidumbre_largo1) ** 2 + (largo1 * insertidumbre_largo2) ** 2)



# DIAMETRO

def calcular_promedio_diametro(diametros):
    return sum(diametros) / len(diametros)

def calcular_desviacion_estandar(promedio_diametros, diametros):
    return np.sqrt(sum((d - promedio_diametros) ** 2 for d in diametros) / (len(diametros) - 1))

def calcular_error_estandar(desviacion_estandar, diametros):
    return desviacion_estandar / np.sqrt(len(diametros))

def calcular_insertidumbre_diametro(error_estandar, insertidumbre_medicion):
    return np.sqrt(error_estandar ** 2 + insertidumbre_medicion ** 2)

# VOLUMEN
    
def calcular_volumen(diametro_promedio):
    return (4 / 3) * np.pi * (diametro_promedio / 2) ** 3 

def calcular_insertidumbre_volumen(diametro_promedio, insertidumbre_diametro):
    return (1 / 2) * np.pi * (diametro_promedio) ** 2 * insertidumbre_diametro


def _parse_tupla(cadena):
    """
    Recibe una cadena como "50,50" o "(50,50)" y devuelve (50.0, 50.0).
    Si la cadena está vacía o es inválida, devuelve None.
    """
    if not cadena or cadena.strip() == "":
        return None
    # Quitamos paréntesis y comillas sobrantes si existen
    cadena_limpia = cadena.strip().replace("(", "").replace(")", "").replace("\"", "")
    try:
        numeros = [float(x.strip()) for x in cadena_limpia.split(",") if x.strip()]
        return tuple(numeros)
    except ValueError:
        return None

def _parse_float(valor):
    """
    Recibe una cadena y devuelve el valor float si es posible.
    Si no, devuelve None.
    """
    try:
        return float(valor)
    except:
        return None


def completar_datos_csv():
    """
    Lee los 3 archivos CSV (datos_papel_1, datos_papel_2, datos_papel_3),
    calcula las columnas faltantes a partir de los valores ya presentes y
    reescribe los archivos con la información actualizada, 
    manteniendo intactos los valores en Diametro1/2/3 (tupla) [mm].
    """
    for i in range(1, 4):
        nombre_archivo = f"datos_papel_{i}.csv"
        with open(nombre_archivo, "r", newline="", encoding="utf-8") as f:
            lector = csv.reader(f)
            filas = list(lector)

        # Verifica si hay encabezados
        if not filas:
            continue
        encabezado = filas[0]
        col_idx = {col: idx for idx, col in enumerate(encabezado)}

        for fila_i in range(1, len(filas)):
            fila = filas[fila_i]
            if not fila or len(fila) < len(encabezado):
                continue

            # --- LARGO
            largo_str = fila[col_idx["Largo (tupla) [mm]"]]
            insert_largo_str = fila[col_idx["Insertidumbre Largo [mm]"]]
            largo_tupla = _parse_tupla(largo_str)
            insert_largo = _parse_float(insert_largo_str)

            # --- DIÁMETROS (no se reescriben, sólo se usan para cálculo)
            diam1_str = fila[col_idx["Diametro1 (tupla) [mm]"]]
            diam2_str = fila[col_idx["Diametro2 (tupla) [mm]"]]
            diam3_str = fila[col_idx["Diametro3 (tupla) [mm]"]]

            diam1_tupla = _parse_tupla(diam1_str) or ()
            diam2_tupla = _parse_tupla(diam2_str) or ()
            diam3_tupla = _parse_tupla(diam3_str) or ()

            # Creamos la lista total con todos los valores para calcular UN promedio global
            diametros = list(diam1_tupla) + list(diam2_tupla) + list(diam3_tupla)

            # --- CALCULA ÁREA
            area_idx = col_idx["Area [mm^2]"]
            insert_area_idx = col_idx["Insertidumbre Area [mm^2]"]
            if largo_tupla and len(largo_tupla) == 2:
                # Si la celda de área está vacía
                if fila[area_idx].strip() == "":
                    fila[area_idx] = str(calcular_area(largo_tupla[0], largo_tupla[1]))
                # Insertidumbre área
                if fila[insert_area_idx].strip() == "" and insert_largo is not None:
                    valor_ins_area = calcular_insertidumbre_area(
                        largo_tupla[0],
                        largo_tupla[1],
                        insert_largo,
                        insert_largo
                    )
                    fila[insert_area_idx] = str(valor_ins_area)

            # --- CALCULA PROMEDIO DIÁMETRO
            prom_diam_idx = col_idx["Promedio Diametro [mm]"]
            if diametros and fila[prom_diam_idx].strip() == "":
                promedio = calcular_promedio_diametro(diametros)
                fila[prom_diam_idx] = str(promedio)

            # --- INSERTIDUMBRE DIÁMETRO
            insert_diam_idx = col_idx["Insertidumbre Diametro [mm]"]
            if diametros and fila[insert_diam_idx].strip() == "":
                promedio = calcular_promedio_diametro(diametros)
                desv_est = calcular_desviacion_estandar(promedio, diametros)
                error_est = calcular_error_estandar(desv_est, diametros)
                inc_medicion = 1.0  # valor asumido
                inc_diam = calcular_insertidumbre_diametro(error_est, inc_medicion)
                fila[insert_diam_idx] = str(inc_diam)

            # --- VOLUMEN
            vol_idx = col_idx["Volumen [mm^3]"]
            insert_vol_idx = col_idx["Insertidumbre Volumen [mm^3]"]
            if diametros and fila[vol_idx].strip() == "":
                promedio_str = fila[prom_diam_idx]
                promedio_val = _parse_float(promedio_str)
                if promedio_val is not None:
                    vol = calcular_volumen(promedio_val)
                    fila[vol_idx] = str(vol)

            if diametros and fila[insert_vol_idx].strip() == "":
                inc_diam_str = fila[insert_diam_idx]
                inc_diam_val = _parse_float(inc_diam_str)
                promedio_str = fila[prom_diam_idx]
                promedio_val = _parse_float(promedio_str)
                if inc_diam_val is not None and promedio_val is not None:
                    inc_vol = calcular_insertidumbre_volumen(promedio_val, inc_diam_val)
                    fila[insert_vol_idx] = str(inc_vol)

            # Actualiza la fila en memoria
            filas[fila_i] = fila

        # Reescribe el CSV con las columnas originales intactas, 
        # excepto las que acabamos de completar.
        with open(nombre_archivo, "w", newline="", encoding="utf-8") as f:
            escritor = csv.writer(f)
            escritor.writerows(filas)

        print(f"Archivo '{nombre_archivo}' actualizado con columnas Diametro1/2/3 intactas.")

def generar_tabla_imagen(csv_file, output_image="tabla_resultados.png",
                         figsize=(12, 5), fontsize=8, scale=(1.2, 1.2)):
    """
    Lee un CSV y genera una imagen de la tabla usando matplotlib.
    """
    df = pd.read_csv(csv_file)
    df = df.round(3)
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    tabla = ax.table(cellText=df.values, colLabels=df.columns, loc="center")
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(fontsize)
    tabla.scale(*scale)
    plt.savefig(output_image, bbox_inches="tight")
    plt.close()
    print(f"Tabla guardada en {output_image}")

def generar_grafico_masa_diametro(archivos, output_image="grafico_combinado.png", figsize=(8,5)):
    """
    Combina los CSV indicados y genera un gráfico más estético donde:
      - Eje X: Masa [g] (con error horizontal usando Insertidumbre Masa).
      - Eje Y: Promedio Diametro [mm] (con error vertical usando Insertidumbre Diametro).
    Se mejoran las visualizaciones de las incertidumbres.
    """
    df_list = [pd.read_csv(a) for a in archivos]
    df_combined = pd.concat(df_list, ignore_index=True)
    df_combined = df_combined.round(3)
    masa = df_combined["Masa [g]"]
    inc_masa = df_combined["Insertidumbre Masa [g]"]
    diametro = df_combined["Promedio Diametro [mm]"]
    inc_diametro = df_combined["Insertidumbre Diametro [mm]"]

    plt.style.use("seaborn-darkgrid")
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.errorbar(
        masa, diametro, 
        xerr=inc_masa, yerr=inc_diametro,
        fmt="o", markersize=6, markerfacecolor="white", markeredgecolor="black",
        ecolor="black", elinewidth=1.5, capsize=5, label="Masa vs. Diámetro",
        alpha=0.8
    )
    
    ax.set_xlabel("Masa [g]", fontsize=12)
    ax.set_ylabel("Promedio Diametro [mm]", fontsize=12)
    ax.set_title("Gráfico combinado de los tres archivos", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_image, dpi=200)
    plt.close()
    print(f"Gráfico guardado en {output_image}")


if __name__ == "__main__":
    generar_grafico_masa_diametro(["datos_papel_1.csv", "datos_papel_2.csv", "datos_papel_3.csv"])