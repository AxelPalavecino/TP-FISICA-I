import csv 
import numpy as np
import pandas as pd
import textwrap
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm
from cycler import cycler

# Configuración estética general mejorada
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16

# Colores mejorados para la visualización
colores_gramaje = {
    80: "#1f77b4",    # Azul más profundo
    150: "#2ca02c",   # Verde más elegante
    240: "#9467bd"    # Púrpura más suave
}

# Configurar tema personalizado para los gráficos
sns.set_theme(style="darkgrid", palette="deep")

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

def generar_tabla_imagen(csv_file, output_image=None,
                         figsize=(14, 12), fontsize=11, scale=(1.2, 1.2),
                         header_row_height=0.25):
    """
    Lee un CSV y genera una imagen de la tabla usando matplotlib con estilo mejorado,
    asegurando coherencia visual con los gráficos y mostrando información de manera más clara.
    """
    # Extraer la base del nombre del archivo primero
    base_name = csv_file.rsplit(".", 1)[0]
    
    if output_image is None:
        output_image = f"tabla_{base_name}.png"
    
    # Extraer el número de papel y determinar el gramaje
    papel_num = base_name.split('_')[-1]
    gramajes = {'1': 80, '2': 150, '3': 240}
    gramaje = gramajes.get(papel_num, "Desconocido")
    color_gramaje = colores_gramaje.get(gramaje, "midnightblue")
    
    # Leer y preparar los datos
    df = pd.read_csv(csv_file)
    
    # Redondear números para mejor presentación
    df_display = df.copy()
    for col in df_display.columns:
        if "Diametro" in col and "tupla" in col:
            continue  # No redondear las tuplas
        if df_display[col].dtype in ['float64', 'float32']:
            if "Volumen" in col:
                df_display[col] = df_display[col].round(1)  # Volumen con 1 decimal
            elif "Insertidumbre" in col:
                df_display[col] = df_display[col].round(3)  # Insertidumbres con 3 decimales
            else:
                df_display[col] = df_display[col].round(2)  # Resto con 2 decimales
    
    # Dividir las columnas en dos partes más equilibradas
    total_cols = df_display.shape[1]
    half = total_cols // 2
    # Asegurar que "Volumen" e "Insertidumbre Volumen" están en la segunda tabla
    vol_columns = [col for col in df_display.columns if "Volumen" in col]
    non_vol_columns = [col for col in df_display.columns if "Volumen" not in col]
    
    # Reorganizar columnas para mantener coherencia temática
    first_set = non_vol_columns[:len(non_vol_columns)//2 + len(non_vol_columns)%2]
    second_set = non_vol_columns[len(non_vol_columns)//2 + len(non_vol_columns)%2:] + vol_columns
    
    df1 = df_display[first_set]
    df2 = df_display[second_set]
    
    # Crear figura con tamaño adecuado (incrementado para dar más espacio)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    fig.suptitle(f"Datos de papel {papel_num} (Gramaje: {gramaje} g/m²)", 
                fontsize=16, fontweight='bold', color=color_gramaje)
    
    # Abreviar algunos nombres de columnas particularmente largos
    rename_map = {
        "Insertidumbre Volumen [mm^3]": "Ins. Volumen [mm³]",
        "Volumen [mm^3]": "Volumen [mm³]",
        "Insertidumbre Diametro [mm]": "Ins. Diámetro [mm]",
        "Insertidumbre Area [mm^2]": "Ins. Área [mm²]",
        "Insertidumbre Largo [mm]": "Ins. Largo [mm]",
        "Insertidumbre Masa [g]": "Ins. Masa [g]",
        "Area [mm^2]": "Área [mm²]"
    }
    
    # Función para crear y estilizar tablas
    def create_styled_table(ax, data, title):
        ax.axis("off")
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10, color=color_gramaje)
        
        # Renombrar columnas con nombres cortos para la visualización
        display_data = data.copy()
        display_data.columns = [rename_map.get(col, col) for col in display_data.columns]
        
        # Preparar datos y encabezados
        wrapped_headers = []
        for col in display_data.columns:
            # Cortar el texto en varios renglones usando saltos de línea
            if len(col) > 12:
                parts = col.split(" ")
                if len(parts) > 1:
                    middle = len(parts) // 2
                    wrapped = "\n".join([" ".join(parts[:middle]), " ".join(parts[middle:])])
                else:
                    wrapped = textwrap.fill(col, width=10)
            else:
                wrapped = col
            wrapped_headers.append(wrapped)
        
        cell_text = display_data.values
        
        # Ajustar anchos de columna basado en contenido y tipo de dato
        col_widths = []
        for i, col in enumerate(display_data.columns):
            if "tupla" in col.lower():
                col_widths.append(0.20)  # Columnas de tuplas más anchas
            elif "volumen" in col.lower():
                col_widths.append(0.16)  # Columnas de volumen más anchas
            elif "diametro" in col.lower() or "diámetro" in col.lower():
                col_widths.append(0.17)  # Diámetros también necesitan más espacio
            elif "ins" in col.lower():
                col_widths.append(0.15)  # Incertidumbres
            else:
                col_widths.append(0.14)  # Columnas estándar
        
        # Crear tabla (con más espacio para encabezados)
        tabla = ax.table(
            cellText=cell_text,
            colLabels=wrapped_headers,
            loc="center",
            cellLoc="center",
            colWidths=col_widths
        )
        
        # Estilizar tabla
        tabla.auto_set_font_size(False)
        tabla.set_fontsize(fontsize)
        tabla.scale(*scale)
        
        # Estilizar celdas
        for (row, col), cell in tabla.get_celld().items():
            cell.set_edgecolor("gray")
            cell.set_text_props(va="center")
            
            # Encabezados
            if row == 0:
                cell.set_facecolor(color_gramaje)
                cell.set_text_props(color="white", weight="bold")
                cell.get_text().set_fontsize(fontsize - 1)
                cell.set_height(header_row_height)  # Mayor altura para encabezados
                cell.pad = 8  # Más padding para encabezados
                # Asegurar que el texto está centrado
                cell.get_text().set_horizontalalignment('center')
            # Celdas normales
            else:
                # Filas alternas con colores suaves
                if row % 2 == 0:
                    cell.set_facecolor("#f0f0f0")
                else:
                    cell.set_facecolor("#ffffff")
                
                # Estilo para celdas específicas
                if col == len(data.columns) - 1 and "Ins. Masa" in display_data.columns[col]:
                    cell.set_text_props(weight='bold')
                if col == len(data.columns) - 2 and "Masa [g]" in display_data.columns[col]:
                    cell.set_text_props(weight='bold')
                    cell.set_facecolor("#e6f2ff")
                
                # Para diámetros promedio y sus insertidumbres
                if "Promedio Diametro" in data.columns[col]:
                    cell.set_facecolor("#e6ffe6")
                    cell.set_text_props(weight='bold')
                if "Ins. Diámetro" in display_data.columns[col]:
                    cell.set_facecolor("#e6ffe6")
                    cell.set_text_props(style='italic')
                
        return tabla
    
    # Crear tablas en ambos ejes
    tabla1 = create_styled_table(ax1, df1, "Dimensiones y mediciones directas")
    tabla2 = create_styled_table(ax2, df2, "Mediciones calculadas y derivadas")
    
    # Añadir información explicativa
    fig.text(0.5, 0.01, 
             f"Tabla de mediciones para papel de gramaje {gramaje} g/m².\n"
             f"Las celdas destacadas muestran valores que se utilizan en los análisis de ley de escala.",
             ha='center', fontsize=10, style='italic')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_image, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Tabla mejorada para papel de {gramaje} g/m² guardada en {output_image}")

def generar_grafico_masa_diametro(archivos, output_image="grafico_combinado.png", figsize=(10,7)):
    """
    Combina los CSV indicados y genera un gráfico estético donde:
      - Eje X: Masa [g] (con error horizontal usando Insertidumbre Masa).
      - Eje Y: Promedio Diametro [mm] (con error vertical usando Insertidumbre Diametro).
    Versión mejorada visualmente.
    """
    # Mapeo de archivos a gramajes
    gramajes = {
        "datos_papel_1.csv": 80,
        "datos_papel_2.csv": 150,
        "datos_papel_3.csv": 240
    }
    
    # Crear figura con mejor relación de aspecto
    fig, ax = plt.subplots(figsize=figsize)
    
    # Añadir una sutil cuadrícula de fondo
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Personalizar el aspecto del gráfico
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    
    # Graficar cada archivo con su propio estilo y color
    for i, archivo in enumerate(archivos):
        df = pd.read_csv(archivo).round(3)
        gramaje = gramajes.get(archivo.split("/")[-1], "Desconocido")
        color = colores_gramaje.get(gramaje, "dimgray")
        
        masa = df["Masa [g]"]
        inc_masa = df["Insertidumbre Masa [g]"]
        diametro = df["Promedio Diametro [mm]"]
        inc_diametro = df["Insertidumbre Diametro [mm]"]
        
        # Graficar datos con barras de error más elegantes
        ax.errorbar(
            masa, diametro, 
            xerr=inc_masa, yerr=inc_diametro,
            fmt='o', markersize=8, markerfacecolor='white', 
            markeredgecolor=color, markeredgewidth=1.5,
            ecolor=color, elinewidth=1.2, capsize=4, 
            label=f"Papel {gramaje} g/m²",
            alpha=0.85
        )
    
    # Mejorar las etiquetas de los ejes
    ax.set_xlabel("Masa [g]", fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel("Diámetro promedio [mm]", fontsize=12, fontweight='bold', labelpad=10)
    
    # Título más atractivo con subtítulo explicativo
    ax.set_title("Relación entre Masa y Diámetro para Diferentes Gramajes", 
                 fontsize=14, fontweight='bold', pad=15)
    
    # Añadir un texto explicativo sutil
    ax.text(0.5, -0.15, 
            "El gráfico muestra cómo el diámetro de las bolas de papel\nvaría en función de su masa para distintos gramajes.", 
            transform=ax.transAxes, ha='center', fontsize=9, 
            style='italic', alpha=0.8)
    
    # Leyenda mejorada
    legend = ax.legend(title="Tipo de papel", loc='upper left', framealpha=0.9)
    legend.get_title().set_fontweight('bold')
    
    # Ajustar los límites para que se vea mejor
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    ax.set_xlim(x_min - 0.1, x_max * 1.05)
    ax.set_ylim(y_min - 1, y_max * 1.05)
    
    plt.tight_layout()
    plt.savefig(output_image, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Gráfico combinado mejorado guardado en {output_image}")

def generar_grafico_ley_escala(archivos, output_image="grafico_ley_escala.png", figsize=(10,8)):
    """
    Genera un gráfico para comparar la Ley de Escala:
       D = k · M^alpha
    Con diseño mejorado y claridad visual.
    """
    # Mapeo de archivos a gramajes
    gramajes = {
        "datos_papel_1.csv": 80,
        "datos_papel_2.csv": 150,
        "datos_papel_3.csv": 240
    }
    
    # Estilos para cada tipo de papel (usando marcadores más distintivos)
    estilos = {
        80: {"color": colores_gramaje[80], "marker": "o", "label": "Papel 80 g/m²"},
        150: {"color": colores_gramaje[150], "marker": "s", "label": "Papel 150 g/m²"},
        240: {"color": colores_gramaje[240], "marker": "^", "label": "Papel 240 g/m²"}
    }
    
    # Crear figura con estilo mejorado
    fig, ax = plt.subplots(figsize=figsize)
    
    # Eliminar líneas del marco superior y derecho
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    
    # Almacenar todos los datos para el ajuste
    all_masa = []
    all_diametro = []
    
    # Graficar cada archivo con su propio estilo
    for archivo in archivos:
        df = pd.read_csv(archivo).round(3)
        gramaje = gramajes.get(archivo.split("/")[-1], "Desconocido")
        estilo = estilos.get(gramaje, {"color": "dimgray", "marker": "x", "label": f"Papel {gramaje} g/m²"})
        
        masa = df["Masa [g]"]
        inc_masa = df["Insertidumbre Masa [g]"]
        diametro = df["Promedio Diametro [mm]"]
        inc_diametro = df["Insertidumbre Diametro [mm]"]
        
        all_masa.extend(masa)
        all_diametro.extend(diametro)
        
        # Graficar datos con barras de error mejoradas
        ax.errorbar(
            masa, diametro,
            xerr=inc_masa, yerr=inc_diametro,
            fmt=estilo["marker"], markersize=9, 
            markerfacecolor="white", markeredgecolor=estilo["color"],
            markeredgewidth=1.5,
            ecolor=estilo["color"], elinewidth=1.2, capsize=4, 
            label=estilo["label"],
            alpha=0.85, zorder=2
        )
        
        # Añadir etiquetas a puntos específicos con fondo semitransparente para mejor lectura
        if len(masa) > 0:
            ax.annotate(f"{gramaje} g/m²", 
                        xy=(masa.iloc[0], diametro.iloc[0]),
                        xytext=(7, 7), textcoords="offset points",
                        fontsize=8, color="white", fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.3", fc=estilo["color"], alpha=0.8),
                        zorder=3)
    
    # Convertir a arrays para poder usar en el ajuste
    all_masa = np.array(all_masa)
    all_diametro = np.array(all_diametro)
    
    # Ajuste lineal sobre los logaritmos
    log_masa = np.log(all_masa)
    log_diametro = np.log(all_diametro)
    slope, intercept = np.polyfit(log_masa, log_diametro, 1)
    alpha = slope
    k = np.exp(intercept)
    
    # Generación de la recta ajustada con estilo mejorado
    masa_fit = np.linspace(min(all_masa), max(all_masa), 200)
    diametro_fit = k * masa_fit ** alpha
    ax.plot(masa_fit, diametro_fit, color="#e41a1c", linestyle="-", linewidth=2.5,
            label=f"Ajuste: D = {k:.2f}·M^{alpha:.2f}", zorder=1, alpha=0.8)
    
    # Añadir región sombreada para indicar intervalo de confianza (efecto visual)
    ax.fill_between(masa_fit, diametro_fit*0.95, diametro_fit*1.05, 
                    color="#e41a1c", alpha=0.1, zorder=0)
    
    # Añadir R² con diseño mejorado
    correlation_matrix = np.corrcoef(log_masa, log_diametro)
    correlation_xy = correlation_matrix[0,1]
    r_squared = correlation_xy**2
    ax.text(0.72, 0.22, f"R² = {r_squared:.4f}\n(Coeficiente de determinación)", 
            transform=ax.transAxes, fontsize=10, 
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.9))
    
    # Añadir una descripción de alfa en el gráfico
    ax.text(0.72, 0.12, f"α = {alpha:.4f}\n(Exponente de escala)", 
            transform=ax.transAxes, fontsize=10, 
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.9))
    
    # Mejorar las etiquetas de los ejes
    ax.set_xlabel("Masa [g]", fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel("Diámetro promedio [mm]", fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title("Ley de Escala: Relación entre Masa y Diámetro por Gramaje", 
                 fontsize=14, fontweight='bold', pad=15)
    
    # Leyenda mejorada
    legend = ax.legend(title="Leyenda", loc='upper left', framealpha=0.9, 
                    bbox_to_anchor=(0, 1), ncol=1)
    legend.get_title().set_fontweight('bold')
    
    # Añadir texto explicativo mejorado
    fig.text(0.5, 0.01, 
             "La ley de escala D = k·M^α relaciona el diámetro (D) con la masa (M) del papel.\n"
             f"Un valor α = {alpha:.4f} indica que cuando la masa aumenta, el diámetro aumenta proporcionalmente a M^{alpha:.4f}.", 
             ha='center', fontsize=9, style='italic', alpha=0.8)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    plt.savefig(output_image, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Gráfico ley de escala mejorado guardado en {output_image}")

def generar_grafico_ley_escala_loglog(archivos, output_image="grafico_ley_escala_loglog.png", figsize=(10,8)):
    """
    Genera un gráfico en escala logarítmica en ambos ejes para comparar la Ley de Escala.
    Versión mejorada visualmente.
    """
    # Mapeo de archivos a gramajes
    gramajes = {
        "datos_papel_1.csv": 80,
        "datos_papel_2.csv": 150,
        "datos_papel_3.csv": 240
    }
    
    # Estilos para cada tipo de papel (usando marcadores más distintivos)
    estilos = {
        80: {"color": colores_gramaje[80], "marker": "o", "label": "Papel 80 g/m²"},
        150: {"color": colores_gramaje[150], "marker": "s", "label": "Papel 150 g/m²"},
        240: {"color": colores_gramaje[240], "marker": "^", "label": "Papel 240 g/m²"}
    }
    
    # Crear figura con estilo mejorado
    fig, ax = plt.subplots(figsize=figsize)
    
    # Eliminar líneas del marco superior y derecho
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    
    # Almacenar todos los datos para el ajuste
    all_masa = []
    all_diametro = []
    
    # Graficar cada archivo con su propio estilo
    for archivo in archivos:
        df = pd.read_csv(archivo).round(3)
        gramaje = gramajes.get(archivo.split("/")[-1], "Desconocido")
        estilo = estilos.get(gramaje, {"color": "dimgray", "marker": "x", "label": f"Papel {gramaje} g/m²"})
        
        masa = df["Masa [g]"]
        inc_masa = df["Insertidumbre Masa [g]"]
        diametro = df["Promedio Diametro [mm]"]
        inc_diametro = df["Insertidumbre Diametro [mm]"]
        
        all_masa.extend(masa)
        all_diametro.extend(diametro)
        
        # Graficar datos con barras de error mejoradas
        ax.errorbar(
            masa, diametro,
            xerr=inc_masa, yerr=inc_diametro,
            fmt=estilo["marker"], markersize=9, 
            markerfacecolor="white", markeredgecolor=estilo["color"],
            markeredgewidth=1.5,
            ecolor=estilo["color"], elinewidth=1.2, capsize=4, 
            label=estilo["label"],
            alpha=0.85, zorder=2
        )
        
        # Añadir etiquetas a puntos específicos con fondo semitransparente para mejor lectura
        if len(masa) > 0:
            ax.annotate(f"{gramaje} g/m²", 
                        xy=(masa.iloc[-1], diametro.iloc[-1]),
                        xytext=(7, 7), textcoords="offset points",
                        fontsize=8, color="white", fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.3", fc=estilo["color"], alpha=0.8),
                        zorder=3)
    
    # Convertir a arrays para poder usar en el ajuste
    all_masa = np.array(all_masa)
    all_diametro = np.array(all_diametro)
    
    # Ajuste lineal sobre los logaritmos
    log_masa = np.log(all_masa)
    log_diametro = np.log(all_diametro)
    slope, intercept = np.polyfit(log_masa, log_diametro, 1)
    alpha_val = slope
    b = intercept  # b = log(k)
    
    # Generación de la recta ajustada con estilo mejorado
    masa_fit = np.linspace(min(all_masa), max(all_masa), 200)
    diametro_fit = np.exp(b) * masa_fit**alpha_val
    ax.plot(masa_fit, diametro_fit, color="#e41a1c", linestyle="-", linewidth=2.5,
            label=f"Ajuste: log(D) = {b:.2f} + {alpha_val:.2f}·log(M)", zorder=1, alpha=0.8)
    
    # Configurar ambos ejes en escala logarítmica con grilla más clara
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.grid(True, which="major", linewidth=1.2, alpha=0.5)
    
    # Añadir R² con diseño mejorado
    correlation_matrix = np.corrcoef(log_masa, log_diametro)
    correlation_xy = correlation_matrix[0,1]
    r_squared = correlation_xy**2
    ax.text(0.70, 0.22, f"R² = {r_squared:.4f}\n(Coeficiente de determinación)", 
            transform=ax.transAxes, fontsize=10, 
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.9))
    
    # Añadir una descripción de alfa en el gráfico
    ax.text(0.70, 0.12, f"α = {alpha_val:.4f}\n(Exponente de escala)", 
            transform=ax.transAxes, fontsize=10, 
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.9))
    
    # Mejorar las etiquetas de los ejes
    ax.set_xlabel("Masa [g]", fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel("Diámetro [mm]", fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title("Ley de Escala en Escala Log-Log", 
                 fontsize=14, fontweight='bold', pad=15)
    
    # Leyenda mejorada
    legend = ax.legend(title="Leyenda", loc='upper left', framealpha=0.9, 
                    bbox_to_anchor=(0, 1), ncol=1)
    legend.get_title().set_fontweight('bold')
    
    # Añadir texto explicativo mejorado
    fig.text(0.5, 0.01, 
             "En escala logarítmica, la relación D = k·M^α se visualiza como una línea recta.\n"
             f"La pendiente α = {alpha_val:.4f} indica cómo cambia el diámetro al aumentar la masa.", 
             ha='center', fontsize=9, style='italic', alpha=0.8)
    
    # Añadir anotaciones que expliquen la escala logarítmica
    ax.text(0.05, 0.05, 
            "Escala logarítmica: la línea recta confirma\nla relación potencial entre variables", 
            transform=ax.transAxes, fontsize=8, style='italic',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    plt.savefig(output_image, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Gráfico ley de escala log-log mejorado guardado en {output_image}")

def grafico_dependiendo_gramaje(archivo, gramaje, output_image=None, figsize=(10,8)):
    """
    Genera un gráfico para un papel con un gramaje específico.
    Versión mejorada visualmente.
    """
    # Generar nombre de archivo por defecto si no se proporciona
    if output_image is None:
        base_name = archivo.rsplit(".", 1)[0]
        output_image = f"grafico_gramaje_{gramaje}_{base_name.split('_')[-1]}.png"
    
    # Obtener color para el gramaje específico
    color = colores_gramaje.get(gramaje, "dimgray")
    
    # Leer el archivo CSV
    df = pd.read_csv(archivo).round(3)
    
    masa = df["Masa [g]"]
    inc_masa = df["Insertidumbre Masa [g]"]
    diametro = df["Promedio Diametro [mm]"]
    inc_diametro = df["Insertidumbre Diametro [mm]"]

    # Crear figura con estilo mejorado
    fig, ax = plt.subplots(figsize=figsize)
    
    # Eliminar líneas del marco superior y derecho
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    
    # Graficar los datos con barras de error mejoradas
    ax.errorbar(
        masa, diametro,
        xerr=inc_masa, yerr=inc_diametro,
        fmt='o', markersize=10, markerfacecolor="white", markeredgecolor=color,
        markeredgewidth=1.5, ecolor=color, elinewidth=1.2, capsize=4, 
        label=f"Papel de {gramaje} g/m²",
        alpha=0.85, zorder=2
    )
    
    # Añadir etiquetas a puntos significativos con mejor diseño
    for i in range(len(masa)):
        if i == 0 or i == len(masa)-1 or i == len(masa)//2:
            ax.annotate(f"M = {masa.iloc[i]}g",
                      xy=(masa.iloc[i], diametro.iloc[i]),
                      xytext=(10, 0), textcoords="offset points",
                      fontsize=9, color="white", fontweight="bold",
                      bbox=dict(boxstyle="round,pad=0.2", fc=color, alpha=0.8),
                      arrowprops=dict(arrowstyle="->", color=color, alpha=0.8))
    
    # Ajuste lineal sobre los logaritmos
    log_masa = np.log(masa)
    log_diametro = np.log(diametro)
    slope, intercept = np.polyfit(log_masa, log_diametro, 1)
    alpha_val = slope
    b = intercept  # b = log(k)
    
    # Línea de ajuste mejorada
    masa_fit = np.linspace(masa.min() * 0.95, masa.max() * 1.05, 200)
    diametro_fit = np.exp(b) * masa_fit**alpha_val
    ax.plot(masa_fit, diametro_fit, color="#e41a1c", linestyle="-", linewidth=2.5,
            label=f"Ajuste: D = {np.exp(b):.2f}·M^{alpha_val:.2f}", zorder=1, alpha=0.8)
    
    # Añadir región sombreada para indicar intervalo de confianza (efecto visual)
    ax.fill_between(masa_fit, diametro_fit*0.97, diametro_fit*1.03, 
                    color="#e41a1c", alpha=0.1, zorder=0)
    
    # Añadir R² y alfa con diseño mejorado
    correlation_matrix = np.corrcoef(log_masa, log_diametro)
    correlation_xy = correlation_matrix[0,1]
    r_squared = correlation_xy**2
    ax.text(0.72, 0.22, 
            f"R² = {r_squared:.4f}\n(Coeficiente de determinación)\n\n"
            f"α = {alpha_val:.4f}\n(Exponente de escala)", 
            transform=ax.transAxes, fontsize=10, 
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.9))
    
    # Mejorar las etiquetas de los ejes
    ax.set_xlabel("Masa [g]", fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel("Diámetro promedio [mm]", fontsize=12, fontweight='bold', labelpad=10)
    
    # Título mejorado con valor de gramaje destacado - aumentando el pad para dejar más espacio
    ax.set_title(f"Ley de Escala: Papel de {gramaje} g/m²", 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Añadir un subtítulo con información relevante - movido más arriba para evitar superposición
    ax.text(0.5, 1.12, 
           f"Comportamiento de papel con gramaje {gramaje} g/m²", 
           transform=ax.transAxes, fontsize=11, ha='center', 
           style='italic', alpha=0.8)
    
    # Leyenda mejorada
    legend = ax.legend(loc='upper left', framealpha=0.9, 
                     bbox_to_anchor=(0, 1), ncol=1)
    
    # Ajustar los límites para que se vea mejor
    x_min, x_max = masa.min(), masa.max()
    y_min, y_max = diametro.min(), diametro.max()
    ax.set_xlim(x_min * 0.9, x_max * 1.1)
    ax.set_ylim(y_min * 0.9, y_max * 1.1)
    
    # Añadir una anotación que explique el significado físico
    if gramaje == 80:
        ax.text(0.05, 0.05, 
                "Papel fino: menor rigidez\nfavorece mayor compactación", 
                transform=ax.transAxes, fontsize=9, 
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    elif gramaje == 240:
        ax.text(0.05, 0.05, 
                "Papel grueso: mayor rigidez\ndificulta la compactación", 
                transform=ax.transAxes, fontsize=9, 
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # Ajustar el layout con más espacio en la parte superior
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    plt.savefig(output_image, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Gráfico mejorado para papel de {gramaje} g/m² guardado en {output_image}")

if __name__ == "__main__":
    # Completar los datos en los CSV
    # completar_datos_csv()
    
    # Generar tablas de imagen
    generar_tabla_imagen("datos_papel_1.csv", output_image="tabla_papel_1.png")
    generar_tabla_imagen("datos_papel_2.csv", output_image="tabla_papel_2.png")
    generar_tabla_imagen("datos_papel_3.csv", output_image="tabla_papel_3.png")

    # Generar gráfico combinado
    generar_grafico_masa_diametro(
        ["datos_papel_1.csv", "datos_papel_2.csv", "datos_papel_3.csv"],
        output_image="grafico_combinado.png"
    )
    # Generar gráfico ley de escala
    generar_grafico_ley_escala(
        ["datos_papel_1.csv", "datos_papel_2.csv", "datos_papel_3.csv"],
        output_image="grafico_ley_escala.png"
    )
    # Generar gráfico ley de escala log-log
    generar_grafico_ley_escala_loglog(
        ["datos_papel_1.csv", "datos_papel_2.csv", "datos_papel_3.csv"],
        output_image="grafico_ley_escala_loglog.png"
    )

    # Generar gráficos dependiendo del gramaje
    grafico_dependiendo_gramaje("datos_papel_1.csv", 80, output_image="grafico_papel_1.png")
    grafico_dependiendo_gramaje("datos_papel_2.csv", 150, output_image="grafico_papel_2.png")
    grafico_dependiendo_gramaje("datos_papel_3.csv", 240, output_image="grafico_papel_3.png")