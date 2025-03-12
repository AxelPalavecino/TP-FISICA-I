""" ------------------------ PRACTICA 1 ------------------------ """

"""
Vamos a buscar una Ley de Escala. El objetivo de la propuesta es que midan varias veces las magnitudes, reportando incertezas experimentales y estadísticas de sus mediciones. Detallen las consideraciones que tomaron para realizar cada medición. Todas las magnitudes reportadas deben incluir su incerteza. Deben escribir un informe técnico reportando los resultados de los experimentos. El informe debe contar con una introducción, descripción de los métodos experimentales, análisis de los resultados y conclusiones.

Las tareas a realizar son: 

1. Tomen hojas cuadradas de lado L y masa M.
2. Reporten el valor del área de cada hoja utilizada.
3. Hagan bollos con ellas lo mas compactos posibles.
4. Midan el diámetro D del bollo obtenido.
5. Reporten el valor del volumen de cada bollo.
6. Repitan todo para diversos tamaños L.
7. Grafiquen el tamaño D del bollo en función de la masa M.

Observando los gráficos obtenidos:
-   ¿Qué forma tienen los datos? (p. ej.: recta, cuadrática, raíz cuadrada, etc).
-   ¿Es posible realizar un ajuste lineal de los datos que resulte en una buena descripción de la relación
entre variables?

Ahora repitan los gráficos de los ítems anteriores, pero esta vez utilizando un gráfico con escalas logarítmicas en ambos ejes coordenados. Observen los nuevos gráficos y reflexionen acerca de las siguientes
preguntas:
-   ¿Qué forma adoptan en esta nueva representación?
-   ¿Qué información es posible obtener de un ajuste lineal en esta representación?
-   ¿Cómo interpretan físicamente lo obtenido para la dependencia, por ejemplo, D vs M?

"""

import os
import matplotlib.pyplot as plt

def calculo_area(lado):
    return lado**2

def calculo_volumen(diametro):
    return (4/3)*3.14159265359*(diametro/2)**3

def datos_almacenar():
    import os

    # Crear cabecera si el archivo no existe o está vacío
    if not os.path.exists("datos_pruebas.txt") or os.path.getsize("datos_pruebas.txt") == 0:
        with open("datos_pruebas.txt", "w") as f:
            f.write("Lado\tMasa\tDiametro\tArea\tVolumen\n")

    with open("datos_pruebas.txt", "a") as f:
        while True:
            opcion = input("\nIngrese 'q' para salir o presione Enter para continuar: ")
            if opcion.lower() == 'q':
                break
            try:
                lado = float(input("Ingrese el lado (L) en centímetros: "))
                masa = float(input("Ingrese la masa (M) en gramos: "))
                diam = float(input("Ingrese el diámetro (D) en centímetros: "))
            except ValueError:
                print("Por favor, ingrese valores numéricos válidos.")
                continue

            area = calculo_area(lado)
            volumen = calculo_volumen(diam)
            f.write(f"{lado}\t{masa}\t{diam}\t{area}\t{volumen}\n")


def main():
    # Listas para almacenar datos
    M = []
    D = []

    # Verificar si existe datos.txt
    if os.path.exists("datos.txt"):
        print("Leyendo datos existentes de datos.txt...\n")
        with open("datos.txt", "r") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                line = line.strip()
                print(line)  # Imprime los datos existentes
                if i > 0:  # Ignora la cabecera y almacena datos
                    partes = line.split()
                    if len(partes) == 2:
                        try:
                            valM = float(partes[0])
                            valD = float(partes[1])
                            M.append(valM)
                            D.append(valD)
                        except ValueError:
                            pass  # Ignora líneas mal formateadas
    else:
        print("datos.txt no existe, se creará un archivo nuevo.")

    # Abrir archivo en modo 'append' para agregar datos nuevos
    with open("datos.txt", "a") as f:
        # Si está vacío, escribir la cabecera
        if os.path.getsize("datos.txt") == 0:
            f.write("M\tD\n")

        # Bucle para ingresar datos hasta que el usuario decida salir
        while True:
            opcion = input("\nIngrese 'q' para salir o presione Enter para continuar: ")
            if opcion.lower() == 'q':
                break
            try:
                masa = float(input("Ingrese la masa (M) en gramos: "))
                diam = float(input("Ingrese el diámetro (D) en centimetros: "))
            except ValueError:
                print("Por favor, ingrese valores numéricos válidos.")
                continue

            # Agregar los datos a las listas y al archivo
            M.append(masa)
            D.append(diam)
            f.write(f"{masa}\t{diam}\n")

    # Graficar los datos si se cuenta con al menos un punto
    if M and D:
        plt.figure()
        plt.scatter(M, D, color="blue", label="Datos (M vs D)")
        plt.xlabel("Masa (M)")
        plt.ylabel("Diámetro (D)")
        plt.title("Gráfico de D vs M")
        plt.legend()
        plt.show()

if __name__ == "__main__":
    datos_almacenar()

