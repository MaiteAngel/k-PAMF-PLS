# Trabajo Final Optimización: k-PAMF-PLS

_Paper guía: Discrete optimization methods to fit
piecewise-affine models to data points. E. Amaldi + , S. Coniglio , L. Taccari (2016)_

## Introducción

Proyecto final: replicar manualmente una version del algoritmo k-PAMF-PLS.

### Datos

_Solicitar_

### Objetivo

El objetivo será encontrar una función de baja complejidad que describa el comportamiento de las observaciones $`a_{i}`$ con respuesta $b_{i}$.

En este trabajo nos centraremos en una versión general de los modelos afines a trozos con una función de error lineal $||L_{1}||$ y una partición del dominio de modo que resulte linealmente separable en subdominios, notaremos este enfoque como k-PAMF-PLS (k-Piecewise Affine Model Fitting with Pairwise Linear Separability Problem).

### Problema

Dados puntos $a_{i}$ reales con observación $b_{i}$ y un entero $k$:

Debemos definir una partición del dominio $A_{1}$, . . . , $A_{k}$, definidos por los conjuntos $D_{1}$, . . . , $D_{k}$ de manera que queden linealmente separados:

* para cada $j$ $A_{j}$ está totalmente contenido en un $D_{j}$

* $D_{i}$ y $D_{j}$ son separables por un hiperplano (una lineal).

Cada $D_{j}$ se define como un vector de parámetros $(y_{j},y_{j_{0}})$ de forma que:
$a_{j}$ en $A_{j}$ pertenece a $D_{j}$ sii para todo $j'$ distinto de $j$, tenemos 
$(y_{j} - y_{j'})a_{j}- (y_{j_{0}} - y_{j'_{0}}) > 0$, así para todo $A_{n}$ $A_{m}$ están separados por el hiperplano

$$H=\{ x : (y_{n} - y_{m})x = (y_{n_{0}}-y_{m_{0}})\}$$

Definidos los $D_{j}$ se buscara encontrar funciones lineales $f_{j}:D_{j}\rightarrow R$, $$f_{j}(x)=w_{j}x-w_{j0}$$ que minimicen el error global:
$$
ErrorGlobal=\sum_{j \in J}|b_{i}-f_{j}(a_{i})|
$$

### Resolución propuesta

Vamos a resolverlo con un modelo de programación lineal entera mixta.

MILP es la colisión entre un problema de hyperplane clustering formulation (consigo $A_{1}$, . . . , $A_{k}$ y un modelo lineal) y multicategory linear classification constraints (consigo $D_{1}$, . . . , $D_{k}$ )

Una heurística de 4 pasos será implementada tratando de:

* minimizar el error de fitteo (identificando aquellos puntos que sean críticos).

* buscar la mejor partición del dominio (a través de un modelo multi-clasificación lineal). 

