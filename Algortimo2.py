# librerias
from random import random
import random
from math import dist
import math
import pandas as pd
import numpy as np

# Librerias de Machine Learning
# =================================================================================================================
from matplotlib import pyplot as plot
import statsmodels.api as sm
from statistics import mode
from   scipy import stats
import matplotlib.pyplot       as plt
import seaborn                 as sns
import statsmodels.formula.api as sm
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import statsmodels.formula.api as smf


# Variables Globales
# =================================================================================================================

popSize=100         # Se define el tamaño de la población
Clusters = 5        # Numero de Clusters, donde se va a realizar las agrupaciones 
P = []              # Lista vacia donde se va almacenar la poblacion.
FitnesPobl = []     # Lista vacia donde se van a alamcenar cada uno los valores de la funcion objetivo de la poblacion
Best = []           # Mejor solucion de ajuste. Inicialmente es una lista vacia
QBest = 0           # Calidad del best Inicial
NEFO = 0            # Contador Numero de Evaluaciones de la Funcion Objetivo

# Variables Globales   - DataFrame
# =================================================================================================================
df = pd.read_excel('prueba.xlsx',engine='openpyxl')
nDim = len(df)      #Numero de Observaciones del dataset.


# Funcion para obtener el valor de Aptitud (Fitness) de cada individuo
# ==============================================================================
def fitnesIndividuo(df,individuo,Clusters,NEFO):

    listFitnes = []
    # Se realizan los respectivos filtros
    for i in range(Clusters):
        vectorArray = np.array(individuo)
        clus =  np.where(vectorArray == (i+1))
        list_cls = clus[0].flatten().tolist()
        # Se filtra el dataset deacuerdo a esos indices
        df_cls = df[df.index.isin(list_cls)==True]
        # Se realiza el llamado a la función lineal
        fitnes_cls = RegresionLineal(df_cls)
        listFitnes.append(fitnes_cls)
    
    # Fitnes solucion | Individuo
    FitnesSoluccion = sum(listFitnes)/len(df)
    # Actualizo el contador
    NEFO = NEFO+1
    return FitnesSoluccion,NEFO



# Funcion para calcular el R-Ajustado asociado a cada cluster | Proceso de selección de Atributos
# ======================================================================================================================================0
def RegresionLineal(df_cls):
    Pond_Radjustado = len(df_cls)
    # Se divide el dataset en conjunto de entrenamiento y pruebas [0.8 : Entrenamiento; 0.2: Testeo]
    data_train, data_test = train_test_split(df_cls, test_size=0.2, random_state=42)
    variables_independientes = '''~ C(MATERIAL_GENETICO) + C(TIPO_MATERIAL) +C(POSICION_PERFIL_RASTA) +C(ESTRUCTURA_RASTA) + C(OBSERVA_COSTRAS_DURAS_RASTA)
                                    + C(OBSERVA_COSTRAS_BLANCAS_RASTA) + C(OBSERVA_COSTAS_NEGRAS_RASTA) + C(OBSERVA_PLANTAS_PEQUENAS_RASTA) + C(RECUBRIMIENTO_VEGETAL__SUELO_RASTA)
                                    + C(TIPO_SIEMBRA)+ C(SEM_TRATADAS) + C(DIAS_EN_FLORECER_A_COSECHAR) +ContEnfQui_Emer_Flor + ContMalQui_Antes_Siem + ContMalQui_Emer_Flor
                                    + ContPlaQui_Siem_Emer + PROFUND_RAICES_VIVAS_RASTA + Porc_Ar + Porc_FAr + Porc_BLANDO + Porc_FIRME + Porc_MUY_PLASTICO
    '''
    # Se crea el modelo
    mod_cls = smf.ols(formula='RDT_AJUSTADO'+variables_independientes, data=data_train).fit()
    #Me retorna el valor de la metrica de desempeño  para ese cluster.
    try:
        fitnes_cls = mod_cls.rsquared_adj * Pond_Radjustado
    except:
        fitnes_cls =0
    return fitnes_cls




#Funcion que genera el individuo/cromosoma nDim (Numero Observaciones)
# ==================================================================================
def vectorSolution(nDim,Clusters):
    s= []
    for i in range(nDim):
        s.append(random.randint(1, Clusters))    
    return s



# Funcion para crear la poblacion 
# ==============================================================================
def crearPoblacion(popSize,nDim,Clusters,NEFO):
    for i in range(popSize):
        #print(i)
        # Creo el individuo
        individuo = vectorSolution(nDim, Clusters)
        P.append(individuo)
        # Evaluo el individuo (Fitness)
        FitInd = fitnesIndividuo(df,individuo,Clusters,NEFO)
        # Agrego el ajuste a lista de Fitness de la población
        NEFO = FitInd[1]
        FitnesPobl.append(FitInd[0])
    

    return P,FitnesPobl,NEFO



# Funcion  Seleccion de Padres  (Aleatoria)     # paso poblacion
# ==============================================================================
def seleccionPadres(popSize,Poblacion):
    # Se genrar una lista con el Tamaño de la poblacion
    tamanoPoblacion = list(range(popSize))
    padres = np.random.choice(tamanoPoblacion, 2, False)
    Padre1 = Poblacion[padres[0]]
    Padre2 = Poblacion[padres[1]]

    return Padre1,Padre2


# Función Seleccion de Padres mediante (Elitismo)     
# =====================================================
def seleccionPadresElitismo(Poblacion,FitnesPobl):
    FitnesPoblOrd = sorted(FitnesPobl, reverse=True)
    
    if (FitnesPoblOrd[0] == FitnesPoblOrd[1]):
        # Indice Padre1
        Indice1 = FitnesPobl.index(FitnesPoblOrd[0]) 
        # Indice Padre2|Madre2
        Indice2 = FitnesPobl.index(FitnesPoblOrd[1],2)
    
    else:
        # Indice Padre1
        Indice1 = FitnesPobl.index(FitnesPoblOrd[0])
        # Indice Padre2|Madre2
        Indice2 = FitnesPobl.index(FitnesPoblOrd[1])
  
    
    Padre1 = Poblacion[Indice1]
    Padre2 = Poblacion[Indice2]
    
    return Padre1,Padre2


#  Cruze entre hijos mediante Clusters
# ===================================================================================
def cruzeHijosClusters(Padre1,Padre2,Nclusters):
    listDimCluster = list(range(1,Nclusters))
    clusterChange = np.random.choice(listDimCluster, 1)
    #print("Cluster a Cambiar: ",clusterChange)
    # Encontramos los indices donde sean iguales cl cluster Seleccionado
    IndexP1 = [indice for indice, dato in enumerate(Padre1) if dato == clusterChange[0]]
    IndexP2 = [indice for indice, dato in enumerate(Padre2) if dato == clusterChange[0]]
    
    # Realizo el intercabio de cluster  - Para Formar los Hijos
    # H1: 1 madre | 2 padre
    for i in range(len(IndexP1)):
        Padre2[IndexP1[i]] = clusterChange[0]

    # H2 #2 madre | 1 padre
    for i in range(len(IndexP2)):
        Padre1[IndexP2[i]] = clusterChange[0]

    c1 = Padre1
    c2 = Padre2
    
    #print("Cruze 1: ",c1)
    #print("Cruze 2: ",c2)
    return c1,c2




# Funcion para realizar el cruze entre los padres   (Cruze en un Punto)
# ==============================================================================
def cruzeHijos(Padre1,Padre2):
    dimPadre = list(range(len(Padre1)))
    # seleccion el punto de cruze
    puntoCruze = np.random.choice(dimPadre, 1)
    #Se realiza el intercambio de genes a travez del punto de cruze
    Padre1[puntoCruze[0]:] ,Padre2[puntoCruze[0]:]= Padre2[puntoCruze[0]:], Padre1[puntoCruze[0]:]
    # Se obtienen los 2 Hijos
    c1 = Padre1
    c2 = Padre2
    return c1,c2



# Funcion para generar un numero aleaorio [1 -5]  tamaño del cluster
# ==============================================================================
def gernararAleatroio():
    return random.randint(1, 5)



# Funcion para realizar Mutacion entre los Hijos - Operador Intercambio
# ==============================================================================
def mutacionIntercambio(c1,c2,NEFO):
    #Elijo 2 posiciones de forma aletoria
    Dim = list(range(len(c1)))
    DimIntercambio = np.random.choice(Dim, 2, False)
    #Realizo el intercambio de los genes para ambos cruzes
    #Cruze 1
    c1[DimIntercambio[0]],c1[DimIntercambio[1]] = c1[DimIntercambio[1]],c1[DimIntercambio[0]]
    #Cruze 2
    c2[DimIntercambio[0]],c2[DimIntercambio[1]] = c2[DimIntercambio[1]],c2[DimIntercambio[0]]
    
    # Reasigno los vectores mutados a una nueva variables
    Mc1 = c1
    Mc2 = c2
    # Obtengo el Fitnes relacionado a cada mutacion
    QMc1 = fitnesIndividuo(df,Mc1,Clusters,NEFO)
    NEFO = QMc1[1]
    QMc2 = fitnesIndividuo(df,Mc2,Clusters,NEFO)
    NEFO = QMc2[1]

    return Mc1,Mc2,QMc1[0],QMc2[0],NEFO


# Funcion para remplazar la nueva Poblacion  - Operador del peor
# ==============================================================================
def RemplazoDelPeor(fitP,fitNP,Poblacion,NuevaPoblacion):
    # Agrupanos en una sola lista el fitness de ambas poblaciones
    fitT = fitP + fitNP
    # Agrupamos en una sola lista las poblaciones
    PT = Poblacion + NuevaPoblacion
    # Se convierte la lista de Fitnes Total  a df
    df =  pd.DataFrame(fitT,columns=["Fitnes"])
    # Obtengo los N mejores individuos de acuerdo al tamaño de la población
    df = df.sort_values('Fitnes',ascending=False)[0:len(fitP)]
    # Lista de los mejores valores de Aptitud
    Nfit = list(df.Fitnes)
    # Lista de los indices de la Poblacioón
    indexNuevaPoblacion = list(df.index)
    # Ubicamos los Individuos de la Nueva Población

    NP = []
    for i in range(len(indexNuevaPoblacion)):
        NP.append(PT[indexNuevaPoblacion[i]])
    
    # Retorna la lista de la nueva Poblacion y los respectivos Fitnes
    return NP, Nfit





#  INICIO DEL ALGORTIMO - OPTIMIZACION MEDIANTE ALGORTIMOS GENETICOS ENFOQUE CLR
#  ======================================================================================

# Se crea la Población
P,FitnesPobl,NEFO = crearPoblacion(popSize, nDim, Clusters,NEFO)
print("Funcion Ajuste: ",FitnesPobl)
#print("Numero de Evaluaciones de la Funcion Objetivo: ",NEFO)



# Se define criterio de parada Qbest = 0.9 o NEFO = 5000 
while (QBest < 0.9):

    if (NEFO == 4000):
        break


    # Best de la Poblacion
    # =======================================================================================
    for z in range(len(P)):
        # Seleccionamos el mejor individuo | El que tiene mayor valor de aptitud (R- Ajustado)
        if (len(Best) == 0) or (FitnesPobl[z] > QBest):
            Best = P[z]
            QBest = FitnesPobl[z]
        

    print("El fitness asociado al Best [Mejor Solucion] es: ",QBest)



    # Variables de la Nueva Poblacion Q 
    # ==============================================================================
    Q = []
    FitnesQ = []


    # Genero la nueva Poblacion Q 
    # ==============================================================================
    for i in range(int(popSize/2)):
        # Seleccion padres
        Padre1,Padre2 = seleccionPadresElitismo(P,FitnesPobl)
        # Cruce
        c1,c2 = cruzeHijos(Padre1, Padre2)
        # Mutacion
        Mc1,Mc2,QMc1,QMc2,NEFO = mutacionIntercambio(c1,c2,NEFO)
        # Agregamos los cruzes a la nueva Poblacion
        Q.append(Mc1)
        Q.append(Mc2)
        # Se obtienen los Fitnes de la nueva Poblacion
        FitnesQ.append(QMc1)
        FitnesQ.append(QMc2)
        #print("==================Termina la iteracion: ", i," =============================")

    
    #print("Ajuste Nueva Poblacion: ",FitnesQ)
    print("Numero de Evaluaciones de la Funcion Objetivo: ",NEFO)

    # Remplazo la nueva poblacion y la Funcion de ajuste 

    P, FitnesPobl= RemplazoDelPeor(FitnesPobl,FitnesQ,P,Q)
    

# Gurdamos los datos de la mejor solucion
print("El maximo R-Ajustado es: ", QBest)
print("La mejor distribucion de datos es: ", Best)