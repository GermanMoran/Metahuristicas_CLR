popSize=100         # Se define el tamaño de la población
Clusters = 5        # Numero de Clusters, donde se va a realizar las agrupaciones 
P = []              # Lista vacia donde se va almacenar la poblacion.
FitnesPobl = []     # Lista vacia donde se van a alamcenar cada uno los valores de la funcion objetivo de la poblacion
Best = []           # Mejor solucion de ajuste. Inicialmente es una lista vacia
QBest = 0           # Calidad del best Inicial
NEFO = 0            # Contador Numero de Evaluaciones de la Funcion Objetivo
nDim = len(df)      #Numero de Observaciones del dataset.

# Crea la poblacion
P,FitnesPobl,NEFO = crearPoblacion(popSize, nDim, Clusters,NEFO)


repeat:
    for each individuo (pi) ∈ P do:
         if (len(Best) == 0) or (FitnesPobl[z] > QBest):
            Best = P[z]
            QBest = FitnesPobl[z]

    if (NEFO == 5000):
        break

    Q = []
    FitnesQ = []

    for i in range(int(popSize/2)):
        # Seleccion padres
        Padre1,Padre2 = seleccionPadres(popSize,P)
        # Cruce
        c1,c2 = cruzeHijos(Padre1,Padre2)
        # Mutacion
        Mc1,Mc2,QMc1,QMc2,NEFO = mutacionIntercambio(c1,c2,NEFO)
        # Agregamos los cruzes a la nueva Poblacion
        Q.append(Mc1)
        Q.append(Mc2)
        # Se obtienen los Fitnes de la nueva Poblacion
        FitnesQ.append(QMc1)
        FitnesQ.append(QMc2)

    # Remplazo la nueva poblacion y la Funcion de ajuste 
    P = Q
    FitnesPobl  = FitnesQ
    
Hasta Best solucion ideal (Best < 0.9) or NEFO = 5000





