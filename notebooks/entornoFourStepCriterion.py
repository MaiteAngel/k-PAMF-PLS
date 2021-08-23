from pulp import *
import numpy as np
import matplotlib.pyplot as plt
from random import randrange
import pandas as pd

# # Funciones globales

# ## 0) Feasible solution: 

def feasible_sets_2(A,B,h=10):
    control=False
    itera=0
    while (control!=True) and itera<10:
        #print("sol factible")
        y=[randrange(-10,10),randrange(-10,10)]
        y0=[randrange(-10,10),randrange(-10,10)]
        Ajs=[[]for i in range(0,len(y))]
        Bjs=[[]for i in range(0,len(y))]
        for i in range(0,len(list(A))):
            perdidas=[abs(B[i]-y[d]*A[i]+y0[d]) for d in range(0,len(y))]
            cambiar=np.argmin(perdidas)
            Ajs[cambiar]=Ajs[cambiar]+[A[i]]
            Bjs[cambiar]=Bjs[cambiar]+[B[i]]
        if (len(Ajs[0])>h and len(Ajs[1])>h):
            control=True
        itera=itera+1
        
    Res={"Ajs": Ajs,"Bjs":Bjs}
    return Res

def feasible_sets(A,B,h=10):
    Ajs=[[]for i in range(0,2)]
    Bjs=[[]for i in range(0,2)]
    for i in range(0,len(list(A))):
        if i <= (len(list(A))/h):
            Ajs[0]=Ajs[0]+[A[i]]
            Bjs[0]=Bjs[0]+[B[i]]
        else:
            Ajs[1]=Ajs[1]+[A[i]]
            Bjs[1]=Bjs[1]+[B[i]]
    #print(f"A1 {len(Ajs[0])}, A2 {len(Ajs[1])}")
    Res={"Ajs": Ajs,"Bjs":Bjs}
    return Res


## i) Submodel Fitting: 

def submodel_fitting(Aj,Bj):
    
    #conjuntos
    conjuntoAj=Aj
    conjuntoBj=Bj
    mj=len(Aj)

    #Definimos variables
    d_vars = LpVariable.dicts("errorMilp",[(i) for i in range(0,mj)], cat='Continuous')
    wj_vars = LpVariable.dicts("coeficientesMilp",[(k) for k in range(0,n)], cat='continuous')
    wj0_vars = LpVariable.dicts("termIndependienteMilp",[(k) for k in range(0,n)], cat='Continuous')

    #Armamos problema y definimos función objetivo
    prob=LpProblem("Milp",LpMinimize)
    prob += lpSum(d_vars[(i)] for i in range(0,mj))

    #Agregamos restricciones

    for i in range(0,mj):
        prob += d_vars[(i)] >=conjuntoBj[(i)]-(wj_vars[(0)]*conjuntoAj[i])+ wj0_vars[(0)]
        prob += d_vars[(i)] >=-conjuntoBj[(i)]+(wj_vars[(0)]*conjuntoAj[i])- wj0_vars[(0)] #CAMBIE UN MENOS

    for i in range(0,mj):
        prob += d_vars[(i)]>=0
    
    resultado=prob.solve()
    
    if resultado==-1:
        print("infactible")

    elif resultado not in [-1,1]:
        print("me rompi")
        
    else:
        error=0
        for i in range(0,mj):
            error=error+d_vars[(i)].varValue
    
        Res={"coef":wj_vars[(0)].varValue, "termInd": wj0_vars[(0)].varValue,"error": error,"Aj": Aj}
        return Res


## Agregar supuesto continuidad


def submodel_fitting_2(A,Ajs,Bjs,punto,n):
    #Definimos variables
    d_vars = LpVariable.dicts("errorMilp",[(i,j) for i in range(0,len(A))for j in range(0,len(Ajs))], cat='Continuous')
    wj_vars = LpVariable.dicts("coeficientesMilp",[(k,j) for k in range(0,n) for j in range(0,len(Ajs))], cat='continuous')
    wj0_vars = LpVariable.dicts("termIndependienteMilp",[(k,j) for k in range(0,n) for j in range(0,len(Ajs))], cat='Continuous')

    #Armamos problema y definimos función objetivo
    prob=LpProblem("Milp",LpMinimize)
    prob += lpSum(d_vars[(i,j)] for i in range(0,len(A)) for j in range(0,len(Ajs)))

    #Agregamos restricciones
    for j in range(0,len(Ajs)):
        conjuntoAj=Ajs[j]
        conjuntoBj=Bjs[j]
        mj=len(conjuntoAj)
        for i in range(0,mj):
            prob += d_vars[i,j] >=conjuntoBj[i]-(wj_vars[0,j]*conjuntoAj[i])+ wj0_vars[0,j]
            prob += d_vars[i,j] >=-conjuntoBj[i]+(wj_vars[0,j]*conjuntoAj[i])- wj0_vars[0,j] #CAMBIE UN MENOS
            
    for j in range(0,len(Ajs)):
        for i in range(0,len(A)):
            prob += d_vars[(i,j)]>=0
      
    prob += ((wj_vars[(0,0)]*punto)- wj0_vars[(0,0)])==((wj_vars[(0,1)]*punto)- wj0_vars[(0,1)])
    
    #prob += sum([((wj_vars[(0,0)]*p)- wj0_vars[(0,0)])==((wj_vars[(0,1)]*p)- wj0_vars[(0,1)]) for p in A])==1
    
    resultado=prob.solve()
    
    if resultado==-1:
        print("infactible")
        
    else:
        error=0
        for j in range(0,len(Ajs)):
            conjuntoAj=Ajs[j]
            conjuntoBj=Bjs[j]
            mj=len(conjuntoAj)

            for i in range(0,mj):
                error=error+d_vars[(i,j)].varValue
        Res={"coef":[wj_vars[0,j].varValue for j in range(0,len(Ajs))], "termInd": [wj0_vars[(0,j)].varValue for j in range(0,len(Ajs))],"error": error}
        return Res


## ii) Point Partition: 

def point_partition(dp,alpha,A,B,Ajs_dic):
    
    #Calculamos Rankings
    rankings=[[] for l in Ajs_dic]
    Ajs=[l["Aj"] for l in Ajs_dic]
    #print(Ajs)
    A_nuevo=Ajs[0]+Ajs[1]
    candidatos=[]
        
    for d in range(0,len(Ajs)):
        conjunto=Ajs[d]
        for a in conjunto:
            i=np.where(np.array(A)==a)[0][0]
            cual_Aj=d
            cual_no_Aj=np.where([True if A[i] not in l else False for l in Ajs])[0]
            
            perdidas=[abs(B[i]-Ajs_dic[cual_no_Aj[d]]["coef"]*A[i]+Ajs_dic[cual_no_Aj[d]]["termInd"]) for d in range(0,len(cual_no_Aj))]
            
            #print(f"a: {a}")
            #print(f"perdidas: {perdidas}")
            cambiar=min(perdidas)
            candidatos=candidatos+[cual_no_Aj[np.argmin(perdidas)]]

            if cambiar!=0:
                ra=abs(B[i]-Ajs_dic[cual_Aj]["coef"]*A[i]+Ajs_dic[cual_Aj]["termInd"])/cambiar
            else:
                ra=100000

            rankings[cual_Aj]=rankings[cual_Aj]+[ra]
            #print(f"error actual: {abs(B[i]-Ajs_dic[cual_Aj]['coef']*A[i]+Ajs_dic[cual_Aj]['termInd'])}")
            #print(f"ranking: {ra}")

    #Calculamos puntos criticos y no criticos
        
    scores=[sorted(l,reverse=True) for l in rankings]
    puntosCriticos=[]
    puntosNOCriticos=[]
    for i in range(0,len(Ajs)):
        puntosCriticos=puntosCriticos+[[x for x in Ajs[i] if rankings[i][np.where(np.array(Ajs[i])==x)[0][0]]
                      in scores[i][0:round(alpha*len(Ajs[i]))]]]
        #print(f"puntos criticos {puntosCriticos[i]}")
        if len(puntosCriticos[i])>0:#por si hay muchos con el mismo score
            puntosCriticos[i]=puntosCriticos[i][0:round(alpha*len(Ajs[i]))]
            #print(f"proporcion: {round(alpha*len(Ajs[i]))}")
            
        puntosNOCriticos=puntosNOCriticos+[[x for x in Ajs[i] if x not in puntosCriticos[i]]]

    #Forzamos la reasignacion de los puntos criticos y los no criticos van al su correspondiente Dj  

    nuevosAjs=[[] for i in range(0,len(Ajs))]
    #print(f"La cantidad de puntos criticos es de A1 {len(puntosCriticos[0])}, de A2 {len(puntosCriticos[1])}")
    for p in range(0,len(puntosCriticos)):
        for punto in puntosCriticos[p]:
            punto_A=np.where(np.array(A_nuevo)==punto)[0][0]
            nuevosAjs[candidatos[punto_A]]=nuevosAjs[candidatos[punto_A]]+[punto]
            
    for p in range(0,len(puntosNOCriticos)):
        #l=nuevosAjs[p]+puntosNOCriticos[p] #Opcion de dejarlo donde esta
        
        for punto in puntosNOCriticos[p]:
            i=np.where(np.array(A)==punto)[0][0]
            perdida=[abs(B[i]-Ajs_dic[d]["coef"]*A[i]+Ajs_dic[d]["termInd"]) for d in range(0,len(Ajs))]
            ind=np.argmin(perdida)
            #if len(nuevosAjs[ind])>(len(A)*0.9):
            #    left=list(set(range(0,len(Ajs)))-set([ind]))[0]
            #    nuevosAjs[left]=nuevosAjs[left]+[punto]
            #else:
            nuevosAjs[ind]=nuevosAjs[ind]+[punto]
        
        #    try:
        #        indice= np.where([sum([(dp["y"][i][0]-dp["y"][j][0])*punto-(dp["y0"][i]-dp["y0"][j])>0 
        #              for j in range(0,len(Ajs)) if j!=i])==(len(Ajs)-1) for i in range(0,len(Ajs))])[0][0]
        #        nuevosAjs[indice]=nuevosAjs[indice]+[punto]
        #    except:
        #        nuevosAjs[p]=nuevosAjs[p]+[punto]
                
    #print(nuevosAjs)
    return({"Ajs":nuevosAjs})


## iii) Domain Partition: 


def domain_partition(A,Ajs,n):

    #Definimos variables
    e_vars = LpVariable.dicts("errorMilp",[(i) for i in range(0,len(A))], cat='Continuous')
    y_vars = LpVariable.dicts("coeficientesMilp",[(k,j) for k in range(0,n) for j in range(0,len(Ajs))], 
                              cat='Continuous')
    y0_vars = LpVariable.dicts("termIndependienteMilp",[(k,j) for k in range(0,1) for j in range(0,len(Ajs))], 
                              cat='Continuous')

    #Armamos problema y definimos función objetivo
    prob=LpProblem("Particion",LpMinimize)

    prob += lpSum(e_vars[(i)] for i in range(0,len(A)))

    #Agregamos restricciones

    for i in range(0,len(A)):
        for j in range(0,len(Ajs)):
            ji=np.where([True if A[i] in l else False for l in Ajs])[0][0]
            if j!=ji:
                prob += e_vars[(i)] >=-(y_vars[0,ji]-y_vars[0,j])*A[i]+y0_vars[0,ji]-y0_vars[0,j]+1

    for i in range(0,len(A)):
        prob += e_vars[(i)]>=0
    
    resultado=prob.solve()
    #print(resultado)
    
    if resultado==-1:
        print("infactible")

    elif resultado not in [-1,1]:
        print("me rompi")
        
    else:
        error=0
        error_completo=[]
        for i in range(0,len(A)):
            error=error+e_vars[(i)].varValue
            error_completo=error_completo+[e_vars[(i)].varValue]
        y=[]
        y0=[]
        for j in range(0,len(Ajs)):
            coef=[]
            for d in range(0,n):
                coef=coef+[y_vars[d,j].varValue]
            y=y+[coef]
            y0=y0+[y0_vars[0,j].varValue]

        Res={"error": error,"y": y,"y0":y0,"Ajs":Ajs,"error completo":error_completo}
        return Res


## iv) Partition Consistency: 
# * Reasigna puntos a su Dj correcto


def partition_consistency(dp,A):
    
    Ajs_actuales=dp["Ajs"]
    Ajs_nuevos=[[] for i in range(0,len(Ajs_actuales))]
    
    for j in range(0,len(Ajs_actuales)):
        for punto in Ajs_actuales[j]:
            #if dp["error completo"][np.where(np.array(A)==punto)[0][0]]>0:
            pasa=np.where([True if punto not in l else False for l in Ajs_actuales])[0][0]
            if ((dp["y"][j][0]-dp["y"][pasa][0])*punto-(dp["y0"][j]-dp["y0"][pasa]))>0:#puse un mayor igual
                Ajs_nuevos[j]=Ajs_nuevos[j]+[punto]
            else:
                Ajs_nuevos[pasa]=Ajs_nuevos[pasa]+[punto]
    
        
    Res={"Ajs":Ajs_nuevos,"y":dp["y"],"y0":dp["y0"]}
    return Res 


# ## Fin del proceso

# Estos 4 puntos se repiten hasta que $\alpha$ sea 0 o se alcance el tiempo máximo.
def FourStepCriterion(inputs,respuesta,alpha,rho,n):
    iteracion=1
    #Generamos solución factible
    
    fs=feasible_sets(inputs,respuesta,h=10)
    A1,A2,B1,B2=fs["Ajs"][0],fs["Ajs"][1],fs["Bjs"][0],fs["Bjs"][1]
    A=A1+A2
    B=B1+B2
    dp=domain_partition(A,[A1,A2],n)
    pc=partition_consistency(dp,A)
    if dp["y"][0][0]==0:
        dp["y"][0][0]=0.000001
    if dp["y0"][1]==0 and dp["y"][1][0]==0:
        try:
            punto=max([x for x in inputs if x<=(dp["y0"][0]/dp["y"][0][0])])
        except:
            punto=dp["y0"][0]/dp["y"][0][0]
    #punto=dp["y0"][0]/dp["y"][0][0]
    A_sol=submodel_fitting_2(A1+A2,[A1,A2],[B1,B2],punto,n)
    A1_ganador=A1
    A2_ganador=A2
    B1_ganador=B1
    B2_ganador=B2
    error=A_sol["error"]

    while (alpha>=0.001):
        
        for count in range(1,6):
            if [] in pc["Ajs"]:
                #print("entre")
                fs=feasible_sets(inputs,respuesta,h=10-count)
                A1,A2,B1,B2=fs["Ajs"][0],fs["Ajs"][1],fs["Bjs"][0],fs["Bjs"][1]
                dp=domain_partition(A,[A1,A2],n)
                pc=partition_consistency(dp,A)
            if [] not in pc["Ajs"]:
                break

        if [] in pc["Ajs"]:
            print("soy lineal asecas")
            break
 
        if dp["y"][0][0]==0:
            dp["y"][0][0]=0.000001
        if dp["y0"][1]==0 and dp["y"][1][0]==0:
            try:
                punto=max([x for x in inputs if x<=(dp["y0"][0]/dp["y"][0][0])])
            except:
                punto=dp["y0"][0]/dp["y"][0][0]
        else:
            try:
                punto=max([x for x in inputs if x<=(dp["y0"][1]/dp["y"][1][0])])
            except:
                punto=dp["y0"][1]/dp["y"][1][0]
        #punto=dp["y0"][0]/dp["y"][0][0]
        #print(punto)
        
        A_sol=submodel_fitting_2(A1+A2,[A1,A2],[B1,B2],punto,n)
        alpha=0.99*(rho**(iteracion))
        iteracion=iteracion+1
        A1_sol={"Aj":A1,"coef":A_sol["coef"][0],'termInd':A_sol['termInd'][0]}
        A2_sol={"Aj":A2,"coef":A_sol["coef"][1],'termInd':A_sol['termInd'][1]}
        pp=point_partition(dp,alpha,A,B,[A1_sol,A2_sol])
        dp=domain_partition(A,pp["Ajs"],n)
        pc=partition_consistency(dp,A)
        A1=pc["Ajs"][0]
        A2=pc["Ajs"][1]
        B1=[B[np.where(np.array(A)==a)[0][0]] for a in A1]
        B2=[B[np.where(np.array(A)==a)[0][0]] for a in A2]
        
        if A_sol["error"]<=error:
            A1_ganador=A1
            A2_ganador=A2
            B1_ganador=B1
            B2_ganador=B2
            error=A_sol["error"]

        #print(f"error: {error}")
        
        #print(iteracion)
        
    A1=A1_ganador
    A2=A2_ganador
    B1=B1_ganador
    B2=B2_ganador
    dp=domain_partition(A,[A1,A2],n)
    if dp["y"][0][0]==0:
        dp["y"][0][0]=0.000001
    if dp["y0"][1]==0 and dp["y"][1][0]==0:
        try:
            punto=max([x for x in inputs if x<=(dp["y0"][0]/dp["y"][0][0])])
        except:
            punto=dp["y0"][0]/dp["y"][0][0]
    else:
        try:
            punto=max([x for x in inputs if x<=(dp["y0"][1]/dp["y"][1][0])])
        except:
            punto=dp["y0"][1]/dp["y"][1][0]
    A_sol=submodel_fitting_2(A1+A2,[A1,A2],[B1,B2],punto,n)
    A1_sol={"Aj":A1,"coef":A_sol["coef"][0],'termInd':A_sol['termInd'][0]}
    A2_sol={"Aj":A2,"coef":A_sol["coef"][1],'termInd':A_sol['termInd'][1]}
    
    return A1_sol,A2_sol,A1,A2,punto


# Plots


def plot_separability(inputs,respuesta,A1_sol,A2_sol,A1,A2,x=[-0.4, 1],y=[-5, 8]):
    
    objective_A1 = np.vectorize(lambda x: A1_sol["coef"]*x-A1_sol["termInd"])
    objective_A2 = np.vectorize(lambda x: A2_sol["coef"]*x-A2_sol["termInd"])
    A_nuevo=A1+A2
    Prediccion = list(objective_A1(np.array(A1)))+list(objective_A2(np.array(A2)))
    dp=domain_partition(A_nuevo,[A1,A2])
    print(f"El error de mala clasificacion es {dp['error']}")
    print(dp["y0"])
    objective_LS = np.vectorize(lambda x: dp["y"][0][0]*x-dp["y0"][0])
    LS = list(objective_LS(np.array(A_nuevo)))
    plt.scatter(A_nuevo, Prediccion)
    plt.plot(A_nuevo,LS, '-o',color="red")
    plt.plot(A_nuevo,LS, '-o',color="red")
    plt.ylim(y)
    plt.xlim(x)
    plt.show()

def plot_prediccion(inputs,respuesta,A1_sol,A2_sol,A1,A2):
    objective_A1 = np.vectorize(lambda x: A1_sol["coef"]*x-A1_sol["termInd"])
    objective_A2 = np.vectorize(lambda x: A2_sol["coef"]*x-A2_sol["termInd"])

    A_nuevo=A1+A2
    Prediccion = list(objective_A1(np.array(A1)))+list(objective_A2(np.array(A2)))

    plt.figure(figsize=(10,10))
    plt.plot(inputs, respuesta, 'o',color="lightgreen", markersize=12)
    plt.plot(A_nuevo, Prediccion, 'o', markersize=6)
    plt.show()