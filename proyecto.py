from __future__ import division
from pyomo.environ import *

from pyomo.opt import SolverFactory
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


Model = ConcreteModel()

# Data de entrada
drones = 2
solar = 2


locs=RangeSet(0, drones)
travelers = RangeSet(1,solar)


#Conjunto
depots = pd.read_csv('data\multi_depots.csv')
clientes = pd.read_csv('data\clients.csv')
costos = pd.read_csv('data\data_vehiculos.csv')
vehiculos = pd.read_csv('data\multi_vehicles.csv')

import numpy as np

# Dimensiones de la matriz
long_depots = len(depots)
long_clientes = len(clientes)
len_total = long_clientes + long_depots

# Inicialización de la matriz de distancias con 999
matriz = np.full((len_total, len_total), 999)

# Cálculo de distancias entre depósitos y clientes
for i in range(long_depots):
    longitud_origen = depots.iloc[i, 1]
    latitud_origen = depots.iloc[i, 2]

    for j in range(long_clientes):
        longitud_destino = clientes.iloc[j, 1]
        latitud_destino = clientes.iloc[j, 2]

        norma = np.sqrt((longitud_origen - longitud_destino)**2 + (latitud_origen - latitud_destino)**2)
        matriz[i, j + long_depots] = norma  # De depósito a cliente
        matriz[j + long_depots, i] = norma  # De cliente a depósito

# Cálculo de distancias entre clientes
for i in range(long_clientes):
    longitud_origen = clientes.iloc[i, 1]
    latitud_origen = clientes.iloc[i, 2]

    for j in range(i + 1, long_clientes):  # Solo para j > i (matriz simétrica)
        longitud_destino = clientes.iloc[j, 1]
        latitud_destino = clientes.iloc[j, 2]

        norma = np.sqrt((longitud_origen - longitud_destino)**2 + (latitud_origen - latitud_destino)**2)
        matriz[i + long_depots, j + long_depots] = norma
        matriz[j + long_depots, i + long_depots] = norma

#Matriz costos
matriz_costo = np.full((len(vehiculos),len(costos.iloc[0])+len(vehiculos.iloc[0])-1),999,dtype="float")
for i in range(len(vehiculos)):

    for j in range(1,len(vehiculos.iloc[0])):
        matriz_costo[i,j] = vehiculos.iloc[i,j]

    for k in range(1,len(costos.iloc[0])):
        print(costos.iloc[1, k])
        if vehiculos.iloc[i, 0] == "Gas Car":
            matriz_costo[i,0] = 1
            try:
                matriz_costo[i,k+len(vehiculos.iloc[0])-1] = costos.iloc[0, k]
            except:
                matriz_costo[i,k+len(vehiculos.iloc[0])] = 0
        if vehiculos.iloc[i, 0] == "EV":
            matriz_costo[i,0] = 2
            try:
                matriz_costo[i,k+len(vehiculos.iloc[0])-1] = costos.iloc[1, k]
            except:
                matriz_costo[i,k+len(vehiculos.iloc[0])-1] = 0
        if vehiculos.iloc[i, 0] == "drone":
            matriz_costo[i,0] = 3
            try:
                matriz_costo[i,k+len(vehiculos.iloc[0])-1] = costos.iloc[2, k]
            except:
                matriz_costo[i,k+len(vehiculos.iloc[0])-1] = 0

#Rangos

range_nodos = RangeSet(0,len_total)
range_vehiculos = RangeSet(0,len(vehiculos))
range_depots = RangeSet(0,long_depots)

 # Decision Variables
Model.x = Var(model.matriz_distancias, model.matriz_distancias, model.matriz_costos, domain=Binary)  # Binary variable for sensor placement

#Variable Auxiliar
Model.u = Var(self.range_nodos, self.range_vehiculos,domain= NonNegativeReals)


#Variable de decisión
Model.x = Var(self.range_nodos, self.range_nodos, self.range_vehiculos, domain= Binary, initialize=0)

#Variable Auxiliar
Model.u = Var(self.range_nodos, self.range_vehiculos,domain= NonNegativeReals)

class SensorPlacementModel:
    def __init__(self, matriz_distancias, matriz_costos, range_nodos, range_vehiculos, range_depots, len_total, matriz_demandas):
        """
        Initialize the model parameters.
        """
        distances = {
            (i, j): matriz_distancias[i, j]
            for i in range(len_total)
            for j in range(len_total)
            if i != j
        }
        
        tam_demandas = matriz_demandas.shape[0]
        demand = {i: matriz_demandas[i] for i in range(tam_demandas)}

        rows, cols = matriz_costos.shape

        costs = {
            (i, j): matriz_costos[i, j]
            for i in range(rows)
            for j in range(cols)
            if i != j
        }

        self.matriz_distancias = distances
        self.matriz_costos = costs
        self.range_nodos = range_nodos
        self.range_vehiculos = range_vehiculos
        self.range_depots = range_depots
        self.len_total = len_total
        self.matriz_demandas = demand

        # Create the Pyomo model
        self.model = ConcreteModel()

    def build_model(self):
        """
        Build the optimization model.
        """
        model = self.model

        # Sets
        model.matriz_distancias = Set(initialize=self.matriz_distancias.keys(), dimen=2)  # Different sensor types
        model.matriz_costos = Set(initialize=self.matriz_costos.keys(), dimen=2)  # Possible installation locations
        
            # Parameters
        model.matriz_distancias_param = Param(
            model.matriz_distancias,
            initialize=self.matriz_distancias
        )
        model.matriz_costos_param = Param(
            model.matriz_costos,
            initialize=self.matriz_costos
        )

        # Decision Variables
        model.x = Var(model.matriz_distancias, model.matriz_distancias, model.matriz_costos, domain=Binary)  # Binary variable for sensor placement
        #Variable Auxiliar
        Model.u = Var(self.range_nodos, self.range_vehiculos,domain= NonNegativeReals)


        #Variable de decisión
        Model.x = Var(self.range_nodos, self.range_nodos, self.range_vehiculos, domain= Binary, initialize=0)

        #Variable Auxiliar
        Model.u = Var(self.range_nodos, self.range_vehiculos,domain= NonNegativeReals)

        # Función objetivo
        def obj_expression(model):
            cosot_distancia = sum(model.x[i,j,k]*self.matriz_distancias[i,j]*self.matriz_costos[k,1] for i in self.range_nodos for j in self.range_nodos for k in self.range_vehiculos)

            costo_tiempo_viaje = sum(model.x[i,j,k]*self.matriz_distancias[i,j]/self.matriz_costos[k,8]*self.matriz_costos[k,4] for i in self.range_nodos for j in self.range_nodos for k in self.range_vehiculos)

            costo_mantenimiento = 0
            w = 0
            while w <= self.range_vehiculos:
                usa_w = 0
                for i in self.range_nodos:
                    for j in self.range_nodos:
                        if model.x[i,j,w] == 1:
                            w += 1
                            usa_w = 1
                            break
                    if model.x[i,j,w] == 1:
                        break
                costo_mantenimiento += usa_w*self.matriz_costos[w,5]

            #lo primero es lo que vale el combustible y lo segundo el tiempo
            costo_refuel = sum(
            model.x[i, j, k] * self.matriz_distancias[i, j] * self.matriz_costos[k, 6] * 
            (self.matriz_costos[k, 9]**-1 if self.matriz_costos[0] == 1 else self.matriz_costos[k, 10]) +
            model.x[i, j, k] * self.matriz_distancias[i, j] * self.matriz_costos[k, 7] * self.matriz_costos[w,4] * 10 *
            (self.matriz_costos[k, 9]**-1 if self.matriz_costos[0] == 1 else self.matriz_costos[k, 10])
            for i in self.range_nodos
            for j in self.range_nodos
            for k in self.range_vehiculos)


            costo_recharge = sum(model.x[i,j,k]*self.matriz_distancias[i,j]*self.matriz_costos[k,6] for i in self.range_nodos for j in self.range_nodos for k in self.range_vehiculos)

            return cosot_distancia + costo_tiempo_viaje + costo_mantenimiento + costo_refuel + costo_recharge

        Model.obj = Objective(rule=obj_expression, sense=minimize)

        #Restricciones
        Model.entrada = ConstraintList()
        Model.salida = ConstraintList()
        Model.entrada_salida = ConstraintList()
        for j in self.range_nodos:
            if j != 0:
                Model.entrada.add(sum(Model.x[i,j,k] for k in self.range_vehiculos for i in self.range_nodos if i!= j) == 1) # and i!=0
                Model.salida.add(sum(Model.x[j,i,k] for k in self.range_vehiculos for i in self.range_nodos if i!= j) == 1) # and i!=0
            else:
                Model.entrada_salida.add(sum(Model.x[i,j,k] for k in self.range_vehiculos for i in self.range_nodos if i!= j) == sum(Model.x[j,i,k] for k in self.range_vehiculos for i in self.range_nodos if i!= j))
             
        Model.unico_repartidor_por_nodos = ConstraintList()
        for k in self.range_vehiculos:
            for j in self.range_nodos:
                Model.unico_repartidor_por_nodos.add(sum(Model.x[i,j,k] for i in self.range_nodos if i!= j) == sum(Model.x[j,i,k]  for i in self.range_nodos if i!= j))

        Model.salida = ConstraintList()
        for k in range_vehiculos:
            for i in range_depots:
                Model.salida.add(sum(Model.x[i,j,k] for j in self.range_nodos if j != 0) == 1)

        Model.MTZ_1 = ConstraintList()
        for k in range_vehiculos:
            for i in range_nodos:
                for j in range_nodos:
                        if i != j and i != 0 and j != 0:
                            Model.MTZ_1.add(Model.u[i,k]-Model.u[j,k]+(self.len_total)*Model.x[i,j,k] <= (self.len_total - 1))

        Model.MTZ_2 = ConstraintList()
        for k in self.range_vehiculos:
            Model.MTZ_2.add(Model.u[0,k] == 1)

        Model.capacidades = ConstraintList()
        Model.rangos = ConstraintList()
        for k in range_vehiculos:
            Model.capacidades.add(sum(Model.x[i,j,k]*self.matriz_demandas[j]) <= self.matriz_costo[1])
            Model.rangos.add(sum(Model.x[i,j,k]*self.matriz_distancias[i, j]) <= self.matriz_costo[2])

        return model
    
    def solve_model(self):
        """
        Solve the model using the given solver.
        """
        solver = pyomo.SolverFactory('highs' )
        results = solver.solve(self.model)
        return results

    def display_results(self):
        """
        Display the results of the optimization.
        """
        self.model.display()
