import matplotlib.pyplot as plt
from pyomo.environ import ConcreteModel, Var, ConstraintList, Objective, minimize, NonNegativeReals, Binary, SolverFactory
import numpy as np
import pandas as pd


class DistributionModel:
    def __init__(self, depots_file, clients_file, vehicles_file, costs_file):
        # Cargar datos
        self.depots = pd.read_csv(depots_file)
        self.clients = pd.read_csv(clients_file)
        self.vehicles = pd.read_csv(vehicles_file)
        self.costs = pd.read_csv(costs_file)

        # Validar columnas
        print("Depots Columns:", self.depots.columns)
        print("Clients Columns:", self.clients.columns)
        print("Vehicles Columns:", self.vehicles.columns)
        print("Costs Columns:", self.costs.columns)

        self.num_depots = len(self.depots)
        self.num_clients = len(self.clients)
        self.num_nodes = self.num_depots + self.num_clients
        self.num_vehicles = len(self.vehicles)
        print("datos cargados")

        # Generar matrices
        self.distance_matrix = self.generate_distance_matrix()
        self.cost_matrix = self.generate_cost_matrix()
        print("matrices generadas")

        # Modelo Pyomo
        self.model = ConcreteModel()

    def generate_distance_matrix(self):
        # Verificar datos de longitud y latitud
        print("Verificando datos de depósitos y clientes...")
        if self.depots[['Longitude', 'Latitude']].isnull().any().any():
            raise ValueError("Existen valores nulos en las coordenadas de los depósitos.")
        if self.clients[['Longitude', 'Latitude']].isnull().any().any():
            raise ValueError("Existen valores nulos en las coordenadas de los clientes.")

        # Asegurarse de que los valores sean numéricos
        self.depots[['Longitude', 'Latitude']] = self.depots[['Longitude', 'Latitude']].astype(float)
        self.clients[['Longitude', 'Latitude']] = self.clients[['Longitude', 'Latitude']].astype(float)

        # Combinar coordenadas de depósitos y clientes
        print("Calculando matriz de distancias...")
        coords = np.vstack([
            self.depots[['Longitude', 'Latitude']].values,
            self.clients[['Longitude', 'Latitude']].values
        ])

        # Calcular distancias Euclidianas
        distance_matrix = np.sqrt(((coords[:, None, :] - coords[None, :, :])**2).sum(axis=2))
        print("Matriz de distancias calculada correctamente.")
        return distance_matrix


    def generate_cost_matrix(self):
        # Imprimir columnas para depuración
        print("Columns in Vehicles:", self.vehicles.columns)
        print("Columns in Costs:", self.costs.columns)

        # Validar que las columnas necesarias existan
        if 'VehicleType' not in self.vehicles.columns:
            raise KeyError("La columna 'VehicleType' no existe en los datos de vehículos.")
        if 'Vehicle' not in self.costs.columns:
            raise KeyError("La columna 'Vehicle' no existe en los datos de costos.")

        # Inicializar la matriz de costos
        cost_matrix = np.full((self.num_vehicles, self.num_nodes), np.inf)

        # Asignar costos basados en las columnas correctas
        for i, vehicle in self.vehicles.iterrows():
            for j, cost in self.costs.iterrows():
                if vehicle['VehicleType'] == cost['Vehicle']:  # Comparar las columnas correctas
                    # Validar el tamaño del arreglo
                    cost_values = cost.values[1:]  # Excluir la primera columna 'Vehicle'
                    if len(cost_values) < self.num_nodes:
                        # Ajustar si los costos son menores que los nodos
                        cost_values = np.pad(cost_values, (0, self.num_nodes - len(cost_values)), constant_values=np.inf)
                    elif len(cost_values) > self.num_nodes:
                        # Recortar si los costos son mayores que los nodos
                        cost_values = cost_values[:self.num_nodes]
                    
                    cost_matrix[i, :] = cost_values  # Asignar los costos ajustados

        return cost_matrix

    
    def build_model(self):
        m = self.model

        # Variables
        m.x = Var(range(self.num_nodes), range(self.num_nodes), range(self.num_vehicles), domain=Binary)
        m.u = Var(range(self.num_nodes), range(self.num_vehicles), domain=NonNegativeReals)

        # Función objetivo
        def obj_expression(model):
            # Costos de distancia
            cost_distance = sum(
                model.x[i, j, k] * self.distance_matrix[i, j] * self.cost_matrix[k, 1]
                for i in range(self.num_nodes)
                for j in range(self.num_nodes)
                for k in range(self.num_vehicles)
            )

            # Costos de tiempo de viaje
            cost_travel_time = sum(
                model.x[i, j, k] * self.distance_matrix[i, j] / self.cost_matrix[k, 8] * self.cost_matrix[k, 4]
                for i in range(self.num_nodes)
                for j in range(self.num_nodes)
                for k in range(self.num_vehicles)
            )

            # Costos de mantenimiento
            cost_maintenance = sum(
                self.cost_matrix[k, 5] * sum(model.x[i, j, k] for i in range(self.num_nodes) for j in range(self.num_nodes))
                for k in range(self.num_vehicles)
            )

            # Costos de recarga/reabastecimiento
            cost_recharge = sum(
                model.x[i, j, k] * self.distance_matrix[i, j] * self.cost_matrix[k, 6]
                for i in range(self.num_nodes)
                for j in range(self.num_nodes)
                for k in range(self.num_vehicles)
            )

            return cost_distance + cost_travel_time + cost_maintenance + cost_recharge

        m.objective = Objective(rule=obj_expression, sense=minimize)

        # Restricciones de entrada y salida
        m.entry_constraints = ConstraintList()
        m.exit_constraints = ConstraintList()
        m.entry_exit_constraints = ConstraintList()

        for j in range(self.num_nodes):
            if j >= self.num_depots:  # Solo para clientes
                m.entry_constraints.add(sum(m.x[i, j, k] for i in range(self.num_nodes) for k in range(self.num_vehicles) if i != j) == 1)
                m.exit_constraints.add(sum(m.x[j, i, k] for i in range(self.num_nodes) for k in range(self.num_vehicles) if i != j) == 1)
            else:  # Depots
                m.entry_exit_constraints.add(
                    sum(m.x[i, j, k] for i in range(self.num_nodes) for k in range(self.num_vehicles) if i != j) ==
                    sum(m.x[j, i, k] for i in range(self.num_nodes) for k in range(self.num_vehicles) if i != j)
                )

        # Restricción de capacidad de vehículos
        m.capacity_constraints = ConstraintList()
        for k in range(self.num_vehicles):
            for j in range(self.num_nodes):
                if j >= self.num_depots:
                    m.capacity_constraints.add(
                        sum(m.x[i, j, k] * self.demands[j] for i in range(self.num_nodes) if i != j) <= self.vehicles['Capacity'].iloc[k]
                    )

        # Restricción de subtours (MTZ)
        m.mtz_constraints = ConstraintList()
        for k in range(self.num_vehicles):
            for i in range(self.num_nodes):
                for j in range(self.num_nodes):
                    if i != j and i >= self.num_depots and j >= self.num_depots:
                        m.mtz_constraints.add(m.u[i, k] - m.u[j, k] + self.num_nodes * m.x[i, j, k] <= self.num_nodes - 1)

        # Inicialización de la variable MTZ para depósitos
        for k in range(self.num_vehicles):
            m.mtz_constraints.add(m.u[0, k] == 0)

        return m

    def solve(self):
        solver = SolverFactory('highs')
        results = solver.solve(self.model)
        return results

    def plot_results(self):
        coords = np.vstack([
            self.depots[['Longitude', 'Latitude']].values,
            self.clients[['Longitude', 'Latitude']].values
        ])
        
        fig, ax = plt.subplots()
        # Ploteo de depósitos
        ax.scatter(self.depots['Longitude'], self.depots['Latitude'], c='red', label='Depots', s=100)
        # Ploteo de clientes
        ax.scatter(self.clients['Longitude'], self.clients['Latitude'], c='blue', label='Clients', s=50)

        # Recuperar rutas
        for k in range(self.num_vehicles):
            for i in range(self.num_nodes):
                for j in range(self.num_nodes):
                    if i != j and self.model.x[i, j, k]() > 0.5:  # Si la ruta está activa
                        ax.plot(
                            [coords[i, 0], coords[j, 0]],
                            [coords[i, 1], coords[j, 1]],
                            label=f'Vehicle {k+1}' if k == 0 else '',
                            alpha=0.7
                        )
        ax.legend()
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('Optimized Routes')
        plt.grid(True)
        plt.show()

    def export_results_to_csv(self, output_file):
        results = []
        for k in range(self.num_vehicles):
            for i in range(self.num_nodes):
                for j in range(self.num_nodes):
                    if i != j and self.model.x[i, j, k]() > 0.5:
                        results.append({'Vehicle': k+1, 'From': i, 'To': j})
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)
        print(f"Results exported to {output_file}")


# Ejecución
model = DistributionModel(
    r'data\multi_depots.csv', 
    r'data\clients.csv', 
    r'data\multi_vehicles.csv', 
    r'data\data_vehiculos.csv'
)
model.build_model()
results = model.solve()

# Graficar resultados
model.plot_results()

# Exportar resultados
model.export_results_to_csv('optimized_routes.csv')
