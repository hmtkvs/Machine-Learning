
import numpy as np
from node import Node

battery = Node("battery")
battery.cardinality = 2
battery.priors = np.array([0.02, 0.98]) #  no=0 yes=1

fuel = Node("fuel")
fuel.cardinality = 2
fuel.priors = np.array([0.05, 0.95]) #  no=0 yes=1

m = np.zeros((2, 2, 2)) 
m[1, 1, 1] = 0.96
m[1, 1, 0] = 0.04
m[0, 1, 1] = 0.9
m[0, 1, 0] = 0.1
m[1, 0, 1] = 0.03
m[1, 0, 0] = 0.97
m[0, 0, 1] = 0.01
m[0, 0, 0] = 0.99
gauge = Node("gauge")
gauge.cardinality = 2
gauge.m = m
gauge.likelihood = np.array([1, 1])

m = np.ones((2, 2)) 

turn = Node("turn over")
turn.cardinality = 2
turn.m = m
turn.likelihood = np.array([1, 1])

gauge.add_parent(battery)
gauge.add_parent(fuel)
turn.add_parent(battery)


gauge.message_to_parent(battery)
gauge.message_to_parent(fuel)

battery.message_to_child(gauge)


gauge.get_priors()
turn.get_priors()

# Gauge is empty

gauge.likelihood = np.array([0, 1])
gauge.message_to_parent(battery)
gauge.message_to_parent(fuel)

print("Fuel belief    (Fuel=True)    :",fuel.get_belief())
print("Battery belief (Battery=True) :",battery.get_belief())

# Gauge is notempty

gauge.likelihood = np.array([1, 0])
gauge.message_to_parent(battery)
gauge.message_to_parent(fuel)

print("Fuel belief    (Fuel=False)   :",fuel.get_belief())
print("Battery belief (Battery=False):",battery.get_belief())

