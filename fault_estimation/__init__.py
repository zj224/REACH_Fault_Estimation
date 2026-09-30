"""Actuator fault estimation for a soft robotic fish with a sigma-point filter.

Modules
-------
config        parameters, sensor layout, noise settings (``Params``, ``Sensor``)
model         FEM fish model: system matrices, controls, hydrodynamics, simulation
measurements  sensor models (GPS, IMU, bend sensors)
spf           sigma-point filter estimating the state and actuator health
experiments   batches of runs (parallel), scoring, saving/loading results
plotting      one ``plot_<experiment>`` function per experiment
"""
