# This script was run using Underworld 2.13 on 144 to 192 CPU's on the NCI Gadi supercomputer
#Written by Youseph Ibrahim

import matplotlib.pyplot as plt
import underworld as uw
from underworld import UWGeodynamics as GEO
import underworld.function as fn
from underworld.UWGeodynamics.surfaceProcesses import SedimentationThreshold
import numpy as np
import scipy
import os.path
from mpi4py import MPI
import argparse
import math

u = GEO.UnitRegistry

#Model solver parameters
GEO.rcParams["initial.nonlinear.tolerance"] = 1e-3
GEO.rcParams["nonlinear.tolerance"] = 5e-4
GEO.rcParams["nonlinear.min.iterations"] = 1
GEO.rcParams["nonlinear.max.iterations"] = 100
GEO.rcParams["CFL"] = 0.1
GEO.rcParams["advection.diffusion.method"] = "SLCN"
GEO.rcParams["shear.heating"] = True
GEO.rcParams["surface.pressure.normalization"] = True  # Make sure the top of the model is approximately 0 Pa
GEO.rcParams["popcontrol.split.threshold"] = 0.95
GEO.rcParams["popcontrol.max.splits"] = 100

#Model Scaling
resolution = (960,320) #750m resolution
half_rate = 18 * u.millimeter / u.year
model_length = 720e3 * u.meter
surfaceTemp = 293.15 * u.degK
baseModelTemp = 1603.15 * u.degK
bodyforce = 3150 * u.kilogram / u.metre**3 * 9.81 * u.meter / u.second**2

KL = model_length
Kt = KL / half_rate
KM = bodyforce * KL**2 * Kt**2
KT = (baseModelTemp - surfaceTemp)

GEO.scaling_coefficients["[length]"] = KL
GEO.scaling_coefficients["[time]"] = Kt
GEO.scaling_coefficients["[mass]"]= KM
GEO.scaling_coefficients["[temperature]"] = KT

#Defining the model bounds
Model = GEO.Model(elementRes=resolution, 
                  minCoord=(0. * u.kilometer, -210. * u.kilometer), 
                  maxCoord=(720. * u.kilometer, 30. * u.kilometer), 
                  gravity=(0.0, -9.81 * u.meter / u.second**2))

#Output directory
Model.outputDir="Inversion_Narrow_Rift"

Model.diffusivity = 9e-7 * u.metre**2 / u.second 
Model.capacity    = 1000. * u.joule / (u.kelvin * u.kilogram)

#Defining materials

#sticky air
air = Model.add_material(name="Air", shape=GEO.shapes.Layer(top=Model.top, bottom=0 * u.kilometer))
air.density = 1. * u.kilogram / u.metre**3
air.diffusivity = 1e-5 * u.metre**2 / u.second
air.capacity = 100. * u.joule / (u.kelvin * u.kilogram)
air.compressibility = 1.e4

#sediments
sediment = Model.add_material(name="Sediment") # This is the sediment that gets deposited below sea level
sediment.density           = GEO.LinearDensity(reference_density=2300. * u.kilogram / u.metre**3,
                                        thermalExpansivity= 2.8e-5 * u.kelvin**-1)
sediment.radiogenicHeatProd   = 0.88e-6 * u.microwatt / u.meter**3

Sediment1 = Model.add_material(name="Sediment Layer 1", shape=GEO.shapes.Layer(top=0. * u.kilometer, bottom=-1. * u.kilometer))
Sediment1.radiogenicHeatProd = 0.88e-6 * u.watt / u.meter**3
Sediment1.density  = GEO.LinearDensity(reference_density  = 2600. * u.kilogram / u.metre**3,
                                        thermalExpansivity= 2.8e-5 * u.kelvin**-1)
Sediment2 = Model.add_material(name="Sediment Layer 2", shape=GEO.shapes.Layer(top=-1. * u.kilometer, bottom=-2. * u.kilometer))
Sediment2.radiogenicHeatProd = 0.88e-6 * u.watt / u.meter**3
Sediment2.density  = GEO.LinearDensity(reference_density  = 2600. * u.kilogram / u.metre**3,
                                        thermalExpansivity= 2.8e-5 * u.kelvin**-1)
Sediment3 = Model.add_material(name="Sediment Layer 3", shape=GEO.shapes.Layer(top=-2. * u.kilometer, bottom=-3. * u.kilometer))
Sediment3.radiogenicHeatProd = 0.88e-6 * u.watt / u.meter**3
Sediment3.density  = GEO.LinearDensity(reference_density  = 2600. * u.kilogram / u.metre**3,
                                        thermalExpansivity= 2.8e-5 * u.kelvin**-1)
Sediment4 = Model.add_material(name="Sediment Layer 4", shape=GEO.shapes.Layer(top=-3. * u.kilometer, bottom=-4. * u.kilometer))
Sediment4.radiogenicHeatProd = 0.88e-6 * u.watt / u.meter**3
Sediment4.density  = GEO.LinearDensity(reference_density  = 2600. * u.kilogram / u.metre**3,
                                        thermalExpansivity= 2.8e-5 * u.kelvin**-1)

Sediment5 = Model.add_material(name="Sediment Layer 5", shape=GEO.shapes.Layer(top=-4. * u.kilometer, bottom=-5. * u.kilometer))
Sediment5.radiogenicHeatProd = 0.88e-6 * u.watt / u.meter**3
Sediment5.density  = GEO.LinearDensity(reference_density  = 2600. * u.kilogram / u.metre**3,
                                        thermalExpansivity= 2.8e-5 * u.kelvin**-1)

Sediment6 = Model.add_material(name="Sediment Layer 6", shape=GEO.shapes.Layer(top=-5. * u.kilometer, bottom=-6. * u.kilometer))
Sediment6.radiogenicHeatProd = 0.88e-6 * u.watt / u.meter**3
Sediment6.density  = GEO.LinearDensity(reference_density  = 2600. * u.kilogram / u.metre**3,
                                        thermalExpansivity= 2.8e-5 * u.kelvin**-1)

#Continental Crust
continentalcrustL3 = Model.add_material(name="Continental Crust Layer3", shape=GEO.shapes.Layer(top=-6. * u.kilometer, bottom=-9. * u.kilometer))
continentalcrustL3.radiogenicHeatProd = 0.88e-6 * u.watt / u.meter**3
continentalcrustL3.density  = GEO.LinearDensity(reference_density  = 2650. * u.kilogram / u.metre**3,
                                        thermalExpansivity= 2.8e-5 * u.kelvin**-1)

continentalcrustL4 = Model.add_material(name="Continental Crust Layer4", shape=GEO.shapes.Layer(top=-9. * u.kilometer, bottom=-12. * u.kilometer))
continentalcrustL4.radiogenicHeatProd = 0.88e-6 * u.watt / u.meter**3
continentalcrustL4.density  = GEO.LinearDensity(reference_density  = 2675. * u.kilogram / u.metre**3,
                                        thermalExpansivity= 2.8e-5 * u.kelvin**-1)

continentalcrustL5 = Model.add_material(name="Continental Crust Layer5", shape=GEO.shapes.Layer(top=-12. * u.kilometer, bottom=-15. * u.kilometer))
continentalcrustL5.radiogenicHeatProd = 0.88e-6 * u.watt / u.meter**3
continentalcrustL5.density  = GEO.LinearDensity(reference_density  = 2700. * u.kilogram / u.metre**3,
                                        thermalExpansivity= 2.8e-5 * u.kelvin**-1)

continentalcrustL6 = Model.add_material(name="Continental Crust Layer6", shape=GEO.shapes.Layer(top=-15. * u.kilometer, bottom=-18. * u.kilometer))
continentalcrustL6.radiogenicHeatProd = 0.88e-6 * u.watt / u.meter**3
continentalcrustL6.density  = GEO.LinearDensity(reference_density  = 2725. * u.kilogram / u.metre**3,
                                        thermalExpansivity= 2.8e-5 * u.kelvin**-1)

continentalcrustL7 = Model.add_material(name="Continental Crust Layer7", shape=GEO.shapes.Layer(top=-18. * u.kilometer, bottom=-21. * u.kilometer))
continentalcrustL7.radiogenicHeatProd = 0.88e-6 * u.watt / u.meter**3
continentalcrustL7.density  = GEO.LinearDensity(reference_density  = 2750. * u.kilogram / u.metre**3,
                                        thermalExpansivity= 2.8e-5 * u.kelvin**-1)

continentalcrustL8 = Model.add_material(name="Continental Crust Layer8", shape=GEO.shapes.Layer(top=-21. * u.kilometer, bottom=-24. * u.kilometer))
continentalcrustL8.radiogenicHeatProd = 0.88e-6 * u.watt / u.meter**3
continentalcrustL8.density  = GEO.LinearDensity(reference_density  = 2775. * u.kilogram / u.metre**3,
                                        thermalExpansivity= 2.8e-5 * u.kelvin**-1)

continentalcrustL9 = Model.add_material(name="Continental Crust Layer9", shape=GEO.shapes.Layer(top=-24. * u.kilometer, bottom=-27. * u.kilometer))
continentalcrustL9.radiogenicHeatProd = 0.88e-6 * u.watt / u.meter**3
continentalcrustL9.density  = GEO.LinearDensity(reference_density  = 2800. * u.kilogram / u.metre**3,
                                        thermalExpansivity= 2.8e-5 * u.kelvin**-1)

continentalcrustL10 = Model.add_material(name="Continental Crust Layer10", shape=GEO.shapes.Layer(top=-27. * u.kilometer, bottom=-36. * u.kilometer))
continentalcrustL10.radiogenicHeatProd = 0.88e-6 * u.watt / u.meter**3
continentalcrustL10.density  = GEO.LinearDensity(reference_density  = 2825. * u.kilogram / u.metre**3,
                                        thermalExpansivity= 2.8e-5 * u.kelvin**-1)

#Mantle
uppermantle = Model.add_material(name="Upper Mantle", shape=GEO.shapes.Layer(top=-36. * u.kilometer, bottom=-110. * u.kilometer))
uppermantle.density = GEO.LinearDensity(reference_density=3370. * u.kilogram / u.metre**3,
                                        thermalExpansivity= 2.8e-5 * u.kelvin**-1)
uppermantle.temperatureLimiter = 1603. * u.kelvin

asthenosphere = Model.add_material(name="Asthenosphere", shape=GEO.shapes.Layer(top=-110. * u.kilometer, bottom=Model.bottom))
asthenosphere.density = GEO.LinearDensity(reference_density=3370. * u.kilogram / u.metre**3,
                                         thermalExpansivity= 2.8e-5 * u.kelvin**-1)
asthenosphere.temperatureLimiter = 1603. * u.kelvin

#Defining material viscosities
rh = GEO.ViscousCreepRegistry()

Model.minViscosity = 1e18 * u.pascal * u.second
Model.maxViscosity = 5e23 * u.pascal * u.second

air.viscosity = 5e18 * u.pascal * u.second
Sediment1.viscosity = 5e19 * u.pascal *u.second
Sediment2.viscosity = 0.01 * rh.Wet_Quartz_Dislocation_Paterson_and_Luan_1990
Sediment3.viscosity = 0.01 * rh.Wet_Quartz_Dislocation_Paterson_and_Luan_1990
Sediment4.viscosity = 0.01 * rh.Wet_Quartz_Dislocation_Paterson_and_Luan_1990
Sediment5.viscosity = 0.01 * rh.Wet_Quartz_Dislocation_Paterson_and_Luan_1990
Sediment6.viscosity = 0.01 * rh.Wet_Quartz_Dislocation_Paterson_and_Luan_1990
continentalcrustL3.viscosity = 0.75 * rh.Wet_Quartz_Dislocation_Paterson_and_Luan_1990
continentalcrustL4.viscosity = 0.25 * rh.Wet_Quartz_Dislocation_Paterson_and_Luan_1990
continentalcrustL5.viscosity = 1 * rh.Wet_Quartz_Dislocation_Paterson_and_Luan_1990
continentalcrustL6.viscosity = 1 * rh.Wet_Quartz_Dislocation_Paterson_and_Luan_1990
continentalcrustL7.viscosity = 2 * rh.Wet_Quartz_Dislocation_Paterson_and_Luan_1990
continentalcrustL8.viscosity = 2 * rh.Wet_Quartz_Dislocation_Paterson_and_Luan_1990
continentalcrustL9.viscosity = 3 * rh.Wet_Quartz_Dislocation_Paterson_and_Luan_1990
continentalcrustL10.viscosity = 3 * rh.Wet_Quartz_Dislocation_Paterson_and_Luan_1990
uppermantle.viscosity = rh.Dry_Olivine_Dislocation_Karato_and_Wu_1993
asthenosphere.viscosity = rh.Dry_Olivine_Dislocation_Karato_and_Wu_1993
sediment.viscosity         = rh.Wet_Quartz_Dislocation_Gleason_and_Tullis_1995

#Defining material plasticity
Sediment1.plasticity = GEO.DruckerPrager(name="Continental Crust",
                                                cohesion=0. * u.megapascal,
                                                cohesionAfterSoftening=0. * u.megapascal,
                                                frictionCoefficient=0.1,
                                                frictionAfterSoftening=0.01,
                                                epsilon1=0.0, epsilon2=0.15)
Sediment1.stressLimiter = 100 * u.megapascal

Sediment2.plasticity = GEO.DruckerPrager(name="Continental Crust",
                                                cohesion=5. * u.megapascal,
                                                cohesionAfterSoftening=1. * u.megapascal,
                                                frictionCoefficient=0.54,
                                                frictionAfterSoftening=0.011,
                                                epsilon1=0.0, epsilon2=0.15)
Sediment2.stressLimiter = 100 * u.megapascal

Sediment3.plasticity = GEO.DruckerPrager(name="Continental Crust",
                                                cohesion=5. * u.megapascal,
                                                cohesionAfterSoftening=1. * u.megapascal,
                                                frictionCoefficient=0.54,
                                                frictionAfterSoftening=0.011,
                                                epsilon1=0.0, epsilon2=0.15)
Sediment3.stressLimiter = 100 * u.megapascal

Sediment4.plasticity = GEO.DruckerPrager(name="Continental Crust",
                                                cohesion=5. * u.megapascal,
                                                cohesionAfterSoftening=1. * u.megapascal,
                                                frictionCoefficient=0.54,
                                                frictionAfterSoftening=0.011,
                                                epsilon1=0.0, epsilon2=0.15)
Sediment4.stressLimiter = 100 * u.megapascal

Sediment5.plasticity = GEO.DruckerPrager(name="Continental Crust",
                                                cohesion=5. * u.megapascal,
                                                cohesionAfterSoftening=1. * u.megapascal,
                                                frictionCoefficient=0.54,
                                                frictionAfterSoftening=0.011,
                                                epsilon1=0.0, epsilon2=0.15)
Sediment5.stressLimiter = 100 * u.megapascal

Sediment6.plasticity = GEO.DruckerPrager(name="Continental Crust",
                                                cohesion=5. * u.megapascal,
                                                cohesionAfterSoftening=1. * u.megapascal,
                                                frictionCoefficient=0.54,
                                                frictionAfterSoftening=0.011,
                                                epsilon1=0.0, epsilon2=0.15)
Sediment6.stressLimiter = 100 * u.megapascal

continentalcrustL3.plasticity = GEO.DruckerPrager(name="Continental Crust",
                                                cohesion=5. * u.megapascal,
                                                cohesionAfterSoftening=1. * u.megapascal,
                                                frictionCoefficient=0.54,
                                                frictionAfterSoftening=0.011,
                                                epsilon1=0.0, epsilon2=0.15)
continentalcrustL3.stressLimiter = 125 * u.megapascal

continentalcrustL4.plasticity = GEO.DruckerPrager(name="Continental Crust",
                                                cohesion=15. * u.megapascal,
                                                cohesionAfterSoftening=1.5 * u.megapascal,
                                                frictionCoefficient=0.54,
                                                frictionAfterSoftening=0.011,
                                                epsilon1=0.0, epsilon2=0.25)
continentalcrustL4.stressLimiter = 150 * u.megapascal

continentalcrustL5.plasticity = GEO.DruckerPrager(name="Continental Crust",
                                                cohesion=15. * u.megapascal,
                                                cohesionAfterSoftening=1.5 * u.megapascal,
                                                frictionCoefficient=0.54,
                                                frictionAfterSoftening=0.011,
                                                epsilon1=0.0, epsilon2=0.25)
continentalcrustL5.stressLimiter = 150 * u.megapascal

continentalcrustL6.plasticity = GEO.DruckerPrager(name="Continental Crust",
                                                cohesion=15. * u.megapascal,
                                                cohesionAfterSoftening=1.5 * u.megapascal,
                                                frictionCoefficient=0.54,
                                                frictionAfterSoftening=0.011,
                                                epsilon1=0.0, epsilon2=0.25)
continentalcrustL6.stressLimiter = 150 * u.megapascal

continentalcrustL7.plasticity = GEO.DruckerPrager(name="Continental Crust",
                                                cohesion=15. * u.megapascal,
                                                cohesionAfterSoftening=1.5 * u.megapascal,
                                                frictionCoefficient=0.54,
                                                frictionAfterSoftening=0.011,
                                                epsilon1=0.0, epsilon2=0.25)
continentalcrustL7.stressLimiter = 150 * u.megapascal

continentalcrustL8.plasticity = GEO.DruckerPrager(name="Continental Crust",
                                                cohesion=15. * u.megapascal,
                                                cohesionAfterSoftening=1.5 * u.megapascal,
                                                frictionCoefficient=0.54,
                                                frictionAfterSoftening=0.011,
                                                epsilon1=0.0, epsilon2=0.25)
continentalcrustL8.stressLimiter = 150 * u.megapascal

continentalcrustL9.plasticity = GEO.DruckerPrager(name="Continental Crust",
                                                cohesion=15. * u.megapascal,
                                                cohesionAfterSoftening=1.5 * u.megapascal,
                                                frictionCoefficient=0.54,
                                                frictionAfterSoftening=0.011,
                                                epsilon1=0.0, epsilon2=0.25)
continentalcrustL9.stressLimiter = 150 * u.megapascal

continentalcrustL10.plasticity = GEO.DruckerPrager(name="Continental Crust",
                                                cohesion=15. * u.megapascal,
                                                cohesionAfterSoftening=1.5 * u.megapascal,
                                                frictionCoefficient=0.54,
                                                frictionAfterSoftening=0.011,
                                                epsilon1=0.0, epsilon2=0.25)
continentalcrustL10.stressLimiter = 150 * u.megapascal

uppermantle.plasticity = GEO.DruckerPrager(name="Continental Crust",
                                           cohesion=15. * u.megapascal,
                                           cohesionAfterSoftening=1.5 * u.megapascal,
                                           frictionCoefficient=0.44,
                                           frictionAfterSoftening=0.011,
                                           epsilon1=0.0, epsilon2=0.15)
uppermantle.stressLimiter = 250 * u.megapascal

asthenosphere.plasticity = GEO.DruckerPrager(name="Continental Crust",
                                          cohesion=15. * u.megapascal,
                                          cohesionAfterSoftening=1.5 * u.megapascal,
                                          frictionCoefficient=0.44,
                                          frictionAfterSoftening=0.011,
                                          epsilon1=0.0, epsilon2=0.15)
asthenosphere.stressLimiter = 250 * u.megapascal

sediment.plasticity = GEO.DruckerPrager(cohesion=1.0 * u.megapascal,
                               cohesionAfterSoftening=0.1 * u.megapascal,
                               frictionCoefficient=0.4,
                               frictionAfterSoftening=0.01,
                               epsilon1=0.05,
                               epsilon2=0.15)
sediment.stressLimiter = 100 * u.megapascal

#Defining solidus and liquidus curves
solidii = GEO.SolidusRegistry()
my_crust_solidus = GEO.Solidus(A1=923 * u.kelvin, A2=-1.2e-07 * u.kelvin / u.pascal, A3=1.2e-16 * u.kelvin / u.pascal**2, A4=0.0 * u.kelvin / u.pascal**3)
mid_crust_solidus = GEO.Solidus(A1=1263 * u.kelvin, A2=-1.2e-07 * u.kelvin / u.pascal, A3=1.2e-16 * u.kelvin / u.pascal**2, A4=0.0 * u.kelvin / u.pascal**3)
mantle_solidus = solidii.Mantle_Solidus

liquidii = GEO.LiquidusRegistry()
my_crust_liquidus = GEO.Liquidus(A1=1423 * u.kelvin, A2=-1.2e-07 * u.kelvin / u.pascal, A3=1.6e-16 * u.kelvin / u.pascal**2, A4=0.0 * u.kelvin / u.pascal**3)
mid_crust_liquidus = GEO.Liquidus(A1=1763 * u.kelvin, A2=-1.2e-07 * u.kelvin / u.pascal, A3=1.6e-16 * u.kelvin / u.pascal**2, A4=0.0 * u.kelvin / u.pascal**3)
mantle_liquidus = liquidii.Mantle_Liquidus

#Defining melt modifiers
continentalcrustL3.add_melt_modifier(my_crust_solidus, my_crust_liquidus, 
                         latentHeatFusion=250.0 * u.kilojoules / u.kilogram / u.kelvin,
                         meltFraction=0.,
                         meltFractionLimit=0.3,
                         meltExpansion=0.13, 
                         viscosityChangeX1 = 0.15,
                         viscosityChangeX2 = 0.30,
                         viscosityChange = 1e-3
                        )

continentalcrustL4.add_melt_modifier(my_crust_solidus, my_crust_liquidus, 
                         latentHeatFusion=250.0 * u.kilojoules / u.kilogram / u.kelvin,
                         meltFraction=0.,
                         meltFractionLimit=0.3,
                         meltExpansion=0.13, 
                         viscosityChangeX1 = 0.15,
                         viscosityChangeX2 = 0.30,
                         viscosityChange = 1e-3
                        )

continentalcrustL5.add_melt_modifier(my_crust_solidus, my_crust_liquidus, 
                         latentHeatFusion=250.0 * u.kilojoules / u.kilogram / u.kelvin,
                         meltFraction=0.,
                         meltFractionLimit=0.3,
                         meltExpansion=0.13, 
                         viscosityChangeX1 = 0.15,
                         viscosityChangeX2 = 0.30,
                         viscosityChange = 1e-3
                        )

continentalcrustL6.add_melt_modifier(my_crust_solidus, my_crust_liquidus, 
                         latentHeatFusion=250.0 * u.kilojoules / u.kilogram / u.kelvin,
                         meltFraction=0.,
                         meltFractionLimit=0.3,
                         meltExpansion=0.13, 
                         viscosityChangeX1 = 0.15,
                         viscosityChangeX2 = 0.30,
                         viscosityChange = 1e-3
                        )


continentalcrustL7.add_melt_modifier(my_crust_solidus, my_crust_liquidus, 
                         latentHeatFusion=250.0 * u.kilojoules / u.kilogram / u.kelvin,
                         meltFraction=0.,
                         meltFractionLimit=0.3,
                         meltExpansion=0.13, 
                         viscosityChangeX1 = 0.15,
                         viscosityChangeX2 = 0.30,
                         viscosityChange = 1e-3
                        )


continentalcrustL8.add_melt_modifier(my_crust_solidus, my_crust_liquidus, 
                         latentHeatFusion=250.0 * u.kilojoules / u.kilogram / u.kelvin,
                         meltFraction=0.,
                         meltFractionLimit=0.3,
                         meltExpansion=0.13, 
                         viscosityChangeX1 = 0.15,
                         viscosityChangeX2 = 0.30,
                         viscosityChange = 1e-3
                        )

continentalcrustL9.add_melt_modifier(my_crust_solidus, my_crust_liquidus,
                         latentHeatFusion=250.0 * u.kilojoules / u.kilogram / u.kelvin,
                         meltFraction=0.,
                         meltFractionLimit=0.3,
                         meltExpansion=0.13,
                         viscosityChangeX1 = 0.15,
                         viscosityChangeX2 = 0.30,
                         viscosityChange = 1e-3
                        )

continentalcrustL10.add_melt_modifier(my_crust_solidus, my_crust_liquidus,
                         latentHeatFusion=250.0 * u.kilojoules / u.kilogram / u.kelvin,
                         meltFraction=0.,
                         meltFractionLimit=0.3,
                         meltExpansion=0.13,
                         viscosityChangeX1 = 0.15,
                         viscosityChangeX2 = 0.30,
                         viscosityChange = 1e-3
                        )

uppermantle.add_melt_modifier(mantle_solidus, mantle_liquidus,
                         latentHeatFusion=250.0 * u.kilojoules / u.kilogram / u.kelvin,
                         meltFraction=0.,
                         meltFractionLimit=0.03,
                         meltExpansion=0.13,
                         viscosityChangeX1 = 0.001,
                         viscosityChangeX2 = 0.03,
                         viscosityChange = 1e-2
                        )

asthenosphere.add_melt_modifier(mantle_solidus, mantle_liquidus,
                         latentHeatFusion=250.0 * u.kilojoules / u.kilogram / u.kelvin,
                         meltFraction=0.,
                         meltFractionLimit=0.03,
                         meltExpansion=0.13,
                         viscosityChangeX1 = 0.001,
                         viscosityChangeX2 = 0.03,
                         viscosityChange = 1e-2
                        )

#Defining temperature conditions
Model.set_temperatureBCs(top=293.15 * u.degK,
                         bottom=1603.15 * u.degK,
                         nodeSets = [(air.shape, 293.15 * u.degK)])

#Defining velocity boundary conditions. -velocity is contraction, +velocity is extension
import underworld.function as fn
velocity = -1.1355 * u.centimeter / u.year

Model.set_velocityBCs(left=[-velocity, 0.0 * u.centimeter / u.year],
                      right=[velocity, 0.0 * u.centimeter / u.year],
                      top=[None, 0.0 * u.centimeter / u.year])

#Airy-like traction basal boundary condition
Model.set_stressBCs(bottom=[0., 6.48174e9 * u.pascal])

#Gaussian damage function
def gaussian(xx, centre, width):
    return ( np.exp( -(xx - centre)**2/width))

maxDamage = 0.25
centre = (GEO.nd(360. * u.kilometer), GEO.nd(-40. * u.kilometer))
width = GEO.nd(75. * u.kilometer)  # this gives a normal distribution

Model.plasticStrain.data[:] = maxDamage * np.random.rand(*Model.plasticStrain.data.shape[:])
Model.plasticStrain.data[:,0] *= gaussian(Model.swarm.particleCoordinates.data[:,0], centre[0], width)
Model.plasticStrain.data[:,0] *= gaussian(Model.swarm.particleCoordinates.data[:,1], centre[1], width*100)

air_mask = Model.swarm.particleCoordinates.data[:,1] > GEO.nd(0 * u.kilometer)

Model.plasticStrain.data[air_mask] = 0.0

#Sediment deposition below 0 m elevation
Model.surfaceProcesses = GEO.surfaceProcesses.SedimentationThreshold(air=[air], sediment=[sediment], threshold=0. * u.metre)

#Tracers for visualisation, not required for running model
npoints_surface = 1000
coords_surface = np.ndarray((npoints_surface, 2))
coords_surface[:, 0] = np.linspace(GEO.nd(Model.minCoord[0]), GEO.nd(Model.maxCoord[0]), npoints_surface)
coords_surface[:, 1] = GEO.nd(0. * u.kilometre)
surface_tracers = Model.add_passive_tracers(name="Surface", vertices=coords_surface)

npoints_moho = 1000
coords_moho = np.ndarray((npoints_moho, 2))
coords_moho[:, 0] = np.linspace(GEO.nd(Model.minCoord[0]), GEO.nd(Model.maxCoord[0]), npoints_moho)
coords_moho[:, 1] = GEO.nd(0. * u.kilometre) - GEO.nd(continentalcrustL10.bottom)
moho_tracers = Model.add_passive_tracers(name="Moho", vertices=coords_moho)

coords_FSE_Crust = GEO.circles_grid(radius = 1.5 * u.kilometer,
                           minCoord=[Model.minCoord[0], continentalcrustL10.bottom],
                           maxCoord=[Model.maxCoord[0], 0.*u.kilometer])

FSE_Crust = Model.add_passive_tracers(name="FSE_Crust", vertices=coords_FSE_Crust)


coords_FSE_Mantle = GEO.circles_grid(radius=1.5 * u.kilometer, 
                    minCoord=[Model.minCoord[0], -110.*u.kilometer], 
                    maxCoord=[Model.maxCoord[0], continentalcrustL10.bottom])

FSE_Mantle = Model.add_passive_tracers(name="FSE_Mantle", vertices=coords_FSE_Mantle)


#Initialise the model
Model.swarm.allow_parallel_nn = True
Model.init_model()

#Solver options
solver = Model.solver

# Decide whether to use mumps or multigrid
if resolution[0] * resolution[1] < 1e6:
    print("Using mumps")
    solver.set_inner_method("mumps")
else:
    print("Using multigrid with coarse mumps")
    solver.options.A11.mg_coarse_pc_factor_mat_solver_package = "mumps"
    solver.options.A11.mg_coarse_pc_type = "lu"
    solver.options.A11.mg_coarse_ksp_type = "preonly"
    
solver.options.A11.ksp_rtol=1e-8
solver.options.A11.ksp_set_min_it_converge = 10
solver.options.A11.use_previous_guess = True
solver.options.scr.ksp_rtol=1e-6
solver.options.scr.use_previous_guess = True
solver.options.scr.ksp_set_min_it_converge = 10
solver.options.scr.ksp_type = "cg"
solver.options.main.remove_constant_pressure_null_space=True
solver.set_penalty(1e5)

def post_hook():  
    coords = fn.input()
    zz = (coords[0] - GEO.nd(Model.minCoord[0])) / (GEO.nd(Model.maxCoord[0]) - GEO.nd(Model.minCoord[0]))
    fact = fn.math.pow(fn.math.tanh(zz*20.0) + fn.math.tanh((1.0-zz)*20.0) - fn.math.tanh(20.0), 4)
    Model.plasticStrain.data[:] = Model.plasticStrain.data[:] * fact.evaluate(Model.swarm)

Model.run_for(8010000 * u.years, checkpoint_interval=100000. * u.year, restartStep=-1, restartDir="Inversion_Narrow_Rift")
