using Pkg

Pkg.activate("gb")

using JuLIP
using ACE1.RPI: SparsePSHDegreeM
using Conda
using PyCall
using ASE
using ACE1
using IPFitting
using LinearAlgebra
using JLD2

# Sigma-3
disl = pyimport("matscipy.dislocation")

a0, C11, C12, C44 = disl.get_elastic_constants("/home/spatel46/w_eam4.fs", delta=1e-3)

lat=pyimport("pymatgen.core")
gb=pyimport("pymatgen.analysis.gb.grain")
converter=pyimport("pymatgen.io.ase")


lattice_101=lat.Lattice.cubic(a0)

structure_101=lat.Structure(lattice_101,["W","W"],[[0,0,0],[0.5,0.5,0.5]])

gb_generate_101=gb.GrainBoundaryGenerator(structure_101)

at_101=gb_generate_101.gb_from_parameters(
    rotation_axis= [1,0,1],
    rotation_angle=70.52877936550931,
    expand_times= 1,
    vacuum_thickness= 2.0,
    ab_shift= [0.0,0.0],
    rm_ratio= 0.0,
    plane=[1,0,0],
    normal=true
)

ase_at_101=converter.AseAtomsAdaptor.get_atoms(at_101)
atp_2_101 = ASE.ASEAtoms(ase_at_101)
atmax_101 = ASE.Atoms(atp_2_101) 

set_calculator!(atmax_101,eam)
set_pbc!(atmax_101,(true,true,true))
minimise!(atmax_101,verbose=2,gtol=1e-4)
