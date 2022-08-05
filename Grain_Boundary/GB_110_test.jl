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

#---------------------------------------------
disl = pyimport("matscipy.dislocation")

a0, C11, C12, C44 = disl.get_elastic_constants("/home/spatel46/w_eam4.fs", delta=1e-3)

lat=pyimport("pymatgen.core")
gb=pyimport("pymatgen.analysis.gb.grain")
converter=pyimport("pymatgen.io.ase")


#Rotation axis-110
lattice=lat.Lattice.cubic(a0)

structure=lat.Structure(lattice,["W","W"],[[0,0,0],[0.5,0.5,0.5]])

gb_generate=gb.GrainBoundaryGenerator(structure)


at=gb_generate.gb_from_parameters(
    rotation_axis= [1,1,0],
    rotation_angle=129.5211963586422,
    expand_times= 1,
    vacuum_thickness= 2.0,
    ab_shift= [0.0,0.0],
    rm_ratio= 0.0,
    plane=[1,0,0],
    normal=true
)


ase_at=converter.AseAtomsAdaptor.get_atoms(at)
atp_2 = ASE.ASEAtoms(ase_at)
atmax = ASE.Atoms(atp_2) 


eam = JuLIP.Potentials.EAM("/home/spatel46/w_eam4.fs")

set_pbc!(atmax,(true,true,true))
set_calculator!(atmax,eam)
minimise!(atmax,verbose=2)


