
using Pkg

Pkg.activate("Scrw_dislocation")
using ACE1
using ACE1.RPI: SparsePSHDegreeM
using IPFitting
using JuLIP
using JuLIP.MLIPs
using PyCall
using ASE


# Generating screw dislocation quadrupole

disl = pyimport("matscipy.dislocation")
a0, C11, C12, C44 = disl.get_elastic_constants("/home/spatel46/w_eam4.fs", delta=1e-3)

bk, disloc_ref, disp=disl.make_screw_quadrupole(a0,left_shift=0,right_shift=0,n1u=5,symbol="W")

atp = ASE.ASEAtoms(disloc_ref)
atmax = ASE.Atoms(atp)

eam = JuLIP.Potentials.EAM("/home/spatel46/w_eam4.fs")

set_calculator!(atmax,eam)
set_pbc!(atmax,(true,true,true))
minimise!(atmax,verbose=2)
#-----------------------------------------------


# Generating training and testing data with random pertubations

function gen_dat(at)
    at_ = deepcopy(at)
    rattle!(at_, 1e-1 * rand())
    return Dat(at_, "disloc"; E = energy(at_), F = forces(at_))
end

train = [ gen_dat(atmax) for _= 1:400 ]
test=[ gen_dat(atmax) for _=1:50]

#---------------------------------

# Generating basis

r0 = rnn(:W)

Dd = Dict(1 => 18,
          2 => 24,
          3 => 14,
          4 => 10,
          5=> 15  )
Dn = Dict("default" => 1.0)
Dl = Dict("default" => 1.5)
Bsel = SparsePSHDegreeM(Dn, Dl, Dd)

basis = rpi_basis(;
      species = :W,
      N = 5,
      maxdeg = 1.0000001,
      D = Bsel,
      r0 = r0,
      rin = 0.65*r0, rcut = 5.5,
      pin = 0)

@show length(basis)

#------------------------------------

# Fitting the ACE

dB = LsqDB("", basis, train)
weights=Dict("Default"=>Dict("E"=>16.0,"F"=>1.0,"V"=>50.0))


solver= Dict(
         "solver" => :ard,
         "ard_tol" => 1e-3,
         "ard_threshold_lambda" => 10000)

# Training

IP, lsqinfo = lsqfit(dB; weights = weights, error_table = true,solver=solver)
rmse_table(lsqinfo["errors"])


# Testing
IPFitting.add_fits_serial!(IP, test)
IPFitting.rmse_table(test)
