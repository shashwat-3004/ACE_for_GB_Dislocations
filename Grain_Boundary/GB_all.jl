using Pkg

Pkg.activate("gb")

#Pkg.add("Conda")
#Pkg.add("JuLIP")
#Pkg.add("ACE1")
#Pkg.add("IPFitting")
#Pkg.add("ASE")
#Pkg.add("PyCall")
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


# sigma-11
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


function gen_dat_110(at)
    at_ = deepcopy(at)
    rattle!(at_, 1e-1 * rand())
    return Dat(at_, "gb_110"; E = energy(at_), F = forces(at_))
end

train_110 = [ gen_dat_110(atmax) for _= 1:150 ]  # How many configuratiosn(training data)
test_110=[ gen_dat_110(atmax) for _=1:15]
#------------------------------------------------------

r0 = rnn(:W)

Dd = Dict(1 => 12,
          2 => 14,
          3 => 15,
          4 => 20,
          5=> 21  )
Dn = Dict("default" => 1.0)
Dl = Dict("default" => 1.5)
Bsel = SparsePSHDegreeM(Dn, Dl, Dd)

basis = rpi_basis(;
      species = :W,
      N = 4,                        
      maxdeg = 1.0000001,                  
      D = Bsel,
      r0 = r0,                      
      rin = 0.65*r0, rcut = 5.5,    
      pin = 0)

@show length(basis)

#--------------------------------------------------

dB = LsqDB("", basis, combined_train_210)
weights=Dict("Default"=>Dict("E"=>15.0,"F"=>1.0,"V"=>55.0))


solver= Dict(
         "solver" => :ard,
         "ard_tol" => 1e-3,
         "ard_threshold_lambda" => 10000)


IP, lsqinfo = lsqfit(dB; weights = weights, error_table = true,solver=solver)

rmse_table(lsqinfo["errors"])
         
#----------------------------------------------------------
         
         
IPFitting.add_fits_serial!(IP, combined_test_210)
IPFitting.rmse_table(combined_test_210)

#------------------------------------------------------
#Adding Rotation axis [1,1,1]

lattice_111=lat.Lattice.cubic(a0)

structure_111=lat.Structure(lattice_111,["W","W"],[[0,0,0],[0.5,0.5,0.5]])

gb_generate_111=gb.GrainBoundaryGenerator(structure_111)

at_111=gb_generate_111.gb_from_parameters(
    rotation_axis= [1,1,1],
    rotation_angle=38.21321070173819,
    expand_times= 1,
    vacuum_thickness= 2.0,
    ab_shift= [0.0,0.0],
    rm_ratio= 0.0,
    plane=[1,0,0],
    normal=true
)


ase_at_111=converter.AseAtomsAdaptor.get_atoms(at_111)
atp_2_111 = ASE.ASEAtoms(ase_at_111)
atmax_111 = ASE.Atoms(atp_2_111) 


set_pbc!(atmax_111,(true,true,true))
set_calculator!(atmax_111,eam)
minimise!(atmax_111,verbose=2)


function gen_dat_111(at)
    at_ = deepcopy(at)
    rattle!(at_, 1e-1 * rand())
    return Dat(at_, "gb_111"; E = energy(at_), F = forces(at_))
end


train_111 = [ gen_dat_111(atmax_111) for _= 1:150 ]  # How many configuratiosn(training data)
test_111=[ gen_dat_111(atmax_111) for _=1:15]

#--------------------------------------------------
# Rotation axis 100

lattice_100=lat.Lattice.cubic(a0)

structure_100=lat.Structure(lattice_100,["W","W"],[[0,0,0],[0.5,0.5,0.5]])

gb_generate_100=gb.GrainBoundaryGenerator(structure_100)

at_100=gb_generate_100.gb_from_parameters(
    rotation_axis= [1,0,0],
    rotation_angle=126.86989764584402,
    expand_times= 3,
    vacuum_thickness= 2.0,
    ab_shift= [0.0,0.0],
    rm_ratio= 0.0,
    plane=[1,0,0],
    normal=true
)


ase_at_100=converter.AseAtomsAdaptor.get_atoms(at_100)
atp_2_100 = ASE.ASEAtoms(ase_at_100)
atmax_100 = ASE.Atoms(atp_2_100) 


set_calculator!(atmax_100,eam)
set_pbc!(atmax_100,(true,true,true))
minimise!(atmax_100,verbose=2)


function gen_dat_100(at)
    at_ = deepcopy(at)
    rattle!(at_, 1e-1 * rand())
    return Dat(at_, "gb_100"; E = energy(at_), F = forces(at_))
end


train_100 = [ gen_dat_100(atmax_100) for _= 1:150 ]  # How many configuratiosn(training data)
test_100=[ gen_dat_100(atmax_100) for _=1:15]






#--------------------------------------------------
# Joining the trainig and test data
combined_train=cat(combined_train,train_100,dims=1)
combined_test=cat(combined_test,test_100,dims=1)

@save "training_data.jld2" combined_train 

@load "training_data.jld2" combined_train


@save "test_data.jld2" combined_test 

@load "test_data.jld2" combined_test


#---------------------------------------------------
save_dict("grain_boundary.json", Dict("IP" => write_dict(IP), "info" => lsqinfo))


V_gb = read_dict(load_dict("grain_boundary.json")["IP"]);

#------------------------------------

ase_old=converter.AseAtomsAdaptor.get_atoms(at)
atp_old = ASE.ASEAtoms(ase_old)
atmax_old = ASE.Atoms(atp_old) 

Xat = positions(atmax_old)

Xatnew = positions(atmax)

u = Xatnew - Xat
u2 = norm.(u)

plt=pyimport("matplotlib.pyplot")

x, y, z = xyz(atmax_old)
p = plt.scatter(x, y, c=u2)

plt.savefig("/home/spatel46/gb_u.png")
#----------------------------------------------
# Same case

lattice_sa=lat.Lattice.cubic(a0)

structure_sa=lat.Structure(lattice_sa,["W","W"],[[0,0,0],[0.5,0.5,0.5]])

gb_generate_sa=gb.GrainBoundaryGenerator(structure_sa)

at_sa=gb_generate_sa.gb_from_parameters(
    rotation_axis= [1,1,0],
    rotation_angle=50.478803641357835,
    expand_times= 1,
    vacuum_thickness= 2.0,
    ab_shift= [0.0,0.0],
    rm_ratio= 0.0,
    plane=[1,0,0],
    normal=true
)


ase_at_sa=converter.AseAtomsAdaptor.get_atoms(at_sa)
atp_2_sa = ASE.ASEAtoms(ase_at_sa)
atmax_sa = ASE.Atoms(atp_2_sa) 

set_calculator!(atmax_sa,eam)
set_pbc!(atmax_sa,(true,true,true))
minimise!(atmax_sa,verbose=2)
set_calculator!(atmax_sa,V_gb)
minimise!(atmax_sa,verbose=2)



eam = JuLIP.Potentials.EAM("/home/spatel46/w_eam4.fs")


#----------------------------------------------
#Slightly Diffferent

lattice_sd=lat.Lattice.cubic(a0)

structure_sd=lat.Structure(lattice_sd,["W","W"],[[0,0,0],[0.5,0.5,0.5]])

gb_generate_sd=gb.GrainBoundaryGenerator(structure_sd)

at_sd=gb_generate_sd.gb_from_parameters(
    rotation_axis= [1,1,0],
    rotation_angle=50.478803641357835,
    expand_times= 1,
    vacuum_thickness= 2.0,
    ab_shift= [0.0,0.0],
    rm_ratio= 0.0,
    plane=[1,0,0],
    normal=true
)


ase_at_sd=converter.AseAtomsAdaptor.get_atoms(at_sd)
atp_2_sd = ASE.ASEAtoms(ase_at_sd)
atmax_sd = ASE.Atoms(atp_2_sd) 

set_calculator!(atmax_sd,eam)
set_pbc!(atmax_sd,(true,true,true))
minimise!(atmax_sd,verbose=2)
set_calculator!(atmax_sd,V_gb)
minimise!(atmax_sd,verbose=2)



eam = JuLIP.Potentials.EAM("/home/spatel46/w_eam4.fs")

#-------------------------------------
#Very Different

lattice_vd=lat.Lattice.cubic(a0)

structure_vd=lat.Structure(lattice_vd,["W","W"],[[0,0,0],[0.5,0.5,0.5]])

gb_generate_vd=gb.GrainBoundaryGenerator(structure_vd)

at_vd=gb_generate_vd.gb_from_parameters(
    rotation_axis= [1,1,1],
    rotation_angle=38.21321070173819,
    expand_times= 3,
    vacuum_thickness= 2.0,
    ab_shift= [0.0,0.0],
    rm_ratio= 0.0,
    plane=[1,0,0],
    normal=true
)


ase_at_vd=converter.AseAtomsAdaptor.get_atoms(at_vd)
atp_2_vd = ASE.ASEAtoms(ase_at_vd)
atmax_vd = ASE.Atoms(atp_2_vd) 

set_calculator!(atmax_vd,eam)
set_pbc!(atmax_vd,(true,true,true))
minimise!(atmax_vd,verbose=2,gtol=1e-5)
set_calculator!(atmax_vd,V_gb)
minimise!(atmax_vd,verbose=2,gtol=1e-4)

# Large 100

lattice_100_l=lat.Lattice.cubic(a0)

structure_100_l=lat.Structure(lattice_100_l,["W","W"],[[0,0,0],[0.5,0.5,0.5]])

gb_generate_100_l=gb.GrainBoundaryGenerator(structure_100_l)

at_100_l=gb_generate_100_l.gb_from_parameters(
    rotation_axis= [1,0,0],
    rotation_angle=36.86989764584402,
    expand_times= 12,
    vacuum_thickness= 2.0,
    ab_shift= [0.0,0.0],
    rm_ratio= 0.0,
    plane=[1,0,0],
    normal=true
)

ase_at_100_l=converter.AseAtomsAdaptor.get_atoms(at_100_l)
atp_2_100_l = ASE.ASEAtoms(ase_at_100_l)
atmax_100_l = ASE.Atoms(atp_2_100_l) 

set_calculator!(atmax_100_l,eam)
set_pbc!(atmax_100_l,(true,true,true))
minimise!(atmax_100_l,verbose=2,gtol=1e-4)
set_calculator!(atmax_100_l,V_gb)
minimise!(atmax_100_l,verbose=2,gtol=1e-5)

# 100 not required to add

# Check 101 Sigma-3



lattice_101=lat.Lattice.cubic(a0)

structure_101=lat.Structure(lattice_101,["W","W"],[[0,0,0],[0.5,0.5,0.5]])

gb_generate_101=gb.GrainBoundaryGenerator(structure_101)

at_101=gb_generate_101.gb_from_parameters(
    rotation_axis= [1,0,1],
    rotation_angle=70.52877936550931,
    expand_times= 4,
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
set_calculator!(atmax_101,V_gb)
minimise!(atmax_101,verbose=2,gtol=1e-5)

######
# 210 rotation_axis (15 sigma)

lattice_210=lat.Lattice.cubic(a0)

structure_210=lat.Structure(lattice_210,["W","W"],[[0,0,0],[0.5,0.5,0.5]])

gb_generate_210=gb.GrainBoundaryGenerator(structure_210)

at_210=gb_generate_210.gb_from_parameters(
    rotation_axis= [2,1,0],
    rotation_angle=48.18968510422141,
    expand_times= 1,
    vacuum_thickness= 2.0,
    ab_shift= [0.0,0.0],
    rm_ratio= 0.0,
    plane=[1,0,0],
    normal=true
)

ase_at_210=converter.AseAtomsAdaptor.get_atoms(at_210)
atp_2_210 = ASE.ASEAtoms(ase_at_210)
atmax_210 = ASE.Atoms(atp_2_210) 

set_calculator!(atmax_210,eam)
set_pbc!(atmax_210,(true,true,true))
minimise!(atmax_210,verbose=2,gtol=1e-5)


function gen_dat_210(at)
    at_ = deepcopy(at)
    rattle!(at_, 1e-1 * rand())
    return Dat(at_, "gb_210"; E = energy(at_), F = forces(at_))
end

train_210 = [ gen_dat_210(atmax_210) for _= 1:150 ]  # How many configuratiosn(training data)
test_210=[ gen_dat_210(atmax_210) for _=1:15]


combined_train_210=cat(combined_train,train_210,dims=1)
combined_test_210=cat(combined_test,test_210,dims=1)


@save "training_data_210.jld2" combined_train_210 

@load "training_data_210.jld2" combined_train_210


@save "test_data_210.jld2" combined_test_210 

@load "test_data_210.jld2" combined_test_210

@save "train_210.jld2" train_210
@save "test_210.jld2" test_210 


save_dict("grain_boundary_210.json", Dict("IP" => write_dict(IP), "info" => lsqinfo))


V_gb_210 = read_dict(load_dict("grain_boundary_210.json")["IP"]);




#--------------------
# Sigma 7 210-axis


lattice_210=lat.Lattice.cubic(a0)

structure_210=lat.Structure(lattice_210,["W","W"],[[0,0,0],[0.5,0.5,0.5]])

gb_generate_210=gb.GrainBoundaryGenerator(structure_210)

at_210=gb_generate_210.gb_from_parameters(
    rotation_axis= [2,1,0],
    rotation_angle=73.39845040097977,
    expand_times= 1,
    vacuum_thickness= 2.0,
    ab_shift= [0.0,0.0],
    rm_ratio= 0.0,
    plane=[1,0,0],
    normal=true
)

ase_at_210=converter.AseAtomsAdaptor.get_atoms(at_210)
atp_2_210 = ASE.ASEAtoms(ase_at_210)
atmax_210 = ASE.Atoms(atp_2_210) 

set_calculator!(atmax_210,eam)
set_pbc!(atmax_210,(true,true,true))
minimise!(atmax_210,verbose=2,gtol=1e-3)
set_calculator!(atmax_210,V_gb)
minimise!(atmax_210,verbose=2,gtol=1e-4)



##
#210 15 sigma (large)


lattice_210_l=lat.Lattice.cubic(a0)

structure_210_l=lat.Structure(lattice_210_l,["W","W"],[[0,0,0],[0.5,0.5,0.5]])

gb_generate_210_l=gb.GrainBoundaryGenerator(structure_210_l)

at_210_l=gb_generate_210_l.gb_from_parameters(
    rotation_axis= [2,1,0],
    rotation_angle=48.18968510422141,
    expand_times= 2,
    vacuum_thickness= 2.0,
    ab_shift= [0.0,0.0],
    rm_ratio= 0.0,
    plane=[1,0,0],
    normal=true
)

ase_at_210_l=converter.AseAtomsAdaptor.get_atoms(at_210_l)
atp_2_210_l = ASE.ASEAtoms(ase_at_210_l)
atmax_210_l = ASE.Atoms(atp_2_210_l) 

set_calculator!(atmax_210_l,eam)
set_pbc!(atmax_210_l,(true,true,true))
minimise!(atmax_210_l,verbose=2,gtol=1e-4)
set_calculator!(atmax_210_l,V_gb_210)
minimise!(atmax_210_l,verbose=2,gtol=1e-5)
