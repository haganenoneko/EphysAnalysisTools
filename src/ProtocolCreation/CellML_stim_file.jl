"""
Run existing CellML model and save output membrane potential Vm
"""

using CellMLToolkit, DifferentialEquations, OrdinaryDiffEq, Sundials 
using PyPlot
PyPlot.pygui(true) 

function LoadAndRun(model_name::String; show_params=false, show_states=false)
    """
    Load and run model with model name `model_name`
    """
    # path to cellml models 
    cellml_path = ".//data//CellML_models//$model_name"

    # CellModel factory function creates an ODESystem 
    ml = CellMLToolkit.CellModel(cellml_path; dependency=false)
    
    # model parameters 
    if show_params 
        params = list_params(ml)
        @info params 
    end 

    # model states 
    states = list_states(ml)
    if show_states 
        @info states        
    end 
    
    # find index of membrane voltage in list of state variables 
    idx = findfirst([occursin("V", string(s)) for s in states])
    @info "Index of voltage variable:", idx

    # generate and solve ODEProblem 
    tspan = (0., 0.4)
    ts = collect(0:1e-5:tspan[2])
    prob = CellMLToolkit.ODEProblem(ml, tspan);
    sol = solve(prob, TRBDF2(), saveat=ts)
    # extract membrane voltage 
    Vm = sol[idx, :] 

    # plot 
    f, ax = plt.subplots(figsize=(10, 5))
    ax.set_ylabel("Voltage (mV)")
    ax.set_xlabel("Time (ms)")
    ax.plot(sol.x, Vm, lw=2)
    f.tight_layout()
    PyPlot.show()
end 

model_name = "severi_fantini_charawi_difrancesco_2012.cellml"
LoadAndRun(model_name; show_params=true, show_states=true)