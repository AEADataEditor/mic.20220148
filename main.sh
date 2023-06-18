#!/bin/bash
# Job name:
#SBATCH --job-name=a3757
#
# Request one node:
#SBATCH --nodes=1
#
# Specify number of tasks for use case (example):
#SBATCH --ntasks-per-node=1
#
#SBATCH --mem=0
#
# Processors per task: here, 8 bc we have Stata-MP/8
#SBATCH --cpus-per-task=32
#
# Email?
#SBATCH --mail-type=ALL
#
set -ev

source venv39/bin/activate
echo $PATH
which python3

#nprocs=$(nproc --all)
nprocs=30
echo "Nprocs = $nprocs"
uptime

python=python3
export JULIA_PROJECT=julia16
#julia="julia -p $nprocs " 
julia=julia
R=R

cd code
#$python ./gen_data_one_step_ahead.py
$julia  ./population_predictions.jl
$python ./ML_population_predictions.py
$julia  ./OSAP_predictions.jl
$python ./ML_osap.py
$julia  ./crosstreat_population_predictions.jl
$julia  ./simulating_data_and_testing_procedure.jl
$julia  ./Distributions_ei_sizeRD.jl
$julia  ./sfem_on_learn.jl
$julia  ./sim_perf.jl
$julia  ./sims_sizeRD_and_avg_C.jl
$julia  ./Read_results.jl
$julia  ./generate_OSAP_table.jl
$R CMD BATCH     ./Descriptives_and_plots.R
