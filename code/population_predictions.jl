using DataFrames
using DataFramesMeta
using CSV
using StatsBase
using Statistics
using SplitApplyCombine
using Missings
using BenchmarkTools
using Distributions
using Distributed
using Optim
using Random
using BlackBoxOptim
using JSON
using GLM
using Plots
nunique(xs) = length(unique(xs))
#%%

#%% added
using Pkg
Pkg.instantiate()

n_cores = 30
nprocs() == 1 && addprocs(n_cores -1)


#%%

osap_df =  CSV.File("../clean_data/one_step_ahead_data.csv") |> DataFrame

osap_df[!, :last_half] = (osap_df.inter_from_last./osap_df.max_inter .< 0.5).*1
osap_df[!, :first_half] = (osap_df.inter_from_last./osap_df.max_inter .>= 0.5).*1
osap_df[!,:y] = (osap_df[!,:c] .+ 1)./2


by_rd_df = combine(groupby(osap_df, :treatment), :y => mean, :paper => first, :session=>first, :sizeRD => first, :sizeRD => mean)

to_predict_df = @based_on(groupby(osap_df, [:session, :inter, :round]), y = mean(:y), n = nunique(:id), δ=first(:delta), sizeRD = first(:sizeRD), g = first(:g), l = first(:l), fold = first(:r_fold_1), treatment=first(:treatment), paper_treat=first(:paper_treat), last_third=first(:last_third), initial=first(:initial), last_half=first(:last_half), first_half=first(:first_half), prev_inter_diff_len=first(:prev_inter_diff_len), sgpe=first(:sgpe), sizeSPE=first(:sizeSPE), r_fold_1=first(:r_fold_1), r_fold_2=first(:r_fold_2), r_fold_3=first(:r_fold_3), r_fold_4=first(:r_fold_4), r_fold_5=first(:r_fold_5), r_fold_6=first(:r_fold_6), r_fold_7=first(:r_fold_7), r_fold_8=first(:r_fold_8), r_fold_9=first(:r_fold_9), r_fold_10=first(:r_fold_10), diff_len=first(:diff_len), diff_len_share=first(:diff_len_share), cum_diff_len=first(:cum_diff_len), cum_diff_len_share=first(:cum_diff_len_share), first_third_diff_len=first(:first_third_diff_len), first_third_diff_len_share=first(:first_third_diff_len_share))


to_predict_df = @transform(groupby(to_predict_df, :session), max_inter = maximum(:inter))
to_predict_df = @transform(groupby(to_predict_df, :session), not_initial = :initial .!= 1)
to_predict_df = @transform(groupby(to_predict_df, :treatment), treat_inters = minimum(:max_inter))


to_predict_df[!, :rd_group] = 1 .*(to_predict_df[!, :sgpe] .< 1)  .+ (to_predict_df[!, :sgpe] .> 0).*(to_predict_df[!, :sizeRD] .< 0) .* 2  .+ (to_predict_df[!, :sizeRD] .> 0).*(to_predict_df[!, :sizeRD] .< 0.15) .* 3 .+ (to_predict_df[!, :sizeRD] .> 0.15).*(to_predict_df[!, :sizeRD] .< 0.3) .* 4 .+ (to_predict_df[!, :sizeRD] .> 0.3) .* 5


data_to_predict = map(collect(groupby(to_predict_df, :session))) do gdf
    vals = collect(gdf[1,:])
    namn = names(gdf)
    res = Dict{Any, Any}(Symbol(k) => val for (k, val) in zip(namn, vals))
    res[:round_seq] = @by(gdf, :inter, n_round = maximum(:round)).n_round
    res[:y] = collect(gdf.y)
    res[:last_third] = collect(gdf.last_third)
    res[:initial] = collect(gdf.initial)
    res[:not_initial] = collect(gdf.not_initial)
    res[:last_half] = collect(gdf.last_half)
    res[:first_half] = collect(gdf.first_half)
    res[:all] = (collect(gdf.initial) .!= 16541984) .* 1
    res[:fold] = first(gdf.r_fold_1)
    res
end

y = collect(to_predict_df.y)
last_third = collect(to_predict_df.last_third)


CSV.write("../clean_data/population_play.csv", to_predict_df)


#%%
include("PopulationModels.jl")
import .PopulationModels
PP = PopulationModels

@everywhere begin
    using Pkg
    Pkg.instantiate()   
    include("PopulationModels.jl")
    import .PopulationModels
    PP = PopulationModels
    using Optim
    using BlackBoxOptim
    function y_from_dat(data)
        res = map(data) do dat
            dat[:y]
        end
        vcat(res...)
    end
    function idxs_from_dat(data, idx_key)
        res = map(data) do dat
            dat[idx_key]
        end
        vcat(res...)
    end
    function l3_from_dat(data)
        res = map(data) do dat
            dat[:initial]
        end
        vcat(res...)
    end
end


#%% Function for estimating and testing

function estim_eval_from_srange(namn, model_in, srange; n_sims_wrap=100, n_cands=n_cores, bb_time=200, NM_time=600, print_stuff = false, perf_n=1000, test_n=1500, data=data_to_predict, reduced_avg=false, use_idx=false, idx_key="all", fold_key="r_fold_1")
    n_cols = length(srange)
    to_save_df = DataFrame([Any[] for i in 1:(n_cols+3)], [:test_perf, :train_perf, :fold, [Symbol("x"*string(i)) for i in 1:n_cols]...])
    test_perf_vec = map(1:10) do fold
        @everywhere begin
            perf_n = $perf_n
            test_n = $test_n
            n_models = $n_models
            n_sims_wrap = $n_sims_wrap
            bb_time = $bb_time
            NM_time = $NM_time
            data = $data
            print_stuff = $print_stuff
            fold = $fold
            reduced_avg = $reduced_avg
            use_idx = $use_idx
            fold_key = Symbol($fold_key)
            idx_key = Symbol($idx_key)
            train_data = filter(x -> x[fold_key] != fold, data)
            test_data = filter(x -> x[fold_key] == fold, data)
            train_y = y_from_dat(train_data)
            test_y = y_from_dat(test_data)
            train_idx = idxs_from_dat(train_data, idx_key)
            test_idx = idxs_from_dat(test_data, idx_key)
            model_in = deepcopy($model_in)
            srange = $srange
            function wrap_in(x, model_in=model_in)
                m = PP.set_params_from_opt!(x, model_in)
                if reduced_avg
                    return  PP.avg_C_perf(train_data, train_y, m, n=n_sims_wrap, seed=1066, use_idx=use_idx, idx_key=idx_key)
                else
                    return PP.perf(train_data, train_y, m;n=n_sims_wrap, seed=1066,use_idx=use_idx, idxs=train_idx)
                end
            end
        end
        bb_opts = pmap(1:n_cands) do j
            opt_m = bboptimize(wrap_in, SearchRange = srange, NumDimensions=length(srange), MaxTime=bb_time, TraceInterval=100, TraceMode=:silent)
            bb_opt = PP.get_params_for_opt(PP.set_params_from_opt!(best_candidate(opt_m), model_in))
        end

        println("GC-1")
        @everywhere GC.gc()

        NM_opts = pmap(bb_opts) do init_in
            opt_in = Optim.optimize(wrap_in, init_in, NelderMead(), Optim.Options(time_limit=NM_time, show_trace=false))
            if print_stuff
                println(opt_in.minimum, " - ", opt_in.time_run)
            end
            opt_m = Optim.minimizer(opt_in)
            m = PP.set_params_from_opt!(opt_m, model_in)
            PP.get_params_for_opt(PP.set_params_from_opt!(opt_m, model_in))
        end
        println("GC-2")
        @everywhere GC.gc()

        perfs = pmap(NM_opts) do opt_x
            opt_model = PP.set_params_from_opt!(opt_x, model_in)
            if reduced_avg
                perf =  PP.avg_C_perf(train_data, train_y, opt_model, n=perf_n, seed=1, use_idx=use_idx, idx_key=idx_key)
            else
                perf = PP.perf(train_data, train_y, opt_model, seed=1, n=perf_n, use_idx=use_idx, idxs=train_idx)
            end
        end
        println("GC-3")
        @everywhere GC.gc()


        idx = argmin(perfs)
        train_perf = minimum(perfs)

        model_to_test = PP.set_params_from_opt!(NM_opts[idx], model_in)
        if reduced_avg
            test_perf =  PP.avg_C_perf(test_data, test_y, model_to_test, n=test_n, seed=2, use_idx=use_idx, idx_key=idx_key)
        else
            test_perf = PP.perf(test_data, test_y, model_to_test, seed=2, n=test_n, use_idx=use_idx, idxs=test_idx)
        end
        println("GC-4")
        @everywhere GC.gc()
        println("--------------------------------------------------------------")
        println(train_perf, test_perf)
        push!(to_save_df, [test_perf, train_perf, fold, PP.get_params_for_opt(model_to_test)...])
        CSV.write(namn, to_save_df)
        (test_perf, train_perf, model_to_test, fold)
    end
    return (test_perf_vec, to_save_df)
end


function opt_single(model_in, srange; n_sims_wrap=100, n_cands=n_cores, bb_time=200, NM_time=600, print_stuff = false, perf_n=1000, data=data_to_predict, reduced_avg=false, use_idx=false, key=:all)
    @everywhere begin
        perf_n = $perf_n
        n_sims_wrap = $n_sims_wrap
        bb_time = $bb_time
        NM_time = $NM_time
        print_stuff = $print_stuff
        srange = $srange
        reduced_avg = $reduced_avg
        data = $data
        use_idx = $use_idx
        idx_key = Symbol($key)
        idxs = idxs_from_dat(data, idx_key)
        y = y_from_dat(data)
        model_in = deepcopy($model_in)
        function wrap_in(x, model_in=model_in)
            m = PP.set_params_from_opt!(x, model_in)
            if reduced_avg
                return  PP.avg_C_perf(data, y, m, n=n_sims_wrap, seed=1066, use_idx=use_idx, idx_key=idx_key)
            else
                return PP.perf(data, y, m;n=n_sims_wrap, seed=1066, use_idx=use_idx, idxs=idxs)
            end
        end
    end


    bb_opts = pmap(1:n_cands) do j
        opt_m = bboptimize(wrap_in, SearchRange = srange, NumDimensions=length(srange), MaxTime=bb_time, TraceInterval=100, TraceMode=:silent)
        bb_opt = PP.get_params_for_opt(PP.set_params_from_opt!(best_candidate(opt_m), model_in))
    end

    println("GC-1")
    @everywhere GC.gc()

    NM_opts = pmap(bb_opts) do init_in
        opt_in = Optim.optimize(wrap_in, init_in, NelderMead(), Optim.Options(time_limit=NM_time, show_trace=false))
        if print_stuff
            println(opt_in.minimum, " - ", opt_in.time_run)
        end
        opt_m = Optim.minimizer(opt_in)
        m = PP.set_params_from_opt!(opt_m, model_in)
        PP.get_params_for_opt(PP.set_params_from_opt!(opt_m, model_in))
    end

    println("GC-2")
    @everywhere GC.gc()

    perfs = pmap(NM_opts) do opt_x
        opt_model = PP.set_params_from_opt!(opt_x, model_in)
        if reduced_avg
            perf =  PP.avg_C_perf(data, y, opt_model, n=perf_n, seed=1, use_idx=use_idx, idx_key=idx_key)
        else
            perf = PP.perf(data, y, opt_model, seed=1, n=perf_n, use_idx=use_idx, idxs=idxs)
        end
        perf
    end

    println("GC-3")
    @everywhere GC.gc()

    idx = argmin(perfs)
    perf = minimum(perfs)

    model_to_return = PP.set_params_from_opt!(NM_opts[idx], model_in)
    return (model_to_return, perf)
end

#%%
n_models = 1


srange_single_fpparams = [(-5.,5.), (0.,5.), (1.,3.), (-2.,2.), (-2.,2.), (-3.,0.), (0.,0.99)]
srange_fpparams = [vcat([srange_single_fpparams for i in 1:n_models]...)..., [(0., 2.) for i in 1:n_models]...]
fpparams_in  = PP.Model([PP.FPparams() for i in 1:n_models])


srange_single_fpsemigrim = [(-5.,5.), (0.,5.), (1.,3.), (-2.,1.), (-3.,-1.), (0.,0.99)]
srange_fpsemigrim = [vcat([srange_single_fpsemigrim for i in 1:n_models]...)..., [(0., 2.) for i in 1:n_models]...]
fpsemigrim_in  = PP.Model([PP.FPsemigrim() for i in 1:n_models])

srange_single_fpalld = [(-5.,5.), (0.,5.), (1.,3.), (-2.,1.), (-3.,-1.), (0.,0.99), (-5.,5.), (-5.,5.), (0.,0.4)]
srange_fpalld = [vcat([srange_single_fpalld for i in 1:n_models]...)..., [(0., 2.) for i in 1:n_models]...]
fpalld_in  = PP.Model([PP.FpAllD()])

srange_single_fpsemigrim = [(-5.,5.), (0.,5.), (1.,3.), (-2.,1.), (-3.,-1.), (0.,0.99)]
srange_fpsemigrim_2m = [vcat([srange_single_fpsemigrim for i in 1:2]...)..., [(0., 2.) for i in 1:2]...]
fpsemigrim_2m_in  = PP.Model([PP.FPsemigrim() for i in 1:2])

srange_single_fpsemigrimT = [(-5.,5.), (0.,5.), (1.,3.), (-2.,1.), (-3.,-1.), (0.,0.99), (-2., 2.)]
srange_fpsemigrimT = [vcat([srange_single_fpsemigrimT for i in 1:n_models]...)..., [(0., 2.) for i in 1:n_models]...]
fpsemigrimT_in  = PP.Model([PP.FPsemigrimT() for i in 1:n_models])

srange_single_fpsemigrim2T = [(-5.,5.), (0.,5.), (1.,3.), (-2.,1.), (-3.,-1.), (0.,0.99), (-2., 2.)]
srange_fpsemigrim2T = [vcat([srange_single_fpsemigrim2T for i in 1:2]...)..., [(0., 2.) for i in 1:2]...]
fpsemigrim2T_in  = PP.Model([PP.FPsemigrimT() for i in 1:2])

srange_single_fpsemigrimρ = [(-5.,5.), (0.,5.), (0., 0.99), (1.,3.), (-2.,1.), (-3.,-1.), (0.,0.99)]
srange_fpsemigrimρ = [vcat([srange_single_fpsemigrimρ for i in 1:n_models]...)..., [(0., 2.) for i in 1:n_models]...]
fpsemigrimρ_in  = PP.Model([PP.FPsemigrim_ρ() for i in 1:n_models])

srange_fpsemigrim_AllD = [srange_single_fpsemigrim..., (0.,0.3), (0.,2.), (0., 2.)]
fpsemigrim_AllD_in = PP.MixModel([PP.FPsemigrim(), PP.AllD_type()])

srange_single_fp_avgV = [(-5.,5.), (0.,5.), (0.,0.99)]
srange_fp_avgV = [vcat([srange_single_fp_avgV for i in 1:n_models]...)..., [(0., 2.) for i in 1:n_models]...]
fp_avgV_in  = PP.Model([PP.FP_avgV() for i in 1:n_models])

srange_single_pure = [(0.,5.),(0.,5.),(0.,5.),(0.,5.), (0.,1.), (0.,5.), (0.,5.), (0.,1.), (0.,0.5)]
srange_pure = [vcat([srange_single_pure for i in 1:n_models]...)..., [(0., 2.) for i in 1:n_models]...]
pure_in = PP.Model([PP.PureLearn()])

srange_single_purenoε = [(0.,5.),(0.,5.),(0.,5.),(0.,5.), (0.,1.), (0.,5.), (0.,5.), (0.,1.)]
srange_purenoε = [vcat([srange_single_purenoε for i in 1:n_models]...)..., [(0., 2.) for i in 1:n_models]...]
purenoε_in = PP.Model([PP.PureLearnNoε()])

srange_single_purereinf = [(-5.,5.),(-5.,5.),(-5.,5.),(-5.,5.),(-5.,5.),(-5.,5.),(0.,10.),(0.,1.)]
srange_purereinf = [vcat([srange_single_purereinf for i in 1:n_models]...)..., [(0., 2.) for i in 1:n_models]...]
purereinf_in = PP.Model([PP.PureReinf()])

srange_single_flearn = [[(-5.,5.) for i in 1:5]..., [(0.,5.) for i in 1:5]..., (0.01,1.)]
srange_flearn = [vcat([srange_single_flearn for i in 1:n_models]...)..., [(0., 2.) for i in 1:n_models]...]
flearn_in = PP.Model([PP.fLearn()])

srange_single_flearn2 = [[(-5.,5.) for i in 1:5]..., [(0.,5.) for i in 1:5]..., (0.01,1.), (0.01,1.)]
srange_flearn2 = [vcat([srange_single_flearn2 for i in 1:n_models]...)..., [(0., 2.) for i in 1:n_models]...]
flearn2_in = PP.Model([PP.fLearn2()])


srange_single_fpflex = [[(-5.,5.) for i in 1:5]..., [(0.,5.) for i in 1:5]..., (0.01,1.)]
srange_fpflex = [vcat([srange_single_fpflex for i in 1:n_models]...)..., [(0., 2.) for i in 1:n_models]...]
fpflex_in = PP.Model([PP.FPFlex()])

srange_fpflex_2m = [vcat([srange_single_fpflex for i in 1:2]...)..., [(0., 2.) for i in 1:2]...]
fpflex_2m_in = PP.Model([PP.FPFlex(), PP.FPFlex()])

srange_fpflex_AllD = [srange_single_fpflex..., (0.,0.3), (0.,2.), (0., 2.)]
fpflex_AllD_in = PP.MixModel([PP.FPFlex(), PP.AllD_type()])


#####################################################################
#%% CV pop-preds
#####################################################################

for r in 1:10
    res_fpsemigrim_cv_pop =  estim_eval_from_srange("../temp_data/cv_pop_fpsemigrim_1m"*string(r)*".csv", deepcopy(fpsemigrim_in), srange_fpsemigrim, print_stuff=false, fold_key="r_fold_"*string(r))
end

for r in 1:10
    res_fpparams_cv_pop =  estim_eval_from_srange("../temp_data/cv_pop_fpparams_1m"*string(r)*".csv", deepcopy(fpparams_in), srange_fpparams, print_stuff=false, fold_key="r_fold_"*string(r))
end

for r in 1:10
    res_fpsemigrimT_cv_pop =  estim_eval_from_srange("../temp_data/cv_pop_fpsemigrimT_1m"*string(r)*".csv", deepcopy(fpsemigrimT_in), srange_fpsemigrimT, print_stuff=false, fold_key="r_fold_"*string(r))
end

for r in 1:10
    res_fpsemigrim_2m_cv_pop =  estim_eval_from_srange("../temp_data/cv_pop_fpsemigrim_2m"*string(r)*".csv", deepcopy(fpsemigrim_2m_in), srange_fpsemigrim_2m, print_stuff=false, fold_key="r_fold_"*string(r))
end

for r in 1:10
    res_fpsemigrim_AllD_cv_pop =  estim_eval_from_srange("../temp_data/cv_pop_fpsemigrim_AllD"*string(r)*".csv", deepcopy(fpsemigrim_AllD_in), srange_fpsemigrim_AllD, print_stuff=false, fold_key="r_fold_"*string(r))
end

for r in 1:10
    res_fpalld_cv_pop =  estim_eval_from_srange("../temp_data/cv_pop_fpAllD_1m"*string(r)*".csv", deepcopy(fpalld_in), srange_fpalld, print_stuff=false, fold_key="r_fold_"*string(r))
end

for r in 1:10
    res_fpsemigrimρ_cv_pop =  estim_eval_from_srange("../temp_data/cv_pop_fpsemigrimρ_1m"*string(r)*".csv", deepcopy(fpsemigrimρ_in), srange_fpsemigrimρ, print_stuff=false, fold_key="r_fold_"*string(r))
end

for r in 1:10
    res_flearn_cv_pop = estim_eval_from_srange("../temp_data/cv_pop_flearn_1m"*string(r)*".csv", deepcopy(flearn_in), srange_flearn, print_stuff=false, fold_key="r_fold_"*string(r))
end

for r in 1:10
    res_flearn2_cv_pop = estim_eval_from_srange("../temp_data/cv_pop_flearn2_1m"*string(r)*".csv", deepcopy(flearn2_in), srange_flearn2, print_stuff=false, fold_key="r_fold_"*string(r))
end

for r in 1:10
    res_fpflex_cv_pop = estim_eval_from_srange("../temp_data/cv_pop_fpflex_1m"*string(r)*".csv", deepcopy(fpflex_in), srange_fpflex, print_stuff=false, fold_key="r_fold_"*string(r))
end

for r in 1:10
    res_pure_cv_pop = estim_eval_from_srange("../temp_data/cv_pop_pure_1m"*string(r)*".csv", deepcopy(pure_in), srange_pure, print_stuff=false, fold_key="r_fold_"*string(r))
end

for r in 1:10
    res_purenoε_cv_pop = estim_eval_from_srange("../temp_data/cv_pop_purenoε_1m"*string(r)*".csv", deepcopy(purenoε_in), srange_purenoε, print_stuff=false, fold_key="r_fold_"*string(r))
end

for r in 1:10
    res_purereinf_cv_pop = estim_eval_from_srange("../temp_data/cv_pop_purereinf_1m"*string(r)*".csv", deepcopy(purereinf_in), srange_purereinf, print_stuff=false, fold_key="r_fold_"*string(r))
end




######################################################################
#%% First half on second half predictions
########################################################################

for r in 1:10
    first_half_res_fpsemigrim_cv_pop =  estim_eval_from_srange("../temp_data/cv_first_half_pop_fpsemigrim_1m"*string(r)*".csv", deepcopy(fpsemigrim_in), srange_fpsemigrim, print_stuff=false, fold_key="r_fold_"*string(r), data=data_to_predict, use_idx=true, idx_key="first_half")
end


#%%


function cv_first_pop_avg_last(model_in, srange;fold_n=1)
    fold_key = Symbol("r_fold_", fold_n)
    res = map(1:10) do fold
        train_data = filter(x -> x[fold_key] != fold, data_to_predict)
        test_data = filter(x -> x[fold_key] == fold, data_to_predict)
        y = y_from_dat(test_data)
        fp_res = opt_single(deepcopy(model_in), srange, print_stuff=true, bb_time=200, NM_time=600, use_idx=true, key="first_half", data=train_data)
        avg_perf = PP.avg_C_perf(test_data, y, fp_res[1], n=1000, use_idx=true, idx_key=:last_half)
        @everywhere GC.gc()
        pop_perf = PP.perf(test_data, y, fp_res[1], n=1000, use_idx=true, idxs=idxs_from_dat(test_data, :last_half))
        @everywhere GC.gc()
        println(avg_perf, " - ", pop_perf)
        (avg=avg_perf, pop=pop_perf, fold_n=fold_n, fold=fold)
    end
    res
end


for r in 1:10
    new_res = cv_first_pop_avg_last(fpsemigrim_in, srange_fpsemigrim,fold_n=r)
    res_df = DataFrame(new_res)
    CSV.write("../temp_data/cv_first_last_fpsemigrim_1m"*string(r)*".csv", res_df)
end



#####################################################################
#%% Generate predictions data set
#####################################################################


fpsemigrim_csv = CSV.File("../temp_data/cv_pop_fpsemigrim_1m1.csv") |> DataFrame


per_fold_pred = map(1:10) do fold
    model_fpsemigrim = PP.Model([PP.FPsemigrim()])
    fold_data = filter(x -> x[:fold] == fold, data_to_predict)
    fold_y = y_from_dat(fold_data)
    fpsemigrim_vars = collect(@subset(fpsemigrim_csv, :fold .== fold)[1,4:end])
    model_fpsemigrim = PP.set_params_from_opt!(fpsemigrim_vars, model_fpsemigrim)
    fpsemigrim_preds = PP.sim_data(fold_data, fold_y, model_fpsemigrim, n=1500)
    (;fold, fold_y, fpsemigrim_preds)
end



to_predict_df[!,:fold_y] .= 0.
to_predict_df[!,:fpsemigrim_y] .= 0.

for fold in 1:10
    to_predict_df[to_predict_df.r_fold_1 .== fold,:fold_y] = per_fold_pred[fold][:fold_y]
    to_predict_df[to_predict_df.fold .== fold,:fpsemigrim_y] = per_fold_pred[fold][:fpsemigrim_preds]
end

to_predict_df[!, :mse_fpsemigrim] = (to_predict_df[!,:fpsemigrim_y] - to_predict_df[!,:y]).^2

to_predict_df[!,:rd_group] = (to_predict_df.sizeRD .< -0)*-1
to_predict_df[!,:rd_group] += (to_predict_df.sizeRD .> 0.15)*1 .+ (to_predict_df.sizeRD .> 0.3)*1


CSV.write("../temp_data/pop_preds_predictions.csv", to_predict_df)


################################################################################
#%% Simulate individual level data
################################################################################
#
#
# #%% Opt fpsemigrim
fpsemigrim_params = PP.get_params_from_csv("pop_fpsemigrim_1m", fpsemigrim_in)
opt_fpsemigrim_model = PP.set_params_untransformed!(fpsemigrim_params[:avg], fpsemigrim_in)
ind_data = PP.sim_ind_data(data_to_predict,y, opt_fpsemigrim_model, n=100)
ind_data_df = DataFrame(ind_data)
mean(ind_data_df.c .== 1)

CSV.write("../temp_data/sim_ind_data_fpsemigrim_1m_n200.csv", ind_data_df)

#%% Opt fpsemigrim n 16
fpsemigrim_params = PP.get_params_from_csv("pop_fpsemigrim_1m", fpsemigrim_in)
opt_fpsemigrim_model = PP.set_params_untransformed!(fpsemigrim_params[:avg], fpsemigrim_in)
ind_data = PP.sim_ind_data(data_to_predict,y, opt_fpsemigrim_model, n=8)
ind_data_df = DataFrame(ind_data)
mean(ind_data_df.c .== 1)

CSV.write("../temp_data/sim_ind_data_fpsemigrim_1m_n16.csv", ind_data_df)


################################################################################
#%% Generate long run predictions
################################################################################

treats = unique(@subset(osap_df, :paper .== "Dal Bo and Frechette 2011a")[!, :treatment])



dalbo2011 = @subset(to_predict_df, in.(:treatment, [treats]))


dal_dat = map(treats) do treat
    filter(x -> x[:treatment] == treat, data_to_predict)[1]
end


PP.get_params_from_csv("pop_fpsemigrim_1m", fpsemigrim_in)
h = PP.get_params_from_csv("pop_fpsemigrim_1m", fpsemigrim_in)
fpsemigrim_params = PP.get_params_from_csv("pop_fpsemigrim_1m", fpsemigrim_in)
opt_fpsemigrim_model = PP.set_params_untransformed!(fpsemigrim_params[:avg], fpsemigrim_in)
σ = opt_fpsemigrim_model.types[1]

# Do not try to parallellize, will kill RAM
function gen_sim_preds(σ, n, n_sgs, dat, rng=MersenneTwister(); fixed_rounds=false)
    σs = [deepcopy(σ) for i in 1:n]
    δ = dat[:δ]
    n_half = Int(n/2)
    rands= rand(rng, Int(round(n_sgs*(1/(1-δ))*n*2)))
    PP.adjust!.(σs, dat[:sizeRD], dat[:g], dat[:l], dat[:δ])
    res = map(1:n_sgs) do i
        if fixed_rounds
            n_rs = Int(round(1/(1-δ)))
        else
            n_rs = rand(rng, Distributions.Geometric(1-δ)) + 1
        end
        PP.new_inter_update!.(σs)
        PP.reset_for_inter!.(σs, n_rs)
        PP.shuffle!(rng, σs)
        y = PP.sim_inter(n_rs, σs[1:n_half], σs[n_half+1:end], rands)
        es = [σ.e for σ in σs]
        (;initial=y[1], sum=sum(y), n=n_rs, mean=mean(y), inter=i, es)
    end
    df = DataFrame(res)
end


function gen_mul_sim_preds(σ, n, n_sgs, dat, n_sims)
    res = gen_sim_preds(σ, n, n_sgs, dat)
    for i in 1:(n_sims-1)
        res = res .+ gen_sim_preds(σ, n, n_sgs, dat, MersenneTwister(i))
    end
    res./n_sims
end


function gen_long_run_preds(;σ=σ, dat_vec=dal_dat, pop_n=16, sims_n=1000, t=10000, fixed_rounds=false)
    dfs = map(dat_vec) do dat
        sizeRD = dat[:sizeRD]
        δ = dat[:δ]
        δRD = -(sizeRD - δ)
        g = dat[:g]
        l =dat[:l]
        treat =dat[:treatment]
        treat_df = @subset(to_predict_df, :treatment .== treat, :inter .<= :treat_inters)
        treat_df_ini = @subset(treat_df, :round .== 1)

        y_initial = @based_on(groupby(treat_df_ini, :inter), y=mean(:y))[!,:y]
        treat_df = @subset(to_predict_df, :treatment .== treat, :inter .<= :treat_inters)
        y_mean = @based_on(groupby(treat_df, :inter), y=mean(:y))[!, :y]

        ys = map(1:sims_n) do i
            gen_sim_preds(σ, pop_n, t, dat, MersenneTwister(i), fixed_rounds=fixed_rounds)
        end
        y_means = hcat([y[!,:mean] for y in ys]...)
        y_inis = hcat([y[!,:initial] for y in ys]...)

        sim_95 = map(eachrow(y_means)) do row
            quantile(row, [0.95])[1]
        end

        sim_05 = map(eachrow(y_means)) do row
            quantile(row, [0.05])[1]
        end

        sim_50 = map(eachrow(y_means)) do row
            quantile(row, [0.5])[1]
        end

        sim_mean = map(eachrow(y_means)) do row
            mean(row)
        end

        sim_95_inis = map(eachrow(y_inis)) do row
            quantile(row, [0.95])[1]
        end

        sim_05_inis = map(eachrow(y_inis)) do row
            quantile(row, [0.05])[1]
        end

        sim_50_inis = map(eachrow(y_inis)) do row
            quantile(row, [0.5])[1]
        end

        sim_mean_inis = map(eachrow(y_inis)) do row
            mean(row)
        end


        y_in = zeros(t)
        y_in[1:length(y_initial)] .= y_initial
        y = zeros(t)
        y[1:length(y_initial)] .= y_mean
        real_inter = zeros(Int, t)
        real_inter[1:length(y_initial)] .= 1
        inter = 1:t

        df = DataFrame(y_initial=y_in, y_mean=y, real_inter=real_inter, sim_mean=sim_mean, sim_05=sim_05, sim_95=sim_95, sim_50=sim_50, sim_mean_inis=sim_mean_inis, sim_05_inis=sim_05_inis, sim_95_inis=sim_95_inis, sim_50_inis=sim_50_inis, inter=inter)
        df[!, :δ] .= δ
        df[!, :sizeRD] .= sizeRD
        df[!, :g] .= g
        df[!, :l] .= l
        df[!, :treat] .= treat
        df[!, :δRD] .= δRD
        df
    end
    vcat(dfs...)
end



#%%

dfs = map(1:6) do idx
    sizeRD = dal_dat[idx][:sizeRD]
    δ = dal_dat[idx][:δ]
    δRD = -(sizeRD - δ)
    g = dal_dat[idx][:g]
    l =dal_dat[idx][:l]

    treat =dal_dat[idx][:treatment]

    treat_df = @subset(to_predict_df, :treatment .== treat, :inter .<= :treat_inters, :round .== 1)

    y_initial = @based_on(groupby(treat_df, :inter), y=mean(:y))[!,:y]
    treat_df = @subset(to_predict_df, :treatment .== treat, :inter .<= :treat_inters)
    y_mean = @based_on(groupby(treat_df, :inter), y=mean(:y))[!,:y]

    ys = map(1:1000) do i
        gen_sim_preds(σ, 14, 10000, dal_dat[idx], MersenneTwister(i))
    end
    y_means = hcat([y[!,:mean] for y in ys]...)
    y_inis = hcat([y[!,:initial] for y in ys]...)

    sim_95 = map(eachrow(y_means)) do row
        quantile(row, [0.95])[1]
    end

    sim_05 = map(eachrow(y_means)) do row
        quantile(row, [0.05])[1]
    end

    sim_50 = map(eachrow(y_means)) do row
        quantile(row, [0.5])[1]
    end

    sim_mean = map(eachrow(y_means)) do row
        mean(row)
    end

    sim_95_inis = map(eachrow(y_inis)) do row
        quantile(row, [0.95])[1]
    end

    sim_05_inis = map(eachrow(y_inis)) do row
        quantile(row, [0.05])[1]
    end

    sim_50_inis = map(eachrow(y_inis)) do row
        quantile(row, [0.5])[1]
    end

    sim_mean_inis = map(eachrow(y_inis)) do row
        mean(row)
    end


    y_in = zeros(10000)
    y_in[1:length(y_initial)] .= y_initial
    y = zeros(10000)
    y[1:length(y_initial)] .= y_mean
    real_inter = zeros(Int, 10000)
    real_inter[1:length(y_initial)] .= 1
    inter = 1:10000

    df = DataFrame(y_initial=y_in, y_mean=y, real_inter=real_inter, sim_mean=sim_mean, sim_05=sim_05, sim_95=sim_95, sim_50=sim_50, sim_mean_inis=sim_mean_inis, sim_05_inis=sim_05_inis, sim_95_inis=sim_95_inis, sim_50_inis=sim_50_inis, inter=inter)
    df[!, :δ] .= δ
    df[!, :sizeRD] .= sizeRD
    df[!, :g] .= g
    df[!, :l] .= l
    df[!, :treat] .= treat
    df[!, :δRD] .= δRD
    df
end

sim_preds_df = vcat(dfs...)


CSV.write("../temp_data/dalbo_long_sims_fpsemigrim_opt.csv", sim_preds_df)

#%% Dalbo 100


dfs_100 = map(1:6) do idx
    sizeRD = dal_dat[idx][:sizeRD]
    δ = dal_dat[idx][:δ]
    δRD = -(sizeRD - δ)
    g = dal_dat[idx][:g]
    l =dal_dat[idx][:l]

    treat =dal_dat[idx][:treatment]

    treat_df = @subset(to_predict_df, :treatment .== treat, :inter .<= :treat_inters, :round .== 1)
    y_initial = @based_on(groupby(treat_df, :inter), y=mean(:y))[!,:y]
    treat_df = @subset(to_predict_df, :treatment .== treat, :inter .<= :treat_inters)
    y_mean = @based_on(groupby(treat_df, :inter), y=mean(:y))[!, :y]


    ys = map(1:1000) do i
        if i % 50 == 1
            println("---- ", idx, "-", i, " ----")
        end
        gen_sim_preds(σ, 100, 10000, dal_dat[idx], MersenneTwister(i))
    end
    y_means = hcat([y[!, :mean] for y in ys]...)
    y_inis = hcat([y[!, :initial] for y in ys]...)

    sim_95 = map(eachrow(y_means)) do row
        quantile(row, [0.95])[1]
    end

    sim_05 = map(eachrow(y_means)) do row
        quantile(row, [0.05])[1]
    end

    sim_50 = map(eachrow(y_means)) do row
        quantile(row, [0.5])[1]
    end

    sim_mean = map(eachrow(y_means)) do row
        mean(row)
    end

    sim_95_inis = map(eachrow(y_inis)) do row
        quantile(row, [0.95])[1]
    end

    sim_05_inis = map(eachrow(y_inis)) do row
        quantile(row, [0.05])[1]
    end

    sim_50_inis = map(eachrow(y_inis)) do row
        quantile(row, [0.5])[1]
    end

    sim_mean_inis = map(eachrow(y_inis)) do row
        mean(row)
    end


    y_in = zeros(10000)
    y_in[1:length(y_initial)] .= y_initial
    y = zeros(10000)
    y[1:length(y_initial)] .= y_mean
    real_inter = zeros(Int, 10000)
    real_inter[1:length(y_initial)] .= 1
    inter = 1:10000

    df = DataFrame(y_initial=y_in, y_mean=y, real_inter=real_inter, sim_mean=sim_mean, sim_05=sim_05, sim_95=sim_95, sim_50=sim_50, sim_mean_inis=sim_mean_inis, sim_05_inis=sim_05_inis, sim_95_inis=sim_95_inis, sim_50_inis=sim_50_inis, inter=inter)
    df[!,:δ] .= δ
    df[!,:sizeRD] .= sizeRD
    df[!,:g] .= g
    df[!,:l] .= l
    df[!,:treat] .= treat
    df[!,:δRD] .= δRD
    df
end

sim_preds_df_100 = vcat(dfs_100...)

CSV.write("../temp_data/dalbo_long_sims_fpsemigrim_100p_opt.csv", sim_preds_df_100)



#%% calculate and compare single treatment with 1000 p and fixed vs large
single_dat = [dal_dat[3]]
df_1000_p = gen_long_run_preds(sims_n = 1000, t=1000, pop_n=1000, dat_vec=single_dat)
CSV.write("../temp_data/dalbo_long_sims_fpsemigrim_1000p_opt.csv", df_1000_p)

df_1000_p_fixed = gen_long_run_preds(sims_n = 1000, t=1000, pop_n=1000, dat_vec=single_dat, fixed_rounds=true)
CSV.write("../temp_data/dalbo_long_sims_fpsemigrim_1000p_opt.csv", df_1000_p_fixed)

df_100_p_fixed = gen_long_run_preds(sims_n = 1000, t=1000, pop_n=100, dat_vec=single_dat, fixed_rounds=true)

CSV.write("../temp_data/dalbo_long_sims_fpsemigrim_100p_opt.csv", df_100_p_fixed)

df_1000_p_fixed[end,:]

#%% Testing fixed supergame lengths
dfs_fixed = map(1:6) do idx
    sizeRD = dal_dat[idx][:sizeRD]
    δ = dal_dat[idx][:δ]
    δRD = -(sizeRD - δ)
    g = dal_dat[idx][:g]
    l =dal_dat[idx][:l]

    treat =dal_dat[idx][:treatment]

    treat_df = @subset(to_predict_df, :treatment .== treat, :inter .<= :treat_inters, :round .== 1)
    y_initial = @based_on(groupby(treat_df, :inter), y=mean(:y))[!, :y]
    treat_df = @subset(to_predict_df, :treatment .== treat, :inter .<= :treat_inters)
    y_mean = @based_on(groupby(treat_df, :inter), y=mean(:y))[!, :y]


    ys = map(1:1000) do j
        gen_sim_preds(σ, 14, 10000, dal_dat[idx], MersenneTwister(j), fixed_rounds=true)[!, :mean]
    end
    y_mat = hcat(ys...)

    sim_95 = map(eachrow(y_mat)) do row
        quantile(row, [0.95])[1]
    end

    sim_05 = map(eachrow(y_mat)) do row
        quantile(row, [0.05])[1]
    end

    sim_50 = map(eachrow(y_mat)) do row
        quantile(row, [0.5])[1]
    end

    sim_mean = map(eachrow(y_mat)) do row
        mean(row)
    end


    y_in = zeros(10000)
    y_in[1:length(y_initial)] .= y_initial
    y = zeros(10000)
    y[1:length(y_initial)] .= y_mean
    real_inter = zeros(Int, 10000)
    real_inter[1:length(y_initial)] .= 1
    inter = 1:10000

    df = DataFrame(y_initial=y_in, y_mean=y, real_inter=real_inter, sim_mean=sim_mean, sim_05=sim_05, sim_95=sim_95, sim_50=sim_50, inter=inter)
    df[!, :δ] .= δ
    df[!, :sizeRD] .= sizeRD
    df[!, :g] .= g
    df[!, :l] .= l
    df[!, :treat] .= treat
    df[!, :δRD] .= δRDl

    df
end

sim_preds_df_fixed = vcat(dfs_fixed...)


CSV.write("../temp_data/dalbo_long_sims_fpsemigrim_fixed_opt.csv", sim_preds_df_fixed)



#%%

treat_preds = map(unique(to_predict_df[!,:treatment])) do treat
    dat = filter(x -> x[:treatment] == treat, data_to_predict)[1]
    y_vals = map(1:1000) do i
        res = gen_sim_preds(σ, 16, 10000, dat, MersenneTwister(i))
        (y=res[!, :mean], initial=res[!, :initial])
    end
    ys = [x[:y][end] for x in y_vals]
    initials = [x[:initial][end] for x in y_vals]
    sim_mean = mean(ys)
    initial = mean(initials)

    sim_95 = quantile(ys, [0.95])[1]
    sim_05 = quantile(ys, [0.05])[1]


    (;y=mean(sim_mean), initial, sim_05=mean(sim_05), sim_95=mean(sim_95), sizeRD=dat[:sizeRD], δ=dat[:δ], treat=dat[:treatment], δRD = -(dat[:sizeRD] - dat[:δ]))
end
treat_df = DataFrame(treat_preds)

CSV.write("../temp_data/fpsemigrim_treat_long_preds_t10000.csv", treat_df)
