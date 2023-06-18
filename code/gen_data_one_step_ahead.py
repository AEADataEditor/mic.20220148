import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import GroupKFold
import random

import sys
import os
sys.version_info
os.path.dirname(sys.executable)

pd.options.display.float_format = '{:.4f}'.format
pd.options.display.max_rows = 400

dalbo_df = pd.read_csv("../clean_data/DF_cleaned.csv")
new_data_df = pd.read_csv("../clean_data/new_data_cleaned.csv")

data_df = pd.concat([dalbo_df, new_data_df]).reset_index(drop=True)

data_df.shape
data_df = data_df.sort_values(by=["treatment", "id", "inter", "round"]).reset_index()


data_df["paper_treat"] = data_df["treatment"]
# Some floating point representation errors, therefore rounding
data_df["treatment"] = pd.factorize(round(data_df["g"]*10000) + round(data_df["l"]*1000) + round(data_df["delta"]*100))[0] + 1
data_df["session"] = pd.factorize(data_df["session"])[0] + 1
data_df["id"] = pd.factorize(data_df["id"])[0] + 1

##############################################################################
#%% Generate variables for learning model
##############################################################################

for i in [1,2,3,4,5]:
    data_df.loc[data_df["round"] <= i,"self_t"+str(i)] = 0
    data_df.loc[data_df["round"] <= i,"opp_t"+str(i)] = 0


data_df["hist"] = "initial"
data_df.loc[(data_df["self_t1"] == 1) & (data_df["opp_t1"] == 1), "hist"] = "CC"
data_df.loc[(data_df["self_t1"] == -1) & (data_df["opp_t1"] == 1), "hist"] = "DC"
data_df.loc[(data_df["self_t1"] == 1) & (data_df["opp_t1"] == -1), "hist"] = "CD"
data_df.loc[(data_df["self_t1"] == -1) & (data_df["opp_t1"] == -1), "hist"] = "DD"

data_df["payoff"] = 0
data_df["payoff"] = np.select([(data_df["c"] == 1) & (data_df["opp_c"] == 1)], [1], data_df["payoff"])
data_df["payoff"] = np.select([(data_df["c"] == -1) & (data_df["opp_c"] == 1)], [1 + data_df["g"]], data_df["payoff"])
data_df["payoff"] = np.select([(data_df["c"] == 1) & (data_df["opp_c"] == -1)], [ -data_df["l"]], data_df["payoff"])

copy_df = data_df.copy()

copy_df["prev_occur"] = 1
copy_df["prev_occur"] = copy_df.groupby(["id", "hist"])["prev_occur"].transform("cumsum")
copy_df["prev_occur"] = copy_df.groupby(["id", "hist"])["prev_occur"].transform("shift").fillna(0)
copy_df["max_occur"] = copy_df.groupby(["id", "hist"])["prev_occur"].transform("max")


# Set ρ= 1 to avoid rececny
ρ_vals = np.array([0.1,0.2,0.3,0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.])
ρ_reinforce_names = ["reinforce_" + str(ρ) for ρ in ρ_vals]
ρ_belief_names = ["belief_" + str(ρ) for ρ in ρ_vals]

for i in range(len(ρ_vals)):
    ρ = ρ_vals[i]
    reinf_name = ρ_reinforce_names[i]
    belief_name = ρ_belief_names[i]
    reinf_name_within = reinf_name + "_within_inter"
    belief_name_within = belief_name + "_within_inter"

    # These three lines calculate V_i(t)
    copy_df["inter_payoff"] = copy_df.groupby(["id", "inter"])["payoff"].transform("sum")
    copy_df["round_payoff"] = copy_df.groupby(["id", "inter"])["payoff"].transform("cumsum")
    copy_df["rest_payoff"] = copy_df["inter_payoff"] - copy_df["round_payoff"] + copy_df["payoff"]

    # To get the reinforcement, we calculate a_i(t)*V_i(t)
    copy_df["c*payoff"] = copy_df["c"]*(copy_df["rest_payoff"])

    # To get discounted values, we first multiply with total discount at last occurance and shift one hist forward
    copy_df["c*payoff"] = copy_df["c"]*(copy_df["rest_payoff"])*np.power(ρ, copy_df["max_occur"] - copy_df["prev_occur"])
    copy_df[reinf_name] = copy_df.groupby(["id", "hist"])["c*payoff"].transform("shift").fillna(0)

    # To get reset only between supergames we calculate and subtract the reinf from the current supergame
    copy_df[reinf_name_within] = copy_df.groupby(["id", "hist", "inter"])["c*payoff"].transform("shift").fillna(0)
    copy_df[reinf_name_within] = copy_df.groupby(["id", "hist", "inter"])[reinf_name_within].transform("cumsum")
    copy_df[reinf_name] = copy_df.groupby(["id", "hist"])[reinf_name].transform("cumsum")
    copy_df[reinf_name] = copy_df[reinf_name] - copy_df[reinf_name_within]

    # To get the relevant reinforcement at time t, we divide with remaining discouting.
    copy_df[reinf_name] = copy_df[reinf_name]/np.power(ρ, copy_df["max_occur"] - copy_df["prev_occur"] + 1)



    # Essentially the same logic as for the reinforcement, but with a_{-i}(t) instead of a_i(t)*V_i(t)
    copy_df[belief_name] = copy_df["opp_c"]*np.power(ρ, copy_df["max_occur"] - copy_df["prev_occur"])
    copy_df[belief_name_within] = copy_df.groupby(["id", "hist", "inter"])[belief_name].transform("shift").fillna(0)
    copy_df[belief_name_within] = copy_df.groupby(["id", "hist", "inter"])[belief_name_within].transform("cumsum")
    copy_df[belief_name] = copy_df.groupby(["id", "hist"])[belief_name].transform("cumsum")
    copy_df[belief_name] = copy_df.groupby(["id", "hist"])[belief_name].transform("shift").fillna(0)
    copy_df[belief_name] = copy_df[belief_name]/np.power(ρ, copy_df["max_occur"] - copy_df["prev_occur"] + 1)
    copy_df[belief_name] = copy_df[belief_name] -copy_df[belief_name_within]


for nam in ρ_reinforce_names:
    data_df[nam] = copy_df[nam]

for nam in ρ_belief_names:
    data_df[nam] = copy_df[nam]


copy_df["self_hist_sum"] = copy_df.groupby(["id", "hist"])["c"].transform("cumsum")
copy_df["self_hist_sum"] = copy_df.groupby(["id", "hist"])["self_hist_sum"].transform("shift").fillna(0)
copy_df["opp_hist"] = copy_df.groupby(["id", "hist"])["opp_c"].transform("shift").fillna(0)
copy_df["self_hist"] = copy_df.groupby(["id", "hist"])["c"].transform("shift").fillna(0)


copy_df["prev_occur"] = 1
copy_df["prev_occur"] = copy_df.groupby(["id", "hist"])["prev_occur"].transform("cumsum")
copy_df["prev_occur"] = copy_df.groupby(["id", "hist"])["prev_occur"].transform("shift").fillna(1)
copy_df["belief_mean"] = copy_df["belief_1.0"]/copy_df["prev_occur"]
copy_df["reinforce_mean"] = copy_df["reinforce_1.0"]/copy_df["prev_occur"]
copy_df["self_hist_mean"] = copy_df["self_hist_sum"]/copy_df["prev_occur"]



copy_df["tot_round"] = 1
copy_df["tot_round"] = copy_df.groupby("id")["tot_round"].transform("cumsum")


copy_df["prev_occur"] = 1
copy_df["prev_occur"] = copy_df.groupby(["id", "hist"])["prev_occur"].transform("cumsum")
copy_df["prev_occur"] = copy_df.groupby(["id", "hist"])["prev_occur"].transform("shift").fillna(0)


copy_df["max_inter"] = copy_df.groupby("id")["inter"].transform('max')
copy_df["inter_from_last"] = copy_df["max_inter"] -  copy_df["inter"]  + 1
copy_df[["inter_from_last", "last_4", "inter"]]
copy_df["last_4"] = copy_df["inter_from_last"] <= 4
copy_df["last_third"] =  (copy_df["inter_from_last"]/copy_df["max_inter"]) < 1/3
copy_df["last_half"] =  (copy_df["inter_from_last"]/copy_df["max_inter"]) < 1/2
copy_df["third"] =  1
copy_df["third"] =  copy_df["third"] + ((copy_df["inter_from_last"]/copy_df["max_inter"]) < 2/3)
copy_df["third"] =  copy_df["third"] + ((copy_df["inter_from_last"]/copy_df["max_inter"]) < 1/3)


copy_df["exp_inter_len"] = (1/(1 - copy_df["delta"]))
copy_df["inter_rounds"] = copy_df.groupby(['session', "inter"])["round"].transform('max')

copy_df["by_third_exp_len"] = copy_df.groupby(['session', "third"])["inter"].transform('nunique')*copy_df["exp_inter_len"]
copy_df["by_third_len_temp"] = copy_df.groupby(["third", "round", "id"])["inter_rounds"].transform('sum')
copy_df["by_third_len"] = copy_df.groupby(['session', "third", "inter"])["by_third_len_temp"].transform('first')
copy_df["by_third_diff_len"] = copy_df["by_third_len"] - copy_df["by_third_exp_len"]
copy_df["by_third_diff_len_share"] = copy_df["by_third_len"]/copy_df["by_third_exp_len"]

copy_df["first_third_diff_len"] = copy_df.groupby(['session'])["by_third_diff_len"].transform('first')
copy_df["first_third_diff_len_share"] = copy_df.groupby(['session'])["by_third_diff_len_share"].transform('first')

copy_df["exp_len"] = copy_df.groupby(['session'])["inter"].transform('nunique')*copy_df["exp_inter_len"]
copy_df["tot_sess_rounds"] = copy_df.groupby(['session'])["tot_round"].transform('max')
copy_df["diff_len"] = copy_df["tot_sess_rounds"] - copy_df["exp_len"]
copy_df["diff_len_share"] = copy_df["tot_sess_rounds"]/copy_df["exp_len"]

copy_df["prev_exp_len_cum_temp"] = copy_df.groupby(["round", "id"])["exp_inter_len"].transform('cumsum') - copy_df["exp_inter_len"]
copy_df["prev_exp_len_cum"] = copy_df.groupby(["inter", "id"])["prev_exp_len_cum_temp"].transform('first')
copy_df["prev_len_cum"] = copy_df["tot_round"] - 1
copy_df["cum_diff_len_temp"] = copy_df["prev_exp_len_cum"] - copy_df["prev_len_cum"]
copy_df["cum_diff_len"] = copy_df.groupby(["inter", "id"])["cum_diff_len_temp"].transform('first')

copy_df["cum_diff_len_share_temp"] = (copy_df["prev_len_cum"]/copy_df["prev_exp_len_cum"]).fillna(1)
copy_df["cum_diff_len_share"] = copy_df.groupby(["inter", "id"])["cum_diff_len_share_temp"].transform('first')


#%%
data_df["diff_len"] = copy_df["diff_len"]
data_df["diff_len_share"] = copy_df["diff_len_share"]
data_df["cum_diff_len"] = copy_df["cum_diff_len"]
data_df["cum_diff_len_share"] = copy_df["cum_diff_len_share"]
data_df["first_third_diff_len"] = copy_df["first_third_diff_len"]
data_df["first_third_diff_len_share"] = copy_df["first_third_diff_len_share"]



data_df["last_4"] = copy_df["last_4"]*1
data_df["last_third"] = copy_df["last_third"]*1
data_df["last_half"] = copy_df["last_half"]*1
data_df["max_inter"] = copy_df["max_inter"]
data_df["inter_from_last"] = copy_df["inter_from_last"]
data_df["tot_round"] = copy_df["tot_round"]
data_df["reinforce"] = copy_df["reinforce_1.0"]
data_df["belief"] = copy_df["belief_1.0"]
data_df["reinforce_mean"] = copy_df["reinforce_mean"]
data_df["belief_mean"] = copy_df["belief_mean"]
data_df["self_hist"] = copy_df["self_hist"]
data_df["opp_hist"] = copy_df["opp_hist"]
data_df["prev_occur"] = copy_df["prev_occur"]
data_df["self_hist_sum"] = copy_df["self_hist_sum"]
data_df["self_hist_mean"] = copy_df["self_hist_mean"]

data_df["CC"] = (data_df["hist"] == "CC")*1
data_df["DC"] = (data_df["hist"] == "DC")*1
data_df["CD"] = (data_df["hist"] == "CD")*1
data_df["DD"] = (data_df["hist"] == "DD")*1
data_df["initial"] = (data_df["round"] == 1)*1





#%% Genrate K-folds in order to evaluate all models on the same splits
n_folds = 10
data_df["by_ind_K_fold"] = -1
fold = 1
kf = GroupKFold(n_splits=n_folds)
train_index, test_index = next(kf.split(data_df, data_df["treatment"], data_df["id"]))
for train_index, test_index in kf.split(data_df, data_df["treatment"], data_df["id"]):
    data_df.loc[test_index, "by_ind_K_fold"] = fold
    fold += 1

data_df["by_treat_K_fold"] = -1
fold = 1
kf = GroupKFold(n_splits=n_folds)
for train_index, test_index in kf.split(data_df, groups=data_df["treatment"]):
    data_df.loc[test_index, "by_treat_K_fold"] = fold
    fold += 1

for i in range(1,11):
    # Sums to 161 folds
    folds = np.append(np.repeat(range(1,11),16),[1])
    random.seed(i)
    random.shuffle(folds)
    data_df["r_fold_"+str(i)] = 0
    for sess in np.unique(data_df["session"]):
        f, folds = folds[-1], folds[:-1]
        data_df.loc[data_df["session"] == sess, "r_fold_"+str(i)] = f
data_df.columns[0]

data_df = data_df.drop("index", axis=1)
#%% Save data_frame
data_df.to_csv("../clean_data/one_step_ahead_data.csv", index=False)
