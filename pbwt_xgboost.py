import pbwt_methods
import time
import msprime
import numpy as np
import psutil
import os
import xgboost as xgb
import multiprocessing
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss

#%%
ft = time.time()
st = time.time()

num_samples_train = 500
num_samples_validation = 100
num_samples_test = 100

ts_train = msprime.sim_ancestry(samples=num_samples_train, population_size=40000,
                          recombination_rate = 1.25*10**-8, ploidy=1,sequence_length=10**6)
ts_validation = msprime.sim_ancestry(samples=num_samples_validation, population_size=40000,
                          recombination_rate = 1.25*10**-8, ploidy=1,sequence_length=10**6)
ts_test = msprime.sim_ancestry(samples=num_samples_test, population_size=40000,
                          recombination_rate = 1.25*10**-8, ploidy=1,sequence_length=10**6)


mts_train = msprime.sim_mutations(ts_train, rate=1.25*10**-8)
mts_validation = msprime.sim_mutations(ts_validation, rate=1.25*10**-8)
mts_test = msprime.sim_mutations(ts_test, rate=1.25*10**-8)

bin_data_train = mts_train.genotype_matrix().transpose()
bin_data_validation = mts_validation.genotype_matrix().transpose()
bin_data_test = mts_test.genotype_matrix().transpose()

bin_data_train[bin_data_train > 1] = 1
bin_data_validation[bin_data_validation > 1] = 1
bin_data_test[bin_data_test > 1] = 1


np.random.shuffle(bin_data_train)
np.random.shuffle(bin_data_validation)
np.random.shuffle(bin_data_test)


process = psutil.Process(os.getpid())
print("Tot mem usage: ", process.memory_info().rss)

#%%
du_train = pbwt_methods.get_dual_pbwt(bin_data_train,50)
du_validation = pbwt_methods.get_dual_pbwt(bin_data_validation,50)
du_test = pbwt_methods.get_dual_pbwt(bin_data_test,50)

#%%

side_length = 10
divergence_length = 10

start = time.time()

with multiprocessing.Pool(7) as pool:
    extracted_train = pool.starmap(
        pbwt_methods.both_way_extract,
        zip(itertools.repeat(du_train,num_samples_train),
            [i for i in range(num_samples_train)],
            itertools.repeat(side_length,num_samples_train),
            itertools.repeat(divergence_length,num_samples_train)))

with multiprocessing.Pool(7) as pool:
    extracted_validation = pool.starmap(
        pbwt_methods.both_way_extract,
        zip(itertools.repeat(du_validation,num_samples_validation),
            [i for i in range(num_samples_validation)],
            itertools.repeat(side_length,num_samples_validation),
            itertools.repeat(divergence_length,num_samples_validation)))


with multiprocessing.Pool(7) as pool:
    extracted_test = pool.starmap(
        pbwt_methods.both_way_extract,
        zip(itertools.repeat(du_test,num_samples_test),
            [i for i in range(num_samples_test)],
            itertools.repeat(side_length,num_samples_test),
            itertools.repeat(divergence_length,num_samples_test)))

# extracted_train = []
# for i in range(num_samples_train):
#     extracted_train.append(pbwt_methods.both_way_extract(du_train,i))

# extracted_test = []
# for i in range(num_samples_test):
#     extracted_test.append(pbwt_methods.both_way_extract(du_test,i))

print(time.time()-start)

#%%
freq_bins_train = [0.1,0.2,0.3,0.5,0.7,1,2,3,5,7,10,20,30,50,70,90,100]

# freq_bins_train = [100]

freq_bins_test = [0.1,0.2,0.3,0.5,0.7,1,2,3,5,7,10,20,30,50,70,90,100]


combined_extracted_train = pbwt_methods.combine_extracted_datas(
    du_train.forward_pbwt,extracted_train,freq_bins_train,
    side_window_size=side_length,divergence_window_size = divergence_length)

combined_extracted_validation = pbwt_methods.combine_extracted_datas(
    du_validation.forward_pbwt,extracted_validation,freq_bins_train,
    side_window_size=side_length,divergence_window_size = divergence_length)

combined_extracted_test = pbwt_methods.combine_extracted_datas(
    du_test.forward_pbwt,extracted_test,freq_bins_test,
    side_window_size=side_length,divergence_window_size = divergence_length)


#%%
c = combined_extracted_train
combined_extracted_train = combined_extracted_validation
combined_extracted_validation = c
#%%

trained_models = []
for i in range(len(freq_bins_train)):
    
    num_rounds = 1000
    
    bin_val = freq_bins_train[i]
    
    train_data = pbwt_methods.create_feature_matrices(combined_extracted_train[i])
    validation_data = pbwt_methods.create_feature_matrices(combined_extracted_validation[i])
    
    #test_data = pbwt_methods.create_feature_matrices(combined_extracted_test[i])

    if len(train_data[1]) == 0 or len(validation_data[1]) == 0:
        trained_models.append(None)
        continue
    
    
    
    
    #train_len = len(train_data[1])
    
    #cutoff = int(0.8*train_len)
    
    training_X = train_data[0]
    training_y = train_data[1]
    
    
    validation_X = validation_data[0]
    validation_y = validation_data[1]
    
    # training_X = train_data[0][:cutoff,]
    # training_y = train_data[1][:cutoff]
    
    
    # validation_X = train_data[0][cutoff:]
    # validation_y = train_data[1][cutoff:]
    
    
    
    dtrain_reg = xgb.DMatrix(training_X,training_y)
    dvalidation_reg = xgb.DMatrix(validation_X,validation_y)
    
    evals = [(dtrain_reg, "train"), (dvalidation_reg, "validation")]
             
    constraints = tuple(1 for _ in range(4*side_length))+tuple(0 for _ in range(4*divergence_length))
    params = {"objective": "binary:logistic",
              "tree_method": "hist",
              "monotone_constraints":constraints,
              "colsample_bytree":0.2,
              "gamma":10,
              "max_depth":2}


    
    print(bin_val)
    model = xgb.train(
        params=params,
        dtrain=dtrain_reg,
        num_boost_round=num_rounds,
        evals=evals,
        early_stopping_rounds = 10,
        verbose_eval=10,
        xgb_model=f"./xgboost_models/test_model_{bin_val}.json"
        )
    
    train_preds = model.predict(dtrain_reg)
    
    train_logloss = log_loss(training_y, train_preds)

    print(f"Training Log-loss: {train_logloss:.5f}")
    
    validation_preds = model.predict(dvalidation_reg)
    
    validation_logloss = log_loss(validation_y, validation_preds)

    print(f"Validation Log-loss: {validation_logloss:.5f}")
    
    
    print()
    
    trained_models.append(model)

#%%
for i in range(len(freq_bins_test)):
    
    bin_val = freq_bins_test[i]
    
    
    test_data = pbwt_methods.create_feature_matrices(combined_extracted_test[i])

    
    if len(test_data[1]) == 0:
        continue
    
    dtest_reg = xgb.DMatrix(test_data[0][:,],test_data[1])
    
    #model_number = 0
    model_number = i
    
    
    print(i,"Bin and Model used: ",bin_val,freq_bins_train[model_number])
    model = trained_models[model_number]
    
    if model == None:
        print()
        continue
    
    preds = model.predict(dtest_reg)
    
    predicted_alleles = [1 if preds[i] > 0.5 else 0 for i in range(len(preds))]
    
    logloss = log_loss(test_data[1], preds)

    print(f"Model Log-loss: {logloss:.5f}")
    
    p = np.array([predicted_alleles,predicted_alleles])
    q = np.array([test_data[1],test_data[1]])
    
    pbwt_methods.compare_results(p,q,
    [bin_val/100 for _ in range(len(predicted_alleles))],
    [0 for _ in range(len(predicted_alleles))],[0,1])
    print()

#%%
for i in range(len(trained_models)):
    model = trained_models[i]
    if model != None:
        name = str(freq_bins_train[i])
        print(name)
        model.save_model("./xgboost_models/test_model_"+name+".json")
#%%
xgb.plot_tree(trained_models[9],num_trees=0)

fig = plt.gcf()
fig.set_size_inches(150, 100)
fig.savefig('tree7.png')

# trained_models[11].save_model("./xgboost_models/basic_model.json")
