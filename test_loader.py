import vcf_reader
import pbwt_methods
import multiprocessing
import itertools
import numpy as np
import xgboost as xgb
#%%

genetic_map = pbwt_methods.read_genetic_map("genetic_map/hg38_genetic_map.txt.gz")
#%%
omni_data = vcf_reader.get_vcf_data()
omni_tests = vcf_reader.get_vcf_data("../pbwt_hkhan/vcf_data/omni10.vcf.gz")
#%%
def get_allele_freqs(data):
    
    num_samples = data.shape[0]
    num_alleles = data.shape[1]
    
    counts = [0 for _ in range(num_alleles)]
    
    for i in range(num_alleles):
        for j in data[:,i]:
            if j == 1:
                counts[i] += 1
    
    freqs = [i/num_samples for i in counts]
    
    return freqs

allele_freqs = get_allele_freqs(omni_data[1][1])
    
#%%
full_sites = vcf_reader.read_sites_file("../pbwt_hkhan/vcf_data/omni4k-10.sites")
illu_sites = vcf_reader.read_sites_file("../pbwt_hkhan/vcf_data/illu1M.sites")

common_sites = sorted(list(set(full_sites[0]) & set(illu_sites[0])))

site_intersection = []

for item in common_sites:
    site_intersection.append(vcf_reader.ChromPos("20",item))

index_keep = []
positions_kept = []

for thing in site_intersection:
    index_keep.append(full_sites[1][thing.position])
    positions_kept.append(int(thing.position))

index_keep = sorted(index_keep)    
positions_kept = sorted(positions_kept)

set_keep = set(index_keep)

keep_flags = []

for i in range(len(omni_data[1][0])):
    if i in set_keep:
        keep_flags.append(1)
    else:
        keep_flags.append(0)
    
#%%
omni_pbwts = pbwt_methods.get_dual_spaced_pbwt(omni_data[1][1],keep_flags,100)
#reduced_pbwts = pbwt_methods.get_dual_spaced_pbwt(omni_other,keep_flags,100)
#%%
def seq_to_prediction_data(dual_spaced_pbwt,seq_to_insert):
    
    seq_copy = seq_to_insert.copy()
    
    masked_copy = seq_copy.copy()
    
    keep_sites = dual_spaced_pbwt.forward_pbwt.update_flags
    
    for i in range(len(seq_copy)):
        if keep_sites[i] != 1:
            masked_copy[i] = -1
    
    
    reversed_seq = seq_copy[::-1]
    
    forward_inserted = pbwt_methods.insert_place_spaced(dual_spaced_pbwt.forward_pbwt,seq_copy,keep_sites)
    
    backward_inserted = pbwt_methods.insert_place_spaced(dual_spaced_pbwt.backward_pbwt, reversed_seq, keep_sites[::-1])
    
    dual_insert_positions = (forward_inserted.insert_positions,backward_inserted.insert_positions[::-1])
    dual_insert_neighbours = (forward_inserted.insert_neighbours,backward_inserted.insert_neighbours[::-1])
    dual_insert_neighbours_distance = (forward_inserted.insert_neighbours_distance,backward_inserted.insert_neighbours_distance[::-1])
    
    return (masked_copy,pbwt_methods.DualInsertionData(dual_insert_positions, dual_insert_neighbours, dual_insert_neighbours_distance))
    

#%%

def prediction_data_to_matrix(prediction_data,allele_freqs):
    pred_list = []
    
    sequence = prediction_data[0]
    dual_inserted_vals = prediction_data[1]
    
    length = len(prediction_data[0])
    
    for i in range(length):
        
        freq = allele_freqs[i]
        
        # forward_lower_vals = prediction_data[2][0][i][0]
        # forward_upper_vals = prediction_data[2][0][i][1]
        
        # backward_lower_vals = prediction_data[2][1][i][0]
        # backward_upper_vals = prediction_data[2][1][i][1]
        
        forward_lower_vals = dual_inserted_vals.forward_insert_neighbours[i][0]
        forward_upper_vals = dual_inserted_vals.forward_insert_neighbours[i][1]
       
        backward_lower_vals = dual_inserted_vals.backward_insert_neighbours[i][0]
        backward_upper_vals = dual_inserted_vals.backward_insert_neighbours[i][1]
       
        
        corrected_forward_lower = [freq if x == -1 else x for x in forward_lower_vals]
        corrected_forward_upper = [freq if x == -1 else x for x in forward_upper_vals]
        
        corrected_backward_lower = [freq if x == -1 else x for x in backward_lower_vals]
        corrected_backward_upper = [freq if x == -1 else x for x in backward_upper_vals]
        
        
        # forward_lower_distances = prediction_data[3][0][i][0]
        # forward_upper_distances = prediction_data[3][0][i][1]
        
        # backward_lower_distances = prediction_data[3][1][i][0]
        # backward_upper_distances = prediction_data[3][1][i][1]
        
        forward_lower_distances = dual_inserted_vals.forward_insert_neighbours_distance[i][0]
        forward_upper_distances = dual_inserted_vals.forward_insert_neighbours_distance[i][1]
       
        backward_lower_distances = dual_inserted_vals.backward_insert_neighbours_distance[i][0]
        backward_upper_distances = dual_inserted_vals.backward_insert_neighbours_distance[i][1]
       
        
        
        list_seq_vals = corrected_forward_lower+ corrected_forward_upper+ corrected_backward_lower+ corrected_backward_upper

        list_seq_distances = forward_lower_distances+forward_upper_distances+backward_lower_distances+backward_upper_distances

        
        full_list = list_seq_vals+list_seq_distances
        
        pred_list.append(full_list)
        
    
    preds = []
    
    for i in range(len(pred_list)):
        preds.append(np.array(pred_list[i]))
            
    
        
    return np.array(preds,dtype=float)

def impute_sequence(seq,prediction_matrix,allele_freqs,bins,models):

    loaded_models = [xgb.Booster() for _ in range(len(models))]

    for i in range(len(models)):
        loaded_models[i].load_model(models[i])
    
    final_seq = []
    
    tot_len = prediction_matrix.shape[1]
    
    for j in range(len(seq)):
            
        if seq[j] != -1:
            final_seq.append(seq[j])
        else:
            
            freq = allele_freqs[j]
            
            bin_insert = np.searchsorted(bins,freq,side="right")-1
            
            mo = loaded_models[bin_insert]
        
            predict_data = prediction_matrix[j,:].reshape(1,tot_len)
            
            pred_value = mo.predict(xgb.DMatrix(predict_data))

                
            if pred_value > 0.5:
                final_seq.append(1)
            else:
                final_seq.append(0)
    
    return final_seq

def impute_full(dual_pbwt,seq_array,bins,models):
    
    models = [models[16] for _ in range(len(models))]
    
    final_array = []
    
    
    for i in range(len(seq_array)):
        pred_data = seq_to_prediction_data(dual_pbwt,seq_array[i])
        uniq,cts = np.unique(pred_data[0],return_counts=True)
        
        cts_dict = dict(zip(uniq,cts))
        
        pred_matrix = prediction_data_to_matrix(pred_data,dual_pbwt.forward_pbwt.allele_freqs)
        
        imputed = impute_sequence(pred_data[0],pred_matrix,dual_pbwt.forward_pbwt.allele_freqs,bins,models)
        
        final_array.append(imputed)
    
    return final_array


#%%
bins = [0,0.001,0.002,0.003,0.005,0.007,0.01,0.02,
 0.03,0.05,0.07,0.1,0.2,0.3,0.5,0.7,0.9,1.0001]

models = ["./xgboost_models/test_model_5.json",
          "./xgboost_models/test_model_5.json",
          "./xgboost_models/test_model_5.json",
          "./xgboost_models/test_model_5.json",
          "./xgboost_models/test_model_5.json",
          "./xgboost_models/test_model_5.json",
          "./xgboost_models/test_model_5.json",
          "./xgboost_models/test_model_3.json",
          "./xgboost_models/test_model_5.json",
          "./xgboost_models/test_model_7.json",
          "./xgboost_models/test_model_10.json",
          "./xgboost_models/test_model_20.json",
          "./xgboost_models/test_model_30.json",
          "./xgboost_models/test_model_50.json",
          "./xgboost_models/test_model_70.json",
          "./xgboost_models/test_model_90.json",
          "./xgboost_models/test_model_100.json",]
#%%


imputed = impute_full(omni_pbwts,omni_tests[1][1],bins,models)

#%%

impute5_data = vcf_reader.get_vcf_data("impute5/test3/sandbox/testing_full.vcf.gz")
#%%
pbwt_methods.compare_results(imputed,omni_tests[1][1],omni_pbwts.forward_pbwt.allele_freqs,
                omni_pbwts.forward_pbwt.update_flags,bins)
#%%

pbwt_methods.compare_results(impute5_data[1][1],omni_tests[1][1],omni_pbwts.forward_pbwt.allele_freqs,
                omni_pbwts.forward_pbwt.update_flags,bins)
#%%

imp0 = imputed[0]
omni0 = omni_tests[0]

for i in range(len(imp0)):
    if allele_freqs[i] > 0.7 or allele_freqs[i] < 0.5:
        continue
    if imp0[i] == 0 and omni0[i] == 1:
        print(i)
        
#%%
p = seq_to_prediction_data(omni_pbwts,omni0)
#%%
mo = xgb.Booster()
mo.load_model("./xgboost_models/test_model_30.json")
#%%
index = 188

d = p[1].forward_insert_neighbours[index][0] \
+ p[1].forward_insert_neighbours[index][1] \
+ p[1].backward_insert_neighbours[index][0] \
+ p[1].backward_insert_neighbours[index][1] \
+ p[1].forward_insert_neighbours_distance[index][0] \
+ p[1].forward_insert_neighbours_distance[index][1] \
+ p[1].backward_insert_neighbours_distance[index][0] \
+ p[1].backward_insert_neighbours_distance[index][1]

d = np.array(d).reshape((1,len(d)))
#%%
result = mo.predict(xgb.DMatrix(d))

print(f"{result[0]:.3}")
#%%

