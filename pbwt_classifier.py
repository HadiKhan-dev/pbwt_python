import numpy as np
import msprime
import sklearn
import math
import random
import scipy.optimize
import sys
import log_res_classifier
import pympler.asizeof as sizer
import time
import bisect
import multiprocessing
import pickle 
import itertools
import psutil
import os
import functools

np.set_printoptions(threshold=sys.maxsize)

#%%
class Pbwt:
    def __init__(self,alleles,count_list,occ_list,fm_gap):
        self.alleles = alleles
        self.count_list = count_list
        self.occ_list = occ_list
        self.num_samples = alleles.shape[0]
        self.num_sites = alleles.shape[1]
        self.fm_gap = fm_gap

class DualPbwt:
    def __init__(self,forward,backward):
        self.forward_pbwt = forward
        self.backward_pbwt = backward

#%%

def gen_data(num_samples):
    ts = msprime.sim_ancestry(samples=num_samples, population_size=40000,
                              recombination_rate = 1.25*10**-8, ploidy=1,sequence_length=10**7)
    mts = msprime.sim_mutations(ts, rate=1.25*10**-8)

    
    bin_data = mts.genotype_matrix().transpose()

    bin_data[bin_data > 1] = 1
    
    np.random.shuffle(bin_data)
    
    return bin_data
    
    
def degrade_data(data_array,degradation_rate):
    data_shape = data_array.shape
    
    random_data = np.random.binomial(1,degradation_rate,data_shape).astype("int8")

    degraded = np.logical_xor(data_array,random_data).astype("int8")
    
    return degraded


#%%
ft = time.time()
st = time.time()


bin_data = gen_data(100)

test_data = bin_data[80:,:]
bin_data = bin_data[:80,:]


bin_data = degrade_data(bin_data,0.00)

rev_bin_data = np.fliplr(bin_data)

process = psutil.Process(os.getpid())
print("Tot mem usage: ", process.memory_info().rss)

#%%

def pbwt(data_array,fm_gap=1):
    size = data_array.shape
    
    M = size[0]
    N = size[1]
    
    
    ppa = list(range(M))

    div = [0 for _ in range(M)]
    
    ppa_list = []
    allele_list = []
    div_list = []
    count_list = []
    space_occ_list = []
    
    ppa_list.append(ppa)
    div_list.append(div)
    
    for i in range(N):
        alleles = []
        a = []
        b = []
        d = []
        e = []
        p = i+1
        q = i+1
        
        zero_count_val = 0
        space_occ_positions = [{-1:0},{-1:0}]
        
        zero_tot = 0
        one_tot = 0
        
        zero_occ_val = 0
        one_occ_val = 0
        
        ct = 0
        
        for idx,pos in zip(ppa,div):
            cur_allele = data_array[(idx,i)]
            alleles.append(cur_allele)
            
            if pos > p:
                p = pos
            if pos > q:
                q = pos
                
            if cur_allele == 0:
                a.append(idx)
                d.append(p)
                p = 0
                
                zero_count_val += 1
                zero_occ_val += 1
                zero_tot += 1
                
                if zero_occ_val == fm_gap:
                    zero_occ_val = 0
                
            if cur_allele == 1:
                b.append(idx)
                e.append(q)
                q = 0
                
                one_occ_val += 1
                one_tot += 1
                
                if one_occ_val == fm_gap:
                    one_occ_val = 0
                    
            if ct % fm_gap == 0:
                space_occ_positions[0][ct] = zero_tot
                space_occ_positions[1][ct] = one_tot
            
            ct += 1
                
        ppa = a+b
        div = d+e
        
        #ppa_list.append(ppa)
        #div_list.append(div)
        allele_list.append(np.array(alleles,dtype="int8"))
        count_list.append(zero_count_val)
        #occ_list.append(occ_positions)
        space_occ_list.append(space_occ_positions)
        
    
    alleles_full = np.array(allele_list).transpose()
        
    return Pbwt(alleles_full,count_list,space_occ_list,fm_gap)

def get_position(pbwt_data,i,location,val):
    
    fm_gap = pbwt_data.fm_gap
    
    alleles = pbwt_data.alleles
    

    if val == 0:
        
        occ_index = pbwt_data.occ_list[i][0]
        cur_loc = location
        rem = cur_loc % fm_gap      
        
        check_data = alleles[cur_loc-rem+1:cur_loc+1,i]        
        tot_add = np.count_nonzero(check_data == 0)
        cur_loc -= rem
        
        final_position = tot_add+occ_index[cur_loc]-1
        
    
    if val == 1:
        occ_index = pbwt_data.occ_list[i][1]
        cur_loc = location
        rem = cur_loc % fm_gap  
        
        check_data = alleles[cur_loc-rem+1:cur_loc+1,i]        
        tot_add = np.count_nonzero(check_data == 1)
        cur_loc -= rem
        
        final_position = tot_add+occ_index[cur_loc]+pbwt_data.count_list[i]-1
    
    return final_position
        
def insert_place(pbwt_data,test_sequence):
    
    insert_positions = [len(pbwt[0])-1]
    
    for i in range(len(test_sequence)):
        cur_pos = insert_positions[-1]
        cur_val = test_sequence[i]

        next_pos = get_position(pbwt_data,i,cur_pos,cur_val)
        insert_positions.append(next_pos)
    
    return insert_positions

def get_dual_pbwt(reference_panel,fm_gap):
    
    reverse_reference = np.fliplr(reference_panel)
    
    li = [reference_panel,reverse_reference]
    
    def helper(panel):
        return pbwt(panel,fm_gap)
    
    with multiprocessing.Pool(2) as pool:
        (forward_pbwt,reverse_pbwt) = pool.starmap(pbwt,zip(li,[fm_gap,fm_gap]))
    #forward_pbwt = pbwt(reference_panel,fm_gap)
    #reverse_pbwt = pbwt(reverse_reference,fm_gap)
    
    return DualPbwt(forward_pbwt,reverse_pbwt)

def extract_side_data(pbwt,extract_seq,window_size=3):
    n = pbwt.num_samples
    m = pbwt.num_sites
    
    alleles = pbwt.alleles
    
    seq_data = []
    extracted_data = [[],[]]
    
    new_extracted_data = [[] for i in range(2*window_size)]
    
    cur_pos = extract_seq
    
    for j in range(m):
        
        
        lower_bound = max(0,cur_pos-window_size)
        upper_bound = min(n,cur_pos+window_size+1)
        
        lower_diff = cur_pos-lower_bound
        
        all_vals = alleles[lower_bound:upper_bound,j]
        
        val = all_vals[lower_diff]
        
        lower_vals = all_vals[:lower_diff]
        upper_vals = all_vals[lower_diff+1:]
        
        lower_vals = lower_vals[::-1]
        
        for i in range(window_size):
            if i < len(lower_vals):
                new_extracted_data[i].append(lower_vals[i])
            else:
                new_extracted_data[i].append(-1)
                
            if i < len(upper_vals):
                new_extracted_data[i+window_size].append(upper_vals[i])
            else:
                new_extracted_data[i+window_size].append(-1)
        
        
        seq_data.append(val)
        extracted_data[0].append(lower_vals)
        extracted_data[1].append(upper_vals)
        
        
        cur_pos = get_position(pbwt,j,cur_pos,val)
    
    seq_data = np.array(seq_data,dtype="int8")
    for i in range(len(new_extracted_data)):
        new_extracted_data = np.array(new_extracted_data,dtype="int8")
        
    return [seq_data,new_extracted_data]

def both_way_extract(dual_pbwt,extract_seq,window_size=3):
    
    forward_extract = extract_side_data(dual_pbwt.forward_pbwt,extract_seq,window_size)
    backward_extract = extract_side_data(dual_pbwt.backward_pbwt,extract_seq,window_size)

    backward_extract[0] = backward_extract[0][::-1]
    for i in range(len(backward_extract[1])):
        backward_extract[1][i] = backward_extract[1][i][::-1]
    

    together = (forward_extract[0],(forward_extract[1],backward_extract[1]))
    del forward_extract
    del backward_extract
    return together


def combine_single(alt_freqs,extract_list,frequency_bins,cutoff=10,window_size=3):
    
    bin_final_seq = [[] for _ in range(len(frequency_bins))]
    new_bin_final_extract = [[[] for w in range(4*window_size)] for _ in range(len(frequency_bins))]
    
    seq = extract_list[0]
    extract = extract_list[1]
    
    forward_extract = extract[0]
    backward_extract = extract[1]
    
    side_size = int(len(forward_extract)/2)


    
    for idx in range(cutoff,len(seq)-cutoff):
        idx_freq = alt_freqs[idx]
        bin_number = bisect.bisect_left(frequency_bins,100*idx_freq)


        bin_final_seq[bin_number].append(seq[idx])
        
        for k in range(window_size):
            if k < side_size:
                new_bin_final_extract[bin_number][k].append(forward_extract[k][idx])
                new_bin_final_extract[bin_number][window_size+k].append(forward_extract[side_size+k][idx])
                new_bin_final_extract[bin_number][2*window_size+k].append(backward_extract[k][idx])
                new_bin_final_extract[bin_number][3*window_size+k].append(backward_extract[side_size+k][idx])
            else:
                new_bin_final_extract[bin_number][k].append(-1)
                new_bin_final_extract[bin_number][window_size+k].append(-1)
                new_bin_final_extract[bin_number][2*window_size+k].append(-1)
                new_bin_final_extract[bin_number][3*window_size+k].append(-1)
    
    
    for i in range(len(frequency_bins)):
        bin_final_seq[i] = np.array(bin_final_seq[i],dtype="int8")
        new_bin_final_extract[i] = np.array(new_bin_final_extract[i],dtype="int8")
    
    return (bin_final_seq,new_bin_final_extract)

def combine_extracted_datas(pbwt,extract_list, frequency_bins,
                            cutoff=10,window_size=3):
    alt_freqs = []
    for i in range(pbwt.num_sites):
        alt_freqs.append(1-(pbwt.count_list[i]/pbwt.num_samples))
    
    final_bin_final_seq = [[] for _ in range(len(frequency_bins))]
    final_new_bin_final_extract = [[[] for w in range(4*window_size)] for _ in range(len(frequency_bins))]
    
    buffer_bin_final_seq = [[] for _ in range(len(frequency_bins))]
    buffer_new_bin_final_extract = [[] for _ in range(len(frequency_bins))]
    
    itertimes = len(extract_list)
    
    with multiprocessing.Pool(7) as pool:
        singles = pool.starmap(combine_single,zip(itertools.repeat(alt_freqs,itertimes),extract_list,
                                                  itertools.repeat(frequency_bins,itertimes),
                                                  itertools.repeat(cutoff,itertimes),
                                                  itertools.repeat(window_size,itertimes)))

    for item in singles:
        for m in range(len(frequency_bins)):
            buffer_bin_final_seq[m].append(item[0][m])
            buffer_new_bin_final_extract[m].append(item[1][m])


            
    for m in range(len(frequency_bins)):
        if len(buffer_bin_final_seq[m]) > 0:
            final_bin_final_seq[m] = np.concatenate(buffer_bin_final_seq[m],axis=0)
            final_new_bin_final_extract[m] = np.concatenate(buffer_new_bin_final_extract[m],axis=1)        
            
    
    combined = list(zip(final_bin_final_seq,final_new_bin_final_extract))

    return combined


    

    
#%%
#full_pb = pbwt(full_bin_data,100)

st = time.time()

du = get_dual_pbwt(bin_data,10)
test_du = get_dual_pbwt(test_data,10)

print("PBWT Done; PBWT Size: ", sizer.asizeof(du))

process = psutil.Process(os.getpid())
print("Tot mem usage: ", process.memory_info().rss)
#%%
extract_raw = []
test_extract_raw = []

buckets = [0.1,0.2,0.3,0.5,0.7,1,2,3,5,7,10,20,30,50,70,90,100]
#buckets = [100]

def get_ith_extract(i):
    return both_way_extract(du,i,window_size=10)

with multiprocessing.Pool(7) as pool:
    extract_raw = pool.map(get_ith_extract,list(range(du.forward_pbwt.num_samples)))
with multiprocessing.Pool(7) as pool:
    test_extract_raw = pool.map(get_ith_extract,list(range(test_du.forward_pbwt.num_samples)))


s = combine_extracted_datas(du.forward_pbwt,extract_raw,
                            buckets,window_size=10)

test_s = combine_extracted_datas(test_du.forward_pbwt,test_extract_raw,
                            buckets,window_size=10)
del extract_raw
del test_extract_raw

#%%

write_list = []
def run_log_res(combined,bucket_name,bucket_midpoint,test_data=None):

    if len(set(combined[0])) < 2:
        return (0,[],[],[])
    
    #print("Bucket:",bucket_name)
    
    features = log_res_classifier.create_feature_matrix(
        combined[1],bucket_midpoint,width=10)
    
    (pca_features,pca) = log_res_classifier.get_matrix_pca(features,10)
    
    
    
    t = log_res_classifier.logist_res(pca_features,combined[0])
    
    ones = []
    totals = []
    ratios = []
    
    if test_data != None and test_data[1].shape[1] != 0:
        
        test_features = log_res_classifier.create_feature_matrix(
            test_data[1],bucket_midpoint,width=10)
        
        predictions = t.predict_proba(pca.transform(test_features))
        
        actuals = test_data[0]
        
        predict_bins = [0.1,0.2,0.3,0.5,0.7,1,2,3,5,7,10,20,30,50,70,90,100]
        predict_mids = []
        
        for i in range(len(predict_bins)):
            if i == 0:
                predict_mids.append(predict_bins[i]/2)
            else:
                predict_mids.append((predict_bins[i-1]+predict_bins[i])/2)
                
        ones = [0 for _ in range(len(predict_bins))]
        totals = [0 for _ in range(len(predict_bins))]
        ratios = []
        
        for r in range(predictions.shape[0]):
            row = predictions[r,:]
            one_prob = row[1]
            location = bisect.bisect_left(predict_bins, 100*one_prob)
            
            if actuals[r] == 1:
                ones[location] += 1
            totals[location] += 1
            
        for i in range(len(ones)):
            if totals[i] != 0:
                ratios.append(100*ones[i]/totals[i])
            else:
                ratios.append(float("nan"))
        print("The bucket:",bucket_name)
        print(list(zip(predict_mids,ones,totals,ratios)))   
        print()
        
    sys.stdout.flush()
    
    return (t,ones,totals,ratios)

def evaluate(func,arg1,arg2,arg3):
    return func(bucket_name=arg1,bucket_midpoint=arg2,test_data=arg3)

run_args = []
eval_funcs = []
bucket_mids = []

for i in range(len(s)):
    eval_funcs.append(functools.partial(run_log_res,combined=s[i]))
    if i == 0:
        bucket_mids.append(buckets[i]/200)
    else:
        bucket_mids.append((buckets[i]+buckets[i-1])/200)
        
run_args.extend(zip(eval_funcs,buckets,bucket_mids,test_s))

with multiprocessing.Pool(4) as pool:
    results = pool.starmap(evaluate,run_args)
    
#%%
    
ones = [0 for _ in range(len(results[8][1]))]
totals = [0 for _ in range(len(results[8][1]))]
ratios = []

print(len(results))
for item in results:
    for i in range(len(item[1])):
        ones[i] += item[1][i]
        totals[i] += item[2][i]

for i in range(len(ones)):
    if totals[i] != 0:
        ratios.append(100*ones[i]/totals[i])
    else:
        ratios.append(float("nan"))

buk = [100*i for i in bucket_mids]
print(list(zip(buk,ratios)))
print("Full time: ", time.time()-ft)


# %%
