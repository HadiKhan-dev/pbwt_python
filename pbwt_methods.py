import numpy as np
import msprime
import sklearn
import math
import random
import scipy.optimize
import sys
import log_res
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
class InsertionData:
    """
    Insertion data for a sequence through a PBWT
    """
    def __init__(self,insert_positions,insert_neighbours,insert_neighbours_distance):
        self.insert_positions = insert_positions
        self.insert_neighbours = insert_neighbours
        self.insert_neighbours_distance = insert_neighbours_distance

class DualInsertionData:
    """
    Insertion data for a sequence both going forward and backwards
    through a PBWT
    """
    
    def __init__(self,dual_insert_positions,dual_insert_neighbours,dual_insert_neighbours_distance):
        self.forward_insert_positions = dual_insert_positions[0]
        self.backward_insert_positions = dual_insert_positions[1]
        
        self.forward_insert_neighbours = dual_insert_neighbours[0]
        self.backward_insert_neighbours = dual_insert_neighbours[1]
        
        self.forward_insert_neighbours_distance = dual_insert_neighbours_distance[0]
        self.backward_insert_neighbours_distance = dual_insert_neighbours_distance[1]
#%%
class Pbwt:
    def __init__(self,alleles,count_list,occ_list,
                 fm_gap,allele_freqs,divergence_array = None):
        
        self.alleles = alleles
        self.count_list = count_list
        self.occ_list = occ_list
        self.num_samples = alleles.shape[1]
        self.num_sites = alleles.shape[0]
        self.fm_gap = fm_gap
        self.divergence_array = divergence_array
        self.allele_freqs = allele_freqs
        self.update_flags = [1 for _ in range(self.num_sites)]
        

class SpacedPbwt:
    def __init__(self,alleles,update_flags,count_list,occ_list,
                 fm_gap,allele_freqs,divergence_array = None):
        
        self.alleles = alleles
        self.update_flags = update_flags
        self.count_list = count_list
        self.occ_list = occ_list
        self.num_samples = alleles.shape[1]
        self.num_sites = alleles.shape[0]
        self.fm_gap = fm_gap
        self.divergence_array = divergence_array
        self.allele_freqs = allele_freqs
        
class DualPbwt:
    def __init__(self,forward,backward):
        self.forward_pbwt = forward
        self.backward_pbwt = backward

#%%
def degrade_data(data_array,degradation_rate):
    data_shape = data_array.shape
    
    random_data = np.random.binomial(1,degradation_rate,data_shape).astype("int8")

    degraded = np.logical_xor(data_array,random_data).astype("int8")
    
    return degraded

#%%

def pbwt(data_array,fm_gap=100):   
    
    """
    Computes the pbwt of the data with the given FM gap
    """
    
    size = data_array.shape
    
    M = size[0]
    N = size[1]
    
    
    ppa = list(range(M))

    div = [0 for _ in range(M)]
    
    ppa_list = []
    allele_list = []
    div_list = []
    count_list = []
    allele_freqs = []
    occ_list = []
    
    ppa_list.append(ppa)
    div_list.append(div)
    
    for i in range(N):
        
        # if i % 1000 == 0:
        #     print(i)
            
        alleles = []
        a = []
        b = []
        d = []
        e = []
        p = i+1
        q = i+1
        
        zero_count_val = 0
        occ_positions = [{-1:0},{-1:0}]
        
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
                occ_positions[0][ct] = zero_tot
                occ_positions[1][ct] = one_tot
            
            ct += 1
                
        ppa = a+b
        div = d+e
        
        #ppa_list.append(ppa)
        div_list.append(div)
        allele_list.append(np.array(alleles,dtype="int8"))
        count_list.append(zero_count_val)
        allele_freqs.append(1-(zero_count_val/M))
        occ_list.append(occ_positions)
        
    
    alleles_full = np.array(allele_list)
        
    return Pbwt(alleles_full,count_list,occ_list,fm_gap,allele_freqs,div_list)

def spaced_pbwt(data_array,update_flags,fm_gap=100):   
    
    """
    Computes the SpacedPBWT of the data with the given FM gap
    """
    
    
    size = data_array.shape
    
    M = size[0]
    N = size[1]
    
    
    ppa = list(range(M))

    div = [0 for _ in range(M)]
    
    ppa_list = []
    allele_list = []
    div_list = []
    count_list = []
    allele_freqs = []
    occ_list = []
    
    ppa_list.append(ppa)
    div_list.append(div)
        
    
    for i in range(N):
        
        if update_flags[i] != 1:
            alleles = []
            
            zero_count_val = 0
            occ_positions = [{-1:0},{-1:0}]
        
            zero_tot = 0
            one_tot = 0
        
            ct = 0
            
            for idx,pos in zip(ppa,div):
                cur_allele = data_array[(idx,i)]
                alleles.append(cur_allele)
                
                if cur_allele == 0:
                
                    zero_count_val += 1
                    zero_tot += 1
                
                
                if cur_allele == 1:
                    
                    one_tot += 1
                    
                    
                if ct % fm_gap == 0:
                    occ_positions[0][ct] = zero_tot
                    occ_positions[1][ct] = one_tot
            
                ct += 1
                
                
            
            div_list.append(div)
            allele_list.append(np.array(alleles,dtype="int8"))
            count_list.append(zero_count_val)
            allele_freqs.append(1-(zero_count_val/M))
            occ_list.append(occ_positions)
            
        else:
            
            alleles = []
            a = []
            b = []
            d = []
            e = []
            p = i+1
            q = i+1
        
            zero_count_val = 0
            occ_positions = [{-1:0},{-1:0}]
        
            zero_tot = 0
            one_tot = 0
        
        
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
                    zero_tot += 1
                
                
                if cur_allele == 1:
                    b.append(idx)
                    e.append(q)
                    q = 0
                    
                    one_tot += 1
                    
                if ct % fm_gap == 0:
                    occ_positions[0][ct] = zero_tot
                    occ_positions[1][ct] = one_tot
            
                ct += 1
                
            ppa = a+b
            div = d+e
        
            #ppa_list.append(ppa)
            div_list.append(div)
            allele_list.append(np.array(alleles,dtype="int8"))
            count_list.append(zero_count_val)
            allele_freqs.append(1-(zero_count_val/M))
            occ_list.append(occ_positions)
        
    
    alleles_full = np.array(allele_list)
    
    
        
    return SpacedPbwt(alleles_full,update_flags,
                      count_list,occ_list,fm_gap,
                      allele_freqs,div_list)



def get_dual_pbwt(reference_panel,fm_gap):
    """
    Computes a two way PBWT of a panel.
    """
    
    reverse_reference = np.fliplr(reference_panel)
    
    li = [reference_panel,reverse_reference]
    
    with multiprocessing.Pool(2) as pool:
        (forward_pbwt,reverse_pbwt) = pool.starmap(pbwt,zip(li,[fm_gap,fm_gap]))
    
    return DualPbwt(forward_pbwt,reverse_pbwt)

def get_dual_spaced_pbwt(reference_panel,update_site_flags,fm_gap):
    """
    Computes a two way SpacedPBWT of a panel.
    """
    
    reverse_reference = np.fliplr(reference_panel)
    
    li = [reference_panel,reverse_reference]
    
    with multiprocessing.Pool(2) as pool:
        (forward_pbwt,reverse_pbwt) = pool.starmap(spaced_pbwt,zip(li,[update_site_flags,update_site_flags[::-1]],[fm_gap,fm_gap]))
    
    return DualPbwt(forward_pbwt,reverse_pbwt)


def get_position(pbwt_data,i,location,val,is_update_col):
    """
    Helper function to get the updated position of a 
    sequence given its current position in the PBWT and
    the upcoming value in the sequence.

    """
    
    if not is_update_col:
        return location
    
    fm_gap = pbwt_data.fm_gap
    
    alleles = pbwt_data.alleles
    

    if val == 0:
        
        occ_index = pbwt_data.occ_list[i][0]
        cur_loc = location
        if cur_loc != -1:
            rem = cur_loc % fm_gap      
        
            check_data = alleles[i][cur_loc-rem+1:cur_loc+1]        
            tot_add = np.count_nonzero(check_data == 0)
            cur_loc -= rem
        
            final_position = tot_add+occ_index[cur_loc]-1
        else:
            final_position = occ_index[cur_loc]-1
    
    if val == 1:

        occ_index = pbwt_data.occ_list[i][1]
        cur_loc = location
        
        if cur_loc != -1:
            rem = cur_loc % fm_gap  
        
            check_data = alleles[i][cur_loc-rem+1:cur_loc+1]       
            tot_add = np.count_nonzero(check_data == 1)
            cur_loc -= rem
        
            final_position = tot_add+occ_index[cur_loc]+pbwt_data.count_list[i]-1
        else:
            final_position = occ_index[cur_loc]+pbwt_data.count_list[i]-1
    
    return final_position
        
def insert_place(pbwt_data,test_sequence,side_window_size=10,divergence_window_size=10):
    """
    Returns the positions a new test sequence would 
    insert into a PBWT, also returns a list of vectors
    showing how many positions the num_neighbours
    nearest neighbours on both sides have been moving
    with our sequence, as well as a list of the neighbouring
    values themselves
    """
    
    assert side_window_size >= divergence_window_size
    
    
    
    insert_positions = [len(pbwt_data.alleles[0])-1]
    insert_neighbours = []
    insert_neighbours_distance = [[[0 for _ in range(divergence_window_size)],[0 for _ in range(divergence_window_size)]]]
    
    for i in range(len(test_sequence)):
        cur_pos = insert_positions[-1]
        cur_val = test_sequence[i]

        next_pos = get_position(pbwt_data,i,cur_pos,cur_val,True)
        
        up_min = max(0,cur_pos-side_window_size+1)
        down_max = min(pbwt_data.num_samples,cur_pos+side_window_size+1)
        
        lower_pbwt_vals = pbwt_data.alleles[i][up_min:cur_pos+1][::-1]
        upper_pbwt_vals = pbwt_data.alleles[i][cur_pos+1:down_max]
        
        lower_pbwt_vals = list(lower_pbwt_vals)
        upper_pbwt_vals = list(upper_pbwt_vals)
        
        lower_pbwt_vals.extend([-1 for _ in range(side_window_size-len(lower_pbwt_vals))])
        upper_pbwt_vals.extend([-1 for _ in range(side_window_size-len(upper_pbwt_vals))])
        
        
        new_neighbours_lower = []
        new_neighbours_upper = []
        
        for j in range(min(divergence_window_size,len(lower_pbwt_vals))):
            if lower_pbwt_vals[j] == cur_val:
                new_neighbours_lower.append(insert_neighbours_distance[i][0][j]+1)
        
        for j in range(min(divergence_window_size,len(upper_pbwt_vals))):
            if upper_pbwt_vals[j] == cur_val:
                new_neighbours_upper.append(insert_neighbours_distance[i][1][j]+1)
        
        new_neighbours_lower.extend([0 for _ in range(divergence_window_size-len(new_neighbours_lower))])
        new_neighbours_upper.extend([0 for _ in range(divergence_window_size-len(new_neighbours_upper))])
        
        
        insert_positions.append(next_pos)
        insert_neighbours_distance.append([new_neighbours_lower,new_neighbours_upper])
        insert_neighbours.append([lower_pbwt_vals,upper_pbwt_vals])
        
    return InsertionData(insert_positions,insert_neighbours,insert_neighbours_distance)

def insert_place_spaced(pbwt_data,test_sequence,update_columns,side_window_size=10,divergence_window_size=10):
    """
    Returns the positions a new test sequence would 
    insert into a SpacedPBWT, also returns a list of vectors
    showing how many positions the num_neighbours
    nearest neighbours on both sides have been moving
    with our sequence, as well as a list of the neighbouring
    values themselves
    """
    
    assert side_window_size >= divergence_window_size
    
    
    
    insert_positions = [len(pbwt_data.alleles[0])-1]
    #insert_positions = [-1]
    insert_neighbours = []
    insert_neighbours_distance = [[[0 for _ in range(divergence_window_size)],[0 for _ in range(divergence_window_size)]]]
    
    for i in range(len(test_sequence)):
        
        cur_pos = insert_positions[-1]
        cur_val = test_sequence[i]

        if update_columns[i] == 1:
            next_pos = get_position(pbwt_data,i,cur_pos,cur_val,True)
        else:
            next_pos = get_position(pbwt_data,i,cur_pos,cur_val,False)
            
            
        up_min = max(0,cur_pos-side_window_size+1)
        down_max = min(pbwt_data.num_samples,cur_pos+side_window_size+1)
        
        lower_pbwt_vals = pbwt_data.alleles[i][up_min:cur_pos+1][::-1]
        upper_pbwt_vals = pbwt_data.alleles[i][cur_pos+1:down_max]
        
        
        lower_pbwt_vals = list(lower_pbwt_vals)
        upper_pbwt_vals = list(upper_pbwt_vals)
        
        lower_pbwt_vals.extend([-1 for _ in range(side_window_size-len(lower_pbwt_vals))])
        upper_pbwt_vals.extend([-1 for _ in range(side_window_size-len(upper_pbwt_vals))])
        
        new_neighbours_lower = []
        new_neighbours_upper = []
        
        if pbwt_data.update_flags[i] == 1:
        
            for j in range(min(divergence_window_size,len(lower_pbwt_vals))):
                if lower_pbwt_vals[j] == cur_val:
                    new_neighbours_lower.append(insert_neighbours_distance[i][0][j]+1)
        
            for j in range(min(divergence_window_size,len(upper_pbwt_vals))):
                if upper_pbwt_vals[j] == cur_val:
                    new_neighbours_upper.append(insert_neighbours_distance[i][1][j]+1)
        

            new_neighbours_lower.extend([0 for _ in range(divergence_window_size-len(new_neighbours_lower))])
            new_neighbours_upper.extend([0 for _ in range(divergence_window_size-len(new_neighbours_upper))])
        else:
            
            for j in range(divergence_window_size):
                if lower_pbwt_vals[j] != -1:
                    new_neighbours_lower.append(insert_neighbours_distance[i][0][j]+0)
                else:
                    new_neighbours_lower.append(0)
                
                if upper_pbwt_vals[j] != -1:
                    new_neighbours_upper.append(insert_neighbours_distance[i][1][j]+0)
                else:
                    new_neighbours_upper.append(0)
                

        insert_positions.append(next_pos)
        insert_neighbours_distance.append([new_neighbours_lower,new_neighbours_upper])
        insert_neighbours.append([lower_pbwt_vals,upper_pbwt_vals])
        
            
    return InsertionData(insert_positions,insert_neighbours,insert_neighbours_distance)


def extract_sequence(pbwt_data,extract_index,side_window_size=10,divergence_window_size=10):
    """
    Given a PBWT returns the sequence that was present at an
    index in the original data, also returns a list of this
    sequence's path through the PBWT and returns vectors
    showing how many positions the num_neighbours
    nearest neighbours on both sides have been moving
    with our sequence, as well as a list of the neighbouring
    values themselves
    """
    
    assert side_window_size >= divergence_window_size
    
    seq_data = []
    
    
    insert_positions = [extract_index]
    insert_neighbours = []
    insert_neighbours_distance = [[[0 for _ in range(divergence_window_size)],[0 for _ in range(divergence_window_size)]]]
    
    for i in range(pbwt_data.num_sites):
        cur_pos = insert_positions[-1]
        cur_val = pbwt_data.alleles[(i,cur_pos)]
        
        seq_data.append(cur_val)

        if pbwt_data.update_flags[i] == 1:
            next_pos = get_position(pbwt_data,i,cur_pos,cur_val,True)
        else:
            next_pos = get_position(pbwt_data,i,cur_pos,cur_val,False)
        
        up_min = max(0,cur_pos-side_window_size)
        down_max = min(pbwt_data.num_samples,cur_pos+side_window_size+1)
        
        lower_pbwt_vals = pbwt_data.alleles[i][up_min:cur_pos][::-1]
        upper_pbwt_vals = pbwt_data.alleles[i][cur_pos+1:down_max]
        
        lower_pbwt_vals = np.append(lower_pbwt_vals,[-1 for _ in range(side_window_size-len(lower_pbwt_vals))])
        upper_pbwt_vals = np.append(upper_pbwt_vals,[-1 for _ in range(side_window_size-len(upper_pbwt_vals))])
        
    
        new_neighbours_lower = []
        new_neighbours_upper = []
        
        
        if pbwt_data.update_flags[i] == 1:
        
            for j in range(min(divergence_window_size,len(lower_pbwt_vals))):
                if lower_pbwt_vals[j] == cur_val:
                    new_neighbours_lower.append(insert_neighbours_distance[i][0][j]+1)
        
            for j in range(min(divergence_window_size,len(upper_pbwt_vals))):
                if upper_pbwt_vals[j] == cur_val:
                    new_neighbours_upper.append(insert_neighbours_distance[i][1][j]+1)
        

            new_neighbours_lower.extend([0 for _ in range(divergence_window_size-len(new_neighbours_lower))])
            new_neighbours_upper.extend([0 for _ in range(divergence_window_size-len(new_neighbours_upper))])
        else:
            
            for j in range(divergence_window_size):
                if lower_pbwt_vals[j] != -1:
                    new_neighbours_lower.append(insert_neighbours_distance[i][0][j]+0)
                else:
                    new_neighbours_lower.append(0)
                
                if upper_pbwt_vals[j] != -1:
                    new_neighbours_upper.append(insert_neighbours_distance[i][1][j]+0)
                else:
                    new_neighbours_upper.append(0)
                    
        insert_positions.append(next_pos)
        insert_neighbours_distance.append([new_neighbours_lower,new_neighbours_upper])
        insert_neighbours.append([lower_pbwt_vals,upper_pbwt_vals])
        
    return (seq_data,InsertionData(insert_positions,insert_neighbours,insert_neighbours_distance))


def both_way_extract(dual_pbwt,extract_index,side_window_size=10,divergence_window_size=10):
    """
    Extract the neighbours of an element of our panel going both 
    forward and backwards, as well as an indicator of how long these
    neighbours have been moving with our sequence
    """
    forward_extract = extract_sequence(dual_pbwt.forward_pbwt,
                                       extract_index,side_window_size,divergence_window_size)
    backward_extract = extract_sequence(dual_pbwt.backward_pbwt,
                                         extract_index,side_window_size,divergence_window_size)
    
    sequence = forward_extract[0]
    
    
    dual_insert_positions = (forward_extract[1].insert_positions,backward_extract[1].insert_positions[::-1])
    dual_insert_neighbours = (forward_extract[1].insert_neighbours,backward_extract[1].insert_neighbours[::-1])
    dual_insert_neighbours_distance = (forward_extract[1].insert_neighbours_distance,backward_extract[1].insert_neighbours_distance[::-1])
    
    return (sequence,DualInsertionData(dual_insert_positions, dual_insert_neighbours, dual_insert_neighbours_distance))
    
    

def combine_single(alt_freqs,extracted_seq,frequency_bins,cutoff=10,side_window_size=10,divergence_window_size=10):
    """
    Bin extracted data from a single sequence into bins by
    alternate frequency as well as put it into a format suitable
    for a learning model
    """
    bin_seq = [[] for _ in range(len(frequency_bins))]
    
    bin_extract = [[[[] for _ in range(4*side_window_size)],[[] for _ in range(4*divergence_window_size)]] for _ in range(len(frequency_bins))]
    
    seq = extracted_seq[0]
    
    forward_seq_neighbours = extracted_seq[1].forward_insert_neighbours
    backward_seq_neighbours = extracted_seq[1].backward_insert_neighbours
    
    forward_seq_neighbours_divergence = extracted_seq[1].forward_insert_neighbours_distance
    backward_seq_neighbours_divergence = extracted_seq[1].backward_insert_neighbours_distance
    

    
    for idx in range(cutoff,len(seq)-cutoff):
        idx_freq = alt_freqs[idx]
        bin_number = bisect.bisect_left(frequency_bins,100*idx_freq)


        bin_seq[bin_number].append(seq[idx])
        
        for k in range(side_window_size):
            if forward_seq_neighbours[idx][0][k] != -1:
                bin_extract[bin_number][0][k].append(forward_seq_neighbours[idx][0][k])
            else:
                bin_extract[bin_number][0][k].append(idx_freq)
            
            if forward_seq_neighbours[idx][1][k] != -1:
                bin_extract[bin_number][0][side_window_size+k].append(forward_seq_neighbours[idx][1][k])
            else:
                bin_extract[bin_number][0][side_window_size+k].append(idx_freq)
                
            if backward_seq_neighbours[idx][0][k] != -1:
                bin_extract[bin_number][0][2*side_window_size+k].append(backward_seq_neighbours[idx][0][k])
            else:
                bin_extract[bin_number][0][2*side_window_size+k].append(idx_freq)
                
            if backward_seq_neighbours[idx][1][k] != -1:
                bin_extract[bin_number][0][3*side_window_size+k].append(backward_seq_neighbours[idx][1][k])
            else:
                bin_extract[bin_number][0][3*side_window_size+k].append(idx_freq)
                
        for k in range(divergence_window_size):
            bin_extract[bin_number][1][k].append(forward_seq_neighbours_divergence[idx][0][k])
            bin_extract[bin_number][1][divergence_window_size+k].append(forward_seq_neighbours_divergence[idx][1][k])
            bin_extract[bin_number][1][2*divergence_window_size+k].append(backward_seq_neighbours_divergence[idx][0][k])
            bin_extract[bin_number][1][3*divergence_window_size+k].append(backward_seq_neighbours_divergence[idx][1][k])
            
    for i in range(len(frequency_bins)):
        bin_seq[i] = np.array(bin_seq[i],dtype="int8")
        bin_extract[i][0] = np.array(bin_extract[i][0],dtype="float16")
        bin_extract[i][1] = np.array(bin_extract[i][1],dtype="int32")
        
    
    
    return (bin_seq,bin_extract)

def combine_extracted_datas(pbwt,extract_list, frequency_bins,
                            cutoff=10,side_window_size=10,
                            divergence_window_size=10):
    alt_freqs = []
    for i in range(pbwt.num_sites):
        alt_freqs.append(1-(pbwt.count_list[i]/pbwt.num_samples))
    
    
    bin_seq = [[] for _ in range(len(frequency_bins))]
    bin_extract = [[[[] for _ in range(4*side_window_size)],[[] for _ in range(4*divergence_window_size)]] for _ in range(len(frequency_bins))]
    
    buffer_bin_seq = [[] for _ in range(len(frequency_bins))]
    buffer_bin_extract = [[[[] for _ in range(4*side_window_size)],[[] for _ in range(4*divergence_window_size)]] for _ in range(len(frequency_bins))]
    
    itertimes = len(extract_list)
    
    with multiprocessing.Pool(7) as pool:
        singles = pool.starmap(combine_single,zip(itertools.repeat(alt_freqs,itertimes),extract_list,
                                                  itertools.repeat(frequency_bins,itertimes),
                                                  itertools.repeat(cutoff,itertimes),
                                                  itertools.repeat(side_window_size,itertimes),
                                                  itertools.repeat(divergence_window_size,itertimes)))

    for item in singles:
        for m in range(len(frequency_bins)):
            buffer_bin_seq[m].append(item[0][m])
            for s in range(4*side_window_size):
                buffer_bin_extract[m][0][s].append(item[1][m][0][s])
            for s in range(4*divergence_window_size):
                buffer_bin_extract[m][1][s].append(item[1][m][1][s])


            
    for m in range(len(frequency_bins)):
        if len(buffer_bin_seq[m]) > 0:
            bin_seq[m] = np.concatenate(buffer_bin_seq[m],axis=0)
            for s in range(4*side_window_size):
                bin_extract[m][0][s] = np.concatenate(buffer_bin_extract[m][0][s],axis=0)        
            for s in range(4*divergence_window_size):
                bin_extract[m][1][s] = np.concatenate(buffer_bin_extract[m][1][s],axis=0)        
            
    
    combined = list(zip(bin_seq,bin_extract))

    return combined

def create_feature_matrices(combined_datas):
    
    predictors = []
    
    actual = combined_datas[0]
    
    len_side = len(combined_datas[1][0])
    
    len_divergence = len(combined_datas[1][1])
    
    for i in range(len_side):
        predictors.append(combined_datas[1][0][i])
        
    for i in range(len_divergence):
        predictors.append(combined_datas[1][1][i])
    
    pred_matrix = np.array(predictors).transpose()

    return (pred_matrix,actual)

def run_log_res(combined,bucket_name,bucket_midpoint):

    if len(set(combined[0])) < 2:
        return (0,0)
    
    print("Bucket:",bucket_name)
    
    features = log_res.create_feature_matrix(
        combined[1],bucket_midpoint,width=10)
    
    pca_features = log_res.get_matrix_pca(features,10)
    
    t = log_res.logist_res(pca_features,combined[0])
    
    
    tot_string = "ID: "+str(bucket_name)+"\n"+str(t.intercept_)+"\n"+str(t.coef_)+"\n\n"

    sys.stdout.flush()
    
    return (t,tot_string)

def compare_results(imputed_data,true_data,allele_freqs, update_flags,bins):
    comps = [[[],[]] for _ in range(len(bins)-1)]
    c_nums = [[0,0,0,0,0,0,0,0,0] for _ in range(len(bins)-1)]
    
    for i in range(len(allele_freqs)):
        
        if update_flags[i] == 1:
            continue
        
        cur_freq = allele_freqs[i]
        
        cur_bin = np.searchsorted(bins,cur_freq,side="left")-1
        if cur_bin == -1:
            cur_bin = 0
        
        
        for seq_num in range(int(len(true_data)/2)):
            impute_first_val = imputed_data[2*seq_num][i]
            impute_second_val = imputed_data[2*seq_num+1][i]
            
            true_first_val = true_data[2*seq_num][i]
            true_second_val = true_data[2*seq_num+1][i]
            
            impute_sum = impute_first_val+impute_second_val
            true_sum = true_first_val+true_second_val
            
            comps[cur_bin][0].append(impute_sum)
            comps[cur_bin][1].append(true_sum)
            
            if impute_sum == 0 and true_sum == 0:
                c_nums[cur_bin][0] += 1
            if impute_sum == 0 and true_sum == 1:
                c_nums[cur_bin][1] += 1
            if impute_sum == 0 and true_sum == 2:
                c_nums[cur_bin][2] += 1
            if impute_sum == 1 and true_sum == 0:
                c_nums[cur_bin][3] += 1
            if impute_sum == 1 and true_sum == 1:
                c_nums[cur_bin][4] += 1
            if impute_sum == 1 and true_sum == 2:
                c_nums[cur_bin][5] += 1
            if impute_sum == 2 and true_sum == 0:
                c_nums[cur_bin][6] += 1
            if impute_sum == 2 and true_sum == 1:
                c_nums[cur_bin][7] += 1
            if impute_sum == 2 and true_sum == 2:
                c_nums[cur_bin][8] += 1
            
    
    tota = 0
    
    
    for j in range(len(comps)):
        bin_val = f"{100*bins[j+1]:.1f}"
        
        corr = np.corrcoef(comps[j][0],comps[j][1])
        
        rsq = corr[0,1]**1
        
        print(f"{bin_val} {c_nums[j]} {rsq :.3f}     {sum(c_nums[j])/5}")
        tota += sum(c_nums[j])
    
