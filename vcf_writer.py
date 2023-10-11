import numpy as np
import msprime
import sklearn
import math
import random
import scipy.optimize
import sys
from Bio import SeqIO
from fasta_parse import get_variations
import log_res
import pympler.asizeof as sizer
import vcf
import os
import pathlib

np.set_printoptions(threshold=sys.maxsize)

#%%

num_samples = 100
num_test = 10

ts = msprime.sim_ancestry(samples=num_samples, population_size=40000,
                          recombination_rate = 1.25*10**-8, ploidy=1,sequence_length=10**7)
mts = msprime.sim_mutations(ts, rate=1.25*10**-8)

bin_data = mts.genotype_matrix().transpose()

bin_data[bin_data > 1] = 1


np.random.shuffle(bin_data)

test_seqs = bin_data[:num_test]
test_seq = bin_data[0]


full_bin_data = bin_data

bin_data = bin_data[num_test:,:]

rev_bin_data = np.fliplr(bin_data)

bin_data = bin_data.transpose()
test_seqs = test_seqs.transpose()

sites = list([m.position for m in mts.sites()])

m = bin_data.shape[0]
n = bin_data.shape[1]
#%%

def write_sites(n,keep_percentage,path):
    file_to_write = open(path,"w")
    num_sample = math.floor(n*keep_percentage/100)
    keep_values = random.sample(list(range(n)),num_sample)
    
    keep_values = sorted(keep_values)
    print(len(keep_values))
    for val in keep_values:
        file_to_write.write("1	"+str(val)+"	A	C\n")
    file_to_write.close()

def write_vcf(data,path):
    
    m = data.shape[0]
    n = data.shape[1]
    num_diploid = int(n/2)
    
    pathlib.Path("new_test.vcf").unlink(missing_ok=True)
    pathlib.Path(path).unlink(missing_ok=True)

    file_to_write = open('new_test.vcf', 'w')
    file_to_write.write("##fileformat=VCFv4.2\n")
    file_to_write.write("#CHROM	POS ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	")

    for idx in range(num_diploid):
        file_to_write.write(str(idx)+"	")
    file_to_write.write("\n")
    file_to_write.close()

    read = vcf.Reader(open('new_test.vcf', 'r'))
    write = vcf.Writer(open(path, 'w'), read)

    call_tuple = vcf.model.make_calldata_tuple(["GT"])
                
    for i in range(m):
        data_row = data[i]
        sample_indexes = [str(i) for i in range(num_diploid)]
        num1 = 0
        sample_list = []
        
        for j in range(len(data_row)):
            thing = data_row[j]
            if thing == 1:
                num1 += 1
                
        for j in range(num_diploid):
            first_val = data_row[2*j]
            second_val = data_row[2*j+1]
            
            call_str = str(first_val)+"|"+str(second_val)
            call_data = call_tuple(call_str)
            call = vcf.model._Call(i,j,call_data)
            
            sample_list.append(call)
        info_dict = {}
        info_dict["AC"] = [num1]
        info_dict["AN"] = n
        
        alter = vcf.model._Substitution("C")
        new_row = vcf.model._Record("1", i, None, "A", [alter], None, [], info_dict, "GT", sample_indexes,samples=sample_list)
    
        write.write_record(new_row)
    #vcf.model._Record(CHROM, POS, ID, REF, ALT, QUAL, FILTER, INFO, FORMAT, sample_indexes)
    write.flush()
    
    wr
    
    pathlib.Path("new_test.vcf").unlink(missing_ok=True)
    
    

#%%
write_vcf(bin_data,"haploid100.vcf")
write_vcf(test_seqs,"haploid100_test.vcf")
#%%
write_sites(97841,100,"haploid10000_fullsites.sites")
write_sites(97841,28,"haploid10000_testsites.sites")
#%%
# read = vcf.Reader(open('omni10.vcf', 'r'))

# for record in read:
#     x = record.samples[0].data[0]
#     print(x)
#     print(type(x))
#     break



