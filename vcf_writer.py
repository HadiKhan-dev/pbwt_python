import numpy as np
import msprime
import sklearn
import math
import random
import scipy.optimize
import sys
import log_res
import pympler.asizeof as sizer
import vcf
import os
import pathlib
import gzip
import Bio.bgzf

import vcf_reader

np.set_printoptions(threshold=sys.maxsize)

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
    "Write a VCF file to the path given"
    
    if path[-3:] == ".gz":
        compressed = True
        full_path = path
        path = path[:-3]
        
    
    m = data.shape[1]
    n = data.shape[0]
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
        data_row = data[:,i]
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
    write.close()
    
    pathlib.Path("new_test.vcf").unlink(missing_ok=True)
    
    if compressed:
        with open(path,"rb") as op:
              with Bio.bgzf.BgzfWriter(full_path,"wb") as compressed_file:
                  
                  read_all = op.read()
                  
                  print(read_all)
                  
                  compressed_file.write(read_all)
                  
        pathlib.Path(path).unlink(missing_ok=True)
        
    
    
def filter_vcf(vcf_path,keep_locations,save_path):
    """
    Filters a VCF to keep only those rows which are in 
    the keep_locations and saves it at save_path
    """
    
    def test_loc_in_set(row,location_set):
        
        if (str(row["#CHROM"]),str(row["POS"])) in location_set:

            return True
        return False
    
    if save_path[-3:] == ".gz":
        compressed = True
        full_path = save_path
        save_path = save_path[:-3]
    
    loc_set = set([])
    
    for item in keep_locations:
        loc_set.add((item.chromosome,item.position))
    
    names = vcf_reader.get_vcf_names(vcf_path)
    
    format_index = names.index("FORMAT")
    
    sample_names = names[format_index+1:]
    

    vm = gzip.open(vcf_path)
    lines = vm.readlines()
    vm.close()
    
    comments = []
    
    for line in lines:
        if line[:2] == b"##":
            comments.append(line)
    comments.append(b"##File filtered to keep only a portion of sites\n")
    
    pathlib.Path("intermediate.vcf").unlink(missing_ok=True)
    
    file_to_write = open("intermediate.vcf", 'w')    
    
    for comment in comments:
        file_to_write.write(comment.decode("utf-8"))
    
    for name in names:
        file_to_write.write(str(name)+"	")
    file_to_write.write("\n")
    file_to_write.close()
    
    vcf_df = vcf_reader.get_raw_vcf_data(vcf_path)[1]
    
    filtered_df = vcf_df[vcf_df.apply(lambda x: test_loc_in_set(x,loc_set),axis=1)]
    
    read = vcf.Reader(open("intermediate.vcf", 'r'))
    write = vcf.Writer(open(save_path, 'w'), read)
    
    call_tuple = vcf.model.make_calldata_tuple(["GT"])
    
    for index,row in filtered_df.iterrows():
        
        alter = vcf.model._Substitution(row["ALT"])
        
        sample_vals = []
        
        for name in sample_names:
            
            call_str = row[name]
            call_data = call_tuple(call_str)
            call = vcf.model._Call(row["POS"],name,call_data)
            
            sample_vals.append(call)
        
        info_dict = {}
        
        for info in row["INFO"].split(";"):
            key,value = info.split("=")
            if key == "AC":
                info_dict[key] = [value]
            else:
                info_dict[key] = value
            
        
        new_row = vcf.model._Record(row["#CHROM"],row["POS"], None, row["REF"], [alter], None, row["FILTER"], info_dict, row["FORMAT"], sample_names,samples=sample_vals)
    
        write.write_record(new_row)
        
    write.flush()
    write.close()
    pathlib.Path("intermediate.vcf").unlink(missing_ok=True)
    
    if compressed:
        with open(save_path,"rb") as op:
              with Bio.bgzf.BgzfWriter(full_path,"wb") as compressed_file:
                  
                  read_all = op.read()
                  
                  print(read_all)
                  
                  compressed_file.write(read_all)
                  
        pathlib.Path(save_path).unlink(missing_ok=True)
        
        


