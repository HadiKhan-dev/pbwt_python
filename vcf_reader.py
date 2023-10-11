import pandas as pd
import numpy as np
import gzip
import csv

def get_vcf_names(vcf_path):
    with gzip.open(vcf_path, "rt") as ifile:
          for line in ifile:
            if line.startswith("#CHROM"):
                  vcf_names = [x for x in line.split('\t')]
                  break
    ifile.close()
    vcf_names[-1] = vcf_names[-1][:-1]
    return vcf_names

def vcf_col_to_genotype_rows(vcf_col):
    full = vcf_col.str.split("|",expand=True)
    
    return [np.array([full[0]]).astype("int8"),np.array([full[1]]).astype("int8")]
    
def vcf_to_matrix(vcf_data):
    positions = vcf_data["POS"]
    cols = list(vcf_data.columns)
    format_index = list(vcf_data.columns).index("FORMAT")
    cols = cols[format_index+1:]
    
    cat_list = []
    for col in cols:
        data_row = vcf_data[col]
    
        f = vcf_col_to_genotype_rows(data_row)
        cat_list.append(np.concatenate((f[0],f[1]),axis=0))
    
    final_mat = np.concatenate(cat_list,axis=0)
    return final_mat
    
def get_omni_data():
    file = "../pbwt_hkhan/vcf_data/omni4k-10.vcf.gz"
    names = get_vcf_names(file)
    vc = pd.read_csv(file,header=None,compression="gzip",comment="#",delim_whitespace=True,names=names)


    base_matrix = vcf_to_matrix(vc)
    
    return base_matrix

def get_vcf_data_from_file(file):
    names = get_vcf_names(file)
    vc = pd.read_csv(file,header=None,compression="gzip",comment="#",delim_whitespace=True,names=names)


    base_matrix = vcf_to_matrix(vc)
    
    return base_matrix

def read_sites_file(file):
    locations = []
    location_dict = {}
    i = 0
    
    with open(file,"r",encoding="utf8") as data:
        tsv_reader = csv.reader(data, delimiter="\t")
        
        for row in tsv_reader:
            (chromosome, location, _,_) = row
            
            locations.append(location)
            location_dict[location] = i
            i += 1
    return (locations,location_dict)
    
        