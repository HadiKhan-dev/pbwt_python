import pandas as pd
import numpy as np
import gzip
import csv


#%%
class ChromPos:
    def __init__(self,chromosome,position):
        self.chromosome = chromosome
        self.position = position
        
    def __eq__(self,other):
        if self.chromosome != other.chromosome:
            return False
        if self.position != other.position:
            return False
        return True
    
#%%
def get_vcf_names(vcf_path):
    """
    Gets the column names for the input VCF file
    """
    with gzip.open(vcf_path, "rt") as ifile:
          for line in ifile:
            if line.startswith("#CHROM"):
                  vcf_names = [x for x in line.split('\t')]
                  break
    ifile.close()
    vcf_names[-1] = vcf_names[-1][:-1]
    return vcf_names

def vcf_col_to_genotype_rows(vcf_col):
    
    genotype_data = vcf_col.str.split(":").apply(lambda x: x[0])
    
    full = genotype_data.str.split("|",expand=True)
    
    return [np.array([full[0]]).astype("int8"),np.array([full[1]]).astype("int8")]
    
def vcf_to_matrix(vcf_data):
    
    chroms = list(vcf_data["#CHROM"])
    positions = list(vcf_data["POS"])
    
    cols = list(vcf_data.columns)
    format_index = list(vcf_data.columns).index("FORMAT")
    cols = cols[format_index+1:]
    
    cat_list = []
    for col in cols:
        data_row = vcf_data[col]
        f = vcf_col_to_genotype_rows(data_row)
        cat_list.append(np.concatenate((f[0],f[1]),axis=0))
    
    final_mat = np.concatenate(cat_list,axis=0)
    
    marker_list = []
    
    for i in range(len(chroms)):
        marker_list.append(ChromPos(chroms[i],positions[i]))
    
    return (marker_list,final_mat)

def get_raw_vcf_data(file="../pbwt_hkhan/vcf_data/omni4k-10.vcf.gz"):
    """
    Extract raw data from a VCF file into a pandas DataFrame
    """
    names = get_vcf_names(file)
    
    vm = gzip.open(file)
    lines = vm.readlines()
    vm.close()
    
    comments = []
    
    for line in lines:
        if line[:2] == b"##":
            comments.append(line)

    vc = pd.read_csv(file,header=None,compression="gzip",comment="#",delim_whitespace=True,names=names)

    return (comments,vc)

def get_vcf_data(file="../pbwt_hkhan/vcf_data/omni4k-10.vcf.gz"):
    """
    Extract data from a VCF file into a numpy matrix

    """
    
    names = get_vcf_names(file)
    
    vm = gzip.open(file)
    lines = vm.readlines()
    vm.close()
    
    comments = []
    
    for line in lines:
        if line[:2] == b"##":
            comments.append(line)

    vc = pd.read_csv(file,header=None,compression="gzip",comment="#",delim_whitespace=True,names=names)


    base_matrix = vcf_to_matrix(vc)
    
    return (comments,base_matrix)

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
    
        