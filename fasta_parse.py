from Bio import SeqIO
import numpy as np


def get_variations(fastas):
    """
    Returns a variation matrix for the sequences in the FASTA
    files passed in
    """
    var_files = []
    
    for file in fastas:
        read = SeqIO.parse(file,"fasta")
        for fs in read:
            #print(fs.id,str(fs.seq))
            var_files.append(fs.seq)
            
    max_len = 0
    for item in var_files:
        max_len = max(max_len,len(item))
            
    print(max_len)
    
    fin_mat = np.zeros((len(var_files),max_len))
    
    i_vals = [{} for _ in range(max_len)]
    
    for seq_idx in range(len(var_files)):
        seq = var_files[seq_idx]
        for i in range(len(seq)):
            curs = i_vals[i]
            cur_nuc = seq[i]
            
            if cur_nuc in curs.keys():
                fin_mat[(seq_idx,i)] = curs[cur_nuc]
            elif cur_nuc == "-":
                fin_mat[(seq_idx,i)] = "-"
            else:
                if curs == {}:
                    add_index = 0
                else:
                    add_index = max([curs[j] for j in curs.keys()])+1
                fin_mat[(seq_idx,i)] = add_index
                curs[cur_nuc] = add_index
            
    
    idxs = np.argwhere(np.all(fin_mat[...,:] == 0,axis=0))
    
    fin_mat = np.delete(fin_mat,idxs,axis=1)
    
    return fin_mat
