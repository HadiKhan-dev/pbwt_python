
from scipy.optimize import Bounds
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
#from clogistic import LogisticRegression as cLogisticRegression

#from sklearn.linear_model import LogisticRegression
import numpy as np
#import pandas as pd

def create_feature_matrix(extracted_data,missing_replace,width=3):
    
    new_extracted_data = [[] for i in range(len(extracted_data))]
    for i in range(len(extracted_data)):
        new_extracted_data[i] = extracted_data[i].astype("float16")
        #print("Shape:",new_extracted_data[i].shape)
    for item in new_extracted_data:
        item[item == -1] = missing_replace

    side_length = int(len(extracted_data)/4)
    use_width = min(width,side_length)
    new_actual_final = [[] for _ in range(use_width)]
    
    for j in range(use_width):
        new_actual_final[j] = new_extracted_data[j]+ \
        new_extracted_data[side_length+j]+ \
        new_extracted_data[2*side_length+j]+ \
        new_extracted_data[3*side_length+j]
            
        new_actual_final[j] = new_actual_final[j]/4
    
    np_mat = np.array(new_actual_final).transpose()
    
    #np_mat = np_mat[:,np.r_[columns]]
    
    return np_mat

def get_matrix_pca(feature_matrix,num_components):
    pca = PCA(num_components)
    
    first_row = feature_matrix[0,:]
    
    L = pca.fit_transform(feature_matrix)
    
    P = pca.components_
    
    L_first = L[0,:]
    
    #print("Mean")
    #print(repr(pca.mean_))
    #print("Components")
    #print(repr(P))
    #print("Variance Explained Ratio: ")
    #print(repr(pca.explained_variance_ratio_))
    

    return (L,pca)
    
def logist_res(predictors,observed):
    
    shape = predictors.shape
    
    bound_list = np.concatenate([np.zeros(shape[1]),[-np.inf]])
    bounds = Bounds(lb=bound_list,ub=[np.inf for i in range(len(bound_list))])
    
    #new_clf = cLogisticRegression(penalty="elasticnet",l1_ratio=0.8,solver="ecos",max_iter=1000).fit(predictors,observed,bounds=bounds)
    clf = LogisticRegression(random_state=0,penalty=None,solver="saga",max_iter=1000).fit(predictors, observed)
    
    #print("Intercept: ")
    #print(repr(clf.intercept_))
    #print("Coefficients: ")
    #print(repr(clf.coef_))

    print()
    
    return clf