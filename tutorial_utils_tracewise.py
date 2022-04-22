import numpy as np
import itertools


def add_trace_wise_noise(d,trace_num,trace_value):  
    alldata=[]
    for k in range(len(d)):    
        clean=d[k]    
        data=np.ones([10,512,256])
        for i in range(len(data)):    
            corr=np.random.randint(0,256,trace_num) 
            data[i]=clean.copy()
            data[i,:,corr]=np.ones([1,512])*trace_value
        alldata.append(data)
    alldata=np.array(alldata) 
    alldata=alldata.reshape(4040,512,256)
    print(alldata.shape)

    return alldata

