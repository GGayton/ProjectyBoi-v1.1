import h5py

def save_h5py_arrays(fileString, array, name = []):
    
    with h5py.File(fileString, 'w') as f:
        
        if name:
            f.create_dataset(name, data=array)
            
        else:
            f.create_dataset(data=array)
            
    f.close()

def num_of_keys(fileString, group =[]):
    
    with h5py.File(fileString, 'r') as f:
        
        if not group:
            key_list = list(f.keys())
        else:
            key_list = list(f[group].keys())
        
    f.close()
        
    return len(key_list)

def return_keys(fileString, group = []):
    with h5py.File(fileString, 'r') as f:
        
        if not group:
            key_list = list(f.keys())
        else:
            key_list = list(f[group].keys())
        
    f.close()
        
    return len(key_list)

def load_h5py_arrays(array, indices):
    
    if isinstance(indices, str):
        
        with h5py.File(array, 'r') as f:
            
            out = f[indices][()]
    
    elif isinstance(indices, tuple):
        out = []
        
        start = indices[0]
        end = indices[1]
        with h5py.File(array, 'r') as f:
            for i in range(start,end):
                
                string = "{:02d}".format(i)
                               
                out.append(f[string][()])
            
    elif isinstance(indices, list):
        
        out = []
        with h5py.File(array, 'r') as f:            
            for index in indices:
                
                string = "{:02d}".format(index)
                               
                out.append(f[string][()])
    
    elif isinstance(indices, int):
                        
        string = "{:02d}".format(indices)
        with h5py.File(array, 'r') as f:               
            out = f[string][()]
    
    return out
            
            