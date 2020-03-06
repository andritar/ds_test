import numpy as np
import pandas as pd

def _make_gen(reader):
    b = reader(1024 * 1024)
    while b:
        yield b
        b = reader(1024*1024)

def _get_num_rows(file_name):
    # ... Calculate number of rows in a file
    # ... input:  file_name as str - name of file to be calculated 
    # ... output: num_lines as int - number of lines (rows) in a file
    f = open(file_name, 'rb')
    f_gen = _make_gen(f.raw.read)
    num_lines=sum(buf.count(b'\n') for buf in f_gen )
    f.close()
    return num_lines

def _select_sample_lines(sample_size, file_lines_num):
    # ... Random mark of rows from dataset which shoul be selected
    # ... input:  sample_size    as int       - number of lines which should be selected
    # ...         file_lines_num as int       - number of all lines in a file
    # ... output: keep_lines     as [boolean] - array with shape equal to 'files_lines_num' where 'sample_size' items marked as True
    keep_lines=[False for x in range(file_lines_num)]
    keep_lines[0]=True
    keep_ix=np.random.choice(range(1,file_lines_num), sample_size, replace=False)
    for ix in keep_ix:
        keep_lines[ix]=True
    return keep_lines

def _get_sample(file_name):
    # ... Sample random selection: in case when dataset not more 500000 rows we use whole dataset
    # ...                       in case when dataset bigger then 500000 rows sample of 500000 items will be selected
    # ...                       (it is not high accurate way but much faster then another ones in case of big fitting datasets)
    # ... input:  file_name as str - name of fit file
    # ... output: sample    as str - data for fitting
    num_lines=_get_num_rows(file_name)
    sample=''
    max_sample=500000     # ... Ideally should be calculated based on free memory and file size
    with open(file_name,"r+") as f:
        clean_lines = (line.replace('\t',',') for line in f)
        if num_lines>max_sample:
            keep_lines=_select_sample_lines(max_sample, num_lines)
            sample=list()
            for is_keep, line in zip(keep_lines, clean_lines):
                if is_keep:
                    sample.append(line)
            sample='\n'.join(sample)
        else:
            sample='\n'.join(clean_lines)
    return sample

def _get_index(feature_vals):
    # ... Count location of max value in a list
    # ... input:  feature_vals as [int] - list of 257 numbers where 256 first values are related feature values per vacancy
    # ...                                 and last value is maximum value
    # ... output: counter      as int   - index of max value (if max value is related for two or more features then lowest index will be chosen)
    max_val=feature_vals[256]
    counter=1
    for val in feature_vals:
        if val==max_val:
            return counter
            break
        counter+=1
    else: 
        raise Exception('Max value was not found')


def _get_max_data(data, scaler):
    # ... Calculate index of max feature value and related absolute deviation
    # ... input:  data   as pd.DataFrame - dataFrame with feature data
    # ...         scaler as object       - already fitted scaler object
    # ... output: data   as pd.DataFrame - dataFrame with two columns which contain max feature index and related absolute deviation
    data['max_feature_2']=data.max(axis=1)
    data['max_feature_2_index']=data.apply(_get_index, axis=1)
    mean_vals=scaler.mean_vals.to_dict()
    data['max_val_feature']=data.max_feature_2_index.map(lambda x: 'feature_2_'+str(x))
    data['max_feature_2_abs_mean_diff']=(data.max_feature_2-data.max_val_feature.map(mean_vals)).round(2)
    data=data[['max_feature_2_index', 'max_feature_2_abs_mean_diff']]
    return data
    
def convert_data(data, scaler, features, file):
    # ... Convert data to output format amd append to a file
    # ... input: data     as pd.DataFrame - dataFrame with job id, feature code and 256 features
    # ...        scaler   as object       - already fitted scaler object
    # ...        features as [str]        - list of feature columns
    # ...        file     as file         - opened file object to append data
    max_data=_get_max_data(data[features], scaler)
    max_data=max_data.astype(str)
    scaled_data=scaler.transform(data[features]).round(5)
    scaled_data=scaled_data.astype(str)
    scaled_data_str=scaled_data.apply(lambda x: ','.join([y for y in x]), axis=1)
    del scaled_data
    res_data=data[['id_job']].astype(str)
    res_data['feature_2_stand']=scaled_data_str
    for x in list(max_data.columns):
        res_data[x]=max_data[x]

    res_data=res_data.apply(lambda x: '\t'.join([y for y in x]), axis=1)
    res_data='\n'.join(list(res_data))
    file.write(res_data)

def write_features(features, file):
    # ... Append feature names to a file
    # ... input: features as [str] - list of feature columns
    # ...        file     as file  - opened file object to append data
    features='\t'.join(features)
    file.write(features)

def write_data(file_name, scaler, features, input_cols):
    # ... Transform data and write them into a file 
    # ... input: file_name  as str          - file name of dataset to transform
    # ...        scaler     as object       - already fitted scaler object
    # ...        features   as [str]        - list of feature columns
    # ...        input_cols as [str]        - list of columns related to input dataframe
    num_lines=_get_num_rows(file_name)
    max_chunk_size=500000     # ... Ideally should be calculated based on free memory and file size
    res=open('test_proc.tsv',"w+")
    write_features(['id_job', 'feature_2_stand', 'max_feature_2_index', 'max_feature_2_abs_mean_diff'], res)
    
    with open(file_name,"r+") as f:
        clean_lines = (line.replace('\t',',') for line in f)
        chunk=list()
        cur_chunk_size=0
        is_first_chunk=1
        for line in clean_lines:
            chunk.append(line)
            cur_chunk_size+=1
            if cur_chunk_size==max_chunk_size:     # ... convert chunks and append them to a file
                data=pd.read_csv(pd.compat.StringIO('\n'.join(chunk)), sep=",", header=None, names=input_cols, skiprows=is_first_chunk)
                convert_data(data, scaler, features, res)
                print(str(data.shape[0])+'rows transformed')
                is_first_chunk=0
                cur_chunk_size=0
                chunk=list()
                
        
        if len(chunk)>0:     # ... convert last chunk and append it to a file
            data=pd.read_csv(pd.compat.StringIO('\n'.join(chunk)), sep=",", header=None, names=input_cols, skiprows=is_first_chunk)
            convert_data(data, scaler, features, res)
            print(str(data.shape[0])+'rows transformed')
            
        res.close()
        print('Done')

class Z_scaler():
    # ... Z-scale object
    # ... properties: mean_vals as pd.Series - mean values for each feature
    # ...             std_vals  as pd.Series - standard deviation values for each feature
    
    def __init__(self):
        # ... Initialize object
        mean_vals=pd.Series()
        std_vals=pd.Series()
    
    def fit(self, data):
        # ... Calculate mean and standard deviation values
        # ... input: data as pd.DataFrame - dataset to get mean and standard deviation values
        self.mean_vals=data.mean()
        self.std_vals=data.std()
        
    def transform(self, data):
        # ... Standardize data
        # ... input:  data as pd.DataFrame - dataset to standardized
        # ... output: data as pd.DataFrame - standardized data
        data=(data-self.mean_vals)/self.std_vals
        return data
    
    def fit_transform(self, data):
        # ... Calculate mean and standard deviation values and standardized data
        # ... input:  data as pd.DataFrame - dataset to get mean and standard deviation values and to be standardized
        # ... output: data as pd.DataFrame - standardized data
        self.fit(data)
        data=self.transform(data)
        return data

class Processing():
    # ... Object to process data
    # ... properties: scaler   as object - scaler to use to standardize data
    # ...             features as [str]  - list of features
    # ...             cols     as [str]  - list of columns related to input dataframe
    
    def __init__(self, feature_scaling_type='z_scale'):
        # ... Initialing
        # ... input: feature_scaling_type as str - type of scaler:
        # ...                                      z_scale - standard z-scaling
        self.features=['feature_2_'+str(x) for x in range(1,257)]
        self.cols=['id_job', 'feature_id']+self.features
        if feature_scaling_type=='z_scale':
            self.scaler=Z_scaler()
        else:
            raise Exception('Unknown feature scaling_type')
            
    def fit(self, file_name):
        # ... Fit scaler
        #... input: file_name as str - name of file used for fitting
        data=pd.read_csv(pd.compat.StringIO(_get_sample(file_name)), sep=",", header=None, names=self.cols, skiprows=1)
        self.scaler.fit(data[self.features])
        
    def transform(self, file_name):
        # ... Tranform data
        #... input: file_name as str - name of file used for scaling
        write_data(file_name, self.scaler, self.features, self.cols)
        
    def fit_tranform(self, file_name):
        # ... Tranform data
        #... input: file_name as str - name of file used for scaling and which will be transformed
        self.fit(file_name)
        self.transform(file_name)
