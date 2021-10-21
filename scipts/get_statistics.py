#get stat result
from pathlib import Path
import numpy as np
import os
import pickle as pkl
import sys
import pandas as pd
def pkl_save(data, file):
    with open(file, "wb") as f:
        pkl.dump(data, f)

def pkl_load(file):
    with open(file, "rb") as f:
        data = pkl.load(f)
    return data

def ann_stat(data_root1):
    dict1=dict()
    for fn in Path(data_root1).glob("*.ann"):
#        i+=1
       # print(fn.stem.split('_')[-1])
 #       file_ids.add(fn)
        fid=fn.stem
        if fid not in dict1.keys():
            dict1.update({fid:{}})
        with open(fn,'r') as f:
            lines=f.readlines()
       # if not lines:
       #     continue
        #else:
            for line in lines:
                line=line.strip()
                try:
                    ann_cate=line.split('\t')[1].split(' ')[0]
                    ann_res=line.split('\t')[2].split('\n')[0]
               # print(ann_cate)
               # print(ann_res)
                    if ann_cate not in dict1[fid].keys():
                        dict1[fid].update({ann_cate:[ann_res]})
                    else:
                        dict1[fid][ann_cate].append(ann_res)
                except:
               # print('except')
               # print(line)
                    continue
    return dict1

data_dir1=sys.argv[1]
data_dir2=sys.argv[2]


dict1=ann_stat(data_dir1)
dict2=ann_stat(data_dir2)

def find_agg_data(dict1):
    null_notes=set()
    notes=set()
    dict_agg=dict()
    for k,v in dict1.items():
        if len(v)==0:
            null_notes.add(k)
        else:
            notes.add(k)
            for k1,v1 in v.items():
                if k1 not in dict_agg.keys():
                    dict_agg.update({k1:set()})
                    dict_agg[k1].add(k)
                else:
                    dict_agg[k1].add(k)
    return null_notes,notes,dict_agg


pd_null,pd_pts,pd_dict_agg=find_agg_data(dict1)
gs_null,gs_pts,gs_dict_agg=find_agg_data(dict2)
sdoh_cate=sorted(list(pd_dict_agg.keys())+list(gs_dict_agg.keys()))
def find_agg_data_3(dict1):
    dict_agg=dict()
    for k in sdoh_cate:
        dict_agg.update({k:[]})
    for k,v in dict1.items():

        for sdoh_label in sdoh_cate:
            if sdoh_label not in v.keys():
                dict_agg[sdoh_label].append(0)
            else:
                dict_agg[sdoh_label].append(len(v[sdoh_label]))

    return dict_agg
pd_dict_2=find_agg_data_3(dict1)
gs_dict_2=find_agg_data_3(dict2)

data={'SDoH_cate':sorted(list(pd_dict_agg.keys())+list(gs_dict_agg.keys())+['null_note'])}
df=pd.DataFrame(data)
def count_pts(x,dict_agg):
    if x in dict_agg.keys():
        return len(dict_agg[x])
    else:
        return 0
def sum_pts_cate(x,dict_agg):
    if x in dict_agg.keys():
        return sum(dict_agg[x])
    else:
        return 0

df['concept_sum_pred']=df.apply(lambda x: sum_pts_cate(x['SDoH_cate'],pd_dict_2),axis=1)
df['concept_sum_ann']=df.apply(lambda x: sum_pts_cate(x['SDoH_cate'],gs_dict_2),axis=1)
df['notes_count_pred']=df.apply(lambda x: count_pts(x['SDoH_cate'],pd_dict_agg),axis=1)
df['notes_count_ann']=df.apply(lambda x: count_pts(x['SDoH_cate'],gs_dict_agg),axis=1)
df.loc[(df.SDoH_cate == 'null_note'),'notes_count_pred']=len(pd_null)
df.loc[(df.SDoH_cate == 'null_note'),'notes_count_pred']=len(gs_null)
Path('../results').mkdir(parents=True, exist_ok=True)
df.to_csv('../results/count_concepts.csv')
#print(df)