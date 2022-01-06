run NER
import sys
sys.path.append("../ClinicalTransformerNER/")
sys.path.append("../NLPreprocessing/")
import os
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
from sklearn.model_selection import train_test_split
import shutil 
import fileinput
from annotation2BIO import generate_BIO, pre_processing, read_annotation_brat, BIOdata_to_file
MIMICIII_PATTERN = "\[\*\*|\*\*\]"

#check number of ann in 150/50 split
train_dev_root1 = Path('../data/training_set_150')
test_root1 = Path('../data/test_set_150')
#data stat
file_ids = set()
enss = []

for fn in test_root1.glob("*.ann"):
    file_ids.add(fn.stem)
    _, ens, _ = read_annotation_brat(fn)
    #print( _)
    enss.extend(ens)
print("150 files as training, test files: ", len(file_ids), list(file_ids)[:5])
print("150 files as training, total test eneitites: ", len(enss))
print("Entities distribution by types:\n", "\n".join([str(c) for c in Counter([each[1] for each in enss]).most_common()]))


file_ids = set()
enss = []

for fn in train_dev_root1.glob("*.ann"):
    file_ids.add(fn.stem)
    _, ens, _ = read_annotation_brat(fn)
    #print( _)
    enss.extend(ens)
print("150 files as training, training files: ", len(file_ids), list(file_ids)[:5])
print("150 files as training, total training eneitites: ", len(enss))
print("Entities distribution by types:\n", "\n".join([str(c) for c in Counter([each[1] for each in enss]).most_common()]))

#check ann in 100/100 split

train_dev_root2 = Path('../data/training_set_100')
test_root1 = Path('../data/test_set_100')
#data stat
file_ids = set()
enss = []

for fn in test_root2.glob("*.ann"):
    file_ids.add(fn.stem)
    _, ens, _ = read_annotation_brat(fn)
    #print( _)
    enss.extend(ens)
print("100 files as training, test files: ", len(file_ids), list(file_ids)[:5])
print("100 files as training, total test eneitites: ", len(enss))
print("Entities distribution by types:\n", "\n".join([str(c) for c in Counter([each[1] for each in enss]).most_common()]))


file_ids = set()
enss = []

for fn in train_dev_root2.glob("*.ann"):
    file_ids.add(fn.stem)
    _, ens, _ = read_annotation_brat(fn)
    #print( _)
    enss.extend(ens)
print("100 files as training, training files: ", len(file_ids), list(file_ids)[:5])
print("100 files as training, total training eneitites: ", len(enss))
print("Entities distribution by types:\n", "\n".join([str(c) for c in Counter([each[1] for each in enss]).most_common()]))