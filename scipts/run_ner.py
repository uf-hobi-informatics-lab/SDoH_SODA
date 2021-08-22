#run NER
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

data_dir=sys.argv[1]
output_name=sys.argv[2]
#data stat
file_ids = set()
enss = []

for fn in Path(data_dir).glob("*.ann"):
    file_ids.add(fn.stem)
    _, ens, _ = read_annotation_brat(fn)
    #print( _)
    enss.extend(ens)
print("test files: ", len(file_ids), list(file_ids)[:5])
print("total test eneitites: ", len(enss))
print("Entities distribution by types:\n", "\n".join([str(c) for c in Counter([each[1] for each in enss]).most_common()]))
# generate bio
test_root = Path(data_dir)
test_bio = "./bio/"+output_name
output_root = Path(test_bio)
output_root.mkdir(parents=True, exist_ok=True)

for fn in test_root.glob("*.txt"):
    txt_fn = fn
    bio_fn = output_root / (fn.stem + ".bio.txt")
    
    txt, sents = pre_processing(txt_fn, deid_pattern=MIMICIII_PATTERN)
    nsents, sent_bound = generate_BIO(sents, [], file_id=txt_fn, no_overlap=False)
    
    BIOdata_to_file(bio_fn, nsents)

