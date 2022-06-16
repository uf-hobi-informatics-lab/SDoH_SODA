# -*- coding: utf-8 -*-
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
import re
import codecs
import unicodedata

data_dir=sys.argv[1]
output_name=sys.argv[2]

encoding_dir = "../temp/encoding_txt/"
#data stat
file_ids = set()
enss = []

for fn in Path(data_dir).glob("*.ann"):
    file_ids.add(fn.stem)
    _, ens, _ = read_annotation_brat(fn)
    #print( _)
    enss.extend(ens)

print("number of test files: ", len(file_ids))
print("total number of test eneitites: ", len(enss))
print("Entities distribution by types:\n", "\n".join([str(c) for c in Counter([each[1] for each in enss]).most_common()]))

# generate bio
test_root = Path(data_dir)
test_bio = "../temp/"+output_name
output_root = Path(test_bio)
output_root.mkdir(parents=True, exist_ok=True)
Path(encoding_dir).mkdir(parents=True, exist_ok=True)


for fn in test_root.glob("*.txt"):
    txt_fn = fn
    file_stem=txt_fn.stem
    myf=open(txt_fn,'r',encoding="utf-8")
    mtxt=myf.read()
    myf.close()
#     txt=codecs.decode(mtxt,'unicode_escape').encode('latin1').decode('utf8')
    txt = unicodedata.normalize("NFKD", mtxt)
    txt=txt.strip()
    with open (encoding_dir+file_stem+'.txt','w',encoding="utf-8") as f:
        f.write(txt)
        f.close()
# test_root
raw_text_ct=0
done_ct=0
for fn in Path(encoding_dir).glob("*.txt"):
    txt_fn = fn
    raw_text_ct+=1
    bio_fn = output_root / (fn.stem + ".bio.txt")
    
    txt, sents = pre_processing(txt_fn, deid_pattern=MIMICIII_PATTERN)
    nsents, sent_bound = generate_BIO(sents, [], file_id=txt_fn, no_overlap=False)
    
    BIOdata_to_file(bio_fn, nsents)
    
for fn in output_root.glob("*.txt"):
    done_ct+=1

print("number of test files: ", raw_text_ct)
print("number of geneated bio files: ", done_ct)
