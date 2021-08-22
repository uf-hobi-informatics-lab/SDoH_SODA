#make relation
from pathlib import Path
import pickle as pkl
from collections import defaultdict, Counter
from itertools import permutations, combinations
from functools import reduce
import numpy as np
import os
def pkl_save(data, file):
    with open(file, "wb") as f:
        pkl.dump(data, f)

        
def pkl_load(file):
    with open(file, "rb") as f:
        data = pkl.load(f)
    return data


def load_text(ifn):
    with open(ifn, "r") as f:
        txt = f.read()
    return txt


def save_text(text, ofn):
    with open(ofn, "w") as f:
        f.write(text)
import sys
# https://github.com/uf-hobi-informatics-lab/NLPreprocessing (git clone this repo to local)
sys.path.append("../NLPreprocessing/")
sys.path.append("../NLPreprocessing/text_process")
from annotation2BIO import pre_processing, read_annotation_brat, generate_BIO
MIMICIII_PATTERN = "\[\*\*|\*\*\]"
from sentence_tokenization import logger as l1
from annotation2BIO import logger as l2
l1.disabled = True
l2.disabled = True
data_dir=sys.argv[1]
output_name=sys.argv[2]
def create_entity_to_sent_mapping(nnsents, entities, idx2e):
    loc_ens = []
    
    ll = len(nnsents)
    mapping = defaultdict(list)
    for idx, each in enumerate(entities):
        en_label = idx2e[idx]
        en_s = each[2][0]
        en_e = each[2][1]
        new_en = []
        
        i = 0
        while i < ll and nnsents[i][1][0] < en_s:
            i += 1
        s_s = nnsents[i][1][0]
        s_e = nnsents[i][1][1]

        if en_s == s_s:
            mapping[en_label].append(i)

            while i < ll and s_e < en_e:
                i += 1
                s_e = nnsents[i][1][1]
            if s_e == en_e:
                 mapping[en_label].append(i)
            else:
                mapping[en_label].append(i)
                print("last index not match ", each)
        else:
            mapping[en_label].append(i)
            print("first index not match ", each)

            while i < ll and s_e < en_e:
                i += 1
                s_e = nnsents[i][1][1]
            if s_e == en_e:
                 mapping[en_label].append(i)
            else:
                mapping[en_label].append(i)
                print("last index not match ", each)
    return mapping


def get_permutated_relation_pairs(eid2idx):
    all_pairs = []
    all_ids = [k for k, v in eid2idx.items()]
    for e1, e2 in permutations(all_ids, 2):
        all_pairs.append((e1, e2))
    return all_pairs
def validate_rels(rels, valid):
    nrels = []
    for rel in rels:
        rtype = rel[0]
        if tuple(rtype) not in valid:
            print("invalid: ", rel)
            continue
        nrels.append(rel)
    return nrels


def check_tags(s1, s2):
    assert EN1_START in s1 and EN1_END in s1, f"tag error: {s1}"
    assert EN2_START in s2 and EN2_END in s2, f"tag error: {s2}"


def format_relen(en, rloc, nsents):
    if rloc == 1:
        spec1, spec2 = EN1_START, EN1_END
    else:
        spec1, spec2 = EN2_START, EN2_END
    sn1, tn1 = en[0][3]
    sn2, tn2 = en[-1][3]
    target_sent = nsents[sn1]
    target_sent = [each[0] for each in target_sent]
    ors =  " ".join(target_sent)
    
    if sn1 != sn2:
#         print("[!!!Warning] The entity is not in the same sentence\n", en)
        tt = nsents[sn2]
        tt = [each[0] for each in tt]
        target_sent.insert(tn1, spec1)
        tt.insert(tn2+1, spec2)
        target_sent = target_sent + tt
#         print(target_sent)
    else:
        target_sent.insert(tn1, spec1)
        target_sent.insert(tn2+2, spec2)
    
    fs = " ".join(target_sent)
    
    return sn1, sn2, fs, ors


def gene_true_relations(rels, mappings, ens, e2i, nnsents, nsents, valid_comb, fid=None):
    true_pairs = set()
    pos_samples = []
    
    for rel in rels:
        rel_type = rel[0]
        enid1, enid2 = rel[1:]
        """
        [['100', (15443, 15446), (16473, 16476), (231, 4), 'B-Strength'], 
        ['mg', (15447, 15449), (16477, 16479), (231, 5), 'I-Strength']] 
        [['Metoprolol', (15422, 15432), (16452, 16462), (231, 2), 'B-Drug'], 
        ['Succinate', (15433, 15442), (16463, 16472), (231, 3), 'I-Drug']]
        """
        enbs1, enbe1 = mappings[enid1]
        en1 = nnsents[enbs1: enbe1+1]
        si1, sii1, fs1, ors1 = format_relen(en1, 1, nsents)
        enbs2, enbe2 = mappings[enid2]
        en2 = nnsents[enbs2: enbe2+1]
        si2, sii2, fs2, ors2 = format_relen(en2, 2, nsents)
        sent_diff = abs(si1 - si2)
        
        en1t = en1[0][-1].split("-")[-1]
        en2t = en2[0][-1].split("-")[-1]

        true_pairs.add((enid1, enid2))
        
        if (en1t, en2t) not in valid_comb:
            continue
        
        if sent_diff <= CUTOFF:
            check_tags(fs1, fs2)
            assert (en1t, en2t) in valid_comb, f"{en1t} {en2t}"
            if DO_BIN:
                pos_samples.append((sent_diff, "pos", fs1, fs2, en1t, en2t, enid1, enid2, fid))
            else:
                pos_samples.append((sent_diff, rel_type, fs1, fs2, en1t, en2t, enid1, enid2, fid))

    return pos_samples, true_pairs
        

def gene_neg_relation(perm_pairs, true_pairs, mappings, ens, e2i, nnsents, nsents, valid_comb, fid=None):
    neg_samples = []
    for each in perm_pairs:
        enid1, enid2 = each
        
        # not in true relation
        if (enid1, enid2) in true_pairs:
            continue
        
        enc1 = ens[e2i[enid1]]
        enc2 = ens[e2i[enid2]]

        enbs1, enbe1 = mappings[enid1]
        en1 = nnsents[enbs1: enbe1+1]
        si1, sii1, fs1, ors1 = format_relen(en1, 1, nsents)
        enbs2, enbe2 = mappings[enid2]
        en2 = nnsents[enbs2: enbe2+1]
        si2, sii2, fs2, ors2 = format_relen(en2, 2, nsents)
        sent_diff = abs(si1 - si2)
        
        en1t = en1[0][-1].split("-")[-1]
        en2t = en2[0][-1].split("-")[-1]
        
        if (en1t, en2t) not in valid_comb:
            continue
        
        if sent_diff <= CUTOFF:
            check_tags(fs1, fs2)
            assert (en1t, en2t) in valid_comb, f"{en1t} {en2t}"
            if fid:
                neg_samples.append((sent_diff, NEG_REL, fs1, fs2, en1t, en2t, enid1, enid2, fid))
            else:
                neg_samples.append((sent_diff, NEG_REL, fs1, fs2, en1t, en2t, enid1, enid2))
    
    return neg_samples

    
def create_training_samples(file_path, valids=None, valid_comb=None):
    fids = []
    root = Path(file_path)
    
    dpos = defaultdict(list)
    dneg = defaultdict(list)
    
    for txt_fn in root.glob("*.txt"):
        fids.append(txt_fn.stem)
        ann_fn = root / (txt_fn.stem+".ann")

        # load text
        txt = load_text(txt_fn)
        pre_txt, sents = pre_processing(txt_fn, deid_pattern=MIMICIII_PATTERN)
        e2i, ens, rels = read_annotation_brat(ann_fn)
        i2e = {v: k for k, v in e2i.items()}
        
        nsents, sent_bound = generate_BIO(sents, ens, file_id="", no_overlap=False, record_pos=True)
        print(nsents)
        total_len = len(nsents)
        nnsents = [w for sent in nsents for w in sent]
        mappings = create_entity_to_sent_mapping(nnsents, ens, i2e)

        pos_samples, true_pairs = gene_true_relations(
            rels, mappings, ens, e2i, nnsents, nsents, valid_comb, fid=txt_fn.stem)
        perm_pairs = get_permutated_relation_pairs(e2i)
        neg_samples = gene_neg_relation(
            perm_pairs, true_pairs, mappings, ens, e2i, nnsents, nsents, valid_comb, fid=txt_fn.stem)
        
        for pos_sample in pos_samples:
            dpos[pos_sample[0]].append(pos_sample)
        for neg_sample in neg_samples:
            dneg[neg_sample[0]].append(neg_sample)
        
    return dpos, dneg


def create_test_samples(file_path, valids=None, valid_comb=None):
    #create a separate mapping file
    rel_mappings = []
    #
    fids = []
    root = Path(file_path)
    preds = defaultdict(list)
    
    for txt_fn in root.glob("*.txt"):
        fids.append(txt_fn.stem)
        ann_fn = root / (txt_fn.stem + ".ann")

        # load text
        txt = load_text(txt_fn)
        pre_txt, sents = pre_processing(txt_fn, deid_pattern=MIMICIII_PATTERN)
        e2i, ens, _ = read_annotation_brat(ann_fn)
        i2e = {v: k for k, v in e2i.items()}
        
        nsents, sent_bound = generate_BIO(sents, ens, file_id="", no_overlap=False, record_pos=True)
        total_len = len(nsents)
        nnsents = [w for sent in nsents for w in sent]
        mappings = create_entity_to_sent_mapping(nnsents, ens, i2e)
        
        perm_pairs = get_permutated_relation_pairs(e2i)
        pred = gene_neg_relation(perm_pairs, set(), mappings, ens, e2i, nnsents, nsents, valid_comb, fid=txt_fn.stem)
        for idx, pred_s in enumerate(pred):
            preds[pred_s[0]].append(pred_s)
            
    return preds
def en_sent_id(en_pos, send_bound):
    e_s = en_pos[0]
    e_e = en_pos[1]
    for k, v in sent_bound.items():
        s_s = v[0]
        s_e = v[1]
        if e_s >= s_s and e_s <= s_e and e_e >s_e :
            print("entity is in two sentence")
        if e_s >= s_s and e_s <= s_e:
            return k
        

def extract_entity_comb_for_relation(e2idx, entities, rels, sent_bound):
    #'T1': 0
    #'meropenem', 'Drug', (4534, 4543)
    #('Strength-Drug', 'T5', 'T39')
    rn = defaultdict(list)
    rl = []
    for rel in rels:
        rtype = rel[0]
        en1 = rel[1]
        en2 = rel[2]
        en1_type = entities[e2idx[en1]][1]
        en2_type = entities[e2idx[en2]][1]
        rn[rtype].append((en1_type, en2_type))
        en1_pos = entities[e2idx[en1]][2]
        e1_n = en_sent_id(en1_pos, sent_bound)
        en2_pos = entities[e2idx[en2]][2]
        e2_n = en_sent_id(en2_pos, sent_bound)
        rl.append(abs(e1_n-e2_n))
    return rn, rl
def to_tsv(data, fn):
    header = "\t".join([str(i+1) for i in range(len(data[0]))])
    with open(fn, "w") as f:
        f.write(f"{header}\n")
        for each in data:
            d = "\t".join([str(e) for e in each])
            f.write(f"{d}\n")


def to_5_cv(data, ofd):
    if not os.path.isdir(ofd):
        os.mkdir(ofd)
    
    np.random.seed(13)
    np.random.shuffle(data)
    
    dfs = np.array_split(data, 5)
    a = [0,1,2,3,4]
    for each in combinations(a, 4):
        b = list(set(a) - set(each))[0]
        n = dfs[b]
        m = []
        for k in each:
            m.extend(dfs[k])
        if not os.path.isdir(os.path.join(ofd, f"sample{b}")):
            os.mkdir(os.path.join(ofd, f"sample{b}"))
        
        to_tsv(m, os.path.join(ofd, f"sample{b}", "train.tsv"))
        to_tsv(n, os.path.join(ofd, f"sample{b}", "dev.tsv"))


def all_in_one(*dd, dn="2018n2c2", do_train=True):
    data = []
    for d in dd:
        for k, v in d.items():
            for each in v:
                data.append(each[1:])
    
    output_path = f"./temp/{dn}_aio_th{CUTOFF}"
    p = Path(output_path)
    p.mkdir(parents=True, exist_ok=True)
    
    if do_train:
        to_tsv(data, p/"train.tsv")
        if OUTPUT_CV:
            to_5_cv(data, p.as_posix())
    else:
        to_tsv(data, p/"test.tsv")
    

def all_in_unique(*dd, dn="2018n2c2", do_train=True):
    for idx in range(CUTOFF+1):
        data = []
        for d in dd:
            for k, v in d.items():
                for each in v:
                    if k == idx:
                        data.append(each[1:])
        
        output_path = f"./temp/{dn}_aiu_th{CUTOFF}"
        p = Path(output_path) / f"cutoff_{idx}"
        p.mkdir(parents=True, exist_ok=True)
        if do_train:
            to_tsv(data, p/"train.tsv")
            if OUTPUT_CV:
                to_5_cv(data, p.as_posix())
        else:
            to_tsv(data, p/"test.tsv")
# general pre-defined special tags
EN1_START = "[s1]"
EN1_END = "[e1]"
EN2_START = "[s2]"
EN2_END = "[e2]"
NEG_REL = "NonRel"
# max valid cross sentence distance
CUTOFF = 1
# output 5-fold cross validation data
OUTPUT_CV = False
# do binary classification (if false, then we do multiclass classification)
DO_BIN = False
sdoh_valid_comb = {
        ('Tobacco_use', 'Substance_use_status'), ('Substance_use_status', 'Smoking_type'),
        ('Substance_use_status', 'Smoking_freq_ppd'), ('Substance_use_status', 'Smoking_freq_py'), 
        ('Substance_use_status', 'Smoking_freq_qy'), ('Substance_use_status', 'Smoking_freq_sy'),
        ('Substance_use_status', 'Smoking_freq_other'), ('Alcohol_use', 'Substance_use_status'),
        ('Substance_use_status', 'Alcohol_freq'), ('Substance_use_status', 'Alcohol_type'), 
        ('Substance_use_status', 'Alcohol_other'), ('Drug_use', 'Substance_use_status'),
        ('Substance_use_status', 'Drug_freq'), ('Substance_use_status', 'Drug_type'),('Substance_use_status', 'Drug_other'), ('Sex_act', 'Sdoh_status'),
        ('Sex_act', 'Partner'), ('Sex_act', 'Protection'), 
        ('Sex_act', 'Sex_act_other'), ('Occupation', 'Employment_status'),
        ('Occupation', 'Employment_location'), ('Gender', 'Sdoh_status'),('Social_cohesion', 'Social_method'), ('Social_method', 'Sdoh_status'),
        ('Physical_act', 'Sdoh_status'), ('Physical_act', 'Sdoh_freq'), 
        ('Living_supply', 'Sdoh_status'), ('Abuse', 'Sdoh_status'),
        ('Transportation', 'Sdoh_status'), ('Health_literacy', 'Sdoh_status'),
        ('Financial_constrain', 'Sdoh_status'), ('Social_cohesion', 'Sdoh_status'),
        ('Social_cohesion', 'Sdoh_freq'), ('Gender', 'Sdoh_status'), 
        ('Race', 'Sdoh_status'), ('Ethnicity', 'Sdoh_status'),
        ('Living_Condition', 'Sdoh_status')
    }
#test_root='/data/datasets/zehao/sdoh/res/lung_cancer_formatted_output'

test_root=f'./result/NER/{output_name}_formatted_output'
preds = create_test_samples(test_root, None, sdoh_valid_comb)
all_in_one(preds, dn=output_name, do_train=False)





