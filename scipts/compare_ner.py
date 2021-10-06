import sys
from pathlib import Path
import shutil

gs_root=Path(str(sys.argv[1]))
bert_root=Path(str(sys.argv[2]))
save_root=Path(str(sys.argv[3]))
save_root.mkdir(parents=True, exist_ok=True)
for k in gs_root.glob('*.ann'):
    fid=k.stem
    txt_fn = gs_root / (fid + ".txt")
    ann_fn = gs_root / (fid + ".ann")
    txt_fn1 = save_root / (fid + ".txt")
    ann_fn1 = save_root / (fid + ".ann")
    shutil.copyfile(txt_fn, txt_fn1)
    shutil.copyfile(ann_fn, ann_fn1)

for k in save_root.glob('*.ann'):
    #print(k.stem)
    with open(bert_root/(k.stem+'.ann')) as f:
        lines=f.readlines()
        lines_used=[]
        i=300
        for line in lines:
            if line[0]=='T':
                entity_name=line.split('\t',2)[1].split(' ',1)[0]
                entity_num=line.split('\t',2)[1].split(' ',1)[1]
                #print(entity_name)
                lines_used = lines_used+['T'+str(i)+'\t'+entity_name+'_predicted '+entity_num+'\t'+line.split('\t',2)[2]]
                i+=1
        with open(k, "a") as f1:
            f1.writelines(lines_used)