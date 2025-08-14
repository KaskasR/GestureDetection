# save as check_dataset.py (run: python check_dataset.py)
import csv, collections
counts = collections.Counter()
with open("windows.csv","r",newline="") as f:
    for row in csv.reader(f):
        if not row: continue
        lab = int(float(row[0]))
        if lab in (0,1,2,3):
            counts[lab]+=1
print("Counts {0:NoGesture,1:Pinch,2:Fist,3:Wave}:", dict(counts))
