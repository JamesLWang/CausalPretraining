import pandas as pd

a = pd.read_csv('/local/vondrick/chengzhi/waterbird_complete95_forest2water2/metadata.csv')

y=a['y'].values
c = a['place'].values
s = a['split'].values

cnt_11=0
cnt_10=0
cnt_01=0
cnt_00=0

for es in range(s.shape[0]):
    if s[es]==0: #train
        if y[es]==0 and c[es]==0:
            cnt_00+=1
        elif y[es]==0 and c[es]==1:
            cnt_01+=1
        elif y[es]==1 and c[es]==0:
            cnt_10+=1
        elif y[es]==1 and c[es]==1:
            cnt_11+=1

print(cnt_00, cnt_01,  cnt_10,cnt_11)
sum = cnt_11+ cnt_10+ cnt_01+ cnt_00
print(cnt_00*1.0/sum, cnt_01*1.0/sum,  cnt_10*1.0/sum,cnt_11*1.0/sum)




