import pandas as pd

def conca(files,name):
    data = pd.DataFrame()
    for i in files:
        data_temp = pd.read_csv(i,index_col=0)
        data = pd.concat([data,data_temp],ignore_index=True)
    data.to_csv(f'{name}.csv')
def sorter(fname,savename):
    data = pd.read_csv(f'{fname}.csv',index_col=0)
    data = data.sort_values(by=['pattern','loop','total_size'],ignore_index=True)
    data.to_csv(f'{savename}.csv')

files = ['data2.csv','data3.csv','data4.csv','data16.csv','HALF.csv','data5.csv','last.csv']
conca(files,'total')
sorter('total','final')
data = pd.read_csv('filtered_final.csv', index_col=0)
# Filter rows where DFS-Effi is not 0
filtered_data = data[data['DFS-Effi'] != 0]
# Save the final updated DataFrame
filtered_data.to_csv('updated_final.csv')