import os 
from tqdm import tqdm 
x=os.listdir("Val_Thermal/")
y=os.listdir("Val_RGB")
x_new = [d[:-5] for d in x]
y_new = [d[:-4] for d in y]
x_new.sort()
y_new.sort()
print("Sorting the data done",x_new)
remove_list = [i for i in x_new if i not in y_new]
length=len(remove_list)
print(length)
remove_list =[i+".jpeg" for i in remove_list]
print(remove_list)
for i in tqdm(range(0,length)):
	os.remove("Val_Thermal/"+remove_list[i]) 
	#print("JPEGImages/"+remove_list[i])