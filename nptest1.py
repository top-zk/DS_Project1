import numpy as np
from mpmath import linspace
#task1
arr1=np.arange(10,30)
arr2=np.zeros((3,3))
arr3=np.full((4,4),7)
arr4=np.linspace(0,1,20)
print(arr1.dtype)
print(arr2.shape)
print(arr3.ndim)
#task2
arr5=np.array([[1,5,9,13],[2,6,10,14],[3,7,11,15],[4,8,12,16]])
element=arr5[2,1]
print(element)
#提取第三行第二列的元素（注意，索引从0开始！）
second_row=arr5[1,:]
print(second_row)
#提取完整的第二行
first_col=arr5[0,:]
print(first_col)
#提取完整的第一列
sub_matrix=arr5[:2,2:]
print(sub_matrix)
#从数组arr的右上角提取2x2子矩阵（该子矩阵应包含[9,13]和[10,14]）
every_other = arr5[1, ::2]
print(every_other)
#使用切片功能可获取第二行的每隔一个元素（结果应为[2,10]）。
every_other1=arr5[1,::3]
print(every_other1)
#arr[index,::3(3-1之间的距离)
#task3
np.random.seed(42)
random_arr=np.random.randint(1,100,15)

print(random_arr)
mask=random_arr>50
print(mask)
greater_than_50=random_arr[mask]
print(greater_than_50)
even_numbers=random_arr[::2]
even_numbers1=random_arr[random_arr%2==0]
print(even_numbers)
print(even_numbers1)
random_arr[random_arr<30]=-1
print(random_arr)
select =random_arr[[0,4,8]]
print(select)


