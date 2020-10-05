import numpy as np
from numpy import *
import scipy.spatial.distance as dist
from matplotlib import pyplot as plt
import matplotlib

# x=np.arange(-3*np.pi,3*np.pi,0.1)
# y_sin=np.sin(x)
# y_cos=np.cos(x)
# # 建立 subplot 网格，高为 2，宽为 1
# # 激活第一个 subplot
# plt.subplot(2,1,1)
# plt.plot(x,y_sin)
# plt.title('sine')
# plt.plot(2,1,2)
# plt.plot(x,y_cos)
# plt.title('cosine')
# plt.show()
# x=[5,8,10]
# y=[12,16,6]
# x2=[6,9,11]
# y2=[6,15,7]
# plt.bar(x,y,align='center')
# plt.bar(x2,y2,color='g',align='center')
# plt.title('Bar graph')
# plt.ylabel('Y axis')
# plt.xlabel('X axis')
# plt.show()
#
# a=sorted([f.name for f in matplotlib.font_manager.fontManager.ttflist])
# for i in a:
#     print(i)

plt.figure(figsize=(8,6),dpi=80)
plt.subplot(1,1,1)
X=np.linspace(-np.pi,np.pi,256,endpoint=True)

C,S=np.cos(X),np.sin(X)
plt.plot(X,C,color='blue',linewidth=2.5,linestyle='-')
plt.plot(X,S,color='green',linewidth=2.5,linestyle='-')

xmin,xmax=X.min(),X.max()
dx=(xmax-xmin)*0.2

plt.xlim(X.min()*1.1,X.max()*1.1)
plt.xticks(np.linspace(-4,4,9,endpoint=True))
plt.ylim(C.min()*1.1,C.max()*1.1)
plt.yticks(np.linspace(-1,1,5,endpoint=True))
plt.show()

matV=mat([[1,1,0,1,0,1,0,0,1],[0,1,1,0,0,0,1,1,1]])
print('dist.jaccard:',dist.pdist(matV,'jaccard'))







