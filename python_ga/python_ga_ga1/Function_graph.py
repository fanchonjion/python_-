import math
import numpy as np
import matplotlib.pyplot as plt
x=np.linspace(0,15,10000)  #这个表示在0到5之间生成10000个x值
y=[-3*(i-30)**2*math.sin(i) for i in x]  #对上述生成的10000个数循环求对应的y
plt.plot(x,y)  #用上述生成的10000个xy值对生成1000个点
plt.xlabel('x')
plt.ylabel('y')
plt.title('function')
plt.axis([-1, 16, -3000, 3000]) 
plt.show()  #绘制图像
