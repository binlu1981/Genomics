python 科学计算

麻烦转载的朋友，请标明出处，作者，让我也小小虚荣一下。。。这都是我花了好多时间整理出来的。

谢谢各位捧场。。。

进行命令行，输入 python ,import  numpy as np 导入函数库。

1、创建数组   c = np.array([[1, 2, 3, 4],[4, 5, 6, 7], [7, 8, 9, 10]])

     c.shape=3,4,通过改变c.shape的值改变数组的组合，某一值为-1时，自动计算，但必须保证总元数是两个值的积。 

>>> d = a.reshape((2,2))     #创建一个改变组合的数组，原数组不变
>>> d, a
array([[1, 2], [3, 4]]), array([1, 2, 3, 4])
      数组a和d其实共享数据存储内存区域，因此修改其中任意一个数组的元素都会同时修改另外一个数组的内容：

>>> np.array([[1, 2, 3, 4],[4, 5, 6, 7], [7, 8, 9, 10]],dtype=np.float)   
 　改变数组元素类型，如int,complex,float等。
2、Numpy 提供了几个更加好用的创建数组的函数：

        1) arange函数类似于python的range函数，通过指定开始值、终值和步长来创建一维数组，注意数组不包括终值:

>>> np.arange(0,1,0.1)
array([ 0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        2）linspace函数通过指定开始值、终值和元素个数来创建一维数组，可以通过endpoint关键字指定是否包括终值，缺省设置是包括终值:

>>> np.linspace(0, 1, 12)
array([ 0. , 0.09090909, 0.18181818, 0.27272727, 0.36363636,0.45454545, 0.54545455, 0.63636364, 0.72727273, 0.81818182,0.90909091, 1. ])
         np.linspace(start,end,num), 产生一等差数组，一共产生num个数，整个数组分成 num-1段，总长度 num-start，所以每段是unit=(num-start)/(num-1),每个数从start开始，每次加unit. 上式是一共产生12个数，分成11段，总长12-0，所以每段12/11。所以从0开始，每次加12/11.

        3）logspace  产生等比数列

>>> np.logspace(0, 2, 20)
array([ 1. , 1.27427499, 1.62377674, 2.06913808,2.6366509 , 3.35981829, 4.2813324 , 5.45559478,
6.95192796, 8.8586679 , 11.28837892, 14.38449888,18.32980711, 23.35721469, 29.76351442, 37.92690191,48.32930239, 61.58482111, 78.47599704, 100. ])
        起点：10^0=1,终点：10^2=100,函数log()以e为底，e=2.718,上例是创建一个1~100的20个等比数组

        4）fromstring()  从字符串里创建数组

         Python的字符串实际上是字节序列，每个字符占一个字节，，因此如果从字符串s创建一个8bit的整数数组的话，所得到的数组正好就是字符串中每个字符的ASCII编码:

>>> np.fromstring(s, dtype=np.int8)
array([ 97, 98, 99, 100, 101, 102, 103, 104], dtype=int8)
         如果从字符串s创建16bit的整数数组，那么两个相邻的字节就表示一个整数，把字节98和字节97当作一个16位的整数，它的值就是98*256+97 = 25185。可以看出内存中是以little endian(低位字节在前)方式保存数据的。

>>> np.fromstring(s, dtype=np.int16)
array([25185, 25699, 26213, 26727], dtype=int16)
>>> 98*256+97
25185
         5）将数组下标转换为数组中对应的值，然后使用此函数创建数组：

>>> def func(i):
... return i%4+1
...
>>> np.fromfunction(func, (10,))
array([ 1., 2., 3., 4., 1., 2., 3., 4., 1., 2.])
          fromfunction函数的第一个参数为计算每个数组元素的函数，第二个参数为数组的大小(shape)，因为它支持多维数组，所以第二个参数必须是一个序列，本例中用(10,)创建一个10元素的一维数组。下面的例子创建一个二维数组表示九九乘法表，输出的数组a中的每个元素a[i, j]都等于func2(i, j)：

>>> def func2(i, j):
... return (i+1) * ( j+1)
...
>>> a = np.fromfunction(func2, (9,9))
           6）取数

复制代码
>>> a
array([ 0, 1, 100, 101, 4, 5, 6, 7, 8, 9])
>>> a[1:-1:2]    # 范围中的第三个参数表示步长，2表示隔一个元素取一个元素
array([ 1, 101, 5, 7])
>>> a[::-1] # 省略范围的开始下标和结束下标，步长为-1，整个数组头尾颠倒
array([ 9, 8, 7, 6, 5, 4, 101, 100, 1, 0])
>>> a[5:1:-2] # 步长为负数时，开始下标必须大于结束下标
array([ 5, 101])
复制代码
      和Python的列表序列不同，通过下标范围获取的新的数组是原始数组的一个视图。它与原始数组共享同一块数据空间：

复制代码
>>> b = a[3:7] # 通过下标范围产生一个新的数组b，b和a共享同一块数据空间
>>> b
array([101, 4, 5, 6])
>>> b[2] = -10 # 将b的第2个元素修改为-10
>>> b
array([101, 4, -10, 6])
>>> a # a的第5个元素也被修改为10
array([ 0, 1, 100, 101, 4, -10, 6, 7, 8, 9])
复制代码
      当使用整数序列对数组元素进行存取时，将使用整数序列中的每个元素作为下标，整数序列可以是列表或者数组。使用整数序列作为下标获得的数组不和原始数组共享数据空间。

复制代码
>>> x = np.arange(10,1,-1)
>>> x
array([10, 9, 8, 7, 6, 5, 4, 3, 2])
>>> x[[3, 3, 1, 8]] # 获取x中的下标为3, 3, 1, 8的4个元素，组成一个新的数组
array([7, 7, 9, 2])
>>> b = x[np.array([3,3,-3,8])] #下标可以是负数   先将取的数转换成数组
>>> b[2] = 100
>>> b
array([7, 7, 100, 2])
>>> x # 由于b和x不共享数据空间，因此x中的值并没有改变
array([10, 9, 8, 7, 6, 5, 4, 3, 2])
>>> x[[3,5,1]] = -1, -2, -3 # 整数序列下标也可以用来修改元素的值
>>> x
array([10, -3, 8, -1, 6, -2, 4, 3, 2])
复制代码
         当使用布尔数组b作为下标存取数组x中的元素时，将收集数组x中所有在数组b中对应下标为True的元素。使用布尔数组作为下标获得的数组不和原始数组共享数据空间。

复制代码
>>> x
array([5, 4, 3, 2, 1])
>>> x[np.array([True, False, True, False, False])]
array([5, 3])
>>> x[[True, False, True, False, False]]
array([4, 5, 4, 5, 5])
>>> x[np.array([True, False, True, True])] = -1, -2, -3   # 布尔数组下标也可以用来修改元素
>>> x
array([-1, 4, -2, -3, 1])
>>> x = np.random.rand(10) # 产生一个长度为10，元素值为0-1的随机数的数组
>>> x
array([ 0.72223939, 0.921226 , 0.7770805 , 0.2055047 , 0.17567449,0.95799412, 0.12015178, 0.7627083 , 0.43260184, 0.91379859])
>>> x>0.5  # 数组x中的每个元素和0.5进行大小比较，得到一个布尔数组，True表示x中对应的值大于0.5
array([ True, True, True, False, False, True, False, True, False, True], dtype=bool)
>>> x[x>0.5]   # 使用x>0.5返回的布尔数组收集x中的元素，因此得到的结果是x中所有大于0.5的元素的数组
array([ 0.72223939, 0.921226 , 0.7770805 , 0.95799412, 0.7627083 ,0.91379859])
复制代码
        7）文件存取

         NumPy提供了多种文件操作函数方便我们存取数组内容。文件存取的格式分为两类：二进制和文本。
而二进制格式的文件又分为NumPy专用的格式化二进制类型和无格式类型。
使用数组的方法函数tofile可以方便地将数组中数据以二进制的格式写进文件。tofile输出的数据没有格
式，因此用numpy.fromfile读回来的时候需要自己格式化数据：

复制代码
>>> a = np.arange(0,12)
>>> a.shape = 3,4
>>> a
array([[ 0, 1, 2, 3],
[ 4, 5, 6, 7],
[ 8, 9, 10, 11]])
>>> a.tofile("a.bin")
>>> b = np.fromfile("a.bin", dtype=np.float) # 按照float类型读入数据
>>> b # 读入的数据是错误的
array([ 2.12199579e-314, 6.36598737e-314, 1.06099790e-313,1.48539705e-313, 1.90979621e-313, 2.33419537e-313])
>>> a.dtype # 查看a的dtype
dtype('int32')
>>> b = np.fromfile("a.bin", dtype=np.int32) # 按照int32类型读入数据
>>> b # 数据是一维的
array([ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
>>> b.shape = 3, 4 # 按照a的shape修改b的shape
>>> b # 这次终于正确了
array([[ 0, 1, 2, 3],[ 4, 5, 6, 7],[ 8, 9, 10, 11]])

>>> np.save("a.npy", a)
>>> c = np.load( "a.npy" )
>>> c
array([[ 0, 1, 2, 3],
[ 4, 5, 6, 7],
[ 8, 9, 10, 11]])

>>> a = np.array([[1,2,3],[4,5,6]])
>>> b = np.arange(0, 1.0, 0.1)
>>> c = np.sin(b)
>>> np.savez("result.npz", a, b, sin_array = c)
>>> r = np.load("result.npz")
>>> r["arr_0"] # 数组a
array([[1, 2, 3],
[4, 5, 6]])
>>> r["arr_1"] # 数组b
array([ 0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
>>> r["sin_array"] # 数组c
array([ 0. , 0.09983342, 0.19866933, 0.29552021, 0.38941834,0.47942554, 0.56464247, 0.64421769, 0.71735609, 0.78332691])
复制代码
        使用numpy.savetxt和numpy.loadtxt可以读写1维和2维的数组：

复制代码
>>> a = np.arange(0,12,0.5).reshape(4,-1)
>>> np.savetxt("a.txt", a) # 缺省按照'%.18e'格式保存数据，以空格分隔
>>> np.loadtxt("a.txt")
array([[ 0. , 0.5, 1. , 1.5, 2. , 2.5],
[ 3. , 3.5, 4. , 4.5, 5. , 5.5],
[ 6. , 6.5, 7. , 7.5, 8. , 8.5],
[ 9. , 9.5, 10. , 10.5, 11. , 11.5]])
>>> np.savetxt("a.txt", a, fmt="%d", delimiter=",") #改为保存为整数，以逗号分隔
>>> np.loadtxt("a.txt",delimiter=",") # 读入的时候也需要指定逗号分隔
array([[ 0., 0., 1., 1., 2., 2.],
[ 3., 3., 4., 4., 5., 5.],
[ 6., 6., 7., 7., 8., 8.],
[ 9., 9., 10., 10., 11., 11.]])
复制代码
2.4. 
       本节介绍所举的例子都是传递的文件名，也可以传递已经打开的文件对象，例如对于load和save函数来说，如果使用文件对象的话，可以将多个数组储存到一个npy文件中：

复制代码
>>> a = np.arange(8)
>>> b = np.add.accumulate(a)
>>> c = a + b
>>> f = file("result.npy", "wb")
>>> np.save(f, a) # 顺序将a,b,c保存进文件对象f
>>> np.save(f, b)
>>> np.save(f, c)
>>> f.close()
>>> f = file("result.npy", "rb")
>>> np.load(f) # 顺序从文件对象f中读取内容
array([0, 1, 2, 3, 4, 5, 6, 7])
>>> np.load(f)
array([ 0, 1, 3, 6, 10, 15, 21, 28])
>>> np.load(f)
array([ 0, 2, 5, 9, 14, 20, 27, 35])
复制代码
函数                    描述
abs(x )                 绝对值
divmod(x ,y )           返回 (int(x / y ), x % y )
pow(x ,y [,modulo ])    返回 (x ** y ) x % modulo
round(x ,[n])           四舍五入，n为小数点位数

math.ceil(),math.floor(),都可以舍去小数。
复制代码
>>> divmod(10,3)
(3, 1)
>>> divmod(3,10)
(0, 3)
>>> divmod(10,2.5)
(4.0, 0.0)
>>> divmod(2.5,10)
(0.0, 2.5)
>>> divmod(2+1j, 0.5-1j)
(0j, (2+1j))
复制代码
　　 进行浮点数运算时，结果不能等于0，否则会出现运算错误，需要引入decimal()处理。

import decimal
decimal.Decimal('0.1+0.1')-decimal.Decimal('0.2')
　　abs()函数返回一个数的绝对值。divmod()函数返回一个包含商和余数的元组。pow()函数可以用于代替 ** 运算，但它还支持三重取模运算(经常用于密码运算)。 round函数总是返回一个浮点数。

下列比较操作有标准的数学解释,返回一个布尔值True,或者False:

运算符                  描述
x < y                   小于
x > y                   大于
x == y                  等于
x != y                  不等于(与<>相同)
x >= y                  大于等于
x <= y                  小于等于
 　　Python的比较运算可以连结在一起，如w < x < y < z 。这个表达式等价于 w < x and x < y and y < z 。

x < y > z这个表达式也是合法的，(注意,这个表达式中 x 和 z 并没有比较操作)。不建议这样的写法，因为这会造成代码的阅读困难。

只可以对复数进行等于(==)及不等于(!=)比较，任何对复数进行其他比较的操作都会引发TypeError异常。

数值操作要求操作数必须是同一类型，若Python发现操作数类型不一致，就会自动进行类型的强制转换，转换规则如下:

1.如果操作数中有一个是复数，另一个也将被转换为复数
2.如果操作数中有一个是浮点数，另一个将被转换为浮点数
3.如果操作数中有一个是长整数数，另一个将被转换为长整数数
4.如果以上都不符合，则这两个数字必然都是整数，不需进行强制转换。
min(s)                  最小元素
max(s)                  最大元素

 

字符              输出格式
d,i             十进制整数或长整数
u               无符号十进制整数或长整数
o               八进制整数或长整数
x               十六进制整数或长整数
X               十六进制整数或长整数(大写字母)
f               浮点数如 [-]m.dddddd
e               浮点数如 [-]m .dddddde ±xx .
E               浮点数如 [-]m .ddddddE ±xx .
g,G             指数小于-4或者更高精确度使用 %e 或 %E; 否则,使用 %f
s               字符串或其他对象,使用str()来产生字符串
r               与 repr() 返回的字符串相同
c               单个字符
%               转换符标识 %
 在 % 和转换字符串之间,允许出现以下修饰符,并且只能按以下顺序:

1.映射对象的 key,如果被格式化对象是一个映射对象却没有这个成分,会引发KeyError异常.
2.下面所列的一个或多个:
    左对齐标志
    +,数值指示必须包含
    0,指示一个零填充
3.指示最小栏宽的数字.转换值会被打印在指定了最小宽度的栏中并且填充在(或者右边).
4. 一个小数点用来分割浮点数
5. A number specifying the maximum number of characters to be printed from a string, the number of digits following the decimal point in a floating-point number, or the minimum number of digits for an integer.
另外,形标(*)字符用于在任意宽度的栏中代替数字. If present, the width will be read from the next item in the tuple.下边的代码给出了几个例子:
Toggle line numbers
   1 a = 42
   2 b = 13.142783
   3 c = "hello"
   4 d = {'x':13, 'y':1.54321, 'z':'world'}
   5 e = 5628398123741234L
   6  
   7 print 'a is %d' % a             #  "a is 42"
   8 print '%10d%f' % (a,b)         #  " 42 13.142783"
   9 print '%+010d%E' % (a,b)       #  "+000000042 1.314278E+01"
  10 print '%(x)-10d%(y)0.3g' % d   #  "13         1.54"
  11 print '%0.4s%s' % (c, d['z'])  #  "hell world"
  12 print '%*.*f' % (5,3,b)         #  "13.143"
  13 print 'e = %d' % e              #  "e = 5628398123741234"
 简单的画图：

       matplotlib 是python最著名的绘图库，它提供了一整套和matlab相似的命令API，十分适合交互式地进行制图。而且也可以方便地将它作为绘图控件，嵌入GUI应用程序中。

      plt.axis([0, 6, 0, 20])   限定区间范围x(0,6),y(0,20)

      matplotlib的pyplot子库提供了和matlab类似的绘图API，方便用户快速绘制2D图表。

复制代码
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 1000)
y = np.sin(x)
z = np.cos(x**2)

plt.figure(figsize=(8,4))

plt.plot(x,y,label="$sin(x)$",color="red",linewidth=2)
plt.plot(x,z,"b--",label="$cos(x^2)$")
plt.xlabel("Time(s)")
plt.ylabel("Volt")
plt.title("PyPlot First Example")
plt.ylim(-1.2,1.2)
plt.legend()   #右上角的标签，即写明x,y对应的曲线，与10，11行有关
plt.show()    #有它才能显示出图形
复制代码
 

 

 figsize():

      通过figsize参数可以指定绘图对象的宽度和高度，单位为英寸；dpi参数指定绘图对象的分辨率，即每
英寸多少个像素，缺省值为80。因此本例中所创建的图表窗口的宽度为8*80 = 640像素。

      绘制一组幂函数：

复制代码
from matplotlib.matlab import * 

  x = linspace(-4, 4, 200) 
  f1 = power(10, x) 
  f2 = power(e, x) 
  f3 = power(2, x)  

  plot(x, f1, 'r',  x, f2, 'b', x, f3, 'g', linewidth=2) 
  axis([-4, 4, -0.5, 8])
  text(1, 7.5, r'$10^x$', fontsize=16)
  text(2.2, 7.5, r'$e^x$', fontsize=16)
  text(3.2, 7.5, r'$2^x$', fonsize=16)
  title('A simple example', fontsize=16)
  
  savefig('asd.png', dpi=75)
  show() 
复制代码


另一个例子：

复制代码
from matplotlib.matlab import *
def  f(x, c):
    m1 = sin(2*pi*x)
    m2 = exp(-c*x)
return multiply(m1, m2)
x = linspace(0, 4, 100)
sigma = 0.5
plot(x, f(x, sigma), 'r', linewidth=2)
xlabel(r'$\rm{time}  \  t$', fontsize=16)
ylabel(r'$\rm{Amplitude} \ f(x)$', fontsize=16)
title(r'$f(x) \ \rm{is \ damping  \ with} \ x$', fontsize=16)
text(2.0, 0.5, r'$f(x) = \rm{sin}(2 \pi  x^2) e^{\sigma x}$', fontsize=20)
savefig('latex.png', dpi=75)
show()
复制代码


        多轴图：subplot函数快速绘制有多个轴的图表。subplot函数的调用形式如下：
                                 subplot(numRows, numCols, plotNum)

       作用：创建子图，numRows表示整个大图要分的行数，即每列有几个图。numcols表示整个大图要分的列数，即每行有几个图，plotnum表示子图在大图中占的位置，顺序为自左到右自上到下。

      下面的程序创建3行2列共6个轴，通过axisbg参数给每个轴设置不同的背景颜色。

复制代码
for idx, color in enumerate("rgbyck"):
plt.subplot(320+idx+1, axisbg=color)
plt.show()

#一个图里分几个子图同时显示,命令行直接输入命令画图，偷懒了，反正是一次：

>>> def f(t):
    return np.exp(-t)*np.cos(2*np.pi*t)
>>> t1 = np.arange(0.0,5.0,0.1)
>>> t2 = np.arange(0.0,5.0,0.02)
>>> plt.figure(1)
<matplotlib.figure.Figure object at 0x0433FF50>
>>> plt.subplot(211)
<matplotlib.axes.AxesSubplot object at 0x02700830>
>>> plt.plot(t1,f(t1),'bo',t2,f(t2),'k')
[<matplotlib.lines.Line2D object at 0x04463310>, <matplotlib.lines.Line2D object at 0x0447E570>]
>>> plt.subplot(212)
<matplotlib.axes.AxesSubplot object at 0x0447E450>
>>> plt.plot(t2,np.cos(2*np.pi*t2),'r--')
[<matplotlib.lines.Line2D object at 0x04530510>]
>>> plt.show()
复制代码


 

标题写箭头：plt.title('$\leftarrow$')  

 柱形图：

复制代码
import numpy as np
import matplotlib.pyplot as plt

mu,sigma = 100,15
x = mu + sigma*np.random.randn(10000)
# the histogram of the data
n,bins,patches = plt.hist(x,50,normed=1,facecolor='g',alpha=0.75)
plt.xlabel('Smarts')
plt.ylabel('Probability')
plt.title('Histogram of IQ')
plt.text(60,.025,r'$\mu=100,\ \sigma=15$')
plt.axis([40,160,0,0.03])
plt.grid(True)
plt.show()
复制代码
 



        这个plt.hist(x,y)整死我了，用法和参数及表示的意思都没有地方所，就连官方网都只是一段代码和几个不明白的参数，这个函数的目的都没有写。研究了一半天就在我快要放弃的时候，我最终从数据看出来了。

最后感觉这个还真是比MATLAB更智能点。。不好看懂，因为有些数据它是直接计算再进行画图的。

       y 是图片里所要分的区间数，X是要处理的数据。运行函数plt.hist(x,y)后，会显示两组数据。

        第二组：是y+1个数组，首尾分别是X组中的最小的和最大的数，里面分成y段即y个区间，成员数据大小呈等差增加

        第一组：X数组中的数据在每个区间分布的个数。

  时间紧就不上图了
