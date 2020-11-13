# class 6 Graphic processing units and CUDA

[toc]

## 1 趣味小知识

GPU现在用来做什么

渲染图片：

1.给出相机位置，计算顶点

2.将顶点分组为原语

3.生成一个片段，每个像素一个原始重叠

4.计算每个片段的原语颜色(基于场景照明和原始材质属性)



早期图形编程OpenGL



把GPU像数据并行编程系统一样使用



## 2 GPU 计算模式

OS干涉较少计算过程



## 3 CUDA

cuda线程实现方法不同：

线程均指cuda线程

cuda程序由一个层次的并发线程组成

\_\_global\_\_表示cuda核函数

\_\_device\_\_表示GPU上SPMD执行

SPMD线程的数量在程序中显式表示



执行模式

内存模式：memcpy复制原语

设备内存模式：有共享



一个栗子：1维卷积

\_\_shared\_\_ 共享空间



cuda同步结构：

\_\_syncthreads()  barrier

原子操作  多个线程对一个数据进行操作时

主机\设备 同步