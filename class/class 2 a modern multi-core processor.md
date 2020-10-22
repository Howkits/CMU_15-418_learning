# class 2 a modern multi-core processor

[TOC]

## 1 how to accelerate

1.使用更多的晶体管，增加核心数量

2.通过多个ALU分摊指令的开销，SIMD单指令多数据

（标量编程->向量编程）

3.有分支时，会产生无效的工作，降低了并行的效率

指令流一致性（核内）

4.CPU：显式的并行

5.GPU：隐式的并行，硬件接口本身就是数据并行的

分支执行产生的低效问题更加严重

6.超标量



## 2 accessing memory

内存延迟、内存带宽

数据相关

->

1.Cache（降低了延迟），处理器需要的数据储存在cache中，Cache还提供了更大的带宽

2.预取（隐藏了延迟），取错了也会造成影响

3.多线程（隐藏），核心为线程安排，交错，超标量的拓展

带宽限制了发挥

少请求数据，多做运算