# class 3 Parallel programming models

[TOC]

## 1 ISPC Program

没有使用线程，而是一组ISPC程序实例

uniform 关键字 不同的instance中都使用相同的以这个关键字修饰的变量

foreach 更高层次的抽象

需要人工划分任务->通过内置的变量programCount等进行任务的控制

都只是在一个核心上进行

task可以实现使用多个核心

## 2 communicating and cooperation

pthreads：create

三种方法：

1.共享地址空间：精巧的结构->大公告板 ->问题在于同步->读者写者问题（加锁、互斥）

**非一致性内存访问**（Non-uniform memory access,NUMA)

硬件上：访问权限相同，访问距离不同->延迟

2.信息传递（Message passing)：高结构化交流

数据私有，靠send/receive沟通

**消息传递接口**(message passing interface,MPI)



两种方法的一致性：消息传递类似内存缓冲区读写；可在不支持共享地址空间实现的硬件上实现这个抽象->速度慢

3.数据并行模型：严格的计算结构



Gather/Scatter



多种角度思考问题->如何实现并行之间的交流沟通？->如何有效分配任务？



