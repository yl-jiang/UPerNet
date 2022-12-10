# Pytorch分布式训练
## DDP
DDP有不同的使用模式。DDP的官方最佳实践是，每一张卡对应一个单独的GPU模型（也就是一个进程），在下面介绍中，都会默认遵循这个pattern。
举个例子：我有两台机子，每台8张显卡，那就是2x8=16个进程，并行数是16。

但是，我们也是可以给每个进程分配多张卡的。总的来说，分为以下三种情况：

+ 每个进程一张卡。这是DDP的最佳使用方法;
+ 每个进程多张卡，复制模式。一个模型复制在不同卡上面，每个进程都实质等同于DP模式。这样做是能跑得通的，但是，速度不如上一种方法，一般不采用;
+ 每个进程多张卡，并行模式。一个模型的不同部分分布在不同的卡上面。例如，网络的前半部分在0号卡上，后半部分在1号卡上。这种场景，一般是因为我们的模型非常大，大到一张卡都塞不下batch size = 1的一个模型;

### 基本概念

在16张显卡，16的并行数下，DDP会同时启动16个进程。下面介绍一些分布式的概念。

#### group

即进程组。默认情况下，只有一个组。这个可以先不管，一直用默认的就行。

#### world size

表示全局的并行数，简单来讲，就是2x8=16。
```python
# 获取world size，在不同进程里都是一样的，得到16
torch.distributed.get_world_size()
```

#### rank

表现当前进程的序号，用于进程间通讯。对于16的world sizel来说，就是0,1,2,…,15。
注意：rank=0的进程就是master进程。
```python
# 获取rank，每个进程都有自己的序号，各不相同
torch.distributed.get_rank()
local_rank
```
又一个序号。这是每台机子上的进程的序号。机器一上有0,1,2,3,4,5,6,7，机器二上也有0,1,2,3,4,5,6,7
```python
# 获取local_rank。一般情况下，你需要用这个local_rank来手动设置当前模型是跑在当前机器的哪块GPU上面的。
torch.distributed.local_rank()
```
## torch.distributed.barrier()工作原理
### 1、背景介绍
在pytorch的多卡训练中，通常有两种方式，一种是单机多卡模式（存在一个节点，通过```torch.nn.DataParallel(model)```实现），一种是多机多卡模式（存在一个节点或者多个节点，通过```torch.nn.parallel.DistributedDataParallel(model)```，在单机多卡环境下使用第二种分布式训练模式具有更快的速度。pytorch在分布式训练过程中，对于数据的读取是采用主进程预读取并缓存，然后其它进程从缓存中读取，不同进程之间的数据同步具体通过```torch.distributed.barrier()```实现。

### 2、通俗理解torch.distributed.barrier()
代码示例如下：

```python
def create_dataloader():
    #使用上下文管理器中实现的barrier函数确保分布式中的主进程首先处理数据，然后其它进程直接从缓存中读取
    with torch_distributed_zero_first(rank):
        dataset = LoadImagesAndLabels()
 
 
from contextlib import contextmanager
 
#定义的用于同步不同进程对数据读取的上下文管理器
@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """
    Decorator to make all processes in distributed training wait for each local_master to do something.
    """
    if local_rank not in [-1, 0]:
        torch.distributed.barrier()
    yield   #中断后执行上下文代码，然后返回到此处继续往下执行
    if local_rank == 0:
        torch.distributed.barrier()
```

#### （1）进程号rank理解
在多进程上下文中，我们通常假定rank 0是第一个进程或者主进程，其它进程分别具有0，1，2不同rank号，这样总共具有4个进程。

#### （2）单一进程数据处理
通常有一些操作是没有必要以并行的方式进行处理的，如数据读取与处理操作，只需要一个进程进行处理并缓存，然后与其它进程共享缓存处理数据，但是由于不同进程是同步执行的，单一进程处理数据必然会导致进程之间出现不同步的现象，为此，torch中采用了barrier()函数对其它非主进程进行阻塞，来达到同步的目的。

#### （3）barrier()具体原理
在上面的代码示例中，如果执行```create_dataloader()```函数的进程不是主进程，即rank不等于0或者-1，上下文管理器会执行相应的```torch.distributed.barrier()```，设置一个阻塞栅栏，让此进程处于等待状态，等待所有进程到达栅栏处（包括主进程数据处理完毕）；如果执行```create_dataloader()```函数的进程是主进程，其会直接去读取数据并处理，然后其处理结束之后会接着遇到```torch.distributed.barrier()```，此时，所有进程都到达了当前的栅栏处，这样所有进程就达到了同步，并同时得到释放。


---
## Reference
1. [torch.distributed.barrier()](https://blog.csdn.net/weixin_41041772/article/details/109820870)