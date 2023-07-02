##### GPU资源消耗情况统计方法
* pynvml（nvidia官方包，使用独立显卡）
   * 使用方式：1.使用nvmlDeviceSetAccountingMode方法开启各个显卡对于进程使用情况的统计模式（需要root权限，sudo命令运行py脚本）；2. 使用mlDeviceGetAccountingStats方法获取进程GPU使用情况的统计信息。
   * 可测的指标：1.显卡相关：每张显卡的显存使用情况、每张显卡的计算能力使用率（前一小段时间内的使用率）； 2.进程相关：每个进程在其生命周期内的对GPU的
   平均使用率（使用GPU的时间/进程生命周期）、生命周期中显存的最大占用量。
   * 问题：无法测量在一小段时间内某个进程的GPU使用率（CPU可以利用Process.cpu_percent计算相邻两次调用之间某个进程的cpu占用率），
   即无法测量在任务执行期间工作进程的GPU使用率。
   * 尝试的解决方法：开启一个监听进程，实时监控GPU的使用情况，将实时统计到的进程的GPU利用率写入一个dict，其他工作进程从dict中读取自己的GPU利用率
   （dict需要实现并发控制）。
   * 出现的问题：工作进程执行任务是事件驱动式的，不知何时任务会到达，可能很长一段时间内工作进程都空闲，没有使用GPU。而pynvml
   只能统计每个进程在其生命周期内的对GPU的平均使用率（使用GPU的时间/进程生命周期），因此过一段时间这个值就会变为0，失去意义。实验还发现，如果工作进程
   一直在执行任务（使用GPU），这个值就非0，是可用的。
   * 参考链接：
   ```url
   https://www.cnblogs.com/devilmaycry812839668/p/15563995.html 
   https://www.programcreek.com/python/example/123631/pynvml.nvmlDeviceGetCount 
   https://docs.nvidia.com/deploy/nvml-api/structnvmlAccountingStats__t.html#structnvmlAccountingStats__t_194e947138ea6659f190393f599d23941 
   https://www.programcreek.com/python/example/123629/pynvml.nvmlDeviceGetUtilizationRates 
   https://nvitop.readthedocs.io/en/latest/api/libnvml.html#nvitop.libnvml.NVMLError_NotFound
   pynvml.py(源码)
   https://github.com/JeremyMain/ngputop/wiki
   ```
* py3nvml（pynvml的补充版）
  * 可测的指标与pynvml基本相同，存在的问题也相同。
  * 参考链接：
  ```url
  https://pypi.org/project/py3nvml/
  ```
* nvitop（第三方库，使用独立显卡）
  * 使用方式：见参考链接中github文档中Quick Start部分.
  * 可测的指标：除了对显卡的使用情况进行监控，也可以对某一时刻、某个GPU上正在运行的进程的情况进行快照（GpuProcess.take_snapshots）；
  其中snapshot.gpu_sm_utilization参数表示在当前时刻，进程使用的sm数量/显卡的sm总数（百分比），一定程度上可以作为进程对GPU计算能力使用情况的指标。
  * 问题：1.进程使用cpu的利用率指的是：进程在cpu上执行的时间/采样时间间隔（时间/时间，多个核则累加），
  但这里gpu_sm_utilization的含义是进程使用的sm数量/显卡的sm总数（数量/数量），两个利用率的物理含义不同；2.这种方法是实时地对GPU上正在运行的进程进行快照，因此需要开启一个监控进程实时地运行此方法。
  但实验发现，如果工作进程一直在执行任务（使用GPU），监控进程可以获取到有效的gpu_sm_utilization值；但如果不是一直执行（事件驱动，或周期性sleep），该值就一直是0，
  似乎在任务执行期间监控进程不会进行采样，但是会在任务执行前后采样，因为在GpuProcess中可以看到工作进程，但gpu_sm_utilization为0.
  * 思考：对于事件驱动型的任务执行方式，在当前的调度系统设计方式下，难以获取在任务执行期间的GPU利用率。
  * 参考链接：
  ```url
  https://zhuanlan.zhihu.com/p/577533593
  https://github.com/XuehaiPan/nvitop/tree/main
  https://nvitop.readthedocs.io/en/latest/api/device.html
  ```
* jetson-stats（jtop，nano/tx2）
  * 使用方式：见参考链接中的字段说明，可获得的参数就是命令行执行jtop看到的字段.
  * 可测的指标：某个时刻GPU的计算负载、显存占用情况；某个时刻GPU上运行的所有进程的显存占用情况。
  * 问题：1.无法获取某个进程的GPU计算能力占用率；2.即使是获取进程占用的显存，也需要开始监控进程，不断执行jtop指令。
  * 参考链接：
  ```url
  https://pypi.org/project/jetson-stats/
  https://rnext.it/jetson_stats/reference/jtop.html
  https://rnext.it/jetson_stats/reference/jtop.html#jtop.jtop.__init__
  https://forums.developer.nvidia.com/t/how-to-read-tegrastats-gpu-utilisation-values/147424
  https://docs.nvidia.com/drive/drive_os_5.1.6.1L/nvvib_docs/index.html#page/DRIVE_OS_Linux_SDK_Development_Guide/Utilities/util_tegrastats.html
  https://developer.nvidia.com/docs/drive/drive-os/latest/linux/sdk/common/topics/util_setup/Toruntegrastats15.html
  https://www.jianshu.com/p/37f97517ef97
  https://forums.developer.nvidia.com/t/how-to-read-tegrastats-gpu-utilisation-values/147424
  ```
* GPU相关知识：
  * 参考链接：
  ```url
  https://zhuanlan.zhihu.com/p/396658689
  https://blog.csdn.net/asasasaababab/article/details/80447254
  ```

##### 其他资源消耗情况统计方法
* cpu：
  ```url
  https://hellowac.github.io/psutil-doc-zh/processes/process_class/cpu_percent.html
  https://psutil.readthedocs.io/en/latest/#psutil.Process.cpu_percent
  https://blog.csdn.net/gymaisyl/article/details/101274862
  https://blog.csdn.net/niaolianjiulin/article/details/82726126
  https://blog.csdn.net/Hubz131/article/details/94414013
  ```
* 内存：
  ```url
  https://hellowac.github.io/psutil-doc-zh/processes/process_class/memory_full_info.html
  https://hellowac.github.io/psutil-doc-zh/processes/process_class/memory_info.html
  ```
* cgroupsy：
  ```url
  https://github.com/cloudsigma/cgroupspy
  https://www.cnblogs.com/lsdb/p/13140210.html
  https://pypi.org/project/cgroupspy/
  https://access.redhat.com/documentation/en-us/red_hat_enterprise_linux/6/html/resource_management_guide/sec-cpuset
  https://www.kernel.org/doc/Documentation/cgroup-v1/
  ```
  * 注意：
    * 若服务器重启，则需要root用户对目录/sys/fs/cgroup下的文件夹提供权限（chmod 777 cpu，cpuset和memory类似）
    * cfs_quota_us的设置表示进程对所有核的利用率之和，而不是对每个核各自的利用率。
    如果不指定进程使用的cpuset（即默认可以使用所有的cpu核），且想要限制cpu对每个核的利用率均为50%，则cfs_period_us设置为1000000，
    cfs_quota_us设置为500000*核数，而不是500000；如果设置为500000则表示进程对所有核的利用率加起来是50%而不是每个核50%；
    如果希望进程对每个核的利用率不受限制的话，cfs_quota_us设置为-1，而不是1000000。

* 资源问题总结：
  * 当前系统可以准确测量的指标：1. 当前时刻设备整体的资源情况均可准确测量（cpu利用率、内存占用率、GPU显存占用率、GPU计算负载）； 2. 执行任务过程中
  的cpu利用率。
  * 当前系统无法准确测量的指标：1. 任务执行过程中对内存的消耗（即执行前后工作进程的内存占用差），对于一个工作进程，第一次执行任务前后内存差很明显，
  但后续执行任务时内存差为0。对于这种现象，考虑将第一次执行任务前的占用内存值m保存下来，统计每次执行任务之后的占用内存值n_i，将(n_i-m)作为执行任务对内存的消耗。
  这种方法依然无法刻画“任务执行过程中对内存的消耗”这个指标；2. 任务执行过程中对显存的消耗（即执行前后工作进程的显存占用差），现象与解决方法和内存相同；3. 
  任务执行过程中对GPU计算能力的占用率。
  * 资源消耗情况不好测量的根本原因是：在我们的系统中，工作进程是预装好的，这意味着执行任务前工作进程就已经占用了一定的cpu、内存资源；而内存、显存的消耗设计
  操作系统的页面替换控制，我们无法强制系统在每次执行任务之后就将工作进程在内存中的页面替换出去，所以出现了后续执行任务时内存差为0的情况.
  * 是否可以考虑将任务执行过程中工作进程占用的内存、显存量作为任务执行过程中对内存和显存的消耗，但这样粒度很粗，不一定能看出资源消耗随视频流内容的变化，
  操作系统的优化也是一个重要影响因素。