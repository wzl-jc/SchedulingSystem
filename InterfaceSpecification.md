### <center>接口说明文档 </center>
#### 一、提交数据获取执行结果的接口：
* 最初版本接口如下（配合最初版本的face_detection封装代码）：
```url
/execute_task/<string:task_name>  methods=["POST"]
```
* 新版本的任务执行接口（配合新版本的face_detection封装代码）：
```url
/execute_task_new/<string:task_name>  methods=["POST"]
```
* 新版本的任务并行执行接口（配合新版本的face_detection封装代码，多目标任务并行执行并汇总结果）：
```url
/concurrent_execute_task_new/<string:task_name>  methods=["POST"]
```

* 其中，task_name为用户执行的任务名称，用户可执行的任务如下：
```python
task_name_list = ["face_detection", "face_alignment", "car_detection", "helmet_detection"]
```

* 使用json格式向接口传递数据，具体格式说明如下：
    * face_detection的输入json格式如下：
    ```json5
    {
        "image": [[1, 1, 1], [1, 1, 1]]  // 待推理的图像，list类型，可使用array.tolist()获得
    }
    ```
    * 并行执行任务接口的输入json格式如下：
    ```json5
    {
        "image_list": []  // 待推理的图像列表，list类型，每一个元素是编码后的图片(string类型)
    }
    ```
    * 新版封装代码，输入json格式如下：
    ```json5
    {
        "image": "\x003\x001..." // 待推理的图像，字符串类型，使用特殊编码方式降低传输开销
    }
    ```
    * face_detection的输出json格式如下：
    ```json5
    {
        // "image": [[1, 1, 1], [1, 1, 1]],  // 图像，list类型，可使用array.tolist()获得
        "bbox": [],     // 图像中的bounding box坐标，list类型
        "prob": []      // 图像中各个bounding box中包含人脸的概率，list类型
    }
    ```
    * 并行执行任务接口的输出json格式如下：
    ```json5
    {
        "output_ctx_list": [   // 每一帧图像运行结果组成的列表，列表元素为dict（即为之前的output_ctx）
            {
                "bbox": [],     // 图像中的bounding box坐标，list类型
                "prob": []      // 图像中各个bounding box中包含人脸的概率，list类型
            }
        ],
    }
    ```
    * 新版封装代码，输出json格式如下：
    ```json5
    {
        "faces": ["\x003\x001..."],  // 检测出来的人脸图像，字符串列表
        "bbox": [[1,2,3,4], [1,2,3,4]],
        "prob": []
    }
    ```
    * face_alignment的输入json格式如下：
    ```json5
    {
        "image": [[1, 1, 1], [1, 1, 1]],  // 图像，list类型，可使用array.tolist()获得
        "bbox": [],     // 图像中的bounding box坐标，list类型
        "prob": []      // 图像中各个bounding box中包含人脸的概率，list类型
    }
    ```
    * 并行执行任务接口的输入json格式如下：
    ```json5
    {
        "image_list": []  // 待推理的图像列表，list类型，每一个元素是编码后的图片(string类型)，即小的人脸图片，而非整张图+bbox
    }
    ```
    * 新版封装代码，输入json格式如下：
    ```json5
    {
        "faces": ["\x003\x001..."],  // 待姿态估计的人脸图像，字符串列表
        "bbox": [[1,2,3,4], [1,2,3,4]],
        "prob": []
    }
    ```
    * face_alignment的输出json格式如下：
    ```json5
    {
        // "image": [[1, 1, 1], [1, 1, 1]],  // 图像，list类型，可使用array.tolist()获得
        "bbox": [],          // 图像中的bounding box坐标，list类型
        "head_pose": []      // 图像中各个bounding box中包含人脸的姿态，list类型
    }
    ```
    * 并行执行任务接口的输出json格式如下：
    ```json5
    {
        "output_ctx_list": [   // 每一帧图像运行结果组成的列表，列表元素为dict（即为之前的output_ctx）
            {
                "head_pose": []      // 人脸图像的姿态，类型待确定
            }
        ],
    }
    ```
    * 新版封装代码，输出json格式如下：
    ```json5
    {
        "count_result": {  // 可以显示的数值结果
            "up": 6,
            "total": 8
        },
        // 其余字段可另行添加
    }
    ```
    * car_detection的输入json格式如下：
    ```json5
    {
        "image": [[1, 1, 1], [1, 1, 1]],  // 图像，list类型，可使用array.tolist()获得
    }
    ```
    * 并行执行任务接口的输入json格式如下：
    ```json5
    {
        "image_list": []  // 待推理的图像列表，list类型，每一个元素是编码后的图片(string类型)，即小的人脸图片，而非整张图+bbox
    }
    ```
    * car_detection的输出json格式如下：
    ```json5
    {
        // "image": [[1, 1, 1], [1, 1, 1]],  // 图像，list类型，可使用array.tolist()获得
        "result": [[1029.0, 489.0, 1150.0, 580.0, 0.8599833250045776, 2.0]]  
        // 包含所有检测结果的list，每个检测结果包含了一个检测框的坐标、置信度、类别
    }
    ```
    * 并行执行任务接口的输出json格式如下：
    ```json5
    {
        "output_ctx_list": [   // 每一帧图像运行结果组成的列表，列表元素为dict（即为之前的output_ctx）
            {
                "result": []      // 包含所有检测结果的list，每个检测结果包含了一个检测框的坐标、置信度、类别
            }
        ],
    }
    ```
    * helmet_detection的输入json格式如下：
    ```json5
    {
        // 图像列表，list类型；每一个元素是一帧图像，list类型，可使用array.tolist()获得
        "image_list": [[[1, 1, 1], [1, 1, 1]], [[2, 2, 2], [2, 2, 2]]]
    }
    ```
    * helmet_detection的输出json格式如下：
    ```json5
    {
        "bbox_list": [[0, 0, 100, 100], [200, 200, 300, 300]]
    }
    ```

* 另外，为了实现运行时情境感知，所有类型任务的输出json都包含了以下字段：
```json5
    {
        "proc_resource_info_list": [  // 执行当前任务的所有进程的资源消耗情况，list
            {  // 执行当前任务的某个进程的资源消耗情况
                "cpu_util_limit": 1.0,  // 进程可以使用的cpu利用率上限（通过cgroup设置）
                "cpu_util_use": 0.48875,  // 进程在执行任务时的实际cpu利用率
                "latency": 0.15465092658996582,  // 进程执行该任务的时延 
                "pid": 3009737  // 进程pid
            }
        ]
    }
```

#### 二、获取当前服务列表的接口：
* 接口如下：
```url
/get_service_list  methods=["GET"]
```
* 无需任何参数，以json格式返回当前可执行任务的名称列表，如下：
```python
res = ["face_detection", "face_alignment", "car_detection"]
```

#### 三、获取某一服务所有可用的url：
* 接口如下：
```url
/get_execute_url/<string:task_name>  methods=["GET"]
```
* 其中，task_name为用户查询的任务名称；以json格式返回，示例如下：
```json5
{
    "114.212.81.11:5500": {
        "bandwidth": 1,
        "cpu": 1,
        "mem": 1,
        "url": "http://114.212.81.11:5500/execute_task/face_detection"
    },
    "172.27.138.183:5501": {
        "bandwidth": 1,
        "cpu": 1,
        "mem": 1,
        "url": "http://172.27.138.183:5501/execute_task/face_detection"
    },
    "172.27.151.135:5501": {
        "bandwidth": 1,
        "cpu": 1,
        "mem": 1,
        "url": "http://172.27.151.135:5501/execute_task/face_detection"
    }
}
```

#### 四、获取当前系统情况的接口：
* server端接口如下：
```url
/get_system_status  methods=["GET"]
```
```json5
{
    "114.212.81.11:5500": {  //以ip:port为key，标记一个节点
        "cpu_ratio": [0.2, 0.0, 0.2, 0.1, 0.3, 0.2, 0.0, 0.7, 0.9],  //节点各个cpu的占用百分比列表
        "mem_ratio": 5.4,  //节点的内存占用百分比
        "net_ratio(MBps)": 0.31806,  //节点的带宽
        "swap_ratio": 0.0, //节点交换内存使用情况
        "gpu_mem":  {  //节点各个GPU的显存占用百分比字典
            "0": 0.012761433919270834, // 第0张显卡
            "1": 0.012761433919270834, // 第1张显卡
            "2": 0.012761433919270834, 
            "3": 0.012761433919270834
        },
        "gpu_utilization": {  //节点各个GPU的计算能力利用率百分比字典
            "0": 7, // 第0张显卡；nano或tx2没有显卡，因此只有"0"这一个键；服务器有多张显卡
            "1": 8, // 第1张显卡
            "2": 9, 
            "3": 10
        },
        "car_detection": {  //以服务名为key，表示节点上某类服务的情况
            "1194975": {  //以进程pid为key，表示节点上某类服务某个工作进程的情况
                "cpu_ratio": 0.0,  //某个进程的cpu使用率
                "mem_ratio": 0.20176213870657234,  //某个进程的内存占用率
                "task_to_do": 0  //某个进程待做的任务数量
            }
        },
        "face_alignment": {
            "1196516": {
                "cpu_ratio": 0.0,
                "mem_ratio": 0.14147854986806502,
                "task_to_do": 0
            }
        },
        "face_detection": {
            "1196515": {
                "cpu_ratio": 0.0,
                "mem_ratio": 0.11392224417874179,
                "task_to_do": 0
            }
        }
    },
    "172.27.142.109:5501": {
        "car_detection": {
            "784": {
                "cpu_ratio": 0.0,
                "mem_ratio": 1.7194944849511828,
                "task_to_do": 0
            }
        },
        "cpu_ratio": [2.7, 0.7, 2.0, 1.3, 1.8, 0.7, 1.4],
        "face_alignment": {
            "1896": {
                "cpu_ratio": 0.0,
                "mem_ratio": 0.9882168921417035,
                "task_to_do": 0
            }
        },
        "face_detection": {
            "10832": {
                "cpu_ratio": 0.0,
                "mem_ratio": 0.8926369264778068,
                "task_to_do": 0
            }
        },
        "mem_ratio": 72.9,
        "net_ratio(MBps)": 0.10514,
        "swap_ratio": 89.9,
        "gpu_ratio":  {  
            "0": 0.012761433919270834, 
            "1": 0.012761433919270834, 
            "2": 0.012761433919270834, 
            "3": 0.012761433919270834
        }
    }
}
```
#### 五、增加/减少某类任务工作进程的接口：
* server/client端接口如下：
```url
/add_work_process  methods=["POST"]
```
* 发起请求的json格式说明如下：
```json5
{
    "task_name": "car_detection",  // 任务名
    // 创建该类任务时的模型参数，非必选参数，如果用户未提供model_ctx则使用系统中该类任务的工作进程正在使用的模型参数
    "model_ctx": {  
        "weights": "yolov5s.pt",
        "device": "cpu"
    }
}
```
```url
/decrease_work_process  methods=["POST"]
```
* 发起请求的json格式说明如下：
```json5
{
    "task_name": "car_detection",  // 任务名
}
```

#### 六、设置某个工作进程占用资源的接口：
* server/client端接口如下：
```url
/limit_process_resource  methods=["POST"]
```
```json5
{
    "pid": 10832,  // 进程pid
    "task_name": "face_detection",  // 任务名称
    //"mem_limit": 1024000000,  // 进程内存用量上限，单位：byte；-1表示不调整内存限制
    "cpu_util_limit": 0.45,  // 进程cpu占用率（所有核），取值[0,1]
    //"cpu_set_cpus":[0, 1],  // 进程使用的cpu，list类型
    //"cpu_set_mems": [0, 1]  // 进程使用的mem node，list类型
}
```
#### 七、运行时情境相关字段：
* sniffer.py，函数describe_runtime()，资源情境（runtime_resource变量）：
```json5
{  // 以各个子任务的名字为key，分别获取各个子任务的资源情境
    "face_detection": {
        "flow_mapping": {  // 当前子任务调用系统中哪个节点上的计算服务
            "model_id": 0,
            "node_ip": "192.168.56.102",
            // node_role有三种可能：host、edge、cloud，前端只区分cloud和非cloud，非cloud显示为“边端”
            // host表示在当前节点本地做，cloud表示在云端做，edge表示在其他边缘节点做（边边协同，暂未实现）
            "node_role": "host" 
        },
        "proc_resource_info_list": [  // 执行当前任务的所有进程的资源消耗情况，list
            {  // 执行当前任务的某个进程的资源消耗情况
                "cpu_util_limit": 1.0,  // 进程可以使用的cpu利用率上限（通过cgroup设置）
                "cpu_util_use": 0.48875,  // 进程在执行任务时的实际cpu利用率
                "latency": 0.15465092658996582,  // 进程执行该任务的时延 
                "pid": 3009737  // 进程pid
            },
            {  
                "cpu_util_limit": 1.0,  
                "cpu_util_use": 0.2175,  
                "latency": 0.37465092658996582,   
                "pid": 3009738 
            },
        ]
    },
    "face_alignment": {
        // 格式同上
    }
}
```
* sniffer.py，函数describe_runtime()，资源情境（runtime_desc变量）：
```json5
{
    "runtime_resource_info": {
        "cur_runtime_resource_info": {
        // 当前调度周期内的资源情境，value即为runtime_resource变量
        },
        "pre_runtime_resource_info": {
        // 前一调度周期内的资源情境，同时传给scheduler，便于对比调整资源的效果，从而决定下一个周期要不要继续调资源配置
        }
    },
}
```
* 调度器函数（resource_alloc变量）：
```json5
{   
    "face_detection": {
        // 资源分配操作类型，为1表示保持现有进程资源分配（用于冷启动、时延满足要求、修改其他调度参数等资源分配不变的情况）；
        // 为2表示调整现有进程资源分配方式；为3表示增加新的进程；为4表示减少进程
        "opt_type": 1  // 为1，则只有opt_type字段
    },
    "face_alignment": {
        "opt_type": 2,
        "proc_resource_limit": {  // 每个进程新的资源分配上限
            "12345": {  // key为进程pid
                "cpu_util_limit": 0.5  // 进程新的cpu利用率上限
            },
            "12346": {
                "cpu_util_limit": 0.6
            }
        }
    },
    "car_detection": {
        "opt_type": 3  // 为3，则只有opt_type字段
    },
    "car_classy": {
        "opt_type": 4,
        "node_ip": "114.212.81.11"  // 减少进程数的节点ip
    },
    
}
```