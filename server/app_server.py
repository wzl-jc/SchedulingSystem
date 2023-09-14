import platform

import requests
from flask import Flask, make_response, request, render_template, jsonify, send_from_directory
import json
import zipfile
import os
import multiprocessing as mp
import sys
import importlib
from werkzeug.serving import WSGIRequestHandler
from flask_apscheduler import APScheduler
from field_codec_utils import decode_image, encode_image
import psutil
import time
import argparse


def work_func(task_name, q_input, q_output, model_ctx=None):
    task_name_split = task_name.split('_')  # 以下划线分割
    task_class_name = ''  # 任务对应的类名
    for name_split in task_name_split:
        temp_name = name_split.capitalize()   # 各个字符串的首字母大写
        task_class_name = task_class_name + temp_name

    # 使用importlib动态加载任务对应的类
    path1 = os.path.abspath('.')
    path2 = os.path.join(path1, task_name)
    path3 = os.path.join(path2, '')
    sys.path.append(path3)   # 当前任务代码所在的文件夹，并将其加入模块导入搜索路径
    module_path = task_name + '.' + task_name  # 绝对导入
    module_name = importlib.import_module(module_path)  # 加载模块，例如face_detection
    class_obj = getattr(module_name, task_class_name)  # 从模块中加载执行类，例如FaceDetection

    # 创建执行类对应的对象
    if model_ctx is not None:
        work_obj = class_obj(model_ctx)
    else:
        work_obj = class_obj()

    # 开始监听，并循环执行任务
    '''
    task_num = 0  # 记录执行任务的次数
    total_gpu_memory = 1  # 显存的总值
    old_gpu_memory = 1  # 记录第一次执行任务之前的gpu显存占用
    total_memory = 1  # 物理内存的总值
    start_memory = 1  # 记录第一次执行任务之前的内存占用
    '''
    while True:
        # 从队列中获取任务
        input_ctx = q_input.get()
        '''
        task_num += 1
        # 计算第一次执行任务之前的gpu显存占用
        # if task_num == 1:
        if 'device' in model_ctx and model_ctx['device'] != 'cpu' and torch.cuda.is_available():
            # 若模型指定了设备，且设备不是cpu，且gpu可用，则统计gpu消耗情况
            if platform.uname().machine[0:5] == 'aarch':
                # 如果是aarch架构，则没有独立显卡
                pass
            else:
                gpu_index = int(model_ctx['device'][-1])
                import pynvml
                UNIT = 1024 * 1024
                pynvml.nvmlInit()  # 初始化
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
                old_memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                total_gpu_memory = old_memory_info.total / UNIT
                old_gpu_memory = old_memory_info.used / UNIT
        '''

        pid = os.getpid()  # 获得当前进程的pid
        p = psutil.Process(pid)  # 获取当前进程的Process对象

        '''
        # 第一次执行任务之前的内存占用
        if task_num == 1:
            total_memory = psutil.virtual_memory().total  # 总物理内存
            start_memory = p.memory_full_info().uss  # 任务执行前进程占用的内存值
        before_memory = p.memory_full_info().uss
        before_data_memory = p.memory_info().data
        '''

        # cpu占用的计算
        start_cpu_per = p.cpu_percent(interval=None)  # 任务执行前进程的cpu占用率
        # start_cpu_time = p.cpu_times()  # 任务执行前进程的cpu_time

        # 延时的计算
        start_time = time.time()  # 计时，统计延时

        # 执行任务
        output_ctx = work_obj(input_ctx)

        # 延时的计算
        end_time = time.time()

        # cpu占用的计算
        end_cpu_per = p.cpu_percent(interval=None)  # 任务执行后进程的cpu占用率
        end_cpu_time = p.cpu_times()  # 任务执行后进程的cpu_time

        '''
        # 内存消耗的计算
        end_memory = p.memory_full_info().uss  # 任务执行之后进程占用的内存值，每次执行任务之后都重新计算
        after_data_memory = p.memory_info().data
        '''
        '''
        # gpu消耗的计算
        if 'device' in model_ctx and model_ctx['device'] != 'cpu' and torch.cuda.is_available():
            # 若模型指定了设备，且设备不是cpu，且gpu可用，则统计gpu消耗情况
            if platform.uname().machine[0:5] == 'aarch':
                pass
            else:
                gpu_index = int(model_ctx['device'][-1])
                import pynvml
                UNIT = 1024 * 1024
                pynvml.nvmlInit()  # 初始化
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
                new_memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                # total_gpu_memory = new_memory_info.total
                new_gpu_memory = new_memory_info.used / UNIT  # 任务执行之后显存的占用率，每次执行任务之后都重新计算
                # print(new_gpu_memory, old_gpu_memory, total_gpu_memory)
                output_ctx['gpu_mem_use'] = (new_gpu_memory - old_gpu_memory) / total_gpu_memory * 100
                output_ctx['gpu_mem_use_1'] = (new_gpu_memory - old_gpu_memory) / 1024
                output_ctx['gpu_mem_use_2'] = new_gpu_memory / 1024
        '''


        '''
        output_ctx['cpu_all_time'] = (end_cpu_time.user - start_cpu_time.user) + \
                                     (end_cpu_time.system - start_cpu_time.system)  # 任务执行的cpu占用时间，包括用户态+内核态
        output_ctx['cpu_user_time'] = end_cpu_time.user - start_cpu_time.user
        output_ctx['cpu_sys_time'] = end_cpu_time.system - start_cpu_time.system
        output_ctx['cpu_use_1'] = output_ctx['cpu_all_time'] / output_ctx['latency']
        '''
        '''
        output_ctx['mem_use'] = (end_memory - start_memory) / total_memory * 100  # 任务执行消耗的内存占总物理内存的百分比
        output_ctx['mem_use_1'] = (end_memory - start_memory) / 1024
        output_ctx['mem_use_2'] = end_memory / 1024
        output_ctx['mem_use_3'] = (end_memory - before_memory) / 1024
        output_ctx['mem_use_data'] = (after_data_memory - before_data_memory) / 1024
        output_ctx['mem_use_data_1'] = after_data_memory / 1024
        '''

        proc_resource_info = dict()  # 执行任务过程中的资源消耗情况
        proc_resource_info['pid'] = pid
        # 任务执行的cpu占用率，各个核上占用率之和的百分比->平均每个核的占用率，范围[0,1]
        proc_resource_info['cpu_util_use'] = end_cpu_per / 100 / psutil.cpu_count()
        proc_resource_info['latency'] = end_time - start_time  # 任务执行的延时

        output_ctx['proc_resource_info'] = proc_resource_info
        q_output.put(output_ctx)


'''
def monitor_gpu(lock, pid_gpu_dict):
    # 监听各个工作进程在执行任务过程中对GPU计算能力的利用率
    if torch.cuda.is_available():
        if platform.uname().machine[0:5] == 'aarch':  # nano、tx2
            from jtop import jtop
            gpu_total_mem = 3.9 * 1024 * 1024  # nano显存总量为3.9G，未找到获取显存总量的api，直接写死，单位kB
            with jtop(0.05) as jetson:
                while jetson.ok():
                    res_dict = dict()
                    for pro in jetson.processes:
                        temp_pid = pro[0]
                        temp_dict = dict()
                        temp_dict['memoryUtilization'] = pro[-2] / gpu_total_mem * 100  # 百分比
                        res_dict[temp_pid] = temp_dict
                    lock.acquire()
                    for key in res_dict:
                        pid_gpu_dict[key] = res_dict[key]
                    lock.release()
        else:
            import pynvml
            pynvml.nvmlInit()
            gpu_device_count = pynvml.nvmlDeviceGetCount()
            while True:  # 开始监听
                res_dict = dict()
                for i in range(gpu_device_count):  # 遍历所有显卡设备
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    pid_all_info = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                    for pidInfo in pid_all_info:  # 遍历所有在显卡上正在运行的进程
                        try:
                            account_stats = pynvml.nvmlDeviceGetAccountingStats(handle, pidInfo.pid)
                        except pynvml.NVMLError:  # NVMLError_NotFound
                            pass
                        try:
                            account_stats = pynvml.nvmlDeviceGetAccountingStats(handle, pidInfo.pid)
                            temp_dict = dict()
                            temp_dict['gpuUtilization'] = account_stats.gpuUtilization
                            temp_dict['memoryUtilization'] = account_stats.memoryUtilization
                            temp_dict['maxMemoryUsage'] = account_stats.maxMemoryUsage
                            res_dict[pidInfo.pid] = temp_dict
                        except pynvml.NVMLError:  # NVMLError_NotFound
                            pass
                lock.acquire()
                for key in res_dict:
                    pid_gpu_dict[key] = res_dict[key]
                lock.release()
            pynvml.nvmlShutdown()
'''


def trigger_update_server_status():
    # 定时触发更新server状态
    temp_url = "http://" + server_manager.server_ip + ":" + str(server_manager.server_port) + "/update_server_status"
    requests.get(temp_url)
    print("trigger_update_server_status!")


def trigger_update_clients_status():
    # 定时触发获取所有边缘节点状态，心跳机制
    temp_url = "http://" + server_manager.server_ip + ":" + str(server_manager.server_port) + "/update_clients_status"
    requests.get(temp_url)
    print("trigger_get_clients_status!")


class ServerManager(object):
    def __init__(self):
        # 云端系统配置相关
        self.server_ip = '114.212.81.11'  # 默认设置需要与服务器ip保持一致，供定时事件使用
        self.server_port = 5500
        self.edge_ip_set = set()  # 存放所有边缘端的ip，用于向边缘端发送请求
        self.edge_port = 5500  # 边缘端服务器的端口号，所有边缘端统一
        self.edge_get_task_url = "/task-register"  # 边缘端接受软件下装的接口
        self.server_codebase = os.path.join(os.path.abspath('.'), "")   # 为了导入工作进程代码，需要将代码下载到当前工作目录下

        # 系统状态相关
        self.server_status = dict()  # 记录server的状态，资源状态、各任务进程状态
        self.clients_status = dict()  # 记录各个边缘端的状态

        # 工作进程执行相关
        self.process_dict = dict()  # 存放每个任务对应的工作进程，key为任务名，value为进程列表，方便动态增加各个任务工作进程数量
        self.pid_set = set()  # 存放所有工作进程的pid
        self.code_set = set()  # 云端已存储的代码对应的应用的名字
        self.input_queue_dict = dict()  # 存放每个工作进程所需的输入输出队列，value为队列列表
        self.output_queue_dict = dict()
        self.model_ctx_dict = dict()  # 存放每一类任务工作进程的模型参数，key为任务名，value为model_ctx
        self.work_process_num = 0  # 当前节点上的工作进程数

        # 工作进程资源控制相关
        self.cpu_count = psutil.cpu_count()  # 当前节点上的cpu核数
        self.process_mem_group_dict = dict()  # 记录各个进程对应的memory cgroup组，key为pid(int), value为group对象
        self.process_cpu_group_dict = dict()  # 记录各个进程对应的cpu cgroup组，key为pid(int), value为group对象
        self.process_cpu_set_group_dict = dict()  # 记录各个进程对应的cpuset cgroup组，key为pid(int), value为group对象
        self.default_resource_limit = {  # 创建工作进程时默认的进程对各类资源的使用上限
            'cpu_util_limit': 1.0
        }
        self.resource_limit_dict = dict()  # 记录各类任务各类资源的使用上限，用于在创建新进程时使用，key为任务名，value为dict

    # 云端设置相关
    def init_server_param(self, server_ip, server_port, edge_port):
        self.server_ip = server_ip
        self.server_port = server_port
        self.edge_port = edge_port

    # 工作进程管理相关
    def create_task_process(self, task_dict):
        # 以用户提交的配置文件内容为输入，为DAG图中每一个子任务都创建一个工作进程，仅在代码初次下装时调用
        task_flow = task_dict['flow']
        for task_name in task_flow:
            if task_name not in self.process_dict:  # 只有边缘节点没有当前任务的工作进程时才创建一个工作进程
                temp_input_q = mp.Queue(maxsize=10)
                temp_output_q = mp.Queue(maxsize=10)
                temp_process = mp.Process(target=work_func, args=(task_name, temp_input_q, temp_output_q,
                                                                  task_dict['model_ctx'][task_name]))
                self.process_dict[task_name] = [temp_process]
                self.input_queue_dict[task_name] = [temp_input_q]
                self.output_queue_dict[task_name] = [temp_output_q]
                temp_process.start()  # 必须先启动工作进程，只有启动之后temp_process才有pid，否则为None
                self.pid_set.add(temp_process.pid)
                self.model_ctx_dict[task_name] = task_dict['model_ctx'][task_name]

                # 使用cgroupspy限制进程使用的资源
                '''
                from cgroupspy import trees
                task_set = set()
                task_set.add(temp_process.pid)
                group_name = "process_" + str(temp_process.pid)
                t = trees.Tree()  # 实例化一个资源树
                '''
                # 限制进程使用的cpuset
                # cpuset_resource_item = "cpuset"
                # cpuset_limit_obj = t.get_node_by_path("/{0}/".format(cpuset_resource_item))  # 获取cpuset配置对象
                # cpuset_group = cpuset_limit_obj.create_cgroup(group_name)
                # cpu_id = self.work_process_num % self.cpu_count  # 为当前工作进程绑定cpu核
                # cpuset_group.controller.cpus = set([cpu_id])
                # cpuset_group.controller.mems = set([0, 1])  # 设置使用的memNode
                # cpuset_group.controller.tasks = task_set
                # self.process_cpu_set_group_dict[temp_process.pid] = cpuset_group

                # 限制进程使用的内存上限
                # memory_resource_item = "memory"
                # memory_limit_obj = t.get_node_by_path("/{0}/".format(memory_resource_item))
                # memory_group = memory_limit_obj.create_cgroup(group_name)
                # memory_group.controller.limit_in_bytes = 256 * 1024 * 1024  # 进程初始时设置内存上限为512MB
                # memory_group.controller.tasks = task_set
                # self.process_mem_group_dict[temp_process.pid] = memory_group

                # 限制进程的cpu使用率
                '''
                cpu_resource_item = "cpu"
                cpu_limit_obj = t.get_node_by_path("/{0}/".format(cpu_resource_item))
                cpu_group = cpu_limit_obj.create_cgroup(group_name)
                cpu_group.controller.cfs_period_us = 1000000
                cpu_group.controller.cfs_quota_us = int(self.default_resource_limit['cpu_util_limit'] * cpu_group.controller.cfs_period_us *
                                                        psutil.cpu_count())
                cpu_group.controller.tasks = task_set
                self.process_cpu_group_dict[temp_process.pid] = cpu_group
                self.resource_limit_dict[task_name] = dict()
                self.resource_limit_dict[task_name]['cpu_util_limit'] = self.default_resource_limit['cpu_util_limit']
                '''
                self.work_process_num += 1

    def add_work_process(self, task_info):
        # 以调度器提交请求时的json为输入，添加某类任务的工作进程，当某类任务做不过来时调用
        task_name = task_info['task_name']
        if task_name not in self.code_set:  # 如果某类任务目前没有下发到当前节点（当前节点没有执行对应任务的代码），则无法添加
            return False
        else:  # 某类任务已经下发到当前节点，可以添加
            if 'model_ctx' in task_info:
                model_ctx = task_info['model_ctx']
            else:
                assert task_name in self.model_ctx_dict
                model_ctx = self.model_ctx_dict[task_name]
            temp_input_q = mp.Queue(maxsize=10)
            temp_output_q = mp.Queue(maxsize=10)
            temp_process = mp.Process(target=work_func, args=(task_name, temp_input_q, temp_output_q,
                                                              model_ctx))
            self.process_dict[task_name].append(temp_process)
            self.input_queue_dict[task_name].append(temp_input_q)
            self.output_queue_dict[task_name].append(temp_output_q)
            temp_process.start()  # 必须先启动工作进程，只有启动之后temp_process才有pid，否则为None
            self.pid_set.add(temp_process.pid)

            # 使用cgroupspy限制进程使用的资源
            '''
            from cgroupspy import trees
            task_set = set()
            task_set.add(temp_process.pid)
            group_name = "process_" + str(temp_process.pid)
            t = trees.Tree()  # 实例化一个资源树
            '''

            # 限制进程使用的cpuset
            # cpuset_resource_item = "cpuset"
            # cpuset_limit_obj = t.get_node_by_path("/{0}/".format(cpuset_resource_item))  # 获取cpuset配置对象
            # cpuset_group = cpuset_limit_obj.create_cgroup(group_name)
            # cpu_id = self.work_process_num % self.cpu_count  # 为当前工作进程绑定cpu核
            # cpuset_group.controller.cpus = set([cpu_id])
            # cpuset_group.controller.mems = set([0, 1])  # 设置使用的memNode
            # cpuset_group.controller.tasks = task_set
            # self.process_cpu_set_group_dict[temp_process.pid] = cpuset_group

            # 限制进程使用的内存上限
            # memory_resource_item = "memory"
            # memory_limit_obj = t.get_node_by_path("/{0}/".format(memory_resource_item))
            # memory_group = memory_limit_obj.create_cgroup(group_name)
            # memory_group.controller.limit_in_bytes = 256 * 1024 * 1024  # 进程初始时设置内存上限为512MB
            # memory_group.controller.tasks = task_set
            # self.process_mem_group_dict[temp_process.pid] = memory_group

            # 限制进程的cpu使用率
            '''
            cpu_resource_item = "cpu"
            cpu_limit_obj = t.get_node_by_path("/{0}/".format(cpu_resource_item))
            cpu_group = cpu_limit_obj.create_cgroup(group_name)
            cpu_group.controller.cfs_period_us = 1000000
            cpu_group.controller.cfs_quota_us = int(self.resource_limit_dict[task_name]['cpu_util_limit'] * cpu_group.controller.cfs_period_us *
                                                    psutil.cpu_count())
            cpu_group.controller.tasks = task_set
            self.process_cpu_group_dict[temp_process.pid] = cpu_group
            '''

            self.work_process_num += 1
            return True

    def decrease_work_process(self, task_info):
        task_name = task_info['task_name']
        if task_name not in self.code_set:  # 如果某类任务目前没有下发到当前节点（当前节点没有执行对应任务的代码），则无法添加
            return False
        if len(self.process_dict[task_name]) <= 1:
            print("task:{} has only one work process, decrease failed!".format(task_name))
            return False

        # 终止一个工作进程，将其从process_dict中删除，并将其输入输出队列删除
        del_pid = self.process_dict[task_name][-1].pid
        self.process_dict[task_name][-1].terminate()
        self.process_dict[task_name][-1].join()
        del self.process_dict[task_name][-1]
        del self.input_queue_dict[task_name][-1]
        del self.output_queue_dict[task_name][-1]

        assert len(self.process_dict[task_name]) == len(self.input_queue_dict[task_name])
        assert len(self.process_dict[task_name]) == len(self.output_queue_dict[task_name])

        # 将pid从pid_set中删除
        assert del_pid in self.pid_set
        self.pid_set.remove(del_pid)

        self.work_process_num -= 1

        # 删除工作进程对应的各类资源控制组，并在/sys/fs/cgroup中各类资源目录下删除该进程对应的文件夹
        if del_pid in self.process_mem_group_dict:
            print("del_pid in self.process_mem_group_dict")
            temp_command = "rmdir /sys/fs/cgroup/memory/process_{}".format(del_pid)
            os.system(temp_command)
            del self.process_mem_group_dict[del_pid]

        if del_pid in self.process_cpu_group_dict:
            print("del_pid in self.process_cpu_group_dict")
            temp_command = "rmdir /sys/fs/cgroup/cpu/process_{}".format(del_pid)
            os.system(temp_command)
            del self.process_cpu_group_dict[del_pid]

        if del_pid in self.process_cpu_set_group_dict:
            print("del_pid in self.process_cpu_set_group_dict")
            temp_command = "rmdir /sys/fs/cgroup/cpuset/process_{}".format(del_pid)
            os.system(temp_command)
            del self.process_cpu_set_group_dict[del_pid]

        return True

    def limit_process_resource(self, process_resource_info):
        process_pid = process_resource_info['pid']
        if process_pid in self.pid_set:  # pid是当前机器上正在运行的进程
            if 'cpu_util_limit' in process_resource_info and process_resource_info['cpu_util_limit'] > 0:
                self.process_cpu_group_dict[process_pid].controller.cfs_quota_us = int(process_resource_info['cpu_util_limit'] *
                                                                                       self.process_cpu_group_dict[process_pid].controller.cfs_period_us *
                                                                                       psutil.cpu_count())
                assert process_resource_info['task_name'] in self.resource_limit_dict
                self.resource_limit_dict[process_resource_info['task_name']]['cpu_util_limit'] = process_resource_info['cpu_util_limit']
            '''
            if 'mem_limit' in process_resource_info and process_resource_info['mem_limit'] > 0:
                self.process_mem_group_dict[process_pid].controller.limit_in_bytes = process_resource_info['mem_limit']
            if 'cpu_set_cpus' in process_resource_info and len(process_resource_info['cpu_set_cpus']) > 0:
                self.process_cpu_set_group_dict[process_pid].controller.cpus = set(process_resource_info['cpu_set_cpus'])
            if 'cpu_set_mems' in process_resource_info and len(process_resource_info['cpu_set_mems']) > 0:
                self.process_cpu_set_group_dict[process_pid].controller.mems = set(process_resource_info['cpu_set_mems'])
            '''
            return True
        else:
            return False

    def get_process_cpu_util_limit(self, proc_pid):
        # 获取某个进程的cpu利用率上限，取值范围[0,1]，-1表示异常情况
        if proc_pid in self.process_cpu_group_dict:
            # 该进程使用cgroup限制了资源
            proc_cfs_quota_us = self.process_cpu_group_dict[proc_pid].controller.cfs_quota_us
            if proc_cfs_quota_us == -1:
                return 1.0
            proc_cfs_period_us = self.process_cpu_group_dict[proc_pid].controller.cfs_period_us
            return proc_cfs_quota_us / self.cpu_count / proc_cfs_period_us
        elif proc_pid in self.pid_set:
            # 该进程在当前机器上运行，但并没有使用cgroup限制资源（即资源无限）
            return 1.0
        return -1  # 表示当前机器上没有这个进程

    # 系统状态相关
    def add_edge_ip(self, edge_ip):
        self.edge_ip_set.add(edge_ip)

    def get_service_list(self):
        service_list = []
        for task_name in self.process_dict.keys():
            service_list.append(task_name)
        return service_list

    def update_server_status(self):
        self.server_status['cpu_ratio'] = psutil.cpu_percent(interval=None, percpu=False)  # 所有cpu的使用率
        self.server_status['n_cpu'] = self.cpu_count
        self.server_status['mem_total'] = psutil.virtual_memory().total / 1024 / 1024 / 1024
        self.server_status['mem_ratio'] = psutil.virtual_memory().percent

        self.server_status['swap_ratio'] = psutil.swap_memory().percent
        # 发起请求时再对网络情况进行采样
        old_net_bytes = psutil.net_io_counters().bytes_sent + psutil.net_io_counters().bytes_recv
        sec_interval = 0.3
        time.sleep(sec_interval)
        new_net_bytes = psutil.net_io_counters().bytes_sent + psutil.net_io_counters().bytes_recv
        self.server_status['net_ratio(MBps)'] = round((new_net_bytes - old_net_bytes) / (1024.0 * 1024)
                                                                / sec_interval, 5)

        # 获取GPU使用情况
        # 不同类型设备的GPU使用方式不同，统计方式也不同
        gpu_mem_total = dict()
        gpu_mem_utilization = dict()
        gpu_compute_utilization = dict()
        if platform.uname().machine[0:5] == 'aarch':
            from jtop import jtop
            with jtop() as jetson:
                while jetson.ok():
                    gpu_compute_utilization['0'] = jetson.stats['GPU']  # 计算负载百分比，5，7
                    gpu_total_mem = 3.9 * 1024 * 1024   # nano显存总量为3.9G，未找到获取显存总量的api，直接写死，单位kB
                    gpu_mem_total['0'] = 3.9
                    process_list = jetson.processes
                    gpu_used_mem = 0
                    for pro in process_list:
                        gpu_used_mem += pro[-2]  # 进程占用的显存大小，单位kB
                    gpu_mem_utilization['0'] = gpu_used_mem / gpu_total_mem * 100  # 百分比
                    break
        else:
            import pynvml
            pynvml.nvmlInit()  # 初始化
            gpu_device_count = pynvml.nvmlDeviceGetCount()  # 获取Nvidia GPU块数
            for i in range(gpu_device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)  # 获取GPU i的handle，后续通过handle来处理
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)  # 通过handle获取GPU i的信息
                gpu_mem_utilization[str(i)] = memory_info.used / memory_info.total * 100  # GPU i的显存占用比例
                gpu_compute_utilization[str(i)] = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu  # GPU i 计算能力的使用率，百分比
                gpu_mem_total[str(i)] = memory_info.total / 1024 / 1024 / 1024
            pynvml.nvmlShutdown()  # 最后关闭管理工具
        self.server_status['gpu_mem_utilization'] = gpu_mem_utilization
        self.server_status['gpu_compute_utilization'] = gpu_compute_utilization
        self.server_status['gpu_mem_total'] = gpu_mem_total

        # 更新云端各类工作进程的信息
        '''
        for task_name in self.process_dict.keys():
            task_info = dict()  # 关于某一类服务的所有信息
            task_process_list = self.process_dict[task_name]  # 执行某一类服务的所有进程
            for index in range(len(task_process_list)):
                temp_pid = task_process_list[index].pid
                temp_process_info = dict()  # 当前工作进程的信息
                # 当前工作进程的cpu、内存占用率
                temp_p = psutil.Process(temp_pid)
                temp_process_info['cpu_ratio'] = temp_p.cpu_percent(interval=0.5)
                temp_process_info['mem_ratio'] = temp_p.memory_percent()
                # 当前工作进程未完成的任务数量
                temp_process_info['task_to_do'] = self.input_queue_dict[task_name][index].qsize()
                task_info[str(temp_pid)] = temp_process_info
            self.server_status[task_name] = task_info
        '''
        return self.server_status

    def get_server_status(self):
        return self.server_status

    def update_clients_status(self):
        for edge_ip in self.edge_ip_set:
            edge_url = edge_ip + ":" + str(self.edge_port)  # "114.212.81.11:5501"
            temp_url = "http://" + edge_url + "/update_client_status"
            temp_res = requests.get(temp_url).content.decode()
            temp_res_dict = json.loads(temp_res)
            self.clients_status[edge_ip] = temp_res_dict
        return self.clients_status

    def get_clients_status(self):
        return self.clients_status

    def get_system_status(self):
        system_status_dict = dict()
        system_status_dict['cloud'] = dict()
        system_status_dict['host'] = dict()
        # 由于这里用到了self.server_ip，因此命令行--server_ip参数不能设置为0.0.0.0
        system_status_dict['cloud'][self.server_ip] = self.server_status
        for edge_ip in self.clients_status.keys():
            system_status_dict['host'][edge_ip] = self.clients_status[edge_ip]
        return system_status_dict

    def get_cluster_info(self):
        cluster_info = dict()
        cluster_info[self.server_ip] = self.server_status
        assert isinstance(cluster_info[self.server_ip], dict)
        cluster_info[self.server_ip]['node_role'] = 'cloud'

        for edge_ip in self.clients_status.keys():
            cluster_info[edge_ip] = self.clients_status[edge_ip]
            cluster_info[edge_ip]['node_role'] = 'edge'

        return cluster_info

    def get_execute_url(self, task_name):
        res = dict()
        # 边缘端可执行任务的接口
        for edge_ip in self.edge_ip_set:
            edge_ip_port = edge_ip + ':' + str(self.edge_port)
            edge_url = "http://" + edge_ip_port + "/execute_task/" + task_name
            edge_dict = {
                "url": edge_url
            }
            # 由于云端是定期请求边缘端的资源情况的，因此在第一次请求之前边缘端的资源情况为空，需要进行特判
            if "mem_ratio" in self.clients_status[edge_ip]:
                edge_dict["mem_ratio"] = self.clients_status[edge_ip]["mem_ratio"]
            if "cpu_ratio" in self.clients_status[edge_ip]:
                edge_dict["cpu_ratio"] = self.clients_status[edge_ip]["cpu_ratio"]
            if "n_cpu" in self.clients_status[edge_ip]:
                edge_dict["n_cpu"] = self.clients_status[edge_ip]["n_cpu"]
            res[edge_ip] = edge_dict

        # 服务端可执行任务的接口
        server_ip_port = self.server_ip + ':' + str(self.server_port)
        server_url = "http://" + server_ip_port + "/execute_task/" + task_name
        server_dict = {
            "url": server_url
        }
        if "mem_ratio" in self.server_status:
            server_dict["mem_ratio"] = self.server_status["mem_ratio"]
        if "cpu_ratio" in self.server_status:
            server_dict["cpu_ratio"] = self.server_status["cpu_ratio"]
        if "n_cpu" in self.server_status:
            server_dict["n_cpu"] = self.server_status["n_cpu"]
        res[self.server_ip] = server_dict

        return res


class ServerAppConfig(object):
    # flask定时任务的配置类
    JOBS = [
        {
            'id': 'job1',
            'func': 'app_server:trigger_update_server_status',
            'trigger': 'interval',  # 间隔触发
            'seconds': 11,  # 定时器时间间隔
        },
        {
            'id': 'job2',
            'func': 'app_server:trigger_update_clients_status',
            'trigger': 'interval',  # 间隔触发
            'seconds': 17,  # 定时器时间间隔
        }
    ]
    SCHEDULER_API_ENABLED = True


WSGIRequestHandler.protocol_version = "HTTP/1.1"
app = Flask(__name__)
server_manager = ServerManager()


# 用户编辑json、提交任务的界面
@app.route("/edit-json")
def edit_json():
    return render_template('edit_json.html')


# 路由到dist目录
@app.route("/dist/<path:path>")
def send_dist(path):
    return send_from_directory("dist", path)


@app.route('/upload-json-and-codefiles-api', methods=['POST'])
def upload_json_and_codefiles_api():
    try:
        # 根据用户上传的json文件构造应用相关信息
        received_files = request.files
        task_json = received_files.get('task_json')  # 用户提交的json文件，前端的key为'task_json'
        task_dict = json.load(task_json)
        task_dict['task_code_path'] = dict()  # 存储各个阶段代码的路径，以精简json文件

        for file in received_files:   # file为前端设置的、各个文件的key，为任务中各个阶段的名字，与代码文件对应
            if file == 'task_json':
                continue
            if file not in server_manager.code_set:
                server_manager.code_set.add(file)
                save_path = server_manager.server_codebase + file + '.zip'
                received_files[file].save(save_path)
                with zipfile.ZipFile(save_path, 'r') as zip_ref:
                    zip_ref.extractall(server_manager.server_codebase + file)
            task_dict['task_code_path'][file] = server_manager.server_codebase + file

        # 向边缘端发送应用相关信息
        headers = {"Content-type": "application/json"}
        for edge_ip in server_manager.edge_ip_set:
            edge_url = "http://" + edge_ip + ":" + str(server_manager.edge_port) + server_manager.edge_get_task_url
            res1 = requests.post(edge_url, data=json.dumps(task_dict), headers=headers)

        # 代码下装时，创建云端工作进程（为DAG图中每一个子任务都创建一个工作进程）
        server_manager.create_task_process(task_dict)

        # 响应
        res = make_response("提交成功！")
        res.status = '200'  # 设置状态码
        res.headers['Access-Control-Allow-Origin'] = "*"  # 设置允许跨域
        res.headers['Access-Control-Allow-Methods'] = 'PUT,GET,POST,DELETE'
        return res

    except:
        res = make_response("提交失败！")
        res.status = '200'
        res.headers['Access-Control-Allow-Origin'] = "*"
        res.headers['Access-Control-Allow-Methods'] = 'PUT,GET,POST,DELETE'
        return res


@app.route("/register_edge")
def register_edge():
    edge_ip = request.remote_addr
    server_manager.add_edge_ip(edge_ip)
    return jsonify(edge_ip)


@app.route('/execute_task/<string:task_name>', methods=['POST'])
def execute_task(task_name):
    # 始终为并发执行、配合最新应用软件的接口，配合调度器
    output_ctx = dict()
    if task_name in server_manager.process_dict:
        input_ctx = request.get_json()
        if task_name == 'face_detection':
            input_ctx['image'] = decode_image(input_ctx['image'])
            server_manager.input_queue_dict[task_name][0].put(input_ctx)  # detection任务单进程执行
            output_ctx = server_manager.output_queue_dict[task_name][0].get()
            for i in range(len(output_ctx['faces'])):
                output_ctx['faces'][i] = encode_image(output_ctx['faces'][i])
            output_ctx['proc_resource_info']['cpu_util_limit'] = server_manager.get_process_cpu_util_limit(
                                                                 output_ctx['proc_resource_info']['pid'])
            output_ctx['proc_resource_info_list'] = [output_ctx['proc_resource_info']]
            del output_ctx['proc_resource_info']
        elif task_name == 'face_alignment':
            for i in range(len(input_ctx['faces'])):
                input_ctx['faces'][i] = decode_image(input_ctx['faces'][i])
            task_num = len(input_ctx['faces'])  # 任务数量
            work_process_num = len(server_manager.process_dict[task_name])  # 执行该任务的工作进程数量
            output_ctx_list = []
            proc_resource_info_list = []
            if task_num <= work_process_num:  # 任务数量小于工作进程数量，则并发的分给各个进程，每个进程执行一个任务
                # 将任务并发的分发给各个工作进程
                for i in range(task_num):
                    temp_input_ctx = dict()
                    temp_input_ctx['faces'] = [input_ctx['faces'][i]]
                    temp_input_ctx['bbox'] = [input_ctx['bbox'][i]]
                    temp_input_ctx['prob'] = []
                    server_manager.input_queue_dict[task_name][i].put(temp_input_ctx)
                # 按序获取各个工作进程的执行结果
                for i in range(task_num):
                    temp_output_ctx = server_manager.output_queue_dict[task_name][i].get()
                    output_ctx_list.append(temp_output_ctx)
            else:
                ave_task_num = int(task_num / work_process_num)  # 平均每个进程要执行的任务数量
                more_task_num = task_num % work_process_num  # more_task_num个进程要做ave_task_num+1个任务
                # 将任务并发的分发给各个工作进程
                for i in range(more_task_num):
                    temp_input_ctx = dict()
                    temp_start_index = i * (ave_task_num + 1)
                    temp_end_index = (i + 1) * (ave_task_num + 1)
                    temp_input_ctx['faces'] = input_ctx['faces'][temp_start_index:temp_end_index]
                    temp_input_ctx['bbox'] = input_ctx['bbox'][temp_start_index:temp_end_index]
                    temp_input_ctx['prob'] = []
                    server_manager.input_queue_dict[task_name][i].put(temp_input_ctx)
                for i in range(more_task_num, work_process_num):
                    temp_input_ctx = dict()
                    temp_start_index = more_task_num * (ave_task_num + 1) + (i - more_task_num) * ave_task_num
                    temp_end_index = more_task_num * (ave_task_num + 1) + (i - more_task_num + 1) * ave_task_num
                    temp_input_ctx['faces'] = input_ctx['faces'][temp_start_index:temp_end_index]
                    temp_input_ctx['bbox'] = input_ctx['bbox'][temp_start_index:temp_end_index]
                    temp_input_ctx['prob'] = []
                    server_manager.input_queue_dict[task_name][i].put(temp_input_ctx)
                # 按序获取各个工作进程的执行结果
                for i in range(work_process_num):
                    temp_output_ctx = server_manager.output_queue_dict[task_name][i].get()
                    output_ctx_list.append(temp_output_ctx)
            output_ctx["count_result"] = {"up": 0, "total": 0}
            for t_output_ctx in output_ctx_list:
                output_ctx["count_result"]["up"] += t_output_ctx["count_result"]["up"]
                output_ctx["count_result"]["total"] += t_output_ctx["count_result"]["total"]
                t_output_ctx['proc_resource_info']['cpu_util_limit'] = server_manager.get_process_cpu_util_limit(
                                                                       t_output_ctx['proc_resource_info']['pid'])
                proc_resource_info_list.append(t_output_ctx['proc_resource_info'])
            output_ctx['proc_resource_info_list'] = proc_resource_info_list
        else:
            input_ctx['image'] = decode_image(input_ctx['image'])
            server_manager.input_queue_dict[task_name][0].put(input_ctx)  # detection任务单进程执行
            output_ctx = server_manager.output_queue_dict[task_name][0].get()
            output_ctx['proc_resource_info']['cpu_util_limit'] = server_manager.get_process_cpu_util_limit(
                output_ctx['proc_resource_info']['pid'])
            output_ctx['proc_resource_info_list'] = [output_ctx['proc_resource_info']]
            del output_ctx['proc_resource_info']

        output_ctx['execute_flag'] = True
    else:
        output_ctx['execute_flag'] = False
    return jsonify(output_ctx)


@app.route('/execute_task_old/<string:task_name>', methods=['POST'])
def execute_task_old(task_name):
    # 最初版本的任务执行接口（配合最初版本的face_detection封装代码）
    output_ctx = dict()
    if task_name in server_manager.process_dict:
        input_ctx = request.get_json()
        input_ctx['image'] = decode_image(input_ctx['image'])
        server_manager.input_queue_dict[task_name][0].put(input_ctx)  # 单进程串行执行所有任务
        output_ctx = server_manager.output_queue_dict[task_name][0].get()
        output_ctx['execute_flag'] = True
    else:
        output_ctx['execute_flag'] = False
    return jsonify(output_ctx)


@app.route('/execute_task_new/<string:task_name>', methods=['POST'])
def execute_task_new(task_name):
    # 新版本的任务执行接口（配合新版本的face_detection封装代码）
    output_ctx = dict()
    if task_name in server_manager.process_dict:
        input_ctx = request.get_json()
        if task_name == 'face_detection':
            input_ctx['image'] = decode_image(input_ctx['image'])
            server_manager.input_queue_dict[task_name][0].put(input_ctx)  # 单进程串行执行所有任务
            output_ctx = server_manager.output_queue_dict[task_name][0].get()
            for i in range(len(output_ctx['faces'])):
                output_ctx['faces'][i] = encode_image(output_ctx['faces'][i])
        elif task_name == 'face_alignment':
            for i in range(len(input_ctx['faces'])):
                input_ctx['faces'][i] = decode_image(input_ctx['faces'][i])
            server_manager.input_queue_dict[task_name][0].put(input_ctx)  # 单进程串行执行所有任务
            output_ctx = server_manager.output_queue_dict[task_name][0].get()
        output_ctx['execute_flag'] = True
    else:
        output_ctx['execute_flag'] = False
    return jsonify(output_ctx)


@app.route('/concurrent_execute_task_new/<string:task_name>', methods=['POST'])
def concurrent_execute_task_new(task_name):
    # 新版本的任务并行执行接口（配合新版本的face_detection封装代码，多目标任务并行执行并汇总结果）
    output_ctx = dict()
    if task_name in server_manager.process_dict:
        input_ctx = request.get_json()
        if task_name == 'face_detection':
            input_ctx['image'] = decode_image(input_ctx['image'])
            server_manager.input_queue_dict[task_name][0].put(input_ctx)  # detection任务单进程执行
            output_ctx = server_manager.output_queue_dict[task_name][0].get()
            for i in range(len(output_ctx['faces'])):
                output_ctx['faces'][i] = encode_image(output_ctx['faces'][i])
            output_ctx['proc_resource_info']['cpu_util_limit'] = server_manager.get_process_cpu_util_limit(
                                                                 output_ctx['proc_resource_info']['pid'])
            output_ctx['proc_resource_info_list'] = [output_ctx['proc_resource_info']]
            del output_ctx['proc_resource_info']
        elif task_name == 'face_alignment':
            for i in range(len(input_ctx['faces'])):
                input_ctx['faces'][i] = decode_image(input_ctx['faces'][i])
            task_num = len(input_ctx['faces'])  # 任务数量
            work_process_num = len(server_manager.process_dict[task_name])  # 执行该任务的工作进程数量
            output_ctx_list = []
            proc_resource_info_list = []
            if task_num <= work_process_num:  # 任务数量小于工作进程数量，则并发的分给各个进程，每个进程执行一个任务
                # 将任务并发的分发给各个工作进程
                for i in range(task_num):
                    temp_input_ctx = dict()
                    temp_input_ctx['faces'] = [input_ctx['faces'][i]]
                    temp_input_ctx['bbox'] = [input_ctx['bbox'][i]]
                    temp_input_ctx['prob'] = []
                    server_manager.input_queue_dict[task_name][i].put(temp_input_ctx)
                # 按序获取各个工作进程的执行结果
                for i in range(task_num):
                    temp_output_ctx = server_manager.output_queue_dict[task_name][i].get()
                    output_ctx_list.append(temp_output_ctx)
            else:
                ave_task_num = int(task_num / work_process_num)  # 平均每个进程要执行的任务数量
                more_task_num = task_num % work_process_num  # more_task_num个进程要做ave_task_num+1个任务
                # 将任务并发的分发给各个工作进程
                for i in range(more_task_num):
                    temp_input_ctx = dict()
                    temp_start_index = i * (ave_task_num + 1)
                    temp_end_index = (i + 1) * (ave_task_num + 1)
                    temp_input_ctx['faces'] = input_ctx['faces'][temp_start_index:temp_end_index]
                    temp_input_ctx['bbox'] = input_ctx['bbox'][temp_start_index:temp_end_index]
                    temp_input_ctx['prob'] = []
                    server_manager.input_queue_dict[task_name][i].put(temp_input_ctx)
                for i in range(more_task_num, work_process_num):
                    temp_input_ctx = dict()
                    temp_start_index = more_task_num * (ave_task_num + 1) + (i - more_task_num) * ave_task_num
                    temp_end_index = more_task_num * (ave_task_num + 1) + (i - more_task_num + 1) * ave_task_num
                    temp_input_ctx['faces'] = input_ctx['faces'][temp_start_index:temp_end_index]
                    temp_input_ctx['bbox'] = input_ctx['bbox'][temp_start_index:temp_end_index]
                    temp_input_ctx['prob'] = []
                    server_manager.input_queue_dict[task_name][i].put(temp_input_ctx)
                # 按序获取各个工作进程的执行结果
                for i in range(work_process_num):
                    temp_output_ctx = server_manager.output_queue_dict[task_name][i].get()
                    output_ctx_list.append(temp_output_ctx)
            output_ctx["count_result"] = {"up": 0, "total": 0}
            for t_output_ctx in output_ctx_list:
                output_ctx["count_result"]["up"] += t_output_ctx["count_result"]["up"]
                output_ctx["count_result"]["total"] += t_output_ctx["count_result"]["total"]
                t_output_ctx['proc_resource_info']['cpu_util_limit'] = server_manager.get_process_cpu_util_limit(
                                                                       t_output_ctx['proc_resource_info']['pid'])
                proc_resource_info_list.append(t_output_ctx['proc_resource_info'])
            output_ctx['proc_resource_info_list'] = proc_resource_info_list
        elif task_name == 'car_detection':
            input_ctx['image'] = decode_image(input_ctx['image'])
            server_manager.input_queue_dict[task_name][0].put(input_ctx)  # detection任务单进程执行
            output_ctx = server_manager.output_queue_dict[task_name][0].get()
            output_ctx['proc_resource_info']['cpu_util_limit'] = server_manager.get_process_cpu_util_limit(
                output_ctx['proc_resource_info']['pid'])
            output_ctx['proc_resource_info_list'] = [output_ctx['proc_resource_info']]
            del output_ctx['proc_resource_info']

        output_ctx['execute_flag'] = True
    else:
        output_ctx['execute_flag'] = False
    return jsonify(output_ctx)


@app.route('/add_work_process', methods=['POST'])
def add_work_process():
    # 添加某类任务工作进程的接口
    work_process_info = request.get_json()
    create_res = dict()
    create_res['create_flag'] = server_manager.add_work_process(work_process_info)
    return jsonify(create_res)


@app.route('/decrease_work_process', methods=['POST'])
def decrease_work_process():
    # 减少某类任务工作进程的接口
    decrease_info = request.get_json()
    decrease_res = dict()
    decrease_res['decrease_flag'] = server_manager.decrease_work_process(decrease_info)
    return jsonify(decrease_res)


@app.route('/limit_process_resource', methods=['POST'])
def limit_process_resource():
    process_resource_info = request.get_json()
    limit_res = dict()
    limit_res['limit_flag'] = server_manager.limit_process_resource(process_resource_info)
    return jsonify(limit_res)


@app.route('/get_service_list')
def get_service_list():
    # 获取当前系统中可以执行任务的名称列表
    service_list = server_manager.get_service_list()
    return jsonify(service_list)


@app.route('/get_execute_url/<string:task_name>')
def get_execute_url(task_name):
    # 获取任务task_name在系统中所有计算服务调用的url
    res = server_manager.get_execute_url(task_name)
    return jsonify(res)


@app.route("/update_server_status")
def update_server_status():
    # 更新云端机器整体的资源信息
    server_status = server_manager.update_server_status()
    print("server status change:{}.".format(server_status))
    return jsonify(server_status)


@app.route("/update_clients_status")
def update_clients_status():
    # 获取所有边缘端的状态信息
    clients_status = server_manager.update_clients_status()
    return jsonify(clients_status)


@app.route("/get_system_status")
def get_system_status():
    # 获取整个系统的状态信息，包括云端和边缘端
    system_status_dict = server_manager.get_system_status()
    return jsonify(system_status_dict)


@app.route("/get_resource_info")
def get_resource_info():
    # 配合调度器获取系统资源信息的接口
    resource_info = server_manager.get_system_status()
    return jsonify(resource_info)


@app.route("/get_cluster_info")
def get_cluster_info():
    # 配合调度器获取集群资源信息的接口
    cluster_info = server_manager.get_cluster_info()
    return jsonify(cluster_info)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--server_ip', dest='server_ip', type=str, default='127.0.0.1')
    parser.add_argument('--server_port', dest='server_port', type=int, default=5500)
    parser.add_argument('--edge_port', dest='edge_port', type=int, default=5500)
    args = parser.parse_args()

    server_manager.init_server_param(args.server_ip, args.server_port, args.edge_port)

    app.config.from_object(ServerAppConfig())
    scheduler = APScheduler()  # 利用APScheduler启动定时任务
    scheduler.init_app(app)
    scheduler.start()

    app.run(host=server_manager.server_ip, port=server_manager.server_port)
