import time
from flask_apscheduler import APScheduler
from flask_cors import CORS
from flask import Flask, make_response, request, jsonify
import requests
import os
import paramiko
import stat
import importlib
import sys
import multiprocessing as mp
from werkzeug.serving import WSGIRequestHandler
import psutil
from field_codec_utils import decode_image, encode_image
import argparse


def register_edge_to_server():
    # 主动向服务器发送get请求，实现服务器记录各个边缘节点ip
    url = "http://" + client_manager.server_ip + ":" + str(client_manager.server_port) + client_manager.register_path
    res = requests.get(url)
    print("Edge register success!")


def trigger_update_client_status():
    # 定时触发更新client状态
    temp_url = "http://" + client_manager.edge_ip + ":" + str(client_manager.edge_port) + "/update_client_status"
    requests.get(temp_url)
    print("trigger_update_client_status!")

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
                        temp_dict['memoryUtilization'] = pro[-2] / gpu_total_mem * 100
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


class ClientManager(object):
    def __init__(self):
        # 边端系统参数相关
        self.server_ip = '114.212.81.11'  # 服务器服务端的ip和端口号
        self.server_port = 5500
        self.edge_ip = '127.0.0.1'  # 默认设置为127.0.0.1，供定时事件使用
        self.edge_port = 5500
        self.register_path = "/register_edge"  # 向服务器注册边缘端的接口
        self.server_ssh_port = 22   # 服务端接受ssh连接的端口
        self.server_ssh_username = 'guest'  # 服务端ssh连接的用户名和密码
        self.server_ssh_password = 'dislab@909065'
        self.code_local_dir = os.path.abspath('.')  # 为了导入工作进程代码，需要将代码下载到当前工作目录下

        # 工作进程执行相关
        self.code_set = set()  # 边缘端已存储的代码对应的应用的名字
        self.process_dict = dict()   # 存放每个任务对应的工作进程，value为进程列表，方便动态增加各个任务工作进程数量
        self.pid_set = set()  # 存放所有工作进程的pid
        self.input_queue_dict = dict()   # 存放每个工作进程所需的输入输出队列，value为队列列表
        self.output_queue_dict = dict()
        self.model_ctx_dict = dict()  # 存放每一类任务工作进程的模型参数，key为任务名，value为model_ctx

        # 工作进程控制相关
        self.work_process_num = 0  # 当前节点上的工作进程数
        self.cpu_count = psutil.cpu_count()  # 当前节点上的cpu核数
        self.process_mem_group_dict = dict()  # 记录各个进程对应的memory cgroup组，key为pid(int), value为group对象
        self.default_resource_limit = {  # 创建工作进程时默认的进程对各类资源的使用上限
            'cpu_util_limit': 0.05
        }
        self.process_cpu_group_dict = dict()  # 记录各个进程对应的cpu cgroup组，key为pid(int), value为group对象
        self.process_cpu_set_group_dict = dict()  # 记录各个进程对应的cpuset cgroup组，key为pid(int), value为group对象
        self.resource_limit_dict = dict()  # 记录各类任务各类资源的使用上限，用于在创建新进程时使用，key为任务名，value为dict

        # 边端状态相关
        self.client_status = dict()  # 记录server的状态，资源状态、各任务进程状态

    def init_client_param(self, server_ip, server_port, edge_ip, edge_port):
        self.server_ip = server_ip
        self.server_port = server_port
        self.edge_ip = edge_ip
        self.edge_port = edge_port

    # 软件下装相关
    def get_all_files_in_remote_dir(self, sftp1, remote_dir):
        all_files = list()
        if remote_dir[-1] == '/':
            remote_dir = remote_dir[0:-1]
        files = sftp1.listdir_attr(remote_dir)
        for file in files:
            filename = remote_dir + '/' + file.filename
            if stat.S_ISDIR(file.st_mode):  # 如果是文件夹的话递归处理
                all_files.extend(self.get_all_files_in_remote_dir(sftp1, filename))
            else:
                all_files.append(filename)
        return all_files

        # 远程服务器上指定文件夹下载到本地文件夹

    def sftp_get_dir(self, sftp1, remote_dir, local_dir):
        try:
            all_files = self.get_all_files_in_remote_dir(sftp1, remote_dir)
            for file in all_files:
                local_filename = file.replace(remote_dir, local_dir)
                local_filepath = os.path.dirname(local_filename)
                if not os.path.exists(local_filepath):
                    os.makedirs(local_filepath)
                sftp1.get(file, local_filename)
        except:
            print('ssh get dir from master failed.')

    def download_task_code(self, task_dict):
        if task_dict['name'] not in self.code_set:   # 当前应用未在边缘端提交过才下载代码文件
            self.code_set.add(task_dict['name'])
            remote_task_code_path = task_dict['task_code_path']
            local_task_code_path = {}
            print(remote_task_code_path)

            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(self.server_ip, self.server_ssh_port, self.server_ssh_username, self.server_ssh_password)
            sftp = ssh.open_sftp()
            for task_name in remote_task_code_path.keys():
                self.code_set.add(task_name)
                task_remote_dir = remote_task_code_path[task_name]
                if task_remote_dir[-1] == '/':
                    task_remote_dir = task_remote_dir[0:-1]
                last_dir = (task_remote_dir.split('/'))[-1]
                local_dir = os.path.join(self.code_local_dir, last_dir)
                local_dir = os.path.join(local_dir, '')
                if not os.path.exists(local_dir):
                    os.mkdir(local_dir)
                local_task_code_path[task_name] = local_dir
                self.sftp_get_dir(sftp, task_remote_dir, local_dir)
            print(local_task_code_path)
            print("Get file from server success!")
            ssh.close()

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
            assert task_name in self.resource_limit_dict
            cpu_group.controller.cfs_quota_us = int(self.resource_limit_dict[task_name]['cpu_util_limit'] * cpu_group.controller.cfs_period_us *
                                                    psutil.cpu_count())
            cpu_group.controller.tasks = task_set
            self.process_cpu_group_dict[temp_process.pid] = cpu_group
            '''

            self.work_process_num += 1
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

    # 维护边端状态相关
    def update_client_status(self):
        self.client_status['cpu_ratio'] = psutil.cpu_percent(interval=None, percpu=True)  # 所有cpu的使用率
        self.client_status['mem_ratio'] = psutil.virtual_memory().percent
        '''
        client_manager.client_status['swap_ratio'] = psutil.swap_memory().percent
        # 发起请求时再对网络情况进行采样
        old_net_bytes = psutil.net_io_counters().bytes_sent + psutil.net_io_counters().bytes_recv
        sec_interval = 1.0
        time.sleep(sec_interval)
        new_net_bytes = psutil.net_io_counters().bytes_sent + psutil.net_io_counters().bytes_recv
        client_manager.client_status['net_ratio(MBps)'] = round((new_net_bytes - old_net_bytes) / (1024.0 * 1024)
                                                                / sec_interval, 5)
        '''
        '''
        # 获取GPU使用情况
        # 不同类型设备的GPU使用方式不同，统计方式也不同
        gpu_mem = dict()
        gpu_utilization = dict()
        if platform.uname().machine[0:5] == 'aarch':
            from jtop import jtop
            with jtop() as jetson:
                while jetson.ok():
                    gpu_utilization['0'] = jetson.stats['GPU']  # 计算负载，百分比，5，7
                    gpu_total_mem = 3.9 * 1024 * 1024  # nano显存总量为3.9G，未找到获取显存总量的api，直接写死，单位kB
                    process_list = jetson.processes
                    gpu_used_mem = 0
                    for pro in process_list:
                        gpu_used_mem += pro[-2]  # 进程占用的显存大小，单位kB
                    gpu_mem['0'] = gpu_used_mem / gpu_total_mem * 100  # 百分比
                    break
        else:
            import pynvml
            pynvml.nvmlInit()  # 初始化
            gpu_device_count = pynvml.nvmlDeviceGetCount()  # 获取Nvidia GPU块数
            for i in range(gpu_device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)  # 获取GPU i的handle，后续通过handle来处理
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)  # 通过handle获取GPU i的信息
                gpu_mem[str(i)] = memory_info.used / memory_info.total * 100  # GPU i的显存占用比例
                gpu_utilization[str(i)] = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu  # GPU i 计算能力的使用率，
            pynvml.nvmlShutdown()  # 最后关闭管理工具
        client_manager.client_status['gpu_mem'] = gpu_mem
        client_manager.client_status['gpu_utilization'] = gpu_utilization
        '''
        # 更新边缘端各类工作进程的信息
        # print(server_manager.process_dict)
        '''
        for task_name in client_manager.process_dict.keys():
            task_info = dict()  # 关于某一类服务的所有信息
            task_process_list = client_manager.process_dict[task_name]  # 执行某一类服务的所有进程
            for index in range(len(task_process_list)):
                temp_pid = task_process_list[index].pid
                temp_process_info = dict()  # 当前工作进程的信息
                # 当前工作进程的cpu、内存占用率
                temp_p = psutil.Process(temp_pid)
                temp_process_info['cpu_ratio'] = temp_p.cpu_percent(interval=0.5)
                temp_process_info['mem_ratio'] = temp_p.memory_percent()
                # 当前工作进程未完成的任务数量
                temp_process_info['task_to_do'] = client_manager.input_queue_dict[task_name][index].qsize()
                task_info[str(temp_pid)] = temp_process_info
            client_manager.client_status[task_name] = task_info
        '''
        return self.client_status

    def get_client_status(self):
        return self.client_status


class ClientAppConfig(object):
    # flask定时任务的配置类
    JOBS = [
        {
            'id': 'job1',
            'func': 'app_client:trigger_update_client_status',
            'trigger': 'interval',  # 间隔触发
            'seconds': 11,  # 定时器时间间隔
        }
    ]
    SCHEDULER_API_ENABLED = True


WSGIRequestHandler.protocol_version = "HTTP/1.1"
app = Flask(__name__)
client_manager = ClientManager()   # 用于管理边缘端的代码、工作进程、消息队列


@app.route('/task-register', methods=['POST'])
def task_register():
    # 获取应用信息
    task_dict = request.get_json()
    print(task_dict)
    # print(type(task_dict))

    # 下载应用相关代码
    client_manager.download_task_code(task_dict)

    # 创建执行应用中各个任务的进程
    client_manager.create_task_process(task_dict)

    # 注册成功响应
    res = make_response("Register success!")
    res.status = '200'  # 设置状态码
    res.headers['Access-Control-Allow-Origin'] = "*"  # 设置允许跨域
    res.headers['Access-Control-Allow-Methods'] = 'PUT,GET,POST,DELETE'
    return res


@app.route('/concurrent_execute_task_new/<string:task_name>', methods=['POST'])
def concurrent_execute_task_new(task_name):
    # 新版本的任务并行执行接口（配合新版本的face_detection封装代码，多目标任务并行执行并汇总结果）
    output_ctx = dict()
    if task_name in client_manager.process_dict:
        input_ctx = request.get_json()
        if task_name == 'face_detection':
            input_ctx['image'] = decode_image(input_ctx['image'])
            client_manager.input_queue_dict[task_name][0].put(input_ctx)  # detection任务单进程执行
            output_ctx = client_manager.output_queue_dict[task_name][0].get()
            for i in range(len(output_ctx['faces'])):
                output_ctx['faces'][i] = encode_image(output_ctx['faces'][i])
            output_ctx['proc_resource_info']['cpu_util_limit'] = client_manager.get_process_cpu_util_limit(
                                                                 output_ctx['proc_resource_info']['pid'])
            output_ctx['proc_resource_info_list'] = [output_ctx['proc_resource_info']]
            del output_ctx['proc_resource_info']
        elif task_name == 'face_alignment':
            for i in range(len(input_ctx['faces'])):
                input_ctx['faces'][i] = decode_image(input_ctx['faces'][i])
            task_num = len(input_ctx['faces'])  # 任务数量
            work_process_num = len(client_manager.process_dict[task_name])  # 执行该任务的工作进程数量
            output_ctx_list = []
            proc_resource_info_list = []
            if task_num <= work_process_num:  # 任务数量小于工作进程数量，则并发的分给各个进程，每个进程执行一个任务
                # 将任务并发的分发给各个工作进程
                for i in range(task_num):
                    temp_input_ctx = dict()
                    temp_input_ctx['faces'] = [input_ctx['faces'][i]]
                    temp_input_ctx['bbox'] = [input_ctx['bbox'][i]]
                    temp_input_ctx['prob'] = []
                    client_manager.input_queue_dict[task_name][i].put(temp_input_ctx)
                # 按序获取各个工作进程的执行结果
                for i in range(task_num):
                    temp_output_ctx = client_manager.output_queue_dict[task_name][i].get()
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
                    client_manager.input_queue_dict[task_name][i].put(temp_input_ctx)
                for i in range(more_task_num, work_process_num):
                    temp_input_ctx = dict()
                    temp_start_index = more_task_num * (ave_task_num + 1) + (i - more_task_num) * ave_task_num
                    temp_end_index = more_task_num * (ave_task_num + 1) + (i - more_task_num + 1) * ave_task_num
                    temp_input_ctx['faces'] = input_ctx['faces'][temp_start_index:temp_end_index]
                    temp_input_ctx['bbox'] = input_ctx['bbox'][temp_start_index:temp_end_index]
                    temp_input_ctx['prob'] = []
                    client_manager.input_queue_dict[task_name][i].put(temp_input_ctx)
                # 按序获取各个工作进程的执行结果
                for i in range(work_process_num):
                    temp_output_ctx = client_manager.output_queue_dict[task_name][i].get()
                    output_ctx_list.append(temp_output_ctx)
            output_ctx["count_result"] = {"up": 0, "total": 0}
            for t_output_ctx in output_ctx_list:
                output_ctx["count_result"]["up"] += t_output_ctx["count_result"]["up"]
                output_ctx["count_result"]["total"] += t_output_ctx["count_result"]["total"]
                t_output_ctx['proc_resource_info']['cpu_util_limit'] = client_manager.get_process_cpu_util_limit(
                                                                       t_output_ctx['proc_resource_info']['pid'])
                proc_resource_info_list.append(t_output_ctx['proc_resource_info'])
            output_ctx['proc_resource_info_list'] = proc_resource_info_list
        output_ctx['execute_flag'] = True
    else:
        output_ctx['execute_flag'] = False
    return jsonify(output_ctx)


@app.route('/add_work_process', methods=['POST'])
def add_work_process():
    work_process_info = request.get_json()
    create_res = dict()
    create_res['create_flag'] = client_manager.add_work_process(work_process_info)
    return jsonify(create_res)


@app.route('/limit_process_resource', methods=['POST'])
def limit_process_resource():
    process_resource_info = request.get_json()
    limit_res = dict()
    limit_res['limit_flag'] = client_manager.limit_process_resource(process_resource_info)
    return jsonify(limit_res)


@app.route("/update_client_status")
def update_client_status():
    # 更新边缘端机器整体的资源信息
    client_status = client_manager.update_client_status()
    print("client status change:{}.".format(client_status))
    return jsonify(client_status)


@app.route("/get_client_status")
def get_client_status():
    return jsonify(client_manager.get_client_status())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--server_ip', dest='server_ip', type=str, default='114.212.81.11')
    parser.add_argument('--server_port', dest='server_port', type=int, default=5500)
    parser.add_argument('--edge_ip', dest='edge_ip', type=str, default='192.168.100.5')
    parser.add_argument('--edge_port', dest='edge_port', type=int, default=5500)
    args = parser.parse_args()

    client_manager.init_client_param(args.server_ip, args.server_port, args.edge_ip, args.edge_port)

    register_edge_to_server()  # 向云端注册自己的存在

    app.config.from_object(ClientAppConfig())
    scheduler = APScheduler()  # 利用APScheduler启动定时任务
    scheduler.init_app(app)
    scheduler.start()

    app.run(host=client_manager.edge_ip, port=client_manager.edge_port)
