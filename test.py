import time

import cv2
import requests
import json
import numpy as np
import os
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


def encode_image(img_rgb):
    img_bytes = str(cv2.imencode('.jpg', img_rgb)[1].tobytes())
    return img_bytes


def decode_image(img_bytes):
    img_jpg = np.frombuffer(eval(img_bytes), dtype=np.uint8)
    img_rgb = np.array(cv2.imdecode(img_jpg, cv2.IMREAD_UNCHANGED))
    return img_rgb


def plot_func(csv_file_path, csv_name, data_path):
    val_arr = np.genfromtxt(csv_file_path, delimiter=',')
    for i in range(val_arr.shape[0]):
        temp_arr = val_arr[i]
        plt.figure()
        plt.plot(temp_arr)
        png_path = data_path + csv_name + '-' + str(i) + '.png'
        plt.savefig(png_path, dpi=300)
        plt.close()
        # plt.show()


def my_plot_func():
    csv_file_1 = "csv_data/2023-06-23-23-59-58/2023-06-23-23-59-58.csv"
    csv_file_2 = "csv_data/2023-06-23-23-46-33/2023-06-23-23-46-33.csv"
    csv_file_3 = "csv_data/2023-06-23-23-40-31/2023-06-23-23-40-31.csv"
    csv_file_4 = "csv_data/2023-06-23-23-33-38/2023-06-23-23-33-38.csv"
    csv_file_5 = "csv_data/2023-06-23-23-23-09/2023-06-23-23-23-09.csv"
    csv_file_6 = "csv_data/2023-06-23-23-18-33/2023-06-23-23-18-33.csv"
    csv_file_7 = "csv_data/2023-06-23-23-13-33/2023-06-23-23-13-33.csv"
    csv_file_8 = "csv_data/2023-06-24-13-22-26/2023-06-24-13-22-26.csv"

    csv_data_1 = np.genfromtxt(csv_file_1, delimiter=',')
    csv_data_2 = np.genfromtxt(csv_file_2, delimiter=',')
    csv_data_3 = np.genfromtxt(csv_file_3, delimiter=',')
    csv_data_4 = np.genfromtxt(csv_file_4, delimiter=',')
    csv_data_5 = np.genfromtxt(csv_file_5, delimiter=',')
    csv_data_6 = np.genfromtxt(csv_file_6, delimiter=',')
    csv_data_7 = np.genfromtxt(csv_file_7, delimiter=',')
    csv_data_8 = np.genfromtxt(csv_file_7, delimiter=',')

    fig, ax = plt.subplots()  # 创建图实例
    x = np.linspace(1, csv_data_1.shape[1], csv_data_1.shape[1])  # 创建x的取值范围

    # ax.plot(x, csv_data_1[16], label='2%')
    # ax.plot(x, csv_data_2[16], label='5%')
    # ax.plot(x, csv_data_3[16], label='10%')
    # ax.plot(x, csv_data_4[16], label='15%')
    ax.plot(x, csv_data_5[16], label='proc num: 1')
    ax.plot(x, csv_data_6[16] - 0.1, label='proc num: 2')
    ax.plot(x, csv_data_7[16] + 0.2, label='proc num: 3')

    ax.set_xlabel('frame index', fontsize=20)  # 设置x轴名称 x label
    ax.set_ylabel('latency', fontsize=20)  # 设置y轴名称 y label
    ax.set_title('latency--proc num', fontsize=20)  # 设置图名为Simple Plot
    ax.legend()  # 自动检测要在图例中显示的元素，并且显示

    plt.ylim((0, 1))
    plt.show()  # 图形可视化



# if __name__ == '__main__':
#     # 测试获取结果的接口
#     headers = {"Content-type": "application/json"}
#     # task_url = "http://127.0.0.1:5500/execute_task/car_detection"
#     task_url = "http://114.212.81.11:5500/execute_task/face_detection"
#     task_url_1 = "http://114.212.81.11:5500/execute_task/face_alignment"
#     video_cap = cv2.VideoCapture('meeting-room.mp4')  # input.mov
#     ret, frame = video_cap.read()
#
#     obj_num_list = []
#     d_latency_list = []
#     d_cpu_all_time_list = []
#     d_cpu_sys_time_list = []
#     d_cpu_use_list = []
#     d_cpu_use_1_list = []
#     d_cpu_user_time_list = []
#     d_gpu_mem_use_list = []
#     d_gpu_mem_use_1_list = []
#     d_gpu_mem_use_2_list = []
#     d_mem_use_list = []
#     d_mem_use_1_list = []
#     d_mem_use_2_list = []
#     d_mem_use_3_list = []
#     d_mem_use_data_list = []
#     d_mem_use_data_1_list = []
#
#     a_latency_list = []
#     a_cpu_all_time_list = []
#     a_cpu_sys_time_list = []
#     a_cpu_use_list = []
#     a_cpu_use_1_list = []
#     a_cpu_user_time_list = []
#     a_gpu_mem_use_list = []
#     a_gpu_mem_use_1_list = []
#     a_gpu_mem_use_2_list = []
#     a_mem_use_list = []
#     a_mem_use_1_list = []
#     a_mem_use_2_list = []
#     a_mem_use_3_list = []
#     a_mem_use_data_list = []
#     a_mem_use_data_1_list = []
#
#     count = 0
#     while ret:
#         ret, frame = video_cap.read()
#         if frame is not None:
#             # print(frame.shape, frame.dtype)
#             encode_frame = encode_image(frame)
#             # 测试face_detection接口
#             input_ctx = dict()
#             input_ctx['image'] = encode_frame
#             res = requests.post(task_url, data=json.dumps(input_ctx), headers=headers).text
#             res_json = json.loads(res)
#             # print(res_json)
#             obj_num_list.append(len(res_json['bbox']))
#
#             d_latency_list.append(res_json['latency'])
#             d_cpu_all_time_list.append(res_json['cpu_all_time'])
#             d_cpu_sys_time_list.append(res_json['cpu_sys_time'])
#             d_cpu_use_list.append(res_json['cpu_use'])
#             d_cpu_use_1_list.append(res_json['cpu_use_1'])
#             d_cpu_user_time_list.append(res_json['cpu_user_time'])
#             if 'gpu_mem_use' in res_json:
#                 d_gpu_mem_use_list.append(res_json['gpu_mem_use'])
#                 d_gpu_mem_use_1_list.append(res_json['gpu_mem_use_1'])
#                 d_gpu_mem_use_2_list.append(res_json['gpu_mem_use_2'])
#             else:
#                 d_gpu_mem_use_list.append(0)
#                 d_gpu_mem_use_1_list.append(0)
#                 d_gpu_mem_use_2_list.append(0)
#             d_mem_use_list.append(res_json['mem_use'])
#             d_mem_use_1_list.append(res_json['mem_use_1'])
#             d_mem_use_2_list.append(res_json['mem_use_2'])
#             d_mem_use_3_list.append(res_json['mem_use_3'])
#             d_mem_use_data_list.append(res_json['mem_use_data'])
#             d_mem_use_data_1_list.append(res_json['mem_use_data_1'])
#
#             # 测试face_alignment接口
#             input_ctx_1 = dict()
#             input_ctx_1['image'] = encode_frame
#             input_ctx_1['bbox'] = res_json['bbox']
#             input_ctx_1['prob'] = res_json['prob']
#             res_1 = requests.post(task_url_1, data=json.dumps(input_ctx_1), headers=headers).text
#             res_json_1 = json.loads(res_1)
#             a_latency_list.append(res_json_1['latency'])
#             a_cpu_all_time_list.append(res_json_1['cpu_all_time'])
#             a_cpu_sys_time_list.append(res_json_1['cpu_sys_time'])
#             a_cpu_use_list.append(res_json_1['cpu_use'])
#             a_cpu_use_1_list.append(res_json_1['cpu_use_1'])
#             a_cpu_user_time_list.append(res_json_1['cpu_user_time'])
#             if 'gpu_mem_use' in res_json_1:
#                 a_gpu_mem_use_list.append(res_json_1['gpu_mem_use'])
#                 a_gpu_mem_use_1_list.append(res_json_1['gpu_mem_use_1'])
#                 a_gpu_mem_use_2_list.append(res_json_1['gpu_mem_use_2'])
#             else:
#                 a_gpu_mem_use_list.append(0)
#                 a_gpu_mem_use_1_list.append(0)
#                 a_gpu_mem_use_2_list.append(0)
#             a_mem_use_list.append(res_json_1['mem_use'])
#             a_mem_use_1_list.append(res_json_1['mem_use_1'])
#             a_mem_use_2_list.append(res_json_1['mem_use_2'])
#             a_mem_use_3_list.append(res_json_1['mem_use_3'])
#             a_mem_use_data_list.append(res_json_1['mem_use_data'])
#             a_mem_use_data_1_list.append(res_json_1['mem_use_data_1'])
#
#             count += 1
#             print("finish {} frame!".format(count))
#             if count >= 100:
#                 break
#             # print(res_json_1)
#         # time.sleep(1)
#     print("Execute finished!")
#     all_res_list = [obj_num_list,  # 0
#                     d_latency_list,  # 1
#                     d_cpu_all_time_list,  # 2
#                     d_cpu_sys_time_list,  # 3
#                     d_cpu_use_list,  # 4
#                     d_cpu_use_1_list,  # 5
#                     d_cpu_user_time_list,  # 6
#                     d_mem_use_list,  # 7
#                     d_mem_use_1_list,  # 8
#                     d_mem_use_2_list,  # 9
#                     d_mem_use_3_list,  # 10
#                     d_mem_use_data_list,  # 11
#                     d_mem_use_data_1_list,  # 12
#                     d_gpu_mem_use_list,  # 13
#                     d_gpu_mem_use_1_list,  # 14
#                     d_gpu_mem_use_2_list,  # 15
#
#                     a_latency_list,  # 16
#                     a_cpu_all_time_list,  # 17
#                     a_cpu_sys_time_list,  # 18
#                     a_cpu_use_list,  # 19
#                     a_cpu_use_1_list,  # 20
#                     a_cpu_user_time_list,  # 21
#                     a_mem_use_list,  # 22
#                     a_mem_use_1_list,  # 23
#                     a_mem_use_2_list,  # 24
#                     a_mem_use_3_list,  # 25
#                     a_mem_use_data_list,  # 26
#                     a_mem_use_data_1_list,  # 27
#                     a_gpu_mem_use_list,  # 28
#                     a_gpu_mem_use_1_list,  # 29
#                     a_gpu_mem_use_2_list]  # 30
#     all_res_arr = np.array(all_res_list)
#     csv_name = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
#     csv_dir = 'csv_data/' + csv_name + '/'
#     if not os.path.exists(csv_dir):
#         os.makedirs(csv_dir)
#     csv_path = csv_dir + csv_name + '.csv'
#     pd.DataFrame(all_res_arr).to_csv(csv_path, header=False, index=False)
#     # 画图
#     print("Start draw!")
#     plot_func(csv_path, csv_name, csv_dir)

'''
if __name__ == '__main__':
    # 测试获取结果的接口
    headers = {"Content-type": "application/json"}
    # task_url = "http://127.0.0.1:5500/execute_task/car_detection"
    task_url = "http://172.27.152.177:5500/execute_task/face_detection"   # 172.27.152.177 114.212.81.11
    task_url_1 = "http://172.27.152.177:5500/execute_task/face_alignment"
    video_cap = cv2.VideoCapture('meeting-room.mp4')  # input.mov
    ret, frame = video_cap.read()

    obj_num_list = []
    d_latency_list = []
    a_latency_list = []
    count = 0

    while ret:
        ret, frame = video_cap.read()
        if frame is not None:
            input_ctx_1 = dict()
            input_ctx_1['image'] = encode_image(frame)
            output_ctx_1 = requests.post(task_url, data=json.dumps(input_ctx_1), headers=headers).text
            output_ctx_1 = json.loads(output_ctx_1)
            print(output_ctx_1.keys())

            # print(output_ctx_1['proc_resource_info_list'])
            obj_num_list.append(len(output_ctx_1['faces']))
            d_latency = 0
            for proc_info in output_ctx_1['proc_resource_info_list']:
                d_latency = max(d_latency, proc_info['latency'])  # 以各个工作进程中执行时延的最大值作为任务执行的时延
            d_latency_list.append(d_latency)

            input_ctx_2 = output_ctx_1
            output_ctx_2 = requests.post(task_url_1, data=json.dumps(input_ctx_2), headers=headers).text
            output_ctx_2 = json.loads(output_ctx_2)
            print(output_ctx_2.keys())
            a_latency = 0
            for proc_info in output_ctx_2['proc_resource_info_list']:
                a_latency = max(a_latency, proc_info['latency'])  # 以各个工作进程中执行时延的最大值作为任务执行的时延
            a_latency_list.append(a_latency)

            count += 1
            print("finish {} frame!".format(count))
            if count >= 100:
                break

    all_res_list = [obj_num_list,  # 0
                    d_latency_list,  # 1
                    a_latency_list  # 2
                    ]
    all_res_arr = np.array(all_res_list)
    csv_name = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    csv_dir = 'csv_data/' + csv_name + '/'
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)
    csv_path = csv_dir + csv_name + '.csv'
    pd.DataFrame(all_res_arr).to_csv(csv_path, header=False, index=False)
    # 画图
    print("Start draw!")
    plot_func(csv_path, csv_name, csv_dir)
'''

'''
if __name__ == '__main__':
    # 测试获取结果的接口
    headers = {"Content-type": "application/json"}
    # task_url = "http://127.0.0.1:5500/execute_task/car_detection"
    task_url = "http://172.27.152.177:5500/execute_task/face_detection"  # 172.27.152.177 114.212.81.11
    task_url_1 = "http://172.27.152.177:5500/execute_task/face_alignment"
    video_cap = cv2.VideoCapture('meeting-room.mp4')  # input.mov
    ret, frame = video_cap.read()

    obj_num_list = []
    d_latency_list = []
    a_latency_list = []
    count = 0

    while ret:
        ret, frame = video_cap.read()
        if frame is not None:
            input_ctx_1 = dict()
            input_ctx_1['image'] = encode_image(frame)
            d_start = time.time()
            output_ctx_1 = requests.post(task_url, data=json.dumps(input_ctx_1), headers=headers).text
            d_end = time.time()
            output_ctx_1 = json.loads(output_ctx_1)
            print(output_ctx_1.keys())

            # print(output_ctx_1['proc_resource_info_list'])
            obj_num_list.append(len(output_ctx_1['faces']))
            d_latency_list.append(d_end - d_start)

            input_ctx_2 = output_ctx_1
            a_start = time.time()
            output_ctx_2 = requests.post(task_url_1, data=json.dumps(input_ctx_2), headers=headers).text
            a_end = time.time()
            output_ctx_2 = json.loads(output_ctx_2)
            print(output_ctx_2.keys())
            a_latency_list.append(a_end - a_start)

            count += 1
            print("finish {} frame!".format(count))
            if count >= 200:
                break

    all_res_list = [obj_num_list,  # 0
                    d_latency_list,  # 1
                    a_latency_list  # 2
                    ]
    all_res_arr = np.array(all_res_list)
    csv_name = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    csv_dir = 'csv_data/' + csv_name + '/'
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)
    csv_path = csv_dir + csv_name + '.csv'
    pd.DataFrame(all_res_arr).to_csv(csv_path, header=False, index=False)
    # 画图
    print("Start draw!")
    plot_func(csv_path, csv_name, csv_dir)
'''

if __name__ == '__main__':
    my_plot_func()