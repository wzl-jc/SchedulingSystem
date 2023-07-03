### 代码结构说明：
```
SchedulingSystem
|--README.md
|--InterfaceSpecification.md  // 接口说明文档（运行时情境、软件下装、计算服务调用）
|--RelatedLearning.md  // 项目相关的学习参考链接
|--test.py  // 辅助测试系统功能
|
|--server
|--|--app_server.py  // 云端入口文件
|--|--field_codec_utils.py  // 图像编解码工具文件
|--|--dist  // 旧版前端页面，用于测试软件下装功能，后续可弃用
|--|--templates  // 旧版前端页面，用于测试软件下装功能，后续可弃用
|
|--client
|--|--app_client.py  // 边端入口文件
|--|--field_codec_utils.py  // 图像编解码工具文件
```
### 系统启动方式：
* 启动云端，执行命令如下：
```shell
# --server_ip指定云端提供服务的ip，设置为服务器的公网ip
# 注意，在ServerManager的构造函数中self.server_ip初始设置需要与服务器ip保持一致，供定时事件使用
# --server_port指定云端提供服务的port
# --edge_port指定边端提供服务的port，所有边端保持一致
python3 app_server.py --server_ip=114.212.81.11 --server_port=5500 --edge_port=5500
```
* 启动边缘端，执行命令如下：
```shell
# --server_ip指定云端提供服务的ip, --server_port指定云端提供服务的port
# --edge_ip指定边端提供服务的ip,设置为0.0.0.0以便同时提供定时事件和计算服务
# --edge_port指定边端提供服务的port
python3 app_client.py --server_ip=114.212.81.11 --server_port=5500 --edge_ip=0.0.0.0 --edge_port=5500
```




