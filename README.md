# 2020秋季水下目标检测算法赛  underwater object detection algorithm contest Baseline <font color=red>**A榜 mAP 50.15 B榜 mAP 49.69**</font><br /> 

## base: underwater-object-detection(https://github.com/zhengye1995/underwater-object-detection)

## 代码内容和trick：

+ 基本网络模型
  + cascade rcnn
  + resnet50
  + soft nms
  + 基于mmdetection
  + mmcv==0.2.16
+ 添加的trick
  + dcn
  + global_context
  + RandomRotate90
  + cutout

## 代码运行结果

   + 使用resnet50可以达到A榜50.15mAP，B榜49.06mAP
   + resnet50和resnet101进行模型融合可以达到B榜49.69mAP

## 代码环境及依赖

+ OS: Ubuntu16.10
+ GPU: titan x*1
+ python: python3.7
+ nvidia 依赖:
   - cuda: 10.0.130
   - cudnn: 7.5.1
   - nvidia driver version: 430.14
+ deeplearning 框架: pytorch1.1.0
+ 其他依赖请参考requirement.txt
+ 显卡数量不太重要，大家依据自身显卡数量倍数调整学习率大小即可

## 训练数据准备

- **相应文件夹创建准备**

  - 在代码根目录下新建data文件夹，或者依据自身情况建立软链接
  - 进入data文件夹,创建文件夹:
  
     annotations

     pretrained

     results

     submit

  - (A榜)将官方提供的训练和测试数据解压到data目录中，产生：
    
    train

    test-A-image

  - (B榜)将官方提供的B榜数据解压到data目录中，产生：

    test-B-image
    
- **label文件格式转换**

  - 官方提供的是VOC格式的xml类型label文件，个人习惯使用COCO格式用于训练，所以进行格式转换
  
  - 使用 tools/data_process/xml2coco.py 将label文件转换为COCO格式，新的label文件 train.json 会保存在 data/train/annotations 目录下

  - 为了方便利用mmd多进程测试（速度较快），我们对test数据也生成一个伪标签文件,运行 tools/data_process/generate_test_json.py 生成 testA.json(b榜使用generate_testB_json.py生成testB.json), 伪标签文件会保存在data/train/annotations 目录下

  - A榜总体运行内容：

    - python tools/data_process/xml2coco.py

    - python tools/data_process/generate_test_json.py

  - B榜测试前的准备：

    - python tools/data_process/generate_testB_json.py

- **预训练模型下载**

  - resnet50_dconv_c3-c5[cascade_rcnn_dconv_c3-c5_r50_fpn_1x_20190125-dfa53166.pth](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/dcn/cascade_rcnn_dconv_c3-c5_r50_fpn_1x_20190125-dfa53166.pth)并放置于 data/pretrained 目录下
  - resnet101_dconv_c3-c5[cascade_rcnn_dconv_c3-c5_r101_fpn_1x_20190125-aaa877cc.pth](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/dcn/cascade_rcnn_dconv_c3-c5_r101_fpn_1x_20190125-aaa877cc.pth)并放置于 data/pretrained 目录下

## 依赖安装及编译


- **依赖安装编译**

   1. 创建并激活虚拟环境

        conda create -n underwater python=3.7 -y

        conda activate underwater

   2. 安装 pytorch

        conda install pytorch=1.1.0 torchvision=0.3.0 cudatoolkit=10.0 -c pytorch
        
   3. 安装其他依赖

        pip install cython && pip --no-cache-dir install -r requirements.txt
   
   4. 编译cuda op等：

        python setup.py develop(每次改mmdet的内容都建议重新运行一次此代码)
   

## 模型训练及预测（以下代码A榜和B榜的适配请自行修改）
    
   - **训练**

     1. 运行：
        
        r50:
        
        chmod +x tools/dist_train.sh	

        ./tools/dist_train.sh configs/underwater/cas_r50/cascade_rcnn_r50_fpn_1x_dcn_gc_data_aug_bboxj_testA.py 1
        
        r101:

        chmod +x tools/dist_train.sh	

        ./tools/dist_train.sh configs/underwater/cas_r101/cascade_rcnn_r101_fpn_1x_dcn_gc_data_aug_bboxj_testB.py 1
        
        (上面的1是我的gpu数量，请自行修改)

   	2. 训练过程文件及最终权重文件均保存在config文件中指定的work_dirs目录中

   - **预测**

    1. 运行:
    
       r50:

       chmod +x tools/dist_test.sh

       ./tools/dist_test.sh configs/underwater/cas_r50/cascade_rcnn_r50_fpn_1x_dcn_gc_data_aug_bboxj_testA.py work_dirs/cascade_rcnn_r50_fpn_1x_dcn_gc_data_aug_bboxj_testA/epoch_12.pth 1 --json_out results/cascade_rcnn_r50_testB.json

        (上面的4是我的gpu数量，请自行修改)
        
       r101:

       chmod +x tools/dist_test.sh

       ./tools/dist_test.sh configs/underwater/cas_r101/cascade_rcnn_r101_fpn_1x_dcn_gc_data_aug_bboxj_testB.py work_dirs/cascade_rcnn_r101_fpn_1x_dcn_gc_data_aug_bboxj_testA/epoch_12.pth 1 --json_out results/cascade_rcnn_r101_testB.json
        (上面的1是我的gpu数量，请自行修改)

    2. 预测结果文件会保存在 /results 目录下

    3. 转化mmd预测结果为提交csv格式文件：
       
       python tools/post_process/json2submit.py --test_json cascade_rcnn_r50_testB.bbox.json --submit_file cascade_rcnn_r50_testB.csv
       
       python tools/post_process/json2submit.py --test_json cascade_rcnn_r101_testB.bbox.json --submit_file cascade_rcnn_r101_testB.csv
       
       最终符合官方要求格式的提交文件位于 submit目录下

   - **模型融合(融合的是json文件，修改json文件的名称请打开json_tige.py进行修改)**

       python tools/json_toge.py 
    

## Contact

    author：ymzis69

    email：1750950070@qq.com
# underwater-object-detection-2020
