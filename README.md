# Faster-RCNN-TensorFlow-Python3.5 refered from https://github.com/dBeker/Faster-RCNN-TensorFlow-Python3.5.git
faster-rcnn retrained on video image data

训练前准备  
1.cd ./data/coco/PythonAPI  
执行：  
	python setup.py build_ext --inplace  
	python setup.py build_ext install  
2.download voc2007 from https://github.com/rbgirshick/py-faster-rcnn#beyond-the-demo-installation-for-training-and-testing-models  
解压到：data/VOCDevkit2007/VOC2007  
3.download pre-trained VGG16 from http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz  
解压、重命名到：data\imagenet_weights\vgg16.ckpt  


皮卡丘数据测试  
for train:  
1.使用皮卡丘标注数据  
2.训练记录  

修改1：皮卡丘标注数据替换./data/VOCdevkit2007/VOC2007  
修改2：./lib/datasets/pascal_voc.py  
line 33，self._classes = ('__background__',  # always index 0
                         'aeroplane', 'bicycle', 'bird', 'boat',
                         'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse',
                         'motorbike', 'person', 'pottedplant',
                         'sheep', 'sofa', 'train', 'tvmonitor')  
改为：  
self._classes = ('__background__',  # always index 0
                         'apple','tree','yellow','blue','white','histogram')  
注：classes这里前景类别为'apple','tree','yellow','blue','white','histogram'，可根据标注类别名称替换  

修改3：./lib/config/config.py  
max_iters=15  
snapshot_iterations=10  
训练15次测试，成功,输出10次model文件  

3.extractor.py测试记录  
测试pikachu_1768.jpg  
tag：{'yellow': 0.15671853721141815}  
