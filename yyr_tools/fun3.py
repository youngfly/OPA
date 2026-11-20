import os
import json
import xml.etree.ElementTree as ET
import glob

def dota():
    jsonpath = '/home/airstudio/data/dota_data/train1024/DOTA_train1024.json'
    with open(jsonpath, 'r') as f:
        label = json.load(f)

    images = label['images']
    cate = label['categories']
    data = label['annotations']
    num = len(data)
    area_dic = {}
    w_h_dic = {}
    area_sta = {'100':0, '1000':0, '10000':0, '100000': 0, '1000000':0, '10000000':0}

    a_ma = 0
    a_mi = 100000000
    # todo 统计数据集里面的面积的以及长宽比
    for i in range(len(data)):
        object = data[i]
        area = int(object['area'])
        w = object['bbox'][2]
        h = object['bbox'][3]
        # w_h = float(w/h)
        id = object['category_id']
        if area not in area_dic:
            area_dic.update({"{}".format(area): 0})
        area_dic["{}".format(area)] += 1
        # w_h_dic ['{}'.format(w_h)] +=1
        # todo 寻找最大值以及最小值
        if area > a_ma:
            a_ma = area
        if area <a_mi:
            a_mi = area
        # todo 统计各个段里面的物体数
        if area > -1 and area < 100:
            area_sta["100"] +=1
        elif area >=100 and area < 1000:
            area_sta["1000"] += 1
        elif area >=1000 and area < 10000:
            area_sta["10000"] += 1
        elif area >=10000 and area < 100000:
            area_sta["100000"] += 1
        elif area >= 100000 and area < 1000000:
            area_sta["1000000"] += 1
        elif area >= 1000000 and area < 10000000:
            area_sta["10000000"] += 1
    print(a_ma)
    print(a_mi)
    # todo 实例数
    print(num)
    print(area_sta)

def voc():
    num = 0
    a_ma = 0
    a_mi = 100000000
    area_dic = {}
    area_sta = {'100':0, '1000':0, '10000':0, '100000': 0, '1000000':0, '10000000':0}
    xml_path1 = '/home/airstudio/data/VOCdevkit/VOC2007/Annotations/'
    xml_txt1 = '/home/airstudio/data/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt'
    xml_path2 = '/home/airstudio/data/VOCdevkit/VOC2012/Annotations/'
    xml_txt2 = '/home/airstudio/data/VOCdevkit/VOC2012/ImageSets/Main/trainval.txt'

    # todo 07
    txt1 = []
    with open(xml_txt1,'r') as f:
        for line in f:
            name = line.strip('\n') + '.xml'
            txt1.append(name)
    # todo 12
    txt2 = []
    with open(xml_txt2,'r') as f:
        for line in f:
            name = line.strip('\n') + '.xml'
            txt2.append(name)
    # todo 07
    os.chdir(xml_path1)
    anno = os.listdir('./')
    for i , file in enumerate(anno):
        # print(file)
        # exit()
        if file not in (txt1):
            continue
        in_file = open(file)
        tree = ET.parse(in_file)
        root = tree.getroot()
        for obj in root.iter('object'):
            xml = obj.find('bndbox')
            xmin = int(xml.find('xmin').text)
            xmax = int(xml.find('xmax').text)
            ymin = int(xml.find('ymin').text)
            ymax = int(xml.find('ymax').text)
            area = (xmax - xmin) * (ymax - ymin)
            num +=1
            if area not in area_dic:
                area_dic.update({"{}".format(area): 0})
            area_dic["{}".format(area)] += 1
            # todo 寻找最大值以及最小值
            if area > a_ma:
                a_ma = area
            if area < a_mi:
                a_mi = area
            # todo 统计各个段里面的物体数
            if area > -1 and area < 100:
                area_sta["100"] += 1
            elif area >= 100 and area < 1000:
                area_sta["1000"] += 1
            elif area >= 1000 and area < 10000:
                area_sta["10000"] += 1
            elif area >= 10000 and area < 100000:
                area_sta["100000"] += 1
            elif area >= 100000 and area < 1000000:
                area_sta["1000000"] += 1
            elif area >= 1000000 and area < 10000000:
                area_sta["10000000"] += 1
        # exit(0)

    # todo 12
    os.chdir(xml_path2)
    anno2 = os.listdir('./')
    for i , file in enumerate(anno2):
        # print(file)
        # exit()
        if file not in (txt2):
            continue
        in_file = open(file)
        tree = ET.parse(in_file)
        root = tree.getroot()
        for obj in root.iter('object'):
            xml = obj.find('bndbox')
            xmin = int(xml.find('xmin').text)
            xmax = int(xml.find('xmax').text)
            ymin = int(xml.find('ymin').text)
            ymax = int(xml.find('ymax').text)
            area = (xmax - xmin) * (ymax - ymin)
            num += 1
            if area not in area_dic:
                area_dic.update({"{}".format(area): 0})
            area_dic["{}".format(area)] += 1
            # todo 寻找最大值以及最小值
            if area > a_ma:
                a_ma = area
            if area < a_mi:
                a_mi = area
            # todo 统计各个段里面的物体数
            if area > -1 and area < 100:
                area_sta["100"] += 1
            elif area >= 100 and area < 1000:
                area_sta["1000"] += 1
            elif area >= 1000 and area < 10000:
                area_sta["10000"] += 1
            elif area >= 10000 and area < 100000:
                area_sta["100000"] += 1
            elif area >= 100000 and area < 1000000:
                area_sta["1000000"] += 1
            elif area >= 1000000 and area < 10000000:
                area_sta["10000000"] += 1
        # exit(0)
    print(num)
    print(a_ma)
    print(a_mi)
    print(area_sta)

voc()
