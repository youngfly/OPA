import matplotlib.pyplot as plt
import os


# todo 生成文件夹
def mkdir(path):
    isexists = os.path.exists(path)
    if not isexists:
        os.makedirs(path)
        print("file makes successfully")
    else:
        print("file is exist")
