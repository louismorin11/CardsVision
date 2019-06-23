import tkinter as tk
import os

PathFolderFreshData = './NoLabelData'




class File:
    def __init__(self, path, stats):
        self.path=path.split('/')[-1]
        self.stats=stats
    def isEqual(self,aFile):
        if self.path != aFile.path:
            return False
        if self.stats.st_size==aFile.stats.st_size:
            return True
        else:
            return False


all_files = []
for (root, dirs, files) in os.walk(PathFolderFreshData):
    for f in files:
        all_files.append(os.path.join(root,f))

print(all_files)
all_files_obj=[]
for file in all_files:
    all_files_obj.append(File(file,os.stat(file)))

print(all_files_obj)

for i in range(len(all_files_obj)):
    for j in range(len(all_files_obj)):
        if i!=j:
            print(all_files_obj[i].path + ' and '+ all_files_obj[j].path)
            print(all_files_obj[i].isEqual(all_files_obj[j]))
