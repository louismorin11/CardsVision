import os, os.path

imgs = []
path = "/home/louis/projects/vision/test_severals"
valid_images = [".jpg",".gif",".png",".tga"]
for f in os.listdir(path):
    ext = os.path.splitext(f)[1]
    if ext.lower() not in valid_images:
        continue
    imgs.append((os.path.join(path,f)))

for i in imgs:
    os.system('python scan.py --image '+i+' -s true')
