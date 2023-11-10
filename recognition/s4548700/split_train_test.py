import os, random , shutil
def movfile (fileDir):
    pathDir = os.listdir(fileDir)
    filenumber=len(pathDir)
    rate = 0.8
    picknumber = int(filenumber*rate)
    sample = random.sample(pathDir, picknumber)
    print(sample)
    for name in sample:
        shutil.move(fileDir+name, tarDir+name)
    return

if __name__ =='__main__':
    fileDir = "./test/"
    tarDir = "./train/"
    movfile("./test/")
