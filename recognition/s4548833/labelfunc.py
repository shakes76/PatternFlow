import os
import cv2

def read_directory(directory_name):
    
    array_of_name=[]
    result=[]
    for filename in os.listdir(r"./"+directory_name):
        img = cv2.imread(directory_name + "/" + filename)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        size = img.shape
        x_len=size[1]
        y_len=size[0]
        ret,thresh =cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
        contours, hier = cv2.findContours(thresh,cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            x,y,width,height = cv2.boundingRect(c)
            output="0 "    
            x_center=round((x+width/2)/x_len,6)
            y_center=round((y+height/2)/y_len,6)
            w_normalized=round(width/x_len,6)
            h_normalized=round(height/y_len,6)
            output+=str(x_center)+" "
            output+=str(y_center)+" "
            output+=str(w_normalized)+" "
            output+=str(h_normalized)        
            cv2.rectangle(img, (x,y), (x+width, y+height), (255, 255, 0), 2)
        cv2.drawContours(img, contours, -1, (255, 0, 0), 1)

        print(output)
        result.append(output)
        del img
        array_of_name.append(filename)
    return result,array_of_name

result,names=read_directory("train label")

def write_file():
    array_of_name = []
    for name in names:
        name=name.strip('_segmentation.png')
        array_of_name.append(name)

    if not os.path.exists("./train label output"):
        os.makedirs("./train label output")
    for i in range(len(array_of_name)):
        with open("./train label output/"+array_of_name[i]+".txt","w") as f:
            f.write(result[i])
a = write_file()
    
           
