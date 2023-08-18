import cv2
import numpy as np
import random

# random.seed(0)
if __name__ == '__main__':

    #背景が黒の単色画像を作る
    height = 512
    width = 512

    for type in range(2,13):
        for count in range(10000):
            NN = 1
            black = np.zeros((height, width, 3))
            rand_ii = ([random.randint(int(width*0.4),int(width*0.6)) for i in range(NN)])
            rand_jj = ([random.randint(int(height*0.4),int(height*0.6)) for i in range(NN)])


            for nn_ in range(NN):
                size = random.uniform(width*0.1,width*0.4)
                aspc = random.uniform(0.2,1)
                color = ([random.randint(0,255) for i in range(3)])
                
                if type <= 2:
                    cv2.ellipse(black, ((rand_ii[nn_], rand_jj[nn_]), (size, size*aspc), 0), color, thickness=-1)
                if type >  2:
                    ang_step = 2*np.pi/type
                    points = []
                    for mm_ in range(type):
                        ang_ = ang_step*mm_
                        r__  = np.sqrt(2)*size
                        x_   = int(np.round(r__*np.cos(ang_)     )) + rand_ii[nn_]
                        y_   = int(np.round(r__*np.sin(ang_)*aspc)) + rand_jj[nn_]
                        points.append((x_,y_))
                    cv2.fillPoly(black, [np.array(points)], color)
                    
                ori_ = random.uniform(0,90)
                trans = cv2.getRotationMatrix2D((black.shape[0]/2,black.shape[1]/2), ori_ , 1.0)
                image2 = cv2.warpAffine(black, trans, (width,height))
                
            cv2.imwrite('./img/img_%02d_%04d.png'%(type,count), image2)