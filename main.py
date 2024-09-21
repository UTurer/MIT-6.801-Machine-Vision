print("======= START =======")
import numpy as np
import cv2
import glob
import scipy
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial
from numpy.polynomial.polynomial import polyval

PNTS=np.array([[1,2],[3,4],[4,6]])
N, M = PNTS.shape
A=np.column_stack((PNTS[:,0],np.ones((N,1))))
B=PNTS[:,1]

Ainv=np.linalg.pinv(A)
a=np.dot(Ainv,B)
print(a)

x=np.arange(0,10,1)
y=polyval(x, np.flip(a))

plt.plot(PNTS[:,0],PNTS[:,1],"ro")
plt.plot(x,y,"k")
plt.show()

exit()

filelist = glob.glob("example1\\*.png")

collection = []
for fname in filelist:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    collection.append(np.array(gray))
collection = np.array(collection)

#cv2.imshow("Image",collection[0])
#cv2.waitKey(0)

Ex = np.diff(collection, axis=2)
Ey = np.diff(collection, axis=1)
Et = np.diff(collection, axis=0)
N, rows, cols = Ex.shape
print(rows)
print(cols)

A=np.array([[np.ravel(Ex)],[np.ravel(Ey)]])
B=np.array([np.ravel(Et)])

print("======== END ========")
