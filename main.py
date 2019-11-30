import matplotlib.pyplot as plt
import numpy as np

from reconstruction_3d import reconstruct_3d

dirs = ["150psi1_left", "150psi1_top"]
firstPics = ["150psi1_0001.jpg", "150psi1_0001.jpg"]
pic_to_cms = [1./133.2958, 1./139.428]
framerate = 20000
firstNum = 1
cropSize = 50
splitSymb = "_"
numsplit = 1
dt = 1. / framerate

X, Y, Z, timespan, X2 = reconstruct_3d(90, dirs, firstPics, cropSize, pic_to_cms, framerate, splitSymb="_", numsplit=1)

plt.figure(figsize=(6, 8))
plt.subplot(211)
plt.plot(1000*timespan, X, '.-', label="X")
plt.plot(1000*timespan, X2, '.-', label="X2")
plt.plot(1000*timespan, Y, '.-', label="Y")
plt.plot(1000*timespan, Z, '.-', label="Z")
plt.xlabel("time ($\mu s$)")
plt.ylabel("Position (cm)")

plt.subplot(212)
velx = np.diff(X)/dt
plt.plot(1000*timespan[1:], velx[:], '.-', label="X")
vely = np.diff(Y)/dt
plt.plot(1000*timespan[1:], vely[:], '.-', label="Y")
velz = np.diff(Z)/dt
plt.plot(1000*timespan[1:], velz[:], '.-', label="Z")
plt.xlabel("time (ms)")
plt.ylabel("Velocity (m/s)")
print(velx[5])
plt.savefig("Results.jpg")
plt.legend()
plt.show()

plt.plot(1000*timespan, 1e6*(X2-X))
plt.ylabel("X2- X1 ($\mu m$)")
plt.xlabel("time (ms)")
plt.show()