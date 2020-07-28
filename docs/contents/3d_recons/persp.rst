Accounting for perspective
==================================================

Trajectory reconstruction
**************************************************

Perspective is accounted for by using the common pinhole camera model (see the `opencv documentation
<https://docs.opencv.org/2.4/doc/tutorials/calib3d/camera_calibration/camera_calibration.html>`_
for more detail)

First, a camera calibration must be performed for each camera to obtain both the camera intrinsic matrix :math:`F` and the transformation matrix between the camera coordinate system and the sample's coordinate system (CS) :math:`M`

For each camera, the shot 3D coordinates in the sample CS are therefore related by the following relation:

:math:`\begin{bmatrix} s_1 u_1 \\ s_1 v_1 \\ s_1 \end{bmatrix} = F_1 \cdot M_1 \cdot \begin{bmatrix} X \\ Y \\ Z \\ 1 \end{bmatrix},`

:math:`\begin{bmatrix} s_2 u_2 \\ s_2 v_2 \\ s_2 \end{bmatrix} = F_2 \cdot M_2 \cdot \begin{bmatrix} X \\ Y \\ Z \\ 1 \end{bmatrix},`

for camera 1 and 2 respectively. 

The two equations above can therefore be re-written as a linear system with 4 equations for our three unknowns (X,Y and Z).

Since the two cameras sees approximately the same information toward the y coordinate of the shots, one of the above equation can be ignored to obtain a simple linear system with 3 equations and three unknowns

The function :py:func:`data_treat.reconstruction_3d.get_3d_coor` reconstructs the shot 3D trajectory with this method, assuming that the shot position on each camera and the cameras intrinsic and transformation matrices were already obtained.

.. currentmodule:: data_treat.reconstruction_3d
.. autofunction:: get_3d_coor

Using the method "persp", the function only builds the linear system, ignoring on of the four equations and inverts it to get the shot 3D trajectory.

However, camera calibrations are subjected to errors and the shot position detection on eahc camera is not always perfect. The solution of the linear system might therefore not fully be consistent with the coordinates found on both cameras.

To enhance the obtained solution, the "persp-opti" method uses the linear system solution as a first guess and finds the best (X,Y,Z) solution that minimizes the reprojection error on both cameras using least-square optimization.

.. image:: figures/persp-opti.png
   :align: center

Error indicators
**************************************************