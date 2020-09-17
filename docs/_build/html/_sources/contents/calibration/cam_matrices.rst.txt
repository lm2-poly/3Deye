Camera matrices calibration
==================================================

General principles
*********************************************

The objective of the calibration procedure is to find:

* **The camera intrinsic matrix** which results from the camera geometry :math:`F`.
* **The transormation matrix** between the sample and the camera coordinate systems :math:`M`.

The intrinsic matrix is typically obtained by taking several pictures of a checkboard in different orientations. For more information, see `The openCV documentation
<https://docs.opencv.org/2.4/doc/tutorials/calib3d/camera_calibration/camera_calibration.html>`_.

All the calibration functions are implemented in the :py:mod:`calibration` module and are largely inspired from `the following source
<https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html>`_.

The intrinsic camera matrix is recovered using the provided chessboard picture by the :py:func:`calibration.calib_len.get_cam_matrix` function

.. currentmodule:: calibration.calib_len
.. autofunction:: get_cam_matrix

The transformation matrix between each camera and the sample coordinate system can be obtained by taking a picture of a chessboard positioned in the sample coordiate system, e.g. at a 90° angle from the sample, so that it is seen by both camera simultaneously, as shown in the picture below.

.. image:: figures/transfo_matrix.png
   :align: center
   
The chessboard position is then computed using the `solvePnP` opencv function which provides the rotation Rodrigues vector and translation vector between the two coordinate systems.

Before computing the transformation matrix the chessboard orientation is recovered by detecting the circles positionned on the chessboard with the :py:func:`calibration.chessboard_tools.get_blob_position` function. The chessbard coordinate system is then rotated by the :py:func:`calibration.chessboard_tools.change_chess_ori` function. This is necessary to ensure that the two transformation matrix obtained for the two cameras will chose the same orientation for the chessboard coordinate system.

.. image:: figures/chess_ori.png
   :align: center

All of the step to find a camera transformation matrix with the right orientation are performed by the :py:func:`calibration.find_sample_ori.get_transfo_mat` function.

.. currentmodule:: calibration.find_sample_ori
.. autofunction:: get_transfo_mat

The function :py:func:`calibration.find_sample_ori.plot_proj_origin` plots the reprojected chessboard points as well as the projection of the coordinate system axes and origin to check that the quality of the calibration. One must ensure that

* The projected chess board points match effectively with the chessboard corners
* The projected axes are consistent between the two cameras.

Finally, the whole calibration procedure can be performed at once using the :py:func:`calibration.calibrate_stereo` function.

.. currentmodule:: calibration.main
.. autofunction:: calibrate_stereo

The function will save the calibration files contaning each camera matrices in the calibration folder, in a format that can be read by the analysis and post-processing programs. 

.. note::
   As it is computed by opencv, the camera rotation matrix is given as a Rodrigues vector.

Manual calibration
*********************************************

Sometimes the :py:func:`calibration.find_sample_ori` may fail to detect the chessboard corners. In particular, when performing a calibration on a tilted sample, part of the chessboard might be out of focus for one of the two cameras which results in blury chessboard corners.

When automatic detection does not seem to be possible, the coordinate of the chessboard corners can be manually chosen. The list of point coordinates in the right order can then be given as input to :py:func:`calibration.find_sample_ori` using the imgpoints parameter.

To obtain the corner coordinates list by simply clicking on the picture, use the click_coor.py provided in 3Drecons/calibration/manual.
Simply change the value of the **File** variable for the path of your chessboard picture and click on the corners. When clicking, the point coordinates will automatically be appended in a "coord_list" file as a list.

Please be careful to select the chess corners in a consistent order for the top and left camera to ensure both camera will have the same origin and axes orientation. You can for instance follow the order provided in the pictures below to be consistent with 3Deye default.

.. image:: figures/calib_order.png
   :align: center

Once all the chessboard point are selected for each picture, use the "manual.py" code to compute the calibration files for the two cameras using the manually provided coordinates. Two files "coord_list_top" and "coord_list_left" files containig the coordinate list of the corners should be positioned in the same directory as the code. The following variable values might need to be changed:

* left_lens, right_lens: path of the calibration chessboard pictures for the top and left cameras (intrinsic matrix)
* left_pos, right_pos: path of the left and top chessboard picture for the transormation matrix calculation
* calib_folder: calibration folder where the calibration files should be saved
* chess_dim: chess dimension (number of cases)
* chess_case_length: chess case length (in cm)

