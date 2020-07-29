Camera matrices calibration
==================================================

The objective of teh calibration procedure is to find:

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