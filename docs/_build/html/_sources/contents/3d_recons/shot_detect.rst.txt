Shot detection
==================================================

The shot detection is simply performed by thresholding the difference between the analyzed picture and a reference picture taken without the shot. This is performed by the function :py:func:`data_treat.objectExtract.filter_val` .

.. currentmodule:: data_treat.objectExtract
.. autofunction:: filter_val

.. image:: figures/shot_detec.png

Adjusting the threshold is generally sufficient to obtain a relatively good shot position estimation on the camera, despite the simplicity of teh approach.

The function :py:func:`data_treat.objectExtract.compute_2d_traj` extracts the shot trajectory for every pictures given for a camera

.. currentmodule:: data_treat.objectExtract
.. autofunction:: compute_2d_traj

.. image:: figures/shot_mirror.png
   :align: center

In certain cases, reflection in the sample's surface can induce deviation in the detected trajectory. A simple fix is to mask the part of the image displaying the sample. This is possible by setting the `mask_w` and `mask_h` values of the camera object to the wask width (starting from the left of the picture) and height (starting from the bottom of the picture). This can be performed using the :py:func:`data_treat.cam.Cam.set_mask` of the camera object.

.. currentmodule:: data_treat.cam.Cam
.. autofunction:: set_mask