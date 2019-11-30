from objectExtract import compute_2d_traj

def reconstruct_3d(angle, dirs, firstPics, cropSize, pic_to_cms, framerate, splitSymb="_", numsplit=1):
    """
    Reconstruct the 3D trajectory of a moving object filmed by 2 cameras with a given angle between them

    :param angle:
    :param dir1:
    :param dir2:
    :param firstPic:
    :param cropSize:
    :param pic_to_cm:
    :param framerate:
    :param splitSymb:
    :param numsplit:
    :return:
    """

    traj_2d_1, timespan1 = compute_2d_traj(dirs[0], firstPics[0], cropSize, pic_to_cms[0], framerate, splitSymb="_", numsplit=1)
    traj_2d_2, timespan2 = compute_2d_traj(dirs[1], firstPics[1], cropSize, pic_to_cms[1], framerate, splitSymb="_", numsplit=1)
    minspan_len = min(len(timespan1), len(timespan2))
    X = traj_2d_1[:minspan_len,0]
    X2 = -traj_2d_2[:minspan_len, 1]
    X2 += (X[0]- X2[0])
    Y = traj_2d_2[:minspan_len, 0]
    Z = traj_2d_1[:minspan_len, 1]

    return X, Y, Z, timespan1[:minspan_len], X2