"""`lmj_deploy.py` reimplements the training math in numpy; the two must agree."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

import lmj_deploy as dep
from lmj_real.core.utils import tracking as trk


@pytest.fixture
def frames():
    rng = np.random.default_rng(0)
    quats_xyzw = R.random(6, random_state=1).as_quat()
    quats = np.concatenate([quats_xyzw[:, 3:], quats_xyzw[:, :3]], axis=1)  # MuJoCo order: wxyz
    return rng.normal(size=(6, 3)), quats, R.from_quat(quats_xyzw).as_matrix()


def test_quat_to_mat(frames):
    _, quats, mats = frames
    np.testing.assert_allclose(dep.quat_to_mat(quats), mats, atol=1e-12)
    np.testing.assert_allclose(trk.mat_from_quat(quats, np), mats, atol=1e-6)


def test_6d_encoding_is_columns_0_and_1(frames):
    _, _, mats = frames
    six = dep.mat_to_6d_cols01(mats)
    np.testing.assert_allclose(six, trk.mat_to_6d_cols01(mats, np), atol=1e-12)
    np.testing.assert_allclose(six.reshape(-1, 3, 2), mats[:, :, :2], atol=1e-12)


def test_yaw_rot(frames):
    _, _, mats = frames
    np.testing.assert_allclose(dep.yaw_rot(mats[0]), trk.yaw_rot(mats[0], np), atol=1e-12)


def test_reanchoring(frames):
    pos, _, mats = frames
    robot_pos, robot_mat = np.array([4.0, -2.0, 0.7]), mats[5]
    d_pos, d_rot = dep.anchor_delta(robot_pos, robot_mat, pos[0], mats[0])
    t_pos, t_rot = trk.anchor_delta(robot_pos, robot_mat, pos[0], mats[0], np)
    np.testing.assert_allclose(d_pos, t_pos, atol=1e-12)
    np.testing.assert_allclose(d_rot, t_rot, atol=1e-12)
    a_pos, a_mat = dep.reanchor(pos, mats, pos[0], d_pos, d_rot)
    b_pos, b_mat = trk.reanchor(pos, mats, pos[0], t_pos, t_rot, np)
    np.testing.assert_allclose(a_pos, b_pos, atol=1e-12)
    np.testing.assert_allclose(a_mat, b_mat, atol=1e-12)
    # The re-anchored reference sits at the robot's xy and keeps its own height.
    np.testing.assert_allclose(a_pos[0], [robot_pos[0], robot_pos[1], pos[0, 2]], atol=1e-12)
