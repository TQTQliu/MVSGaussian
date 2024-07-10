import numpy as np
from scipy.spatial.transform import Rotation

def get_interpolate_render_path(c2ws, N_views=30):
    N = len(c2ws)
    rotvec, positions = [], []
    rotvec_inteplat, positions_inteplat = [], []
    weight = np.linspace(1.0, .0, N_views//3, endpoint=False).reshape(-1, 1)
    for i in range(N):
        r = Rotation.from_matrix(c2ws[i, :3, :3])
        euler_ange = r.as_euler('xyz', degrees=True).reshape(1, 3)
        if i:
            mask = np.abs(euler_ange - rotvec[0]) > 180
            euler_ange[mask] += 360.0
        rotvec.append(euler_ange)
        positions.append(c2ws[i, :3, 3:].reshape(1, 3))

        if i:
            rotvec_inteplat.append(weight * rotvec[i - 1] + (1.0 - weight) * rotvec[i])
            positions_inteplat.append(weight * positions[i - 1] + (1.0 - weight) * positions[i])

    rotvec_inteplat.append(weight * rotvec[-1] + (1.0 - weight) * rotvec[0])
    positions_inteplat.append(weight * positions[-1] + (1.0 - weight) * positions[0])

    c2ws_render = []
    angles_inteplat, positions_inteplat = np.concatenate(rotvec_inteplat), np.concatenate(positions_inteplat)
    for rotvec, position in zip(angles_inteplat, positions_inteplat):
        c2w = np.eye(4)
        c2w[:3, :3] = Rotation.from_euler('xyz', rotvec, degrees=True).as_matrix()
        c2w[:3, 3:] = position.reshape(3, 1)
        c2ws_render.append(c2w.copy())
    c2ws_render = np.stack(c2ws_render)
    return c2ws_render


def normalize(v):
    """Normalize a vector."""
    return v / np.linalg.norm(v)


def average_poses(poses):
    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of pose centers.
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.

    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        pose_avg: (3, 4) the average pose
    """
    # 1. Compute the center
    center = poses[..., 3].mean(0)  # (3)
    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0))  # (3)
    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0)  # (3)
    # 4. Compute the x axis
    x = normalize(np.cross(y_, z))  # (3)
    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(z, x)  # (3)
    pose_avg = np.stack([x, y, z, center], 1)  # (3, 4)
    return pose_avg

def center_poses(poses, blender2opencv):
    """
    Center the poses so that we can use NDC.
    See https://github.com/bmild/nerf/issues/34
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        poses_centered: (N_images, 3, 4) the centered poses
        pose_avg: (3, 4) the average pose
    """
    pose_avg = average_poses(poses)  # (3, 4)
    pose_avg_homo = np.eye(4)
    pose_avg_homo[:3] = pose_avg  # convert to homogeneous coordinate for faster computation
    # by simply adding 0, 0, 0, 1 as the last row
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1))  # (N_images, 1, 4)
    poses_homo = np.concatenate([poses, last_row], 1)  # (N_images, 4, 4) homogeneous coordinate

    poses_centered = np.linalg.inv(pose_avg_homo) @ poses_homo  # (N_images, 4, 4)
    poses_centered = poses_centered @ blender2opencv
    poses_centered = poses_centered[:, :3]  # (N_images, 3, 4)

    return poses_centered

# --------------------- Start: Render scriptes for LLFF dataset video generation --------------------
def get_spiral_render_path(c2ws_all, near_far, rads_scale=0.5, N_views=120):
    # center pose
    c2w = poses_avg(c2ws_all)

    # Get average pose
    up = normalize(c2ws_all[:, :3, 1].sum(0))

    # Find a reasonable "focus depth" for this dataset
    close_depth, inf_depth = near_far
    # print(near_far)
    dt = .75
    mean_dz = 1./(((1.-dt)/close_depth + dt/inf_depth))
    focal = mean_dz
    # print(focal)
    # Get radii for spiral path
    shrink_factor = .8
    zdelta = close_depth * .2
    tt = c2ws_all[:, :3, 3] - c2w[:3, 3][None]
    rads = np.percentile(np.abs(tt), 70, 0)*rads_scale
    render_poses = render_path_spiral(c2w, up, rads, focal, zdelta, zrate=.5, N=N_views)
    return np.stack(render_poses)

def poses_avg(poses):
    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = viewmatrix(vec2, up, center)

    return c2w


def normalize(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.eye(4)
    m[:3] = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, N_rots=2, N=120):
    render_poses = []
    rads = np.array(list(rads) + [1.])

    for theta in np.linspace(0., 2. * np.pi * N_rots, N+1)[:-1]:
        c = np.dot(c2w[:3, :4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta*zrate), 1.]) * rads)
        z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.])))
        render_poses.append(viewmatrix(z, up, c))
    return render_poses
# --------------------- End: Render scriptes for LLFF dataset video generation --------------------