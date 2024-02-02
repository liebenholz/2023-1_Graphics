import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np
import os

offsets=np.array(
[
    [ 0. , 0. , 0. ],
    [ 0.      ,  0.      ,  0.      ],
    [ 1.36306 , -1.79463 ,  0.83929 ],
    [ 2.44811 , -6.72613 ,  0.      ],
    [ 2.5622  , -7.03959 ,  0.      ],
    [ 0.15764 , -0.43311 ,  2.32255 ],
    [ 0.      ,  0.      ,  0.      ],
    [-1.30552 , -1.79463 ,  0.83929 ],
    [-2.54253 , -6.98555 ,  0.      ],
    [-2.56826 , -7.05623 ,  0.      ],
    [-0.16473 , -0.45259 ,  2.36315 ],
    [ 0.      ,  0.      ,  0.      ],
    [ 0.02827 ,  2.03559 , -0.19338 ],
    [ 0.05672 ,  2.04885 , -0.04275 ],
    [ 0.      ,  0.      ,  0.      ],
    [-0.05417 ,  1.74624 ,  0.17202 ],
    [ 0.10407 ,  1.76136 , -0.12397 ],
    [ 0.      ,  0.      ,  0.      ],
    [ 3.36241 ,  1.20089 , -0.31121 ],
    [ 4.983   , -0.      , -0.      ],
    [ 3.48356 , -0.      , -0.      ],
    [ 0.      ,  0.      ,  0.      ],
    [ 0.71526 , -0.      , -0.      ],
    [ 0.      ,  0.      ,  0.      ],
    [ 0.      ,  0.      ,  0.      ],
    [-3.1366  ,  1.37405 , -0.40465 ],
    [-5.2419  , -0.      , -0.      ],
    [-3.44417 , -0.      , -0.      ],
    [ 0.      ,  0.      ,  0.      ],
    [-0.62253 , -0.      , -0.      ],
    [ 0.      ,  0.      ,  0.      ],
])

parents= [-1,  0,  1,  2,  3,  4,  0,  6,  7,  8,  9,  0, 11, 12, 13, 14, 15,
    13, 17, 18, 19, 20, 21, 20, 13, 24, 25, 26, 27, 28, 27]

def q_log(q):
    imgs = q[1:]
    lens = np.sqrt(np.sum(imgs**2, axis=-1))
    lens = np.arctan2(lens, q[0]) / (lens + 1e-10)
    return imgs * lens


def q_exp(lq):
    ts = np.sum(lq**2.0, axis=-1, keepdims=True)**0.5
    ts[ts == 0] = 0.001
    ls = np.sin(ts) / ts
    
    qs = np.empty((4,))
    qs[...,0] = np.cos(ts)
    qs[...,1] = lq[0] * ls
    qs[...,2] = lq[1] * ls
    qs[...,3] = lq[2] * ls

    return qs

def q_mul(sq, oq):
    q0 = sq[0]; q1 = sq[1]; 
    q2 = sq[2]; q3 = sq[3]; 
    r0 = oq[0]; r1 = oq[1]; 
    r2 = oq[2]; r3 = oq[3]; 
    
    qs = np.empty((4,))
    qs[0] = r0 * q0 - r1 * q1 - r2 * q2 - r3 * q3
    qs[1] = r0 * q1 + r1 * q0 - r2 * q3 + r3 * q2
    qs[2] = r0 * q2 + r1 * q3 + r2 * q0 - r3 * q1
    qs[3] = r0 * q3 - r1 * q2 + r2 * q1 + r3 * q0

    return qs

def q_inv(q):
    qi = [q[0], -q[1], -q[2], -q[3]]
    return qi
    

def q_slerp(qo, qt, ts):
    # --- slerp code here ---
    q = q_mul(qo, q_exp(ts * q_log(q_mul(q_inv(qo), qt))))
    # q = qo
    return q

def fk(q, pos):
    joint_offsets = offsets[None, :, :, None] # 1, j, 3, 1
    joint_parents = parents
    xz = np.cumsum(pos[:, :, :-1], axis=0)
    print(xz.shape)
    y = pos[:, :, -1:]

    so3 = to_so3(q)

    rot_global= []
    pos = []
    for i, p in enumerate(joint_parents):
        if i==0 and p == -1:
            rot_global.append(so3[:, i])
            pos.append(np.zeros([q.shape[0], 3, 1]))
        else:
            # ------ fk code here ------
            # T_p = np.eye(3)[None, :, :]
            # T = np.eye(3)[None, :, :]
            T_p = rot_global[p]
            T = np.matmul(T_p, so3[:, i])
            
            P_p = pos[p]
            o_c = joint_offsets[:, i]

            rot_global.append(T)
            pos.append(P_p + np.matmul(T_p, o_c))
            # pos.append(P_p + o_c)
            # ------ end ------
            
    gpos = np.stack(pos, axis=1)[..., 0]
    gpos = gpos[:, joints]
    print(gpos.shape)
    gpos[:, :, [0, 2]] += xz
    gpos[:, :, [1]] += y

    return gpos

def qnorm(q):
    return q / np.linalg.norm(q, axis=-1, keepdims=True)

def to_so3(q, ordering='wxyz'):
    quat = qnorm(q)
    assert quat.shape[-1] == 4
    
    if ordering == 'xyzw':
        qx, qy, qz, qw = np.split(quat, 4, axis=-1)
    elif ordering == 'wxyz':
        qw, qx, qy, qz = np.split(quat, 4, axis=-1)
    else:
        raise ValueError(
            "Valid orderings are 'xyzw' and 'wxyz'. Got '{}'.".format(ordering))
    # print(qw[0, 0, 0], qz[0, 0, 0])
    # Form the matrix
    # qw2 = qw * qw
    qx2 = qx * qx
    qy2 = qy * qy
    qz2 = qz * qz

    R00 = 1. - 2. * (qy2 + qz2)
    R01 = 2. * (qx * qy - qw * qz)
    R02 = 2. * (qw * qy + qx * qz)

    R10 = 2. * (qw * qz + qx * qy)
    R11 = 1. - 2. * (qx2 + qz2)
    R12 = 2. * (qy * qz - qw * qx)

    R20 = 2. * (qx * qz - qw * qy)
    R21 = 2. * (qw * qx + qy * qz)
    R22 = 1. - 2. * (qx2 + qy2)

    r0 = np.stack([R00, R01, R02], axis=-1)
    r1 = np.stack([R10, R11, R12], axis=-1)
    r2 = np.stack([R20, R21, R22], axis=-1)
    # print('------------------', np.sum(r0*r1, axis=-1) < 0.00001)
    # print('------------------', np.linalg.norm(r0, axis=-1))
    return np.concatenate([r0, r1, r2], axis=-2)


pos_parents = np.array([-1,0,1,2,3,0,5,6,7,0,9,10,11,11,13,14,15,11,17,18,19])

joints = np.array([
         0,
         2,  3,  4,  5,
         7,  8,  9, 10,
        12, 13, 15, 16,
        18, 19, 20, 22,
        25, 26, 27, 29])


def plot_anim(pos):
    parents = pos_parents
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # map w/ matplot's axis
    x = 2
    y = 0
    z = 1
    lenpos = (np.max(pos[0]) - np.min(pos[0])) * 1.1
    xcentor = pos[0, 0, 0]
    zcentor = pos[0, 0, 2]
    lphalf = lenpos / 2.
    axisidx = {0: 'X', 1: 'Y', 2: 'Z'}
    ax.set_xlim3d([-lphalf+zcentor, lphalf+zcentor])
    ax.set_xlabel(axisidx[x])

    ax.set_ylim3d([-lphalf+xcentor, lphalf+xcentor])
    ax.set_ylabel(axisidx[y])

    ax.set_zlim3d([0, lenpos])
    ax.set_zlabel(axisidx[z])

    sc = ax.scatter(pos[0, :, x], pos[0, :, y], pos[0, :, z])
    lines = []
    for i, p in enumerate(parents):
        if p == -1:
            continue
        line, = ax.plot(pos[0, [i, p], x], pos[0, [i, p], y], pos[0, [i, p], z], c='k')
        lines.append(line)

    def update_scatter(num):
        data=pos[num]
        xcentor = data[0, 0]
        zcentor = data[0, 2]
        ax.set_xlim3d([-lphalf+zcentor, lphalf+zcentor])
        ax.set_ylim3d([-lphalf+xcentor, lphalf+xcentor])
        sc._offsets3d = (data[:, x], data[:, y], data[:, z])
        c = 0
        for i, p in enumerate(parents): 
            if p == -1:
                continue
            lines[c].set_data(data[[i, p], x], data[[i, p], y])
            lines[c].set_3d_properties(data[[i, p], z])
            c += 1

    ani = anim.FuncAnimation(fig, update_scatter, len(pos), interval=int(1000/30), repeat=True)
    plt.show()


if __name__ == "__main__":
    # m.npz 의 경로를 다음과 같이 수정해야 읽혀져서 해당 부분 수정했습니다.
    # 무슨 영문에서인지 './m.npz' 으로 설정해도 읽히지 않습니다..
    # print(os.path.abspath("."))
    # rot = np.load('./m.npz')['rot']
    rot = np.load('univ_lecture/knu23_CG/m.npz')['rot']
    print(rot.shape)

    qo = np.reshape(rot[0][..., :31*4], [31, 4])
    qt = np.reshape(rot[1][..., :31*4], [31, 4])
    xzy = np.zeros([31, 1, 3])
    xzy[:] = rot[0, 31*4:31*4+3] # f, 1, 3
    xzy[:, :, :2] = (0, 0.4)
    
    qs = []
    for i in range(31):
        qs_j = []
        for q0, q1 in zip(qo, qt):
            qs_j.append(q_slerp(q0, q1, i/30))
        qs.append(np.stack(qs_j))
    qs = np.array(qs)
    print(qs.shape)
    pos = fk(qs, xzy)
    plot_anim(pos)