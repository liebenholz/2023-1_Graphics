import colorsys
import matplotlib.pyplot as plt
import numpy as np

# Forward Kinematics : Coordinate Composite Transforamtion
# Moving Lamp with Rigid Transformation.
def draw_line(pT, T, color):
    origin = pT[:-1, 2]
    arm_end = T[:-1, 2]

    plt.plot([origin[0], arm_end[0]] ,[origin[1], arm_end[1]], lw=3, c=color)

def make_T_mat(theta, length):
    mat = np.eye(3)  # identity matrix  
    
    # <<< some code here
    theta = np.radians(theta)
    # Rigid body transformation
    mat[0] = [np.cos(theta), -np.sin(theta), length]
    mat[1] = [np.sin(theta), np.cos(theta), 0]
    #print(mat)
    # ---

    return mat

def draw():
    plt.figure(figsize=(6,6))
    plt.xlim(-0.05, 0.9)
    plt.ylim(-0.05, 0.9)
    lengths = [0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3]
    colors = [colorsys.hsv_to_rgb(i/len(lengths), 1, 1) for i in range(len(lengths))]
    thetas = [40., -20., -40, 30., -120., -120, 0.]

    pT = make_T_mat(30., 0.)

    print(pT)

    for l, c, t in zip(lengths, colors, thetas):
        T = make_T_mat(t, l)
        # Composite Transformation
        T = pT @ T 
        draw_line(pT, T, c)
        pT = T
    
    plt.show()

if __name__ == "__main__":
    draw()