import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import animation

## LOAD IN DATA ##############################################################

room_corners = np.loadtxt('room_corners.mat')
head = np.loadtxt('head.mat')
x_trans = np.loadtxt('x_trans.mat')
y_trans = np.loadtxt('y_trans.mat')
z_trans = np.loadtxt('z_trans.mat')
theta = np.loadtxt('theta.mat')
phi = np.loadtxt('phi.mat')
alpha = np.loadtxt('alpha.mat')

## TRANSFORMATIONS ############################################################

def translational(initial_coordinates, delta_x, delta_y, delta_z):
    
    trans = np.array(([1,0,0,delta_x],[0,1,0,delta_y],[0,0,1,delta_z],[0,0,0,1])) # translational matrix
    new_coordinates = np.matmul(trans,initial_coordinates) # values of new x,y,z
    
    return new_coordinates

def rotational_x(initial_coordinates, angle):
    
    rot = np.array(([1,0,0,0],[0,np.cos(angle),-np.sin(angle),0],[0,np.sin(angle),np.cos(angle),0],[0,0,0,1])) 
    new_coordinates = np.matmul(rot,initial_coordinates) # values of new x,y,z
    
    return new_coordinates

def rotational_y(initial_coordinates, angle):
    
    rot = np.array(([np.cos(angle),0,np.sin(angle),0],[0,1,0,0],[-np.sin(angle),0,np.cos(angle),0],[0,0,0,1])) 
    new_coordinates = np.matmul(rot,initial_coordinates) # values of new x,y,z
    
    return new_coordinates

def rotational_z(initial_coordinates, angle):
    
    rot = np.array(([np.cos(angle),-np.sin(angle),0,0],[np.sin(angle),np.cos(angle),0,0],[0,0,1,0],[0,0,0,1])) 
    new_coordinates = np.matmul(rot,initial_coordinates) # values of new x,y,z
    
    return new_coordinates

## APPLYING TRANSFORMATIONS ##################################################
    
CoM = np.append(np.mean(head,0),1) # centre of mass for initial coordinates
time = 7200 # total running time
initial_coordinates = np.reshape(head,(50,3,1)) # initial head coords reshaped
one_array = np.ones((len(head),1,1)) # make suitable for the matrix multiplication
initial_coordinates = np.concatenate((initial_coordinates,one_array),1)

full_coordinates = np.zeros((len(head),4,len(x_trans)))

for i in range(time):
    
    coordinates = initial_coordinates.copy()  
    coordinates[:,0,:] -= CoM[0] # go to CoM reference frame (x)
    coordinates[:,1,:] -= CoM[1] # go to CoM reference frame (y)
    coordinates[:,2,:] -= CoM[2] # go to CoM reference frame (y)
    
    coordinates = translational(coordinates, x_trans[i],y_trans[i],z_trans[i])
    coordinates = rotational_x(coordinates, theta[i])
    coordinates = rotational_y(coordinates, phi[i])
    coordinates = rotational_z(coordinates, alpha[i])
    
    coordinates[:,0,:] += CoM[0] # go to original reference frame (x)
    coordinates[:,1,:] += CoM[1] # go to original reference frame (y)
    coordinates[:,2,:] += CoM[2] # go to original reference frame (z)
    
    CoM = np.mean(coordinates,0) # update new CoM
    
    full_coordinates[:,:,i] = coordinates[:,:,0]
    
full_coordinates = np.delete(full_coordinates,3,1) # remove redundant data

## PLOTTING ##################################################################

fig = plt.figure()
ax = p3.Axes3D(fig)
ax.scatter(room_corners[:,0],room_corners[:,1],room_corners[:,2], c = 'r', marker = 's')

# create the first plot
point, = ax.plot(full_coordinates[:,0,0], full_coordinates[:,1,0], full_coordinates[:,2,0])


ax.set_xlim([-0.8, 0.8])
ax.set_ylim([-0.8, 0.8])
ax.set_zlim([-0.75, 0.75])

step_size = 10

def update_point(n):
    point.set_data(np.array([full_coordinates[:,0,step_size*n], full_coordinates[:,1,step_size*n]]))
    point.set_3d_properties(full_coordinates[:,2,step_size*n], 'z')
    return point


ani=animation.FuncAnimation(fig, update_point, len(x_trans),interval=8.333333)

plt.show()

# writer_format = animation.writers['ffmpeg']
# writer = writer_format(fps=120, metadata=dict(artist='Me'), bitrate=800)
# ani.save('head_movement.mp4', writer=writer)

