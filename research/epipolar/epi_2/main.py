import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

exec(open('load_vggdata.py').read())

# make 3D points homogeneous and project
X = np.vstack( (points3D, np.ones(points3D.shape[1])) )
x = P[0].project(X)
# plotting the points in view 1
# plt.figure()
# plt.imshow(im1)
# plt.plot(points2D[0][0], points2D[0][1], '*')
# plt.axis('off')
# plt.figure()
# plt.imshow(im1)
# plt.plot(x[0],x[1],'r.')
# plt.axis('off')
# plt.show()

# Plotting a sample 3D plot
# =========================
# fig = plt.figure()
# ax = fig.gca(projection="3d")
# # generate 3D sample data
# X,Y,Z = axes3d.get_test_data(0.25)
# # plot the points in 3D
# ax.plot(X.flatten(),Y.flatten(),Z.flatten(),'o')
# plt.show()
# =========================

# And now, the real data
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot(points3D[0],points3D[1],points3D[2], 'k.')
# plt.show()

# Drawing the epipoles
# =========================
import sfm
# index for points in first two views
ndx = (corr[:,0]>=0) & (corr[:,1]>=0)
# get coordinates and make homogeneous
x1 = points2D[0][:,corr[ndx,0]]
x1 = np.vstack( (x1,np.ones(x1.shape[1])) )
x2 = points2D[1][:,corr[ndx,1]]
x2 = np.vstack( (x2,np.ones(x2.shape[1])) )

# compute F
F = sfm.compute_fundamental(x1,x2)
# compute the epipole
e = sfm.compute_epipole(F)
# plotting
plt.figure()

plt.imshow(im1)

# plot each line individually, this gives nice colors
for i in range(5):
    sfm.plot_epipolar_line(im1,F,x2[:,i],e,False)
plt.axis('off')
plt.figure()
plt.imshow(im2)

# plot each point individually, this gives same colors as the lines
for i in range(5):
    plt.plot(x2[0,i],x2[1,i],'o')
plt.axis('off')
plt.show()
# END - Drawing the epipoles
# =========================

# compute F
F = sfm.compute_fundamental(x2,x1)
# compute the epipole
e = sfm.compute_epipole(F)
# plotting
plt.figure()

plt.imshow(im2)

# plot each line individually, this gives nice colors
for i in range(5):
    sfm.plot_epipolar_line(im2,F,x1[:,i],e,False)
plt.axis('off')
plt.figure()
plt.imshow(im1)

# plot each point individually, this gives same colors as the lines
for i in range(5):
    plt.plot(x1[0,i],x1[1,i],'o')
plt.axis('off')
plt.show()
# END - Drawing the epipoles
# =========================
