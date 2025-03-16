import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

#LOAD IMAGE
I = plt.imread('mandril.jpg')

#DISPLAY
# plt.figure(1)
# plt.imshow(I)
# plt.title('mandril')
# plt.axis('off')

# #STORE IMG
# plt.imsave('mandril.png', I)


# #DISPLAY POINTS
# x = [100, 150, 200, 250]
# y = [50, 100, 150, 200]
# plt.plot(x, y, 'r.', markersize=10)
# plt.show()

# #DISPLAY SHAPE
fig, ax = plt.subplots(1)

# CREATE RECTANGLE (x, y, width, height)
rect = Rectangle((50, 50), 50, 100, fill=True, linewidth=5, edgecolor='r',)
ax.add_patch(rect)
ax.set_xlim(0, 150)
ax.set_ylim(0, 150)
plt.show()