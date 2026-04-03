import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Circle parameters
radius = 8.4127
center = (0, 0)

box_size = 30
box_bottom_left = (-box_size/2, -box_size/2)

fig, ax = plt.subplots()

circle = patches.Circle(center, radius, edgecolor='blue', facecolor='none', linewidth=2, label='Circle (r=16.83)')
ax.add_patch(circle)

rectangle = patches.Rectangle(box_bottom_left, box_size, box_size, edgecolor='red', facecolor='none', linewidth=2, label='Box (d=30)')
ax.add_patch(rectangle)

# Set equal aspect ratio to ensure the circle looks like a circle
ax.set_aspect('equal')

# Set limits slightly larger than the shapes
limit = max(radius, box_size/2) + 2
ax.set_xlim(-limit, limit)
ax.set_ylim(-limit, limit)

# Add title, legend, and show
plt.title(f'cell r={radius}); mech voxel size ({box_size})')
#plt.legend()
#plt.grid(True, linestyle='--')
plt.show()
