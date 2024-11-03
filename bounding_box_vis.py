import numpy as np
import cv2


edges = [
    [0, 1], [1, 2], [2, 3], [3, 0],  # front face
    [4, 5], [5, 6], [6, 7], [7, 4],  # back face
    [0, 4], [1, 5], [2, 6], [3, 7]   # connecting edges
] 

# Define the 3D coordinates of the cube vertices
vertices = np.array([[100, 100, 100],
                     [280, 100, 100],
                     [280, 200, 100],
                     [100, 200, 100],
                     [100, 100, 200],
                     [280, 100, 200],
                     [280, 200, 200],
                     [100, 200, 200]])

# Define the faces of the cube using the vertices
faces = [
    [0, 1, 2, 3],  # front face
    [4, 5, 6, 7],  # back face
    [0, 1, 5, 4],  # left face
    [2, 3, 7, 6],  # right face
    [0, 3, 7, 4],  # top face
    [1, 2, 6, 5]   # bottom face
]

# Function to create a rotation matrix around the X axis
def rotation_matrix_x(theta):
    return np.array([[1, 0, 0],
                     [0, np.cos(theta), -np.sin(theta)],
                     [0, np.sin(theta), np.cos(theta)]])

# Function to create a rotation matrix around the Y axis
def rotation_matrix_y(theta):
    return np.array([[np.cos(theta), 0, np.sin(theta)],
                     [0, 1, 0],
                     [-np.sin(theta), 0, np.cos(theta)]]) 

# Define a simple projection function
def project(vertex):
    scale = 0.8  # Scale factor for perspective
    x = int(vertex[0] * scale + 250)  # Centering in the image
    y = int(vertex[1] * scale + 100)  # Centering in the image
    return (x, y)

# Define colors for each face
colors = [(255, 0, 0),   # red
          (0, 255, 0),   # green
          (0, 0, 255),   # blue
          (255, 255, 0), # yellow
          (255, 0, 255), # magenta
          (0, 255, 255)] # cyan

colors = [(int(25.5*i),int(25.5*i),int(25.5*i)) for i in range(3,10)]
colors[5] = (180,190,255)

def render_bbox(angle_y, edges, vertices, faces, colors):
    # Create a blank image
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    
    # Rotate the cube by a certain angle (in radians)
    angle_x = np.pi / 12  # Rotate 45 degrees around the X axis
    # angle_y = np.pi / 4  # Rotate 45 degrees around the Y axis

    # Apply the rotation 
    rotation_y = rotation_matrix_y(angle_y)
    rotation_x = rotation_matrix_x(angle_x) 

    center_box = vertices.mean(axis=0)
    # Shifting the box to have center at origin 
    vertices = vertices - center_box 

    # Rotate the vertices
    rotated_vertices = np.dot(vertices, rotation_y)
    rotated_vertices = np.dot(rotated_vertices, rotation_x) 

    # Shifting based on the center box 
    rotated_vertices += center_box

    # Draw and fill the faces of the cube
    for i, face in enumerate(faces):
        pts = np.array([project(rotated_vertices[vertex]) for vertex in face], np.int32)
        cv2.fillPoly(img, [pts], colors[i])

    # Optionally draw the edges of the cube
    for edge in edges:
        start = project(rotated_vertices[edge[0]])
        end = project(rotated_vertices[edge[1]])
        cv2.line(img, start, end, (255, 255, 255), 2)

    # Saving image in .png format 
    alpha_channel = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8) * 255

    # Set alpha channel to 0 for black pixels
    black_pixels = (img[:, :, 0] == 0) & (img[:, :, 1] == 0) & (img[:, :, 2] == 0)
    alpha_channel[black_pixels] = 0

    # Merge the original image with the alpha channel
    rgba_image = cv2.merge((img[:, :, 0], img[:, :, 1], img[:, :, 2], alpha_channel))
    
    return rgba_image


# Rending bounding boxes given the particular input angle
angle_y = (4.1 / 6) * 2 * np.pi  
n_steps = 26

for i in range(0,n_steps):
    shift = (np.pi*2 / n_steps) 
    img = render_bbox(np.pi/18 + i*shift, edges, vertices, faces, colors)
    a,b = 100,100
    img_crp = img[220-a:220+a,400-a:400+a,:]

    cv2.imwrite('./bboxes/img'+str(i)+'.png',img_crp)
print("image saved!")


def draw_azimuth(img_shape, azimuth, color):
    global edges 
    global vertices 
    global faces 
    global colors
    
    colors[5] = color
    img = render_bbox(azimuth, edges, vertices, faces, colors)  
    
    a,b = 100,100
    img_crp = img[220-a:220+a,400-a:400+a,:]

    # resizing the image as per the requirement 
    img_resized = cv2.resize(img_crp, (img_shape, img_shape))
    # Adding text for the angle on top of the image 
    cv2.putText(img_resized, str(azimuth), (100,100), cv2.FONT_HERSHEY_SIMPLEX, 5, (255,0,0), 4)
    print("gen-img-shape: {}".format(img_resized.shape))

    return img_resized