import torch
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def generate_pts_on_line(f, n_points, x_range=(0,1)):
    """
    Generates random points on a line defined by the function f.
    Parameters:
    - f: Function defining the line (e.g., lambda x: 2*x + 1)
    - n_points: Number of points to sample
    - x_range: Tuple (min_x, max_x) defining the range of x-values to sample
    Returns:
    - torch.Tensor of shape (n_points, 2) containing the sampled points
    """
    x_values = torch.rand(n_points) * (x_range[1] - x_range[0]) + x_range[0]
    y_values = f(x_values)
    points = torch.stack((x_values, y_values), dim=1)
    return points

def generate_stacked_points(num_pts,num_splits,x_range =(0,1)):
    ranges = []
    for i in range(num_splits+1):
        ranges.append(i*(x_range[1] - x_range[0])/num_splits)    
    stacked_tensors = [
        generate_pts_on_line(
            lambda x: x * 0, num_pts, x_range=(ranges[i],ranges[i+1])
        )[:, 0].to(device)
        for i in range(num_splits)
    ]
    return torch.stack(stacked_tensors,dim = 1).reshape(-1,1)

def generate_points_on_ellipsoid(n_points):
        points = []
        a, b, c = np.sqrt(1.69 / 2), np.sqrt(1.69 / 3), np.sqrt(1.69 / 6)
    
        while len(points) < n_points:
            u = np.random.uniform(0, 2 * np.pi)
            v = np.random.uniform(0, np.pi)
            
            x = a * np.cos(u) * np.sin(v)
            y = b * np.sin(u) * np.sin(v)
            z = c * np.cos(v)
            
            if (2*x**2 + 3*y**2 + 6*z**2 == 1.69):
                points.append([x, y, z])
        
        return torch.tensor(np.array(points).astype('float32'))
    
def generate_random_points_on_cube_surface(n_points,face):
    """
    Generates random points on the face of the cube [-1,1]^3
    n_points number of points to generate
    face (list) : [x,y,z] e.g. [0,0,-1] means face z = -1
    """
    points = []
    if face == [1,0,0]:  # x = 1 face
        for _ in range(n_points):
            x = 1
            y = np.random.uniform(-1, 1)
            z = np.random.uniform(-1, 1)
        points.append([x, y, z])
    elif face == [-1,0,0]:  # x = -1 face
        for _ in range(n_points):
            x = -1
            y = np.random.uniform(-1, 1)
            z = np.random.uniform(-1, 1)
        points.append([x, y, z])
    elif face == [0,1,0]:  # y = 1 face
        for _ in range(n_points):
            x = np.random.uniform(-1, 1)
            y = 1
            z = np.random.uniform(-1, 1)
        points.append([x, y, z])
    elif face == [0,-1,0]:  # y = -1 face
        for _ in range(n_points):
            x = np.random.uniform(-1, 1)
            y = -1
            z = np.random.uniform(-1, 1)
        points.append([x, y, z])
    elif face == [0,0,1]:  # z = 1 face
        for _ in range(n_points):
            x = np.random.uniform(-1, 1)
            y = np.random.uniform(-1, 1)
            z = 1
        points.append([x, y, z])
    elif face == [0,0,-1]:  # z = -1 face
        for _ in range(n_points):
            x = np.random.uniform(-1, 1)
            y = np.random.uniform(-1, 1)
            z = -1
        points.append([x, y, z])
        
    return torch.tensor(np.array(points).astype('float32'))    

    