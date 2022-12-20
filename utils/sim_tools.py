from fipy import Gmsh2D
import numpy as np

def create_circular_mesh(radius,cellSize):
    """
    Function creates circular 2D mesh

    **Input**

    -   radius   =   Radius of mesh
    -   cellSize    =   Size of unit cell

    *Note* : No support for 3D meshes currently and **requires GMSH**
    """

    mesh = Gmsh2D('''
                     cellSize = %g;
                     radius = %g;
                     Point(1) = {0, 0, 0, cellSize};
                     Point(2) = {-radius, 0, 0, cellSize};
                     Point(3) = {0, radius, 0, cellSize};
                     Point(4) = {radius, 0, 0, cellSize};
                     Point(5) = {0, -radius, 0, cellSize};
                     Circle(6) = {2, 1, 3};


                    Circle(7) = {3, 1, 4};
                    Circle(8) = {4, 1, 5};
                    Circle(9) = {5, 1, 2};
                    Line Loop(10) = {6, 7, 8, 9};
                    Plane Surface(11) = {10};

       '''%(cellSize,radius)) # doctest: +GMSH


    return(mesh)

def nucleate_seed(mesh,phi_a,phia_value,nucleus_size=5.0,dimension=2, location=(0,0)):
    """
    Function nucleates spherical nucleus of condensate into mesh

    **Input**

    -   phi_a   =   Phase-field variable
    -   mesh    =   Mesh variable
    -   phia_value  =   Value of dense phase to nucleate
    -   nucleus_size   =   Radius of initial nucleus
    -   dimension   =   Dimension of grid
    """
    a=(mesh.cellCenters)

    xc = (min(mesh.x) + max(mesh.x))*0.5
    yc = (min(mesh.y) + max(mesh.y))*0.5
    
    # modify xc and yc with location coords
    xc += location[0]
    yc += location[1]
    
    if dimension==3:
        zc = (min(mesh.z) + max(mesh.z))*0.5;

    for i in np.arange(a.shape[1]):
        if dimension==2:
            dist = np.sqrt((a.value[0][i]-xc)**2 + (a.value[1][i]-yc)**2)
        elif dimension==3:
            dist = np.sqrt((a.value[0][i]-xc)**2 + (a.value[1][i]-yc)**2 + (a.value[2][i]-zc)**2)

        if (dist<=nucleus_size):
            phi_a.value[i] = phia_value
