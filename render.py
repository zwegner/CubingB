from OpenGL.GL import *
from OpenGL.GLU import *

import solver

# UDRLFB, WYROGB
COLORS = [
    [1, 1, 1],
    [1, 1, 0],
    [1, 0, 0],
    [1, .4, 0],
    [0, 1, 0],
    [0, 0, 1],
    [.1, .1, .1],
]

BG_COLOR = [.7, .7, .7, 1]

# How much spacing is between cubies? This is a factor--1.03 == 3% of piece size
GAP = 1.03

EDGES = [None] * 12
CORNERS = [None] * 8
CENTERS = [None] * 6

# Copy of solver.CENTERS but in a much listier format
DUMB_CENTERS = [[0], [1], [2], [3], [4], [5]]

# Normal vector of each face for partial rotations
AXES = [
    [0, -1, 0],
    [0, 1, 0],
    [-1, 0, 0],
    [1, 0, 0],
    [0, 0, -1],
    [0, 0, 1],
]

FONT = None

def gen_verts():
    # Vertex tables for a single cubie
    vertices = [
        [0, 0, 0],
        [1, 0, 0],
        [1, 0, 1],
        [0, 0, 1],
        [0, 1, 0],
        [1, 1, 0],
        [1, 1, 1],
        [0, 1, 1],
    ]

    faces = [
        [0, 1, 2, 3],
        [7, 6, 5, 4],
        [2, 1, 5, 6],
        [0, 3, 7, 4],
        [3, 2, 6, 7],
        [1, 0, 4, 5],
    ]

    # Translation table: these are the XYZ coordinates of each
    # edge/corner/center in the solver.py order
    edges = [[1, 0, 2], [2, 0, 1], [1, 0, 0], [0, 0, 1],
        [2, 1, 2], [2, 1, 0], [0, 1, 0], [0, 1, 2],
        [1, 2, 2], [2, 2, 1], [1, 2, 0], [0, 2, 1]]
    corners = [[2, 0, 2], [2, 0, 0], [0, 0, 0], [0, 0, 2],
        [2, 2, 0], [2, 2, 2], [0, 2, 2], [0, 2, 0]]
    centers = [[1, 0, 1], [1, 2, 1], [2, 1, 1], [0, 1, 1],
        [1, 1, 2], [1, 1, 0]]

    for x in range(3):
        for y in range(3):
            for z in range(3):
                # Create vertices for this specific cubie
                verts = []
                for [vx, vy, vz] in vertices:
                    vert = [vx + GAP * x, 3 - (vy + GAP * y), vz + GAP * z]
                    verts.append([(c - 1.5) / 3 for c in vert])

                coord = [x, y, z]
                for [table, lookup, result] in [[edges, solver.EDGES, EDGES],
                        [corners, solver.CORNERS, CORNERS],
                        [centers, DUMB_CENTERS, CENTERS]]:
                    # Look up this cubie in the translation table, and use
                    # that to place the vertices in the proper list, with
                    # the faces matching up to the same color ordering as
                    # solver.py
                    if coord in table:
                        index = table.index(coord)
                        piece = lookup[index]
                        cubie = []
                        # Create outside faces
                        for f in piece:
                            cubie.append([verts[v] for v in faces[f]])
                        # Append inside faces
                        for f in range(6):
                            if f not in piece:
                                cubie.append([verts[v] for v in faces[f]])
                        result[index] = cubie
                        break

# Set up window and some rendering state
def setup():
    glDepthFunc(GL_LEQUAL)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_POLYGON_SMOOTH)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

def set_persective(window_size, zoom):
    # Set up view
    [w, h] = window_size
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glViewport(0, 0, w, h)
    gluPerspective(40, (w / h), 0.1, 50.0)
    glTranslatef(0, 0, zoom)

    # Switch back to the model matrix stack for rendering
    glMatrixMode(GL_MODELVIEW)

def set_ortho(window_size, ax, ay):
    # Set up view
    [w, h] = window_size
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glViewport(0, 0, w, h)
    glOrtho(-1, 1, -1, 1, -1, 1)

    # Switch back to the model matrix stack for rendering
    glMatrixMode(GL_MODELVIEW)

    glLoadIdentity()

    glRotatef(ax, 1, 0, 0)
    glRotatef(ay, 0, -1, 0)

def reset():
    glClearColor(*BG_COLOR)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

def set_rotation(matrix):
    glLoadIdentity()
    glMultTransposeMatrixf(matrix)

def rotate_camera(matrix):
    glMatrixMode(GL_PROJECTION)
    glMultTransposeMatrixf(matrix)
    glMatrixMode(GL_MODELVIEW)

def render_cube(cube, turns):
    # Render cube
    dumb_centers = [[c] for c in cube.centers]

    # Render edges/corners/centers
    for [cubie_set, quad_set, face_set] in [[cube.edges, EDGES, solver.EDGES],
            [cube.corners, CORNERS, solver.CORNERS],
            [dumb_centers, CENTERS, DUMB_CENTERS]]:
        for [cubie, quads, faces] in zip(cubie_set, quad_set, face_set):
            # See if the face this cubie is on is in a partial rotation
            glPushMatrix()
            for f in faces:
                if turns[f]:
                    # turns are in 1/9th of a quarter turn, so 10 degrees
                    glRotatef(turns[f] * 10, *AXES[f])
                    break

            glBegin(GL_QUADS)

            # Render all the outside faces of this cubie
            for [color, quad] in zip(cubie, quads):
                color = COLORS[color]
                glColor3fv(color)

                for vertex in quad:
                    glVertex3fv(vertex)

            # Render inside faces: these are at the end of the list
            # after the visible faces
            glColor3fv(COLORS[6])
            for quad in quads[len(cubie):]:
                for vertex in quad:
                    glVertex3fv(vertex)

            glEnd()

            glPopMatrix()

gen_verts()
