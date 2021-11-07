import pygame 
from OpenGL.GL import *
from OpenGL.GLU import *
from pygame.locals import *

import solver

#WINDOW_SIZE = [2560, 1440]
WINDOW_SIZE = [1600, 1000]

COLORS = [
    [1, 1, 0],
    [1, 1, 1],
    [1, 0, 0],
    [1, .4, 0],
    [0, 0, 1],
    [0, 1, 0],
    [.1, .1, .1],
]

# How much spacing is between cubies? This is a factor--1.03 == 3% of piece size
GAP = 1.03

EDGES = [None] * 12
CORNERS = [None] * 8
CENTERS = [None] * 6

# Copy of solver.CENTERS but in a much listier format
DUMB_CENTERS = [[0], [1], [2], [3], [4], [5]]

# Dumb glue code to convert weilong to solver face numbering
FACE_REMAP = [0, 5, 3, 1, 2, 4]

AXES = [
    [0, -1, 0],
    [1, 0, 0],
    [0, 0, -1],
    [-1, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
]

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
    pygame.init()
    mode = OPENGL | DOUBLEBUF 
    if 0:
        mode |= FULLSCREEN
    pygame.display.set_mode(WINDOW_SIZE, mode)
    glDepthFunc(GL_LEQUAL)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_POLYGON_SMOOTH)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    reset()

# Reset the matrix, basically
def reset():
    glLoadIdentity()
    glMatrixMode(GL_PROJECTION)
    gluPerspective(40, (WINDOW_SIZE[0] / WINDOW_SIZE[1]), 0.1, 50.0)
    glTranslatef(0, 0, -4)
    glRotatef(0, 0, 0, 0)

def render(cube, turns):
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # Render cube
    glMatrixMode(GL_MODELVIEW)

    dumb_centers = [[c] for c in cube.centers]

    # Render edges/corners/centers
    for [cubie_set, quad_set, face_set] in [[cube.edges, EDGES, solver.EDGES],
            [cube.corners, CORNERS, solver.CORNERS],
            [dumb_centers, CENTERS, DUMB_CENTERS]]:
        for [cubie, quads, faces] in zip(cubie_set, quad_set, face_set):

            # See if the face this cubie is on is in a partial rotation
            glPushMatrix()
            for f in faces:
                f = FACE_REMAP[f]
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

    glMatrixMode(GL_PROJECTION)

    pygame.display.flip() 

    # Read events
    reset = False
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            reset = True
    return reset

gen_verts()
