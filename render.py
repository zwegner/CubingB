# CubingB, copyright 2021 Zach Wegner
#
# This file is part of CubingB.
#
# CubingB is free software: you can redistribute it and/or modify it under the
# terms of the GNU Affero General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# CubingB is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU Affero General Public License for more
# details.
#
# You should have received a copy of the GNU Affero General Public License
# along with CubingB.  If not, see <https://www.gnu.org/licenses/>.

from OpenGL.GL import *
from OpenGL.GLU import *

import solver

# UDRLFB, WYROGB
COLORS = [
    [1, 1, 1],
    [1, 1, 0],
    [.85, 0, 0],
    [1, .6, .1],
    [0, .7, 0],
    [0, 0, 1],
    [.1, .1, .1],
]

BG_COLOR = [.7, .7, .7, 1]

# How much spacing is between cubies? This is a factor--1.03 == 3% of piece size
GAP = 1.03

EDGE_QUADS = [None] * 12
CORNER_QUADS = [None] * 8
CENTER_QUADS = [None] * 6

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

# Translation table: these are the XYZ coordinates of each
# edge/corner/center in the solver.py order
EDGES = [(1, 0, 2), (2, 0, 1), (1, 0, 0), (0, 0, 1),
    (2, 1, 2), (2, 1, 0), (0, 1, 0), (0, 1, 2),
    (1, 2, 2), (2, 2, 1), (1, 2, 0), (0, 2, 1)]
CORNERS = [(2, 0, 2), (2, 0, 0), (0, 0, 0), (0, 0, 2),
    (2, 2, 0), (2, 2, 2), (0, 2, 2), (0, 2, 0)]
CENTERS = [(1, 0, 1), (1, 2, 1), (2, 1, 1), (0, 1, 1),
    (1, 1, 2), (1, 1, 0)]

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

    for x in range(3):
        for y in range(3):
            for z in range(3):
                # Create vertices for this specific cubie
                verts = []
                for [vx, vy, vz] in vertices:
                    vert = [vx + GAP * x, 3 - (vy + GAP * y), vz + GAP * z]
                    verts.append([(c - 1.5) / 3 for c in vert])

                coord = (x, y, z)
                for [table, lookup, result] in [[EDGES, solver.EDGES, EDGE_QUADS],
                        [CORNERS, solver.CORNERS, CORNER_QUADS],
                        [CENTERS, DUMB_CENTERS, CENTER_QUADS]]:
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
    for [cubie_set, quad_set, face_set] in [[cube.edges, EDGE_QUADS, solver.EDGES],
            [cube.corners, CORNER_QUADS, solver.CORNERS],
            [dumb_centers, CENTER_QUADS, DUMB_CENTERS]]:
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
                glColor3f(*color)

                for vertex in quad:
                    glVertex3f(*vertex)

            # Render inside faces: these are at the end of the list
            # after the visible faces
            glColor3f(*COLORS[6])
            for quad in quads[len(cubie):]:
                for vertex in quad:
                    glVertex3f(*vertex)

            glEnd()

            glPopMatrix()

# SVG diagram generation stuff. Not OpenGL like the rest of this
# file, but main.py is bloated so just chuck it in here I guess.

SVG_COLORS = [
    '#fff',
    '#ff0',
    '#d00',
    '#f92',
    '#0b0',
    '#00f',
]

r3 = 3 ** .5
DXS = [
    [r3, 1],
    [r3, 1],
    [r3, -1],
]
DYS = [
    [-r3, 1],
    [0, 2],
    [0, 2],
]
BASE = [
    [0, -6],
    [-3 * r3, -3],
    [0, 0],
]

EDGE_COORDS = {}
CORNER_COORDS = {}
CENTER_COORDS = {}

def gen_svg_tables():
    face_coords = [
        [solver.W, 0, lambda i, j: (i, 0, j)],
        [solver.G, 1, lambda i, j: (i, j, 2)],
        [solver.R, 2, lambda i, j: (2, j, 2-i)],
    ]

    for [color, direction, coord_xform] in face_coords:
        for x in range(3):
            for y in range(3):
                coord = coord_xform(x, y)

                for [table, lookup, result] in [[EDGES, solver.EDGES, EDGE_COORDS],
                        [CORNERS, solver.CORNERS, CORNER_COORDS],
                        [CENTERS, DUMB_CENTERS, CENTER_COORDS]]:
                    if coord in table:
                        index = table.index(coord)
                        for [i, c] in enumerate(lookup[index]):
                            if c == color:
                                dx = DXS[direction]
                                dy = DYS[direction]
                                b = BASE[direction]
                                sx = x * dx[0] + y * dy[0] + b[0]
                                sy = y * dy[1] + x * dx[1] + b[1]

                                path = (f'M {sx} {sy} l {dx[0]} {dx[1]} '
                                        f'l {dy[0]} {dy[1]} l {-dx[0]} {-dx[1]} z')

                                result[(index, i)] = path

def gen_cube_diagram(cube, transform=''):
    # Render cube
    dumb_centers = [[c] for c in cube.centers]

    # Render edges/corners/centers
    cubies = []
    for [cubie_set, paths] in [[cube.edges, EDGE_COORDS],
            [cube.corners, CORNER_COORDS], [dumb_centers, CENTER_COORDS]]:
        for [i, cubie] in enumerate(cubie_set):
            for [j, c] in enumerate(cubie):
                index = (i, j)
                if index in paths:
                    cubies.append(f'''<path d="{paths[index]}"
                        fill="{SVG_COLORS[c]}" stroke="black" stroke-width=".15" />''') 

    f = 1.015
    d = [3*r3 * f, 3 * f]
    dy = 6*f
    sx = 0
    sy = -6*f
    return f'''<svg viewBox='-6.1 -6.1 12.2 12.2' fill='transparent'>
            <clipPath id='clip'>
            <path d="M 0 {sy} l {d[0]} {d[1]} l 0 {dy}
                  l {-d[0]} {d[1]} l {-d[0]} {-d[1]} l 0 {-dy} z"/>
            </clipPath>
            <g clip-path='url(#clip)' transform="{transform}">
              {' '.join(cubies)}
            </g>
        </svg>'''

# Generate static tables

gen_verts()
gen_svg_tables()
