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

SVG_COLORS = {
    'w': '#fff',
    'y': '#ff0',
    'r': '#d00',
    'o': '#f92',
    'g': '#0b0',
    'b': '#00f',
    '-': '#777',
}

COLOR_MAP = 'wyrogb'

EDGE_COORDS = {}
CORNER_COORDS = {}
CENTER_COORDS = {}
COORDS = [EDGE_COORDS, CORNER_COORDS, CENTER_COORDS]
ALL_COORDS = []

LL_EDGE_COORDS = {}
LL_CORNER_COORDS = {}
LL_CENTER_COORDS = {}
LL_COORDS = [LL_EDGE_COORDS, LL_CORNER_COORDS, LL_CENTER_COORDS]

def gen_svg_tables():
    face_coords = [
        [solver.W, 0, lambda i, j: (i, 0, j)],
        [solver.G, 1, lambda i, j: (i, j, 2)],
        [solver.R, 2, lambda i, j: (2, j, 2-i)],
    ]

    r3 = 3 ** .5
    dxs = [
        [r3, 1],
        [r3, 1],
        [r3, -1],
    ]
    dys = [
        [-r3, 1],
        [0, 2],
        [0, 2],
    ]
    base = [
        [0, -6],
        [-3 * r3, -3],
        [0, 0],
    ]

    # Map cube indices to SVG coordinates
    for [color, direction, coord_xform] in face_coords:
        for y in range(3):
            for x in range(3):
                coord = coord_xform(x, y)

                for [table, lookup, result] in [[EDGES, solver.EDGES, EDGE_COORDS],
                        [CORNERS, solver.CORNERS, CORNER_COORDS],
                        [CENTERS, DUMB_CENTERS, CENTER_COORDS]]:
                    if coord in table:
                        index = table.index(coord)
                        for [i, c] in enumerate(lookup[index]):
                            if c == color:
                                dx = dxs[direction]
                                dy = dys[direction]
                                b = base[direction]
                                sx = x * dx[0] + y * dy[0] + b[0]
                                sy = y * dy[1] + x * dx[1] + b[1]

                                path = (f'M {sx} {sy} l {dx[0]} {dx[1]} '
                                        f'l {dy[0]} {dy[1]} l {-dx[0]} {-dx[1]} z')

                                result[(index, i)] = path
                                ALL_COORDS.append(path)

    # Map again but only for LL pieces
    for x in range(3):
        for z in range(3):
            coord = (x, 0, z)
            for [table, lookup, result] in [[EDGES, solver.EDGES, LL_EDGE_COORDS],
                    [CORNERS, solver.CORNERS, LL_CORNER_COORDS],
                    [CENTERS, DUMB_CENTERS, LL_CENTER_COORDS]]:
                if coord in table:
                    index = table.index(coord)
                    for [i, c] in enumerate(lookup[index]):
                        if c == solver.W:
                            d = 3
                            sx = (x - 1.5) * d
                            sz = (z - 1.5) * d
                            path = (f'M {sx} {sz} l {0} {d} '
                                    f'l {d} {0} l {0} {-d} z')
                        else:
                            d = 3
                            sx = (x - 1.5) * d
                            sz = (z - 1.5) * d
                            if c == solver.G or c == solver.B:
                                dx = d
                                dz = d / 2
                                sz += d if c == solver.G else -dz
                            else:
                                dx = d / 2
                                dz = d
                                sx += d if c == solver.R else -dx
                            path = (f'M {sx} {sz} l {0} {dz} '
                                    f'l {dx} {0} l {0} {-dz} z')

                        result[(index, i)] = path

def gen_cube_diagram(cube, transform='', type='normal'):
    result = []
    # Collect color/coordinate pairs for a regular cube
    if isinstance(cube, solver.Cube):
        coords = LL_COORDS if type in {'pll', 'oll'} else COORDS

        dumb_centers = [[c] for c in cube.centers]

        for [cubie_set, paths] in zip([cube.edges, cube.corners,
                dumb_centers], coords):
            for [i, cubie] in enumerate(cubie_set):
                for [j, c] in enumerate(cubie):
                    index = (i, j)
                    if index in paths:
                        if type == 'oll' and c != solver.Y:
                            c = '-'
                        else:
                            c = COLOR_MAP[c]

                        color = SVG_COLORS[c]
                        result.append((color, paths[index]))
    # Collect color/coordinate pairs for a string diagram
    else:
        if type in {'pll', 'oll'}:
            coords = LL_COORDS 

            # Ugh this code sucks ass
            n = 0
            for [[x, y], paths] in zip([[12, 2], [8, 3], [6, 1]], coords):
                for i in range(x):
                    for j in range(y):
                        index = (i, j)
                        if index in paths:
                            result.append((SVG_COLORS[cube[n]], paths[index]))
                            n += 1
        else:
            for [c, path] in zip(cube, ALL_COORDS):
                result.append((SVG_COLORS[c], path))


    # Render SVG
    cubies = [f'''<path d="{path}"
            fill="{color}" stroke="black" stroke-width=".15" />'''
            for [color, path] in result]

    return f'''<svg viewBox='-6.1 -6.1 12.2 12.2'>
            <g transform="{transform}">
              {' '.join(cubies)}
            </g>
        </svg>'''

# Generate static tables

gen_verts()
gen_svg_tables()
