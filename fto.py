# CubingB, copyright 2022 Zach Wegner
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

import random

[U, D, F, B, BL, R, BR, L] = range(8)
CORNERS = [(U, R, F, L), (U, B, BR, R), (R, BR, D, F),
        (F, D, BL, L), (L, BL, B, U), (B, BL, D, BR)]
EDGES = [(U, B), (U, R), (U, L),
        (F, R), (R, BR), (BR, B), (B, BL), (BL, L), (L, F),
        (D, BR), (D, BL), (D, F)]
TRIANGLES = [f for f in range(8) for i in range(3)]
# A list of uniqueified triangle indices so we can easily see how movements
# affect the various otherwise-indistinguishable triangles
MARKED_TRIANGLES = list(range(24))

FACE_STR = ['U', 'D', 'F', 'B', 'BL', 'R', 'BR', 'L']
TURN_STR = ['', '', "'"]

# For each face, these tables show where the various pieces on that face are.
# Corner, corner index, edge, edge index, face triangle, side triangle
up = [1, 0, 4], [0, 0, 3], [0, 1, 2], [0, 0, 0], [0, 1, 2], [10, 9, 16, 15, 22, 21]
down = [2, 5, 3], [2, 2, 1], [9, 10, 11], [0, 0, 0], [3, 4, 5], [12, 14, 8, 7, 20, 19]
front = [0, 2, 3], [2, 3, 0], [3, 11, 8], [0, 1, 1], [6, 7, 8], [15, 17, 5, 4, 23, 22]
back = [1, 4, 5], [1, 2, 0], [0, 6, 5], [1, 0, 1], [9, 10, 11], [19, 18, 1, 0, 13, 12]
back_left = [5, 4, 3], [1, 1, 2], [6, 7, 10], [1, 0, 1], [12, 13, 14], [11, 10, 21, 23, 4, 3]
right = [0, 1, 2], [1, 3, 0], [1, 4, 3], [1, 0, 1], [15, 16, 17], [2, 1, 18, 20, 7, 6]
back_right = [1, 5, 2], [2, 3, 1], [5, 9, 4], [0, 1, 1], [18, 19, 20], [9, 11, 3, 5, 17, 16]
left = [4, 0, 3], [0, 3, 3], [2, 8, 7], [1, 0, 1], [21, 22, 23], [0, 2, 6, 8, 14, 13]

FACES = [up, down, front, back, back_left, right, back_right, left]

class FTO:
    def __init__(self, corners=CORNERS, edges=EDGES, triangles=TRIANGLES):
        self.corners = corners
        self.edges = edges
        self.triangles = triangles

    def move(self, move):
        [self.corners, self.edges, self.triangles] = MOVE_FN[move](self.corners,
                self.edges, self.triangles)
        return self

    def turn(self, face, turn):
        [self.corners, self.edges, self.triangles] = TURN_FN[face][turn](self.corners,
                self.edges, self.triangles)

    def run_alg(self, alg):
        if isinstance(alg, str):
            alg = alg.split()
        for move in alg:
            self.move(move)
        return self

    def copy(self):
        return FTO(self.corners, self.edges, self.triangles)

SOLVED_FTO = FTO()
SOLVED_MARKED_FTO = FTO(triangles=MARKED_TRIANGLES)

def move_str(face, turn):
    return FACE_STR[face] + TURN_STR[turn]

def rotate_right(l, n):
    return (*l[-n:], *l[:-n])

def find_shift(l, i):
    ii = i
    for s in range(len(i)):
        if i in l:
            return (l.index(i), s)
        i = rotate_right(i, 1)
    assert 0, (l, ii)

# Metaprogramming function generator. This is just like the one in solver.py
# for 3x3. This one has one special requirement, though: since the three
# triangle pieces on each face are indistinguishable from one another, we can't
# determine a unique permutation by comparing two FTO objects. So instead we
# perform turns on a 'marked' FTO (as if each triangle piece was marked
# uniquely), so we can see exactly what pieces have moved where after
# performing turns.
def build_transform_fn(name, target, source=SOLVED_MARKED_FTO):
    corner_idxs = []
    for corner in target.corners:
        # Find where this corner is on a solved FTO
        [c, f] = find_shift(source.corners, corner)
        if f == 0:
            corner_idxs.append('c[%s]' % (c))
        else:
            facelets = ['c[%s][%s]' % (c, (i + f) % 4) for i in range(4)]
            corner_idxs.append('(%s)' % ', '.join(facelets))

    edge_idxs = []
    for [i, edge] in enumerate(target.edges):
        # Find where this edge is on a solved FTO
        [e, f] = find_shift(source.edges, edge)
        if f == 0:
            edge_idxs.append('e[%s]' % e)
        else:
            facelets = ['e[%s][%s]' % (e, (i + f) % 2) for i in range(2)]
            edge_idxs.append('(%s)' % ', '.join(facelets))

    triangle_idxs = []
    for [i, triangle] in enumerate(target.triangles):
        # Find where this triangle is on a solved FTO
        t = 't[%s]' % source.triangles.index(triangle)
        triangle_idxs.append(t)

    code = '''
def {name}(c, e, t):
    return (({c}), ({e}), ({t}))'''.format(name=name,
        c=', '.join(corner_idxs), e=', '.join(edge_idxs), t=', '.join(triangle_idxs))
    ctx = {}
    exec(code, ctx)
    return ctx[name]

# Generate functions for face turns, and a couple rotations used for diagrams
TURN_FN = []
MOVE_FN = {}
T_TURN = [L, BR, R, BL, D, U, B, F]
# Table lookup of triangle position after doing a T turn. The corresponding
# triangles need to match exactly, and the ordering is a bit weird, so it's
# just a big hardcoded list
T_TRI_ROT = [23, 21, 22, 19, 20, 18, 15, 16, 17, 13, 14, 12, 3, 4, 5, 2, 0, 1,
        10, 11, 9, 8, 6, 7]
Y2_TURN = [B, F, D, U, R, BL, L, BR]
Y2_TRI_ROT = [Y2_TURN[i] * 3 + j for i in range(8) for j in range(3)]
def gen_turns():
    def add_move(s, fn, face_1=None, face_2=None):
        MOVE_FN[s] = fn

    # Build turn functions for each face
    for face in range(8):
        table = {}
        TURN_FN.append(table)
        for turn in range(1, 3):
            [cidx, cflip, eidx, eflip, ftri, stri] = FACES[face]

            for [i, f] in zip(cidx, cflip):
                assert SOLVED_FTO.corners[i][f] == face
            for [i, f] in zip(eidx, eflip):
                assert SOLVED_FTO.edges[i][f] == face, (i, f, face)
            for f in ftri:
                assert SOLVED_FTO.triangles[f] == face
            for f in stri:
                assert SOLVED_FTO.triangles[f] != face

            # Rotate corners
            corners = list(CORNERS)
            new_corners = rotate_right(cidx, turn)
            new_cflip = rotate_right(cflip, turn)
            for [i, c, f, nf] in zip(cidx, new_corners, cflip, new_cflip):
                corners[i] = rotate_right(CORNERS[c], (f - nf) % 4)

            # Rotate edges
            edges = list(EDGES)
            new_edges = rotate_right(eidx, turn)
            new_flip = rotate_right(eflip, turn)
            for [i, e, f, nf] in zip(eidx, new_edges, eflip, new_flip):
                edges[i] = rotate_right(EDGES[e], (f - nf) % 2) 

            # Rotate triangles
            triangles = list(MARKED_TRIANGLES)
            new_ftri = rotate_right(ftri, turn)
            new_stri = rotate_right(stri, turn*2)
            for [o, n] in zip(ftri, new_ftri):
                triangles[o] = MARKED_TRIANGLES[n]
            for [o, n] in zip(stri, new_stri):
                triangles[o] = MARKED_TRIANGLES[n]

            fto = FTO(corners=corners, edges=edges, triangles=triangles)

            name = 'turn_%s_%s' % (FACE_STR[face], turn)
            fn = build_transform_fn(name, fto)
            table[turn] = fn
            move = move_str(face, turn)
            add_move(move_str(face, turn), fn, face_1=(face, turn))

    for [n, t, r] in [['T', T_TURN, T_TRI_ROT], ['y2', Y2_TURN, Y2_TRI_ROT]]:
        corners = [tuple(t[c] for c in corner) for corner in CORNERS]
        edges = [tuple(t[c] for c in edge) for edge in EDGES]
        triangles = [r[t] for t in MARKED_TRIANGLES]
        fto = FTO(corners=corners, edges=edges, triangles=triangles)

        MOVE_FN[n] = build_transform_fn('t', fto)

    fto = SOLVED_MARKED_FTO.copy()
    fto.move('T').move('T').move('T')
    MOVE_FN["T'"] = build_transform_fn('t_prime', fto)

gen_turns()

def gen_random_move_scramble(length):
    scramble = []
    all_faces = set(range(8))
    blocked_faces = set()
    turns = [1, 2]
    for i in range(length):
        face = random.choice(list(all_faces - blocked_faces))
        # Only allow one turn of each of an opposing pair of faces in a row.
        # E.g. F B' is allowed, F B' F is not
        if face ^ 1 not in blocked_faces:
            blocked_faces = set()
        blocked_faces.add(face)

        turn = random.choice(turns)
        scramble.append(move_str(face, turn))
    return scramble

################################################################################
## SVG rendering stuff #########################################################
################################################################################

CORNER_POS = [(2, -2), (0, 0), (-2, -2)]
EDGE_POS = [(0, -2), (1, -1), (-1, -1)]
TRIANGLE_POS = [(-1, -2), (1, -2), (0, -1)]

COLORS = [
    '#fff',
    '#ff0',
    '#d00',
    '#00f',
    '#f92',
    '#0b0',
    '#777',
    '#707',
]

R2 = 2 ** .5

def gen_fto_diagram(fto):
    def gen_path(x, y, rot, color):
        if not (x ^ y) & 1:
            points = [(x, y), (x-1, y-1), (x+1, y-1)]
        else:
            points = [(x, y-1), (x+1, y), (x-1, y)]
        color = COLORS[color]
        path = 'M%sz' % 'L'.join('%s %s' % xy for xy in points)
        return f'''<path d="{path}" fill="{color}" stroke="black"
            transform="rotate({rot})" stroke-width=".08" />'''

    def gen_side(fto):
        for rot in range(4):
            [cidx, cflip, eidx, eflip, ftri, stri] = FACES[U]
            for [[x, y], i, f] in zip(CORNER_POS, cidx, cflip):
                yield gen_path(x, y, rot*90, fto.corners[i][f])
            for [[x, y], i, f] in zip(EDGE_POS, eidx, eflip):
                yield gen_path(x, y, rot*90, fto.edges[i][f])
            for [[x, y], t] in zip(TRIANGLE_POS, ftri):
                yield gen_path(x, y, rot*90, fto.triangles[t])

            fto.move("T'")

    fto = fto.copy()
    side_1 = list(gen_side(fto))
    side_2 = list(gen_side(fto.move('y2')))

    return f'''<svg viewBox='-4 -4 16 8'>
            {''.join(side_1)}
            <g transform="translate(8, 0)">
                {''.join(side_2)}
            </g>
        </svg>'''
