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

import array
import os
import random

################################################################################
## Cube logic ##################################################################
################################################################################

[W, Y, R, O, G, B] = range(6)

EDGES = ((W, G), (W, R), (W, B), (W, O),
    (G, R), (R, B), (B, O), (O, G),
    (G, Y), (R, Y), (B, Y), (O, Y))
CORNERS = ((W, G, R), (W, R, B), (W, B, O), (W, O, G),
    (Y, B, R), (Y, R, G), (Y, G, O), (Y, O, B))
CENTERS = (W, Y, R, O, G, B)

# Face tables: for each of the six faces, this table has the coordinates of the
# four edges and four corners that make up the face, as they correspond to the
# lists above of edges and corners in a solved cube. For each face, the four
# lists below are edge position, index in edge, corner position, index in corner 
up    = [2, 1,  0,  3], [0, 0, 0, 0], [2, 1, 0, 3], [0, 0, 0, 0]
down  = [8, 9, 10, 11], [1, 1, 1, 1], [6, 5, 4, 7], [0, 0, 0, 0]
right = [1, 5,  9,  4], [1, 0, 0, 1], [0, 1, 4, 5], [2, 1, 2, 1]
left  = [3, 7, 11,  6], [1, 0, 0, 1], [2, 3, 6, 7], [2, 1, 2, 1]
front = [0, 4,  8,  7], [1, 0, 0, 1], [3, 0, 5, 6], [2, 1, 2, 1]
back  = [2, 6, 10,  5], [1, 0, 0, 1], [1, 2, 7, 4], [2, 1, 2, 1]

faces = [up, down, right, left, front, back]

RY = [0, 1, 4, 5, 3, 2]
RX = [5, 4, 2, 3, 0, 1]
RZ = [2, 3, 1, 0, 4, 5]
ROTATE_FACES = [RY, RX, RZ]

# M and E moves are dumb and have backwards notation
SLICE_FLIP = [0, 2, 0, 2, 2, 0]
SLICE_ROT_FLIP = [1, 1, 0]

COLOR_STR = ['white', 'yellow', 'red', 'orange', 'green', 'blue']
SLICE_STR = 'EMS'
FACE_STR = 'UDRLFB'
INV_FACE_STR = {f: i for [i, f] in enumerate(FACE_STR)}
ROTATE_STR = 'yxz'
TURN_STR = {-1: "'", 1: '', 2: '2', 3: "'"}
INV_TURN_STR = {v: k for [k, v] in TURN_STR.items()}
assert INV_TURN_STR["'"] == 3

class Cube:
    def __init__(self, centers=CENTERS, edges=EDGES, corners=CORNERS):
        self.centers = centers
        self.edges = edges
        self.corners = corners

    def turn(self, face, n):
        [self.edges, self.corners] = TURNS[face][n](self.edges, self.corners)

    def rotate(self, rot, n):
        [self.centers, self.edges, self.corners] = ROTATES[rot][n](self.centers,
                self.edges, self.corners)

    def run_alg(self, alg):
        if isinstance(alg, (str, list)):
            alg = parse_alg(alg)
        for [rot, rn, r, n, r2, n2] in alg:
            if rot is not None:
                self.rotate(rot, rn)
            if r is not None:
                self.turn(r, n)
            if r2 is not None:
                self.turn(r2, n2)
        return self

    def __eq__(self, other):
        return (self.centers == other.centers and self.edges == other.edges and
                self.corners == other.corners)

    # To make a copy, just return a new cube object with the same attributes,
    # since they're all immutable
    def copy(self):
        return Cube(self.centers, self.edges, self.corners)

def rotate(l, n):
    return (*l[-n:], *l[:-n])

def find_shift(l, i):
    ii = i
    for s in range(len(i)):
        if i in l:
            return (l.index(i), s)
        i = rotate(i, 1)
    assert 0, (l, ii)

# Metabrogramming. Generate function for each of the turn and rotate moves.
# This is pretty messy code, just the first random crap I hacked up that worked
TURNS = []
ROTATES = []
def gen_turns():
    for F in range(6):
        FR = {}
        TURNS.append(FR)
        face = faces[F]
        for n in range(1, 4):
            E = list(range(1, 13))
            C = [(x, 0) for x in range(8)]
            [idx, flip, cidx, cflip] = face
            edges = [E[i] for i in idx]
            new_edges = edges[n:] + edges[:n]
            new_flip = flip[n:] + flip[:n]
            for [i, e, f, nf] in zip(idx, new_edges, flip, new_flip):
                if f != nf:
                    e = -e
                E[i] = e

            corners = [C[i] for i in cidx]
            new_corners = corners[n:] + corners[:n]
            new_cflip = cflip[n:] + cflip[:n]
            for [i, [c, _], f, nf] in zip(cidx, new_corners, cflip, new_cflip):
                df = 0
                C[i] = (c, (nf-f)%3)

            idxs = []
            for e in E:
                if e > 0:
                    idxs.append('e[%s]' % (e-1))
                else:
                    e = -e - 1
                    idxs.append('(e[%s][1], e[%s][0])' % (e, e))

            cidxs = []
            for [c, f] in C:
                if f == 0:
                    cidxs.append('c[%s]' % (c))
                else:
                    [x, y, z] = [(i + f) % 3 for i in range(3)]
                    cidxs.append('(c[%s][%s], c[%s][%s], c[%s][%s])' % (c, x, c, y, c, z))

            name = 'turn_%s_%s' % (FACE_STR[F], n)
            code = '''
def {name}(e, c):
    return (({i}),
        ({c}))'''.format(name=name, i=', '.join(idxs), c=', '.join(cidxs))
            ctx = {}
            exec(code, ctx)
            FR[4 - n] = ctx[name]

    for [r, rotation] in enumerate(ROTATE_FACES):
        FR = {}
        ROTATES.append(FR)
        for n in range(1, 4):
            rot = list(range(6))
            for i in range(n):
                rot = [rot[x] for x in rotation]
            E = []
            C = []
            for [a, b] in EDGES:
                edge = (rot[a], rot[b])
                E.append(find_shift(EDGES, edge))
            for [a, b, c] in CORNERS:
                corner = (rot[a], rot[b], rot[c])
                C.append(find_shift(CORNERS, corner))

            cnidxs = ['cn[%s]' % f for f in rot]

            idxs = []
            for [e, f] in E:
                if f == 0:
                    idxs.append('e[%s]' % (e))
                else:
                    idxs.append('(e[%s][1], e[%s][0])' % (e, e))

            cidxs = []
            for [c, f] in C:
                if f == 0:
                    cidxs.append('c[%s]' % (c))
                else:
                    [x, y, z] = [(i + f) % 3 for i in range(3)]
                    cidxs.append('(c[%s][%s], c[%s][%s], c[%s][%s])' % (c, x, c, y, c, z))

            name = 'rotate_%s_%s' % (ROTATE_STR[r], n)
            code = '''
def {name}(cn, e, c):
    return (({cn}),
        ({i}),
        ({c}))'''.format(name=name, cn=', '.join(cnidxs), i=', '.join(idxs), c=', '.join(cidxs))
            ctx = {}
            exec(code, ctx)
            FR[4 - n] = ctx[name]

gen_turns()

################################################################################
## CFOP logic helpers ##########################################################
################################################################################

SOLVED_CUBE = Cube()

# Create CFOP tables the lazy way

CROSS_EDGES = {c: [] for c in range(6)}
for color in range(6):
    for i in range(12):
        for j in range(2):
            if SOLVED_CUBE.edges[i][j] == color:
                CROSS_EDGES[color].append((i, j))

F2L_PAIRS = {c: [] for c in range(6)}
for color in range(6):
    for [i, corner] in enumerate(SOLVED_CUBE.corners):
        if color in corner:
            match = tuple(c for c in corner if c != color)
            xm = match[::-1]
            if match in SOLVED_CUBE.edges:
                j = SOLVED_CUBE.edges.index(match)
            else:
                j = SOLVED_CUBE.edges.index(xm)
            F2L_PAIRS[color].append((i, j))

OLL_CORNERS = {c: [] for c in range(6)}
OLL_EDGES = {c: [] for c in range(6)}
for color in range(6):
    for i in range(8):
        for j in range(3):
            if SOLVED_CUBE.corners[i][j] == color ^ 1:
                OLL_CORNERS[color].append((i, j))
    for i in range(12):
        for j in range(2):
            if SOLVED_CUBE.edges[i][j] == color ^ 1:
                OLL_EDGES[color].append((i, j))

def is_cross_solved(cube, cross_color):
    return all(cube.edges[i][j] == SOLVED_CUBE.edges[i][j]
            for [i, j] in CROSS_EDGES[cross_color])

def is_f2l_solved(cube, cross_color):
    return all(cube.corners[c] == SOLVED_CUBE.corners[c] and
                            cube.edges[e] == SOLVED_CUBE.edges[e]
                for [c, e] in F2L_PAIRS[cross_color])

def is_oll_solved(cube, cross_color):
    ll = cross_color ^ 1
    return (all(cube.corners[i][j] == ll for [i, j] in OLL_CORNERS[cross_color]) and
            all(cube.edges[i][j] == ll for [i, j] in OLL_EDGES[cross_color]))

################################################################################
## Utilities ###################################################################
################################################################################

def move_str(face, turn):
    return FACE_STR[face] + TURN_STR[turn]

def parse_move(move):
    return (INV_FACE_STR[move[0]], INV_TURN_STR[move[1:]])

def parse_rot(m):
    m = m.replace("2'", '2')
    if m.endswith("'"):
        return 3
    elif m.endswith('2'):
        return 2
    # E.g. a Ub perm alg has R3 in it
    elif m.endswith('3'):
        return 3
    return 1

def parse_alg(alg):
    if isinstance(alg, list):
        alg = ' '.join(alg)

    moves = []

    for move in alg.split():
        rot = rn = face = n = f2 = n2 = None

        if move[0].upper() in FACE_STR:
            face = FACE_STR.index(move[0].upper())
            n = parse_rot(move)
            # Wide moves: just rotate our representation, and flip the face
            if move[0].islower():
                rot = face >> 1
                rn = n
                if face & 1:
                    rn = 4 - n
                face ^= 1

        elif move[0] in ROTATE_STR:
            rot = ROTATE_STR.index(move[0])
            rn = parse_rot(move)

        elif move[0] in SLICE_STR:
            rot = SLICE_STR.index(move[0])
            rn = parse_rot(move)
            if SLICE_ROT_FLIP[rot]:
                rn = 4 - rn
            face = rot * 2
            n = 4 - rn
            f2 = face + 1
            n2 = rn
        else:
            assert 0, move

        moves.append((rot, rn, face, n, f2, n2))
    return moves

# Parse all the rotation moves in an alg, including slice/wide moves (which
# change the orientation of centers), and return a list of just the
# rotations. For example, M2 y U f -> x2 y z
def get_alg_rotations(alg):
    moves = []
    for move in alg.split():
        rot = rn = None
        if move[0].islower() and move[0].upper() in FACE_STR:
            face = FACE_STR.index(move[0].upper())
            rot = face >> 1
            rn = parse_rot(move)
            if face & 1:
                rn = 4 - rn

        elif move[0] in ROTATE_STR:
            rot = ROTATE_STR.index(move[0])
            rn = parse_rot(move)

        elif move[0] in SLICE_STR:
            rot = SLICE_STR.index(move[0])
            rn = parse_rot(move)
            if SLICE_ROT_FLIP[rot]:
                rn = 4 - rn

        if rot is not None:
            moves.append(ROTATE_STR[rot] + TURN_STR[rn])
    return moves

def invert_alg(alg, cancel_rotations=False):
    moves = []

    # If we're cancelling rotations, we pull out all the rotations in the given
    # alg apply them at the beginning. Since all the other moves are getting
    # inverted, these cancel out, so the cube will start and end in the same
    # orientation.
    if cancel_rotations:
        moves = get_alg_rotations(alg)

    for m in reversed(alg.split()):
        turn = parse_rot(m[1:])
        moves.append(m[0] + TURN_STR[4 - turn])
    return ' '.join(moves)

def gen_random_move_scramble():
    scramble = []
    all_faces = set(range(6))
    blocked_faces = set()
    turns = [-1, 1, 2]
    # Just do N random moves for now, not random state scrambles
    for i in range(24):
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
## Actual solver stuff #########################################################
################################################################################

# This solver is basically Kociemba's two-phase algorithm. The first phase puts
# a random cube into the G1 group (i.e. cubes that can be solved using only
# <U,D,R2,L2,F2,B2> moves). The second phase solves the cube completely using
# only the G1 moves.
#
# To make this problem much easier, each phase is broken down into subproblems
# that are solved simultaneously. For the first phase, the three subproblems are:
#   * Orienting all the corners, so the top or bottom color matches the top or
#       bottom center
#   * Orienting all the edges so they can be solved with G1 moves
#   * Putting all the E-slice edges in the E slice (i.e. not on the U or D layers)
# For phase 2, the subproblems are:
#   * Permuting all the corners
#   * Permuting all the U/D layer edges
#   * Permuting the E-slice edges
#
# So for each phase, we try to find a sequence of moves that solves all the
# subproblems at once. There are a huge number of possibilities to search here,
# but we can prune the search space considerably by utilizing lookup tables.
# If the table tells us that the current corner orientation needs at least 5
# moves to solve, then the entire phase 1 needs at least 5 moves as well. We
# use larger lookup tables that are based on pairs of subproblems in order to
# prune even more (e.g. for phase 1, we use three tables: corner/edge orientation,
# corner orientation/e-slice position, and edge orientation/e-slice position).
# These are fairly easy/quick to generate (a handful of seconds), but we still
# cache them on disk for faster startup. For phase 2, note that we don't use
# a pruning table for the corner/edge permutation pair, since it'd be pretty big.
#
# The overall search looks iteratively deeper at phase 1 solutions until the
# shortest sequence is found, then the shortest phase 2 solution is searched
# from there using another round of iterative deepening. After a solution is
# found, we can look for shorter solutions by allowing the phase 1 solution
# to get longer (but limiting the maximum overall depth of the phase 2 iterative
# deepening).
#
# To make the search faster, we can use a simplified cube representation for
# each phase. For all the "subproblems" of solving the cube, (e.g. orienting
# all corners), there's an index associated with each cube state for that
# subproblem. For orienting corners, there are 3^7 possibilities (7 corners
# with 3 orientations each, the last corner's orientation is determined by the
# others), and the state of the corners can be mapped to a number 0..2187 for
# each possibility. So we can represent the cube by three indices for each
# phase:
#   * Phase 1: c, e, s (corner orientation, edge orientation, slice position)
#       ranges: (3^7=2187, 2^11=2048, 12 choose 4=24)
#   * Phase 2: c, e, s (corner permutation, edge permutation, slice permutation)
#       ranges: (8!=40320, 8!=40320, 4!=24)
# To update these representations when moves are performed, we use more lookup
# tables. These give, for a given index, a list of successor indices when each
# of the 18 standard moves is applied (one of six faces with 90/180/270 degree
# rotation). Or for phase 2, 10 standard moves (since only 180 degree moves are
# allowed for RLFB).
#
# There's also a transition phase whenever we reach phase 2 to convert between
# the two index-based representations. We simply run the phase 1 solution
# on a normal cube and then convert that representation to the phase 2 indexing.
# This is a bit slow/weird, but is rare enough that indexing is still faster and
# simpler overall.

# Move tables, phase 1
CORNER_MOVES_1 = [None] * 2187
EDGE_MOVES_1 = [None] * 2048
ESLICE_MOVES_1 = [None] * 495
# ...and phase 2
CORNER_MOVES_2 = [None] * 40320
EDGE_MOVES_2 = [None] * 40320
ESLICE_MOVES_2 = [None] * 24

# Tables to convert tuples to dense indices for the indices that aren't easy
# to generate numerically
ESLICE_INDEX_1 = {}
CORNER_INDEX_2 = {}
EDGE_INDEX_2 = {}
ESLICE_INDEX_2 = {}
# And a table to convert corner permutation to permutation of four corners
# (to make a smaller pruning table for phase 2)
CCOMBP_INDEX = []

# Pruning tables, phase 1
CORNER_EDGE_LEN_1 = 2187 * 2048
CORNER_ESLICE_LEN_1 = 2187 * 495
EDGE_ESLICE_LEN_1 = 2048 * 495
CORNER_EDGE_DEPTH_1 = array.array('b', [-1] * CORNER_EDGE_LEN_1)
CORNER_ESLICE_DEPTH_1 = array.array('b', [-1] * CORNER_ESLICE_LEN_1)
EDGE_ESLICE_DEPTH_1 = array.array('b', [-1] * EDGE_ESLICE_LEN_1)
# ...and phase 2
CORNER_ESLICE_LEN_2 = 40320 * 24
EDGE_ESLICE_LEN_2 = 40320 * 24
CCOMBP_EDGE_LEN_2 = 140 * 40320
CORNER_ESLICE_DEPTH_2 = array.array('b', [-1] * CORNER_ESLICE_LEN_2)
EDGE_ESLICE_DEPTH_2 = array.array('b', [-1] * EDGE_ESLICE_LEN_2)
CCOMBP_EDGE_DEPTH_2 = array.array('b', [-1] * CCOMBP_EDGE_LEN_2)

PHASE_1_MOVES = [s + t for s in FACE_STR for t in ['', '2', "'"]]
PHASE_2_MOVES = [m for m in PHASE_1_MOVES if m[0] in 'UD' or m.endswith('2')]
FACE_1 = [f for f in range(6) for t in range(1, 4)]
FACE_2 = [f for f in range(6) for t in range(1, 4) if f >> 1 == 0 or t == 2]

SOLVED_C_1 = 0
SOLVED_E_1 = 175
SOLVED_S_1 = 0
SOLVED_INDICES_1 = (SOLVED_C_1, SOLVED_E_1, SOLVED_S_1)

SOLVED_C_2 = 0
SOLVED_E_2 = 0
SOLVED_S_2 = 0
SOLVED_INDICES_2 = (SOLVED_C_2, SOLVED_E_2, SOLVED_S_2)

INDEX_CACHE_PATH = 'rsrc/solver-indices.bin'

FACTORIAL = [1, 1]
for i in range(2, 13):
    FACTORIAL.append(FACTORIAL[i-1] * i)

# Number of phase 1 solutions to search
MAX_PROBES = 100

# Index helper functions. These convert a regular cube (i.e. the Cube class)
# into the index representations used for searching

# Phase 1 indices

def get_corner_index_1(cube):
    index = 0
    # Find the position of the white or yellow face on the first seven corners
    for [i, corner] in enumerate(cube.corners[:7]):
        if W in corner:
            s = corner.index(W)
        else:
            s = corner.index(Y)
        index += 3 ** i * s
    return index

def get_edge_index_1(cube):
    index = 0
    for [i, edge] in enumerate(cube.edges[:11]):
        index |= (edge[1] > edge[0]) << i
    return index

# It's a bit hard to generate an index for the e-slice orientation, since
# it's permutation-invariant, so we just generate a sorted tuple here. We
# use this to generate a dense lookup table that converts tuple to index.
def get_eslice_sparse_index_1(cube):
    index = [None] * 4
    for [i, edge] in enumerate(EDGES[4:8]):
        (j, s) = find_shift(cube.edges, edge)
        index[i] = j
    return tuple(sorted(index))

def get_eslice_index_1(cube):
    return ESLICE_INDEX_1[get_eslice_sparse_index_1(cube)]

# Phase 2 indices

def get_corner_sparse_index_2(cube):
    index = [None] * 8
    for [i, corner] in enumerate(CORNERS):
        (j, s) = find_shift(cube.corners, corner)
        index[i] = j
    return tuple(index)

def get_edge_sparse_index_2(cube):
    index = [None] * 8
    for [i, edge] in enumerate(EDGES[0:4] + EDGES[8:12]):
        (j, s) = find_shift(cube.edges, edge)
        index[i] = j
    return tuple(index)

def get_eslice_sparse_index_2(cube):
    index = [None] * 4
    for [i, edge] in enumerate(EDGES[4:8]):
        (j, s) = find_shift(cube.edges, edge)
        index[i] = j
    return tuple(index)

def get_corner_index_2(cube):
    return CORNER_INDEX_2[get_corner_sparse_index_2(cube)]

def get_edge_index_2(cube):
    return EDGE_INDEX_2[get_edge_sparse_index_2(cube)]

def get_eslice_index_2(cube):
    return ESLICE_INDEX_2[get_eslice_sparse_index_2(cube)]

# Get position of the first 4 corners
def get_ccomb_index(cube):
    index = [None] * 4
    for [i, corner] in enumerate(CORNERS[:4]):
        (j, s) = find_shift(cube.corners, corner)
        index[i] = j
    return tuple(sorted(index))

def get_parity(index, n):
    parity = 0
    for i in range(2, n + 1):
        [index, m] = divmod(index, i)
        parity ^= m
    return parity & 1

# Permute and orient a piece set (corners or edges). This pulls pieces from
# the <solved> list according to the <perm> number, then orients them
# according to the <orient> number. This works over <n> pieces with <r>
# possible orientations each.
def permute_orient(solved, orient, perm, n, r):
    result = [None] * n
    # Copy to a mutable list so we can pull items out when permuting
    solved = list(solved)
    # Permute
    for i in range(n):
        [d, perm] = divmod(perm, FACTORIAL[n - i - 1])
        result[i] = solved.pop(d)
    # Orient
    total = 0
    for i in range(n - 1):
        [orient, d] = divmod(orient, r)
        result[i] = rotate(result[i], d)
        total += d
    # Orient the last piece so the total orientation is 0 modulo r
    result[n - 1] = rotate(result[n - 1], -total % r)
    return result

# Generate reduced corner (perm of first 4 corners only)/parity index for phase 2
def gen_ccomb_indices():
    global CCOMBP_INDEX
    CCOMBP_INDEX = [None] * FACTORIAL[8]
    ccomb_index = {}
    for cp in range(FACTORIAL[8]):
        corners = permute_orient(SOLVED_CUBE.corners, 0, cp, 8, 3)
        cube = Cube(corners=tuple(corners))

        cc = (get_parity(cp, 8), *get_ccomb_index(cube))
        if cc not in ccomb_index:
            ccomb_index[cc] = len(ccomb_index)
        cc = ccomb_index[cc]
        c = get_corner_index_2(cube)
        CCOMBP_INDEX[c] = cc

# For a given index function, generate all the possiblities and a table to
# transition between indices from the standard moves. If the given index function
# is not a dense integer representation, then we also fill in the index_table to
# convert to one.
def gen_move_tables(get_index, move_set, move_table, index_table=None):
    cube = Cube()
    i = get_index(cube)
    if index_table is not None:
        index_table[i] = 0
        i = 0

    current = [(cube, i)]

    seen = set()

    while current:
        next = []
        while current:
            [cube, i] = current.pop()

            moves = move_table[i] = []

            for [face, turn] in move_set:
                child = cube.copy()
                child.turn(face, turn)

                i = get_index(child)

                if index_table is not None:
                    if i not in index_table:
                        index_table[i] = len(index_table)
                    i = index_table[i]

                moves.append(i)

                if i not in seen:
                    seen.add(i)
                    next.append((child, i))

        current = next

# Fill in a pruning table using a pair of index functions. We use the move
# tables to exhaustively enumerate all the possibilities, by making successive
# moves starting from a solved cube. We keep track of how many moves are
# required and fill the value into the given depth table, which is the minimum
# number of moves required to solve the two given subproblems together.
def gen_prune_tables(i_1, i_2, i_base, move_table_1, move_table_2, depth_table,
        remap=None):
    i = remap[i_1] if remap else i_1
    key = i + i_2 * i_base
    depth_table[key] = 0
    current = [(i_1, i_2)]

    depth = 1
    while current:
        next = []
        # From the current list of positions that take <depth-1> moves to reach,
        # find all the positions that take <depth> moves to reach
        while current:
            [i_1, i_2] = current.pop()

            for [c_1, c_2] in zip(move_table_1[i_1], move_table_2[i_2]):
                c = remap[c_1] if remap else c_1
                key = c + c_2 * i_base

                # Only search further if the position hasn't been found before
                if depth_table[key] == -1:
                    depth_table[key] = depth
                    next.append((c_1, c_2))

        current = next
        depth += 1

def gen_indices():
    global CORNER_MOVES_1, EDGE_MOVES_1, ESLICE_MOVES_1
    global ESLICE_INDEX_1
    global CORNER_EDGE_DEPTH_1, CORNER_ESLICE_DEPTH_1, EDGE_ESLICE_DEPTH_1
    global CORNER_MOVES_2, EDGE_MOVES_2, ESLICE_MOVES_2
    global CORNER_INDEX_2, EDGE_INDEX_2, ESLICE_INDEX_2, CCOMBP_INDEX
    global CORNER_ESLICE_DEPTH_2, EDGE_ESLICE_DEPTH_2, CCOMBP_EDGE_DEPTH_2

    # See if the tables are cached on disk, and deserialize them if so
    if os.path.exists(INDEX_CACHE_PATH):
        with open(INDEX_CACHE_PATH, 'rb') as f:
            data = f.read()

        # Split the binary data into a bunch of chunks of the given lengths
        lengths = [
            # Phase 1 moves
            2*18*2187, 2*18*2048, 2*18*495,
            # Phase 1 index lookups
            2*5*495,
            # Phase 1 pruning tables
            CORNER_EDGE_LEN_1, CORNER_ESLICE_LEN_1, EDGE_ESLICE_LEN_1,
            # Phase 2 moves
            2*10*40320, 2*10*40320, 2*10*24,
            # Phase 2 index lookups
            2*9*40320, 2*9*40320, 5*24, 40320,
            # Phase 2 pruning tables
            CORNER_ESLICE_LEN_2, EDGE_ESLICE_LEN_2, CCOMBP_EDGE_LEN_2,
        ]

        chunks = []
        for l in lengths:
            assert len(data) >= l
            chunks.append(data[:l])
            data = data[l:]
        assert not data, len(data)

        [c_m_1, e_m_1, s_m_1, s_i_1, c_e_d_1, c_s_d_1, e_s_d_1,
                c_m_2, e_m_2, s_m_2, c_i_2, e_i_2, s_i_2, cc_i_2,
                c_s_d_2, e_s_d_2, c_e_d_2] = chunks

        # Helper to create an array of the given type from the given data,
        # splitting it into sublists of a given length if requested
        def make(t, b, split=0):
            a = array.array(t)
            a.frombytes(b)
            if split:
                a = [a[i:i+split] for i in range(0, len(a), split)]
            return a

        # Helper to make an index table from the data in t, into l entries
        # mapping c-1 values to one index
        def make_index(t, l, c, tp='H'):
            t = make(tp, t)
            return {tuple(t[i:i+c-1]): t[i+c-1] for i in range(0, c*l, c)}

        CORNER_MOVES_1 = make('H', c_m_1, split=18)
        EDGE_MOVES_1 = make('H', e_m_1, split=18)
        ESLICE_MOVES_1 = make('H', s_m_1, split=18)

        ESLICE_INDEX_1 = make_index(s_i_1, 495, 5)

        CORNER_EDGE_DEPTH_1 = make('B', c_e_d_1)
        CORNER_ESLICE_DEPTH_1 = make('B', c_s_d_1)
        EDGE_ESLICE_DEPTH_1 = make('B', e_s_d_1)

        CORNER_MOVES_2 = make('H', c_m_2, split=10)
        EDGE_MOVES_2 = make('H', e_m_2, split=10)
        ESLICE_MOVES_2 = make('H', s_m_2, split=10)

        CORNER_INDEX_2 = make_index(c_i_2, 40320, 9)
        EDGE_INDEX_2 = make_index(e_i_2, 40320, 9)
        ESLICE_INDEX_2 = make_index(s_i_2, 24, 5, tp='B')

        CCOMBP_INDEX = make('b', cc_i_2)

        CORNER_ESLICE_DEPTH_2 = make('b', c_s_d_2)
        EDGE_ESLICE_DEPTH_2 = make('b', e_s_d_2)
        CCOMBP_EDGE_DEPTH_2 = make('b', c_e_d_2)

        return

    # Generate tables
    phase_1_moves = [(f, t) for f in range(6) for t in range(1, 4)]
    phase_2_moves = [(f, t) for [f, t] in phase_1_moves
            if f >> 1 == 0 or t == 2]

    gen_move_tables(get_corner_index_1, phase_1_moves, CORNER_MOVES_1)
    gen_move_tables(get_edge_index_1, phase_1_moves, EDGE_MOVES_1)
    gen_move_tables(get_eslice_sparse_index_1, phase_1_moves, ESLICE_MOVES_1,
            index_table=ESLICE_INDEX_1)

    gen_move_tables(get_corner_sparse_index_2, phase_2_moves, CORNER_MOVES_2,
            index_table=CORNER_INDEX_2)
    gen_move_tables(get_edge_sparse_index_2, phase_2_moves, EDGE_MOVES_2,
            index_table=EDGE_INDEX_2)
    gen_move_tables(get_eslice_sparse_index_2, phase_2_moves, ESLICE_MOVES_2,
            index_table=ESLICE_INDEX_2)

    gen_ccomb_indices()

    gen_prune_tables(SOLVED_C_1, SOLVED_E_1, 2187, CORNER_MOVES_1, EDGE_MOVES_1,
            CORNER_EDGE_DEPTH_1)
    gen_prune_tables(SOLVED_C_1, SOLVED_S_1, 2187, CORNER_MOVES_1, ESLICE_MOVES_1,
            CORNER_ESLICE_DEPTH_1)
    gen_prune_tables(SOLVED_E_1, SOLVED_S_1, 2048, EDGE_MOVES_1, ESLICE_MOVES_1,
            EDGE_ESLICE_DEPTH_1)

    gen_prune_tables(SOLVED_C_2, SOLVED_S_2, 40320, CORNER_MOVES_2, ESLICE_MOVES_2,
            CORNER_ESLICE_DEPTH_2)
    gen_prune_tables(SOLVED_E_2, SOLVED_S_2, 40320, EDGE_MOVES_2, ESLICE_MOVES_2,
            EDGE_ESLICE_DEPTH_2)
    gen_prune_tables(SOLVED_C_2, SOLVED_E_2, 140, CORNER_MOVES_2, EDGE_MOVES_2,
            CCOMBP_EDGE_DEPTH_2, remap=CCOMBP_INDEX)

    # Write the generated tables to disk
    with open(INDEX_CACHE_PATH, 'wb') as f:
        # Helper to flatten a list of lists into just a list. I'd usually use
        # sum(ll, []) but that has some O(n^2) behavior apparently
        def flatten(ll):
            return [i for l in ll for i in l]

        # Phase 1
        moves_1 = array.array('H', flatten(CORNER_MOVES_1 + EDGE_MOVES_1 +
                ESLICE_MOVES_1))
        s_i_1 = array.array('H', flatten([[*k, v]
                for [k, v] in ESLICE_INDEX_1.items()]))
        f.write(moves_1.tobytes())
        f.write(s_i_1.tobytes())
        f.write(CORNER_EDGE_DEPTH_1.tobytes())
        f.write(CORNER_ESLICE_DEPTH_1.tobytes())
        f.write(EDGE_ESLICE_DEPTH_1.tobytes())

        # Phase 2
        m2 = flatten(CORNER_MOVES_2 + EDGE_MOVES_2 +
                ESLICE_MOVES_2)
        moves_2 = array.array('H', m2)
        c_i_2 = array.array('H', flatten([[*k, v]
                for [k, v] in CORNER_INDEX_2.items()]))
        e_i_2 = array.array('H', flatten([[*k, v]
                for [k, v] in EDGE_INDEX_2.items()]))
        s_i_2 = array.array('B', flatten([[*k, v]
                for [k, v] in ESLICE_INDEX_2.items()]))
        cc_i_2 = array.array('B', CCOMBP_INDEX)
        f.write(moves_2.tobytes())
        f.write(c_i_2.tobytes() + e_i_2.tobytes() + s_i_2.tobytes())
        f.write(cc_i_2.tobytes())
        f.write(CORNER_ESLICE_DEPTH_2.tobytes())
        f.write(EDGE_ESLICE_DEPTH_2.tobytes())
        f.write(CCOMBP_EDGE_DEPTH_2.tobytes())

gen_indices()

class SolverContext:
    def __init__(self):
        self.probes = 0
        self.nodes = 0
        self.max_depth = 1000
        self.initial_cube = None
        self.remap = None
        self.solution_cache = set()
        self.best_solution = None
        self.best_remap = None

    def set_best(self, moves_1, moves_2):
        self.best_solution = [moves_1, moves_2]
        self.best_remap = self.remap
        # Set overall max depth so we only try to find solutions shorted than this
        self.max_depth = len(moves_1) + len(moves_2)

# Phase 1 recursive search
def phase_1(ctx, c, e, s, last_face, moves, depth):
    ctx.nodes += 1
    # See if we've solved phase 1
    if (c, e, s) == SOLVED_INDICES_1:
        # Make sure we haven't searched this exact sequence before
        key = tuple(moves)
        if key in ctx.solution_cache:
            return
        ctx.solution_cache.add(key)

        # Set up cube for phase 2
        alg = ' '.join(PHASE_1_MOVES[i] for i in moves)
        cube = ctx.initial_cube.copy()
        cube.run_alg(alg)
        c_2 = get_corner_index_2(cube)
        e_2 = get_edge_index_2(cube)
        s_2 = get_eslice_index_2(cube)
        # And search phase 2
        ctx.probes += 1
        for d in range(ctx.max_depth - len(moves)):
            if phase_2(ctx, c_2, e_2, s_2, last_face, moves, [], d):
                return ctx.probes > MAX_PROBES
        if ctx.probes > MAX_PROBES:
            return True
        return False
    if depth == 0:
        return False

    # Look through all the possible moves
    for [m, [c_n, e_n, s_n]] in enumerate(zip(CORNER_MOVES_1[c], EDGE_MOVES_1[e],
            ESLICE_MOVES_1[s])):
        # Don't turn the same face twice in a row, and only turn the opposite face
        # if it's lower (so D U is allowed but not U D)
        face = FACE_1[m]
        if face == last_face or face & ~1 == last_face:
            continue

        # Prune the search if this move leads to a position needing too many
        # moves to solve
        if (CORNER_EDGE_DEPTH_1[c_n + 2187*e_n] >= depth or
                CORNER_ESLICE_DEPTH_1[c_n + 2187*s_n] >= depth or
                EDGE_ESLICE_DEPTH_1[e_n + 2048*s_n] >= depth):
            continue

        if phase_1(ctx, c_n, e_n, s_n, face, moves + [m], depth - 1):
            return True

    return False

# Phase 2 recursive search
def phase_2(ctx, c, e, s, last_face, moves_1, moves_2, depth):
    ctx.nodes += 1
    # See if we've solved phase 2
    if (c, e, s) == SOLVED_INDICES_2:
        ctx.set_best(moves_1, moves_2)
        return True
    if depth == 0:
        return False

    # Look through all the possible moves
    for [m, [c_n, e_n, s_n]] in enumerate(zip(CORNER_MOVES_2[c], EDGE_MOVES_2[e],
            ESLICE_MOVES_2[s])):
        # Don't turn the same face twice in a row, and only turn the opposite face
        # if it's lower (so D U is allowed but not U D)
        face = FACE_2[m]
        if face == last_face or face & ~1 == last_face:
            continue

        # Prune the search if this move leads to a position needing too many
        # moves to solve
        if (CORNER_ESLICE_DEPTH_2[c_n + 40320*s_n] >= depth or
                EDGE_ESLICE_DEPTH_2[e_n + 40320*s_n] >= depth or
                CCOMBP_EDGE_DEPTH_2[CCOMBP_INDEX[c_n] + 140*e_n] >= depth):
            continue

        if phase_2(ctx, c_n, e_n, s_n, face, moves_1, moves_2 + [m], depth - 1):
            return True

    return False

# Random state scramble generator
def gen_random_state_scramble():
    # Generate random corner/edge orientation/permutation
    co = random.randrange(2187) # 3^7
    eo = random.randrange(2048) # 2^11
    while True:
        cp = random.randrange(FACTORIAL[8])
        ep = random.randrange(FACTORIAL[12])
        if get_parity(cp, 8) == get_parity(ep, 12):
            break

    corners = permute_orient(SOLVED_CUBE.corners, co, cp, 8, 3)
    edges = permute_orient(SOLVED_CUBE.edges, eo, ep, 12, 2)

    # Set up cubes to solve to G1 along each axis. The typical Kociemba algorithm
    # first solves the cube into <U,D,R2,L2,F2,B2>, but we also rotate the cube
    # and remap colors to solve to <U2,D2,R,L,F2,B2> and <U2,D2,R2,L2,F,B>.
    cubes = []
    for [alg, remap] in [['', range(6)], ['x', RX], ['z', RZ]]:
        cube = Cube(corners=tuple(corners), edges=tuple(edges))
        cube.run_alg(alg)
        cube.corners = tuple(tuple(remap[f] for f in c) for c in cube.corners)
        cube.edges = tuple(tuple(remap[f] for f in c) for c in cube.edges)

        c = get_corner_index_1(cube)
        e = get_edge_index_1(cube)
        s = get_eslice_index_1(cube)

        cubes.append((cube, c, e, s, remap))

    ctx = SolverContext()
    t = time.time()
    # Main iterative deepening search loop. We interleave searches along all the
    # axes so that if we find a solution of length N on one axis, we can limit
    # the depth on all axes to N-1.
    for depth in range(1, 20):
        for [cube, c, e, s, remap] in cubes:
            ctx.initial_cube = cube
            ctx.remap = remap
            if phase_1(ctx, c, e, s, None, [], depth):
                break
        if ctx.probes > MAX_PROBES:
            break

    # Take best solution and remap faces to match the normal cube
    [m_1, m_2] = ctx.best_solution
    moves = [PHASE_1_MOVES[i] for i in m_1] + [PHASE_2_MOVES[i] for i in m_2]
    alg = invert_alg(' '.join(moves))

    trans = str.maketrans(''.join(FACE_STR[ctx.best_remap[i]]
            for i in range(6)), FACE_STR)
    return alg.translate(trans).split()
