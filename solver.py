import curses
import enum
import random

[W, Y, R, O, G, B, X] = range(7)

EDGES = ((W, G), (W, R), (W, B), (W, O),
    (G, R), (R, B), (B, O), (O, G),
    (G, Y), (R, Y), (B, Y), (O, Y))
CORNERS = ((W, G, R), (W, R, B), (W, B, O), (W, O, G),
    (Y, B, R), (Y, R, G), (Y, G, O), (Y, O, B))
CENTERS = (W, Y, R, O, G, B)

up    = [2, 1,  0,  3], [0, 0, 0, 0], [2, 1, 0, 3], [0, 0, 0, 0], 0
down  = [8, 9, 10, 11], [1, 1, 1, 1], [6, 5, 4, 7], [0, 0, 0, 0], 1
right = [1, 5,  9,  4], [1, 0, 0, 1], [0, 1, 4, 5], [2, 1, 2, 1], 2
left  = [3, 7, 11,  6], [1, 0, 0, 1], [2, 3, 6, 7], [2, 1, 2, 1], 3
front = [0, 4,  8,  7], [1, 0, 0, 1], [3, 0, 5, 6], [2, 1, 2, 1], 4
back  = [2, 6, 10,  5], [1, 0, 0, 1], [1, 2, 7, 4], [2, 1, 2, 1], 5

faces = [up, down, right, left, front, back]

RY = [0, 1, 4, 5, 3, 2]
RX = [5, 4, 2, 3, 0, 1]
RZ = [2, 3, 1, 0, 4, 5]
ROTATE_FACES = [RY, RX, RZ]

FACE_STR = 'UDRLFB'
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
        if isinstance(alg, str):
            alg = parse_alg(alg)
        for [rot, rn, r, n, r2, n2] in alg:
            if rot is not None:
                self.rotate(rot, rn)
            if r is not None:
                self.turn(r, n)
            if r2 is not None:
                self.turn(r2, n2)

# Metabrogramming. Generate function for each of the turn and rotate moves.
# This is pretty messy code, just the first random crap I hacked up that worked
TURNS = []
ROTATES = []
def gen_turns():
    for F in range(6):
        FR = []
        TURNS.append(FR)
        face = faces[F]
        for n in range(1, 4):
            E = list(range(1, 13))
            C = [(x, 0) for x in range(8)]
            [idx, flip, cidx, cflip, _] = face
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
            FR.append(ctx[name])

    def find_shift(l, i):
        ii = i
        for s in range(len(i)):
            if i in l:
                return (l.index(i), s)
            i = (i[-1], *i[:-1])
        assert 0, (l, ii)

    for [r, rotation] in enumerate(ROTATE_FACES):
        FR = []
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
            FR.append(ctx[name])

gen_turns()

def parse_alg(alg):
    moves = []

    def parse_rot(m):
        if move.endswith("'"):
            return 0
        elif move.endswith('2'):
            return 1
        return 2

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
                    rn = 2 - n
                face ^= 1

        elif move[0] in ROTATE_STR:
            rot = ROTATE_STR.index(move[0].lower())
            rn = parse_rot(move)

        moves.append((rot, rn, face, n, f2, n2))
    return moves
