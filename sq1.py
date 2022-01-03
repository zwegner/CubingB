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

import math
import random

[W, Y, R, O, G, B] = range(6)

PIECES = [(Y, R, B), (Y, B), (Y, B, O), (Y, O), (Y, O, G), (Y, G), (Y, G, R), (Y, R),
        (W, O, B), (W, B), (W, B, R), (W, R), (W, R, G), (W, G), (W, G, O), (W, O)]
TOP = [0, None, 1, 2, None, 3, 4, None, 5, 6, None, 7]
BOTTOM = [8, None, 9, 10, None, 11, 12, None, 13, 14, None, 15]

# Rotate the pieces in one layer
def rotate(l, n):
    assert -12 <= n <= 12
    return (l * 3)[12 - n:24 - n]

class Square1:
    def __init__(self, top=TOP, bottom=BOTTOM, flipped=False):
        self.top = top
        self.bottom = bottom
        self.flipped = flipped

    # Rotate top layer by t, bottom layer by b, and optionally slice
    def move(self, t, b, slice=True):
        top = rotate(self.top, t)
        bottom = rotate(self.bottom, b)
        if slice:
            if (top[0] is None or top[6] is None or
                    bottom[5] is None or bottom[11] is None):
                return False
            self.top = top[:6] + bottom[5:11]
            self.bottom = bottom[:5] + top[6:] + bottom[11:]
            self.flipped = not self.flipped
        else:
            self.top = top
            self.bottom = bottom
        return True

    def run_alg(self, moves):
        for [t, b, slice] in moves:
            self.move(t, b, slice=slice)

    def check_top_rot(self, rot):
        return (self.top[-rot % 12] is not None and
                self.top[(-rot + 6) % 12] is not None)

    def check_bottom_rot(self, rot):
        return (self.bottom[(-rot + 5) % 12] is not None and
                self.bottom[(-rot + 11) % 12] is not None)

SOLVED_SQ1 = Square1()

def gen_random_move_scramble(length):
    moves = []
    cube = Square1()
    for i in range(length):
        tops = [i for i in range(12) if cube.check_top_rot(i)]
        bottoms = [i for i in range(12) if cube.check_bottom_rot(i)]
        tb = [(t, b) for t in tops for b in bottoms if (t, b) != (0, 0)]
        [top, bottom] = random.choice(tb)
        cube.move(top, bottom)
        moves.append(((top + 5) % 12 - 5, (bottom + 5) % 12 - 5,
                True if i < length-1 else bool(random.randrange(2))))
    return moves

def alg_str(moves):
    result = []
    for [t, b, slice] in moves:
        result.append('%s,%s%s' % (t, b, '/' if slice else ''))
    return result

################################################################################
## SVG rendering stuff #########################################################
################################################################################

SVG_COLORS = {
    'w': '#fff',
    'y': '#222', # All the cool kids use white/black, not white/yellow
    'r': '#d00',
    'o': '#f92',
    'g': '#0b0',
    'b': '#00f',
    '-': '#777',
}
COLOR_MAP = 'wyrogb'
# How far to stretch the xy points to render the sides of pieces
STRETCH = 1.4

# Some geometry for rendering a square-1 diagram
LENGTH = 1 / math.cos(math.radians(15))
P1 = LENGTH * math.sin(math.radians(15))
R2 = LENGTH * 2 ** .5 / 2

def gen_sq1_diagram(sq1):
    def gen_path(points, rot, color):
        color = SVG_COLORS[COLOR_MAP[color]]
        path = 'M%sz' % 'L'.join('%s %s' % xy for xy in points)
        return f'''<path d="{path}" fill="{color}" stroke="black"
            transform="rotate({rot})" stroke-width=".02" />'''

    def gen_layer(layer):
        for [i, p] in enumerate(layer):
            if p is None:
                continue
            piece = PIECES[p]
            # Render top piece
            if len(piece) == 3:
                points = [(0, 0), (-P1, 1), (-1, 1), (-1, P1)]
            else:
                points = [(0, 0), (-P1, 1), (-R2, R2)]
            yield gen_path(points, i*30, piece[0])

            # Stretch endpoints of top piece to render sides
            stretched = [(x * STRETCH, y * STRETCH) for [x, y] in points]
            for j in range(1, len(points) - 1):
                p = points[j:j+2] + stretched[j+1:j-1:-1]
                yield gen_path(p, i*30, piece[j])

    # Generate top/bottom layers
    layer_1 = '\n'.join(gen_layer(sq1.top))
    layer_2 = '\n'.join(gen_layer(sq1.bottom))

    # Generate middle
    s = 1 - P1
    s2 = 2 - s if sq1.flipped else s
    middle_l = gen_path([(0, 0), (0, s), (s, s), (s, 0)], 0, R)
    middle_r = gen_path([(0, 0), (0, s), (s2, s), (s2, 0)], 0,
            O if sq1.flipped else R)

    return f'''<svg viewBox='-2.5 -2.5 10 5'>
          {layer_1}
          <g transform="translate(5, 0)">
              {layer_2}
          </g>
          <g transform="translate({2.5-s}, {2.2-s})">
              {middle_l}
          </g>
          <g transform="translate(2.5, {2.2-s})">
              {middle_r}
          </g>
        </svg>'''
