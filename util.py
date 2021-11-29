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

import contextlib
import time

from PyQt5.QtWidgets import (QHBoxLayout, QVBoxLayout, QGridLayout)

INF = float('+inf')

@contextlib.contextmanager
def time_execution(label):
    start = time.time()
    yield
    print('%s: %.3fs' % (label, time.time() - start))

def solve_time(solve):
    if solve.dnf:
        return INF
    t = solve.time_ms
    if solve.plus_2:
        t += 2000
    return t

def solve_time_str(solve):
    if solve.dnf:
        return 'DNF'
    if solve.plus_2:
        return '%s +2' % ms_str(solve.time_ms)
    return ms_str(solve.time_ms)

def ms_str(ms, prec=3):
    if ms is None:
        return '-'
    if ms == INF:
        return 'DNF'
    [minutes, ms] = divmod(ms, 60000)
    [seconds, ms] = divmod(ms, 1000)
    if minutes:
        pre = '%d:%02d' % (minutes, seconds)
    else:
        pre = '%d' % seconds
    # Set up a format string since there's no .*f formatting
    fmt = '.%%0%sd' % prec
    return pre + fmt % (ms // (10 ** (3 - prec)))

# Qt helpers

def make_hbox(parent, children):
    layout = QHBoxLayout(parent)
    for c in children:
        layout.addWidget(c)
    return layout

def make_vbox(parent, children):
    layout = QVBoxLayout(parent)
    for c in children:
        layout.addWidget(c)
    return layout

def make_grid(parent, table, stretch=None, widths=None, margin=None):
    layout = QGridLayout(parent)
    if margin is not None:
        layout.setContentsMargins(margin, margin, margin, margin)
    width = max(len(row) for row in table)
    for [i, row] in enumerate(table):
        # Not enough items in the row: assume one cell spanning the whole width
        if len(row) != width:
            [cell] = row
            layout.addWidget(cell, i, 0, 1, width)
        else:
            for [j, cell] in enumerate(row):
                if cell is not None:
                    layout.addWidget(cell, i, j)
    if stretch:
        for [i, s] in enumerate(stretch):
            layout.setColumnStretch(i, s)
    if widths:
        for [i, w] in enumerate(widths):
            layout.setColumnMinimumWidth(i, w)
    return layout
