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

from PyQt5.QtWidgets import (QHBoxLayout, QVBoxLayout, QGridLayout)

INF = float('+inf')

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
    if ms > 60000:
        [minutes, ms] = divmod(ms, 60000)
        # Set up a format string since there's no .*f formatting
        fmt = '%%d:%%0%s.%sf' % (prec + 3, prec)
        return fmt % (minutes, ms / 1000)
    # Set up a format string since there's no .*f formatting
    fmt = '%%.%sf' % prec
    return fmt % (ms / 1000)

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

def make_grid(parent, table, stretch=None, widths=None):
    layout = QGridLayout(parent)
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
