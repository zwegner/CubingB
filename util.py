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

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QHBoxLayout, QVBoxLayout, QGridLayout,
        QTableWidgetItem, QHeaderView, QTabBar)

# Global constants

INF = float('+inf')

STAT_AO_COUNTS = [1, 5, 12, 25, 50, 100, 200, 500, 1000]

# Stat helpers

@contextlib.contextmanager
def time_execution(label):
    start = time.time()
    yield
    print('%s: %.3fs' % (label, time.time() - start))

def stat_str(size):
    if size == 1:
        return 'single'
    else:
        return 'ao%s' % size

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

def cell(text, editable=False, secret_data=None):
    item = QTableWidgetItem(text)
    if not editable:
        item.setFlags(item.flags() & ~Qt.ItemIsEditable)
    # Just set an attribute on the cell to pass data around?
    # Probably not supposed to do this but it works
    if secret_data is not None:
        item.secret_data = secret_data
    return item

def make_hbox(parent, children):
    layout = QHBoxLayout(parent)
    for c in children:
        layout.addWidget(c)
    return layout

def make_vbox(parent, children, margin=None):
    layout = QVBoxLayout(parent)
    if margin is not None:
        layout.setContentsMargins(margin, margin, margin, margin)
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

def make_tabs(*tabs, change=None):
    tab_bar = QTabBar()
    for tab in tabs:
        tab_bar.addTab(tab)
    if change:
        tab_bar.currentChanged.connect(change)
    return tab_bar

# Set the column headers for a table.
# The hacky 'stretch' parameter will stretch all columns if it's negative, or a
# single column of the given id if nonnegative
def set_table_columns(table, cols, stretch=None, visible=True):
    table.setColumnCount(len(cols))
    if visible:
        for [i, item] in enumerate(cols):
            if isinstance(item, str):
                item = cell(item)
            table.setHorizontalHeaderItem(i, item)
    else:
        table.horizontalHeader().hide()

    if stretch is not None:
        if stretch >= 0:
            table.horizontalHeader().setSectionResizeMode(stretch, QHeaderView.Stretch)
        else:
            table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
