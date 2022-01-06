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
import enum
import queue
import time
import threading

from PyQt5.QtCore import (Qt, QSize)
from PyQt5.QtWidgets import (QHBoxLayout, QVBoxLayout, QGridLayout,
        QTableWidgetItem, QHeaderView, QTabBar, QDialog, QLabel, QComboBox,
        QTableWidget, QDialogButtonBox, QWidget, QAbstractItemView, QAction,
        QPushButton, QCheckBox)
from PyQt5.QtGui import QIcon

# Global constants

INF = float('+inf')

ScrambleType = enum.IntEnum('ScrambleType', 'RANDOM_STATE RANDOM_MOVES '
        'ENTER_SCRAMBLE HAND_SCRAMBLE', start=0)
SCRAMBLE_TYPES = ['Random state', 'Random Moves', 'Enter scramble', 'Hand Scramble']

STAT_AO_COUNTS = [1, 3, 5, 12, 25, 50, 100, 200, 500, 1000]

def stat_str(size):
    if size == 1:
        return 'single'
    elif size < 5:
        return 'mo%s' % size
    else:
        return 'ao%s' % size

STAT_AO_STR = [stat_str(size) for size in STAT_AO_COUNTS]
STAT_AO_COUNTS_STR = [(size, stat_str(size)) for size in STAT_AO_COUNTS]

# Stat helpers

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

def session_sort_key(s):
    if s.sort_id is not None:
        return s.sort_id
    return s.id

# For certain tasks that take a long time, we put them in a queue for a background
# thread to run
BACKGROUND_QUEUE = queue.Queue()
def sched_background_task(fn, *args):
    BACKGROUND_QUEUE.put((fn, args))

def run_background_thread():
    while True:
        [task, args] = BACKGROUND_QUEUE.get()
        task(*args)

# Base class for handling various scrambling function for each puzzle type
class PuzzleDefs:
    def __init__(self):
        self.is_bg_scramble_gen = False
        self.cached_state_scramble = None
        self.cached_moves_scramble = None
        self.scramble_message = None
        self.lock = threading.Lock()

    def supported_scrambles(self):
        return [ScrambleType.ENTER_SCRAMBLE, ScrambleType.HAND_SCRAMBLE]

    def clear_cached_scramble(self):
        # Might want to only clear the one that was used?
        self.cached_state_scramble = None
        self.cached_moves_scramble = None

    def async_gen_scramble(self, callback):
        with self.lock:
            if self.is_bg_scramble_gen:
                return
            self.is_bg_scramble_gen = True

        scramble = self.gen_random_state()

        with self.lock:
            self.is_bg_scramble_gen = False

            # Let's hope nothing weird has happened in the meantime
            if self.cached_state_scramble is None:
                self.cached_state_scramble = scramble

                callback()

    def random_state_scramble(self, callback):
        if not self.cached_state_scramble:
            sched_background_task(self.async_gen_scramble, callback)
            self.scramble_message = 'Generating scramble...'
            return None
        return self.cached_state_scramble

    def gen_random_state(self):
        raise NotImplementedError()

    def random_move_scramble(self):
        if not self.cached_moves_scramble:
            self.cached_moves_scramble = self.gen_random_moves()
        return self.cached_moves_scramble

    def gen_random_moves(self):
        raise NotImplementedError()

    # By default just accept a raw string
    def parse_scramble(self, scramble):
        return scramble

    # Convert internal scramble representation to list of string moves
    def alg_list(self, moves):
        return moves

    def alg_str(self, moves):
        return ' '.join(moves)

    # How to join scramble moves in the HTML scramble view. Generally just
    # a space, or a zero-width space to allow breaks
    def html_spacer(self):
        return ' '

    def gen_diagram(self, scramble):
        return None

# Qt helpers

def make_button(text, fn, icon=False, size=None, border=True):
    # For an icon, the text is a path to an icon, also remove border
    if icon or isinstance(text, QIcon):
        if not isinstance(text, QIcon):
            text = QIcon('rsrc/' + text)
        button = QPushButton(text, '')
        if size:
            button.setStyleSheet('border: none; icon-size: %spx %spx;' % (size, size))
        else:
            button.setStyleSheet('border: none;')
    else:
        button = QPushButton(text)
        if not border:
            button.setStyleSheet('border: none;')
    button.clicked.connect(fn)
    return button

def make_checkbox(text, changed_fn, checked=True):
    cb = QCheckBox(text)
    cb.setChecked(checked)
    cb.stateChanged.connect(changed_fn)
    return cb

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

def make_dropdown(items, change=None):
    combo = QComboBox()
    for t in items:
        combo.addItem(t)
    if change:
        combo.currentIndexChanged.connect(change)
    return combo

def add_menu_action(menu, text, fn):
    action = QAction(text, menu)
    action.triggered.connect(fn)
    menu.addAction(action)

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

@contextlib.contextmanager
def block_signals(obj):
    obj.blockSignals(True)
    yield
    obj.blockSignals(False)

# Common widgets

class SessionSelectorDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent)

        self.label = QLabel('Session:')

        self.selector = QComboBox()
        self.multi_selector = QTableWidget()
        set_table_columns(self.multi_selector, [''], visible=False, stretch=-1)
        self.multi_selector.verticalHeader().hide()
        self.multi_selector.setSelectionMode(QAbstractItemView.ExtendedSelection)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept_session)
        buttons.rejected.connect(self.reject)

        top = QWidget()
        make_hbox(top, [self.label, self.selector, self.multi_selector])
        make_vbox(self, [top, buttons])

        self.allow_multi = False
        self.selected_session = None
        self.session_ids = []

    def sizeHint(self):
        return QSize(400, 600)

    def update_data(self, message, sessions, allow_new=False, allow_multi=False):
        self.selected_session = None

        self.label.setText(message)
        self.allow_multi = allow_multi
        if allow_multi:
            self.multi_selector.show()
            self.selector.hide()

            self.multi_selector.clearContents()
            self.multi_selector.setRowCount(len(sessions))
            for [i, s] in enumerate(sessions):
                self.multi_selector.setItem(i, 0, cell(s.name, secret_data=s.id))
        else:
            self.selector.show()
            self.multi_selector.hide()
            self.selector.clear()
            self.session_ids = []
            for s in sessions:
                self.session_ids.append(s.id)
                self.selector.addItem(s.name)
            if allow_new:
                self.session_ids.append(None)
                self.selector.addItem('New...')

    def accept_session(self):
        if self.allow_multi:
            ids = {item.secret_data for item in self.multi_selector.selectedItems()}
            self.selected_session_id = list(ids)
        else:
            i = self.selector.currentIndex()
            self.selected_session_id = self.session_ids[i]
            if self.selected_session_id is None:
                [name, accepted] = QInputDialog.getText(self, 'New session name',
                        'Enter new session name:', text='New Session')
                if not accepted:
                    self.reject()
                self.selected_session_id = name
        self.accept()
