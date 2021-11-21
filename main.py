#!/usr/bin/env python

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

import collections
import enum
import gzip
import math
import random
import struct
import sys
import time

from PyQt5.QtCore import (QSize, Qt, QTimer, pyqtSignal, QAbstractAnimation,
        QVariantAnimation)
from PyQt5.QtWidgets import (QApplication, QMainWindow, QHBoxLayout, QVBoxLayout,
        QWidget, QOpenGLWidget, QLabel, QTableWidget, QTableWidgetItem,
        QSizePolicy, QGridLayout, QComboBox, QDialog, QDialogButtonBox,
        QAbstractItemView, QHeaderView, QFrame, QCheckBox, QPushButton,
        QSlider, QMessageBox, QInputDialog, QMenu, QAction, QPlainTextEdit)
from PyQt5.QtGui import QIcon

import bluetooth
import config
import db
import render
import solver

# Constants

WINDOW_SIZE = [1600, 1000]

State = enum.Enum('State', 'SCRAMBLE SOLVE_PENDING SOLVING SMART_SCRAMBLING '
        'SMART_SCRAMBLED SMART_SOLVING SMART_VIEWING')

SOLVED_CUBE = solver.Cube()

SCRAMBLE_MOVES = 25

STAT_AO_COUNTS = [1, 5, 12, 100]
STAT_OUTLIER_PCT = 5

TIMER_DEBOUNCE = .5

INF = float('+inf')

# Basic header to support future metadata/versioning/etc for smart data
SMART_DATA_VERSION = 2

# Quaternion helper functions

def quat_mul(q1, q2):
    [a, b, c, d] = q1
    [w, x, y, z] = q2
    return [a*w - b*x - c*y - d*z,
        a*x + b*w + c*d - d*y,
        a*y - b*z + c*w + d*x,
        a*z + b*y - c*x + d*w]

def quat_invert(values):
    [w, x, y, z] = values
    f = 1 / (w*w + x*x + y*y + z*z)
    return [w * f, -x * f, -y * f, -z * f]

def quat_normalize(values):
    [w, x, y, z] = values
    f = 1 / (w*w + x*x + y*y + z*z) ** .5
    return [w * f, x * f, y * f, z * f]

def quat_matrix(values):
    [w, x, y, z] = values
    return [
        w*w + x*x - y*y - z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y, 0,
        2*x*y + 2*w*z, w*w - x*x + y*y - z*z, 2*y*z - 2*w*x, 0,
        2*x*z - 2*w*y, 2*y*z + 2*w*x, w*w - x*x - y*y + z*z, 0,
        0, 0, 0, 1,
    ]

def vec_cross_prod(v1, v2):
    [a, b, c] = v1
    [x, y, z] = v2
    return [b*z - c*y, c*x - a*z, a*y - b*x]

def vec_dot_prod(v1, v2):
    [a, b, c] = v1
    [x, y, z] = v2
    return a*x + b*y + c*z

def vec_len_sq(v):
    [a, b, c] = v
    return a*a + b*b + c*c

# UI helpers

def calc_ao(solves, start, size):
    if len(solves) - start < size:
        mean = None
    else:
        times = [solve_time(s) for s in solves[start:start+size]]
        if size > 1:
            times.sort()
            outliers = (size * STAT_OUTLIER_PCT + 99) // 100
            times = times[outliers:-outliers]
        mean = sum(times) / len(times)

    return mean

# Calculate the aoX for all solves. This is more efficient (at least for bigger X)
# since it's incrementally updated, but the code is a bit annoying. We use a
# binary search on the sorted solves within the sliding window, so overall the
# runtime is O(n log size). This is also slightly weird in that the solves are
# ordered new-to-old, so process them with the sliding window ahead of the current
# solve rather than behind.
def calc_rolling_ao(solves, all_times, size):
    if len(solves) < size:
        for _ in range(len(solves)):
            yield None
        return

    outliers = (size * STAT_OUTLIER_PCT + 99) // 100
    low = outliers
    high = size - outliers
    n_samples = high - low

    # Build initial list
    times = list(sorted(all_times[:size]))

    # Binary search helper. Slightly annoying in that it needs to return the
    # insertion position for new elements
    def find(i):
        lo = 0
        hi = len(times) - 1
        while lo < hi:
            mid = (hi + lo + 1) // 2
            if times[mid] > i:
                hi = mid - 1
            elif times[mid] < i:
                lo = mid
            else:
                return mid
        if times[lo] < i:
            return lo + 1
        return lo

    # Keep track of the sum of all solve times within the window
    total = sum(times[low:high])
    yield total / n_samples

    for i in range(len(solves) - size):
        assert not math.isnan(total)
        # Remove old solve time
        old = all_times[i]
        o = find(old)
        assert times[o] == old, (times, old, o)
        times.pop(o)

        # Insert new solve
        new = all_times[i + size]
        n = find(new)
        times.insert(n, new)

        # Recalculate total if the average is moving from DNF to non-DNF. If
        # this happens a lot, it negates the speed advantage of this rolling
        # calculation. I guess that's punishment for having too many DNFs.
        if total == INF and times[high - 1] < INF:
            total = sum(times[low:high])
            yield total / n_samples
            continue

        # Update total based on what solves moved in/out the window. There are
        # 9 cases, with old and new each in high/mid/low

        # Old in low
        if o < low:
            if n < low:
                pass
            elif n < high:
                total += new
                total -= times[low - 1]
            else:
                total += times[high - 1]
                total -= times[low - 1]
        # Old in mid
        elif o < high:
            total -= old
            if n < low:
                total += times[low]
            elif n < high:
                total += new
            else:
                total += times[high - 1]
        # Old in high
        else:
            if n < low:
                total += times[low]
                total -= times[high]
            elif n < high:
                total += new
                total -= times[high]
            else:
                pass

        # Guard against inf-inf
        if math.isnan(total):
            total = INF

        yield total / n_samples

    # Yield None for all the solves before reaching the aoX
    for _ in range(size - 1):
        yield None

def get_ao_str(solves, start, size):
    if len(solves) - start < size or size == 1:
        return ''
    solves = solves[start:start+size]
    times = [(solve_time(s), s.id) for s in solves]
    times.sort()
    outliers = (size * STAT_OUTLIER_PCT + 99) // 100
    outlier_set = {s_id for [_, s_id] in times[:outliers] + times[-outliers:]}
    times = [t for [t, _] in times[outliers:-outliers]]
    mean = sum(times) / len(times)

    result = []
    for solve in solves:
        s = ms_str(solve_time(solve))
        if solve.id in outlier_set:
            s = '(%s)' % s
        result.append(s)

    return '%s = %s' % (' '.join(result), ms_str(mean))

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
    if ms > 60000:
        [minutes, ms] = divmod(ms, 60000)
        # Set up a format string since there's no .*f formatting
        fmt = '%%d:%%0%s.%sf' % (prec + 3, prec)
        return fmt % (minutes, ms / 1000)
    # Set up a format string since there's no .*f formatting
    fmt = '%%.%sf' % prec
    return fmt % (ms / 1000)

def cell(text, editable=False, secret_data=None):
    item = QTableWidgetItem(text)
    if not editable:
        item.setFlags(item.flags() & ~Qt.ItemIsEditable)
    # Just set an attribute on the cell to pass data around?
    # Probably not supposed to do this but it works
    if secret_data is not None:
        item.secret_data = secret_data
    return item

def session_sort_key(s):
    if s.sort_id is not None:
        return s.sort_id
    return s.id

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

# Get the solve number (i.e. the number within the session, not the
# database id)
def get_solve_nb(session, solve):
    return (session.query(db.Solve).filter_by(session=solve.session)
            .filter(db.Solve.created_at <= solve.created_at).count())

# A slightly weird abstraction: show a session selector dialog (already created
# by the caller), and if the user confirms, merge all the solves in the given
# query into the selected session, all in a given db session. Returns whether
# the solves were merged.
def show_merge_dialog(session, session_selector, msg, query, exclude_session=None,
        allow_new=False):
    # Get relevant sessions to merge into
    sessions = session.query(db.Session)
    if exclude_session is not None:
        sessions = sessions.filter(db.Session.id != exclude_session)
    sessions = sorted(sessions.all(), key=session_sort_key)

    # Show dialog and merge sessions if accepted
    session_selector.update_data(msg, sessions, allow_new=allow_new)
    if session_selector.exec():
        # Create a new session if requested
        merge_id = session_selector.selected_session_id
        if isinstance(merge_id, str):
            merge_id = session.insert(db.Session, name=merge_id).id

        # Merge solves
        query.update({db.Solve.session_id: merge_id})

        # Clear stat cache
        new_sesh = session.query_first(db.Session, id=merge_id)
        new_sesh.cached_stats_current = None
        new_sesh.cached_stats_best = None
        return merge_id
    return None

# Smart cube analysis stuff

class SmartEvent:
    def __init__(self, ts=None, cube=None, turns=None, quat=None, face=None,
            turn=None, rotation=None, rot_n=None):
        self.ts = ts
        self.cube = cube
        self.turns = turns
        self.quat = quat
        self.face = face
        self.turn = turn
        self.rotation = rotation
        self.rot_n = rot_n

# This class is basically a bigass constructor
class SmartSolve:
    def __init__(self, solve, solve_nb):
        data = solve.smart_data_raw
        # Parse smart solve header
        [version, *values] = struct.unpack('<B4f4f6b', data[:39])
        assert version == 2
        base_quat = values[:4]
        quat = values[4:8]
        turns = values[8:]

        # Parse smart solve data into event list
        events = []
        data = bytearray(gzip.decompress(data[39:]))
        ts = 0
        while data:
            [b, ts_delta] = struct.unpack('<BH', data[:3])
            data = data[3:]
            ts += ts_delta
            if b & 0x80:
                # Convert lazy half-float to float
                [a, b, c, d] = struct.unpack('<4H', data[:8])
                M = 0x8000
                full = struct.pack('<8H', M, a, M, b, M, c, M, d)

                quat = struct.unpack('<ffff', full)
                data = data[8:]
                events.append((ts, quat, None, None))
            else:
                face = b >> 1
                assert 0 <= face < 6
                turn = [-1, 1][b & 1]
                events.append((ts, None, face, turn))

        # Create an updated list of events, where after each full turn the event
        # contains a new copy of a cube (like keyframes in videos for easy scrubbing).
        # index XXX figure out how to merge this with code in CubeWindow
        # without making a big mess of overabstraction
        cube = solver.Cube()
        cube.run_alg(solve.scramble)
        orient = list(range(6))
        moves = []
        last_face = None
        last_turn = 0
        last_ts = 0
        last_slice_face = None
        last_slice_turn = None
        last_slice_ts = None
        new_events = [SmartEvent(ts=0, cube=cube, turns=turns.copy(), quat=quat)]
        # Variables to track what the cube/turns were before an event
        for [i, [ts, quat, face, turn]] in enumerate(events):
            # Add the updated event
            new_events.append(SmartEvent(ts=ts, quat=quat, face=face, turn=turn))

            if face is not None:
                turns[face] += turn

                # 9 incremental turns make a full quarter turn
                if abs(turns[face]) >= 9:
                    turn = turns[face] // 9
                    turns[face] = 0

                    # Zero out everything but the opposite face as a sanity
                    # check. Use a threshold so that a partial turn doesn't
                    # mess up later turn accounting (if the turning is choppy,
                    # say, one turn might start before the last completes)
                    opp = face ^ 1
                    for f in range(6):
                        if f != opp and abs(turns[f]) > 4:
                            turns[f] = 0

                    alg = solver.FACE_STR[face] + solver.TURN_STR[turn]

                    # Merge moves of the same made within a short time window
                    # for purposes of reconstruction
                    if face == last_face and ts - last_ts < 500 and (last_turn + turn) % 4:
                        last_turn += turn
                        last_turn %= 4
                        moves[-1] = (solver.FACE_STR[orient[face]] +
                                solver.TURN_STR[last_turn])
                    # Likewise with opposite moves into slice moves, but with a
                    # smaller window
                    elif (face ^ 1 == last_face and turn == -last_turn and
                            abs(turn) == 1 and ts - last_ts < 100):
                        slice_str = solver.SLICE_STR[orient[face] >> 1]
                        # Merge with previous slice for M2, S2, E2
                        merge = False
                        if (last_slice_face is not None and
                                last_slice_face >> 1 == face >> 1 and
                                ts - last_slice_ts < 500):
                            if face != last_slice_face:
                                last_slice_turn += 2
                            last_slice_turn += turn
                            last_slice_turn %= 4
                            merge = last_slice_turn != 0
                        if merge:
                            moves[-2:] = [slice_str + solver.TURN_STR[last_slice_turn]]
                            last_slice_face = None
                        else:
                            last_face = None
                            last_slice_face = face
                            last_slice_turn = turn
                            last_slice_ts = ts
                            slice_turn = (turn + solver.SLICE_FLIP[face]) % 4
                            moves[-1] = slice_str + solver.TURN_STR[slice_turn]
                        # Fix up center orientation from the slice
                        rotation = solver.ROTATE_FACES[face >> 1]
                        if not face & 1:
                            turn += 2
                        for i in range(turn % 4):
                            orient = [orient[x] for x in rotation]
                    else:
                        last_face = face
                        last_turn = turn
                        # Look up current name of given face based on rotations
                        moves.append(solver.FACE_STR[orient[face]] +
                                solver.TURN_STR[turn])

                    last_ts = ts
                    if (last_slice_face is not None and
                            face >> 1 != last_slice_face >> 1):
                        last_slice_face = None

                    turns = turns.copy()
                    cube = cube.copy()
                    cube.run_alg(alg)

                    # Copy the new cube/turns to an event if they just changed
                    new_events.append(SmartEvent(ts=ts, cube=cube, turns=turns.copy()))

        self.scramble = solve.scramble
        self.base_quat = base_quat
        self.events = new_events
        self.solve_nb = solve_nb
        self.reconstruction = moves
        self.session_name = solve.session.name

        c = solver.Cube().run_alg(self.scramble)
        c.run_alg(' '.join(self.reconstruction))
        self.solved = (c == SOLVED_CUBE)

# Giant main class that handles the main window, receives bluetooth messages,
# deals with cube logic, etc.
class CubeWindow(QMainWindow):
    # Signal to just run a specified function in a Qt thread, since Qt really
    # cares deeply which thread you're using when starting timers etc.
    schedule_fn = pyqtSignal([object])
    schedule_fn_args = pyqtSignal([object, object])

    playback_events = pyqtSignal([list])

    bt_scan_result = pyqtSignal([str, object])
    bt_status_update = pyqtSignal([object])
    bt_connected = pyqtSignal([object])

    def __init__(self):
        super().__init__()
        self.timer = None
        self.pending_timer = None
        self.smart_device = None

        # Initialize DB and make sure there's a current session
        db.init_db(config.DB_PATH)
        with db.get_session() as session:
            settings = session.upsert(db.Settings, {})
            if not settings.current_session:
                sesh = session.insert(db.Session, name='New Session',
                        scramble_type='3x3')
                settings.current_session = sesh

        # Create basic layout/widgets. We do this first because the various
        # initialization functions can send data to the appropriate widgets to
        # update the view, and it's simpler to just always have them available
        self.gl_widget = GLWidget(self)
        self.scramble_widget = ScrambleWidget(self)
        self.scramble_view_widget = ScrambleViewWidget(self)
        self.instruction_widget = InstructionWidget(self)
        self.timer_widget = TimerWidget(self)
        self.session_widget = SessionWidget(self)
        self.smart_playback_widget = SmartPlaybackWidget(self)
        self.bt_status_widget = BluetoothStatusWidget(self)

        # Set up bluetooth. This doesn't actually scan or anything yet
        self.bt_handler = bluetooth.BluetoothHandler(self)

        self.bt_connection_dialog = BluetoothConnectionDialog(self, self.bt_handler)
        self.settings_dialog = SettingsDialog(self)

        main = QWidget()

        # Create an overlapping widget thingy so the scramble is above the timer
        timer_container = QWidget(main)
        timer_layout = QGridLayout(timer_container)
        timer_layout.addWidget(self.timer_widget, 0, 0)
        timer_layout.addWidget(self.scramble_view_widget, 0, 0,
                Qt.AlignRight | Qt.AlignTop)

        top = QWidget()
        top.setObjectName('top')
        make_hbox(top, [self.instruction_widget, self.smart_playback_widget,
                self.scramble_widget])

        right = QWidget()
        right.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        make_vbox(right, [top, self.gl_widget, timer_container])

        settings_button = QPushButton('Settings')
        settings_button.pressed.connect(self.settings_dialog.exec)
        bt_button = QPushButton('Bluetooth')
        bt_button.pressed.connect(self.bt_connection_dialog.exec)

        buttons = QWidget()
        make_vbox(buttons, [settings_button, bt_button])

        # Make grid and overlapping status widget and buttons
        grid = make_grid(main, [[self.session_widget, right]])
        grid.addWidget(self.bt_status_widget, 0, 1,
                Qt.AlignRight | Qt.AlignBottom)
        grid.addWidget(buttons, 0, 1, Qt.AlignRight | Qt.AlignTop)

        self.setCentralWidget(main)

        # Annoying: set up style here in the parent, so it can be overridden
        # in children
        self.setStyleSheet('TimerWidget { font: 240px Courier; }'
                'BluetoothStatusWidget { font: 24px; '
                '   color: #FFF; background: rgb(80,80,255); padding: 5px; }'
                '#top > * { padding-right: 100px; }')

        self.gen_scramble()

        self.update_state_ui()

        self.setWindowTitle('CubingB')
        self.setFocus()
        self.setFocusPolicy(Qt.StrongFocus)

        # Wire up signals
        self.schedule_fn.connect(self.run_scheduled_fn, type=Qt.QueuedConnection)
        self.schedule_fn_args.connect(self.run_scheduled_fn_args,
                type=Qt.QueuedConnection)
        self.playback_events.connect(self.play_events)
        self.bt_scan_result.connect(self.bt_connection_dialog.update_device)
        self.bt_status_update.connect(self.bt_status_widget.update_status)
        self.bt_connected.connect(self.got_bt_connection)

        # Initialize session state
        self.schedule_fn.emit(self.session_widget.trigger_update)

    def run_scheduled_fn(self, fn):
        fn()

    def run_scheduled_fn_args(self, fn, args):
        fn(*args)

    def got_bt_connection(self, device):
        self.smart_device = device
        self.bt_connection_dialog.set_device(device)
        # Move between smart scrambling and normal scrambling
        # XXX need more logic here?
        if device and self.state == State.SCRAMBLE:
            self.state = State.SMART_SCRAMBLING
        elif not device and self.state == State.SMART_SCRAMBLING:
            self.state = State.SCRAMBLE
        self.update_state_ui()

    def sizeHint(self):
        return QSize(*WINDOW_SIZE)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_C:
            self.calibrate_cube()
        elif event.key() == Qt.Key.Key_R:
            self.reset()
        elif event.key() == Qt.Key.Key_Space and self.state == State.SCRAMBLE:
            self.state = State.SOLVE_PENDING
            self.start_pending()
        elif event.key() == Qt.Key.Key_Escape and self.state == State.SOLVE_PENDING:
            self.state = State.SCRAMBLE
            self.stop_pending(False)
        # Any key stops a solve
        elif self.state == State.SOLVING:
            self.state = State.SCRAMBLE
            final_time = self.stop_solve(dnf=event.key() == Qt.Key.Key_Escape)
            self.stop_solve_ui(final_time)
        # Escape can DNF a smart solve
        elif event.key() == Qt.Key.Key_Escape and self.state == State.SMART_SOLVING:
            self.state = State.SMART_SCRAMBLING
            # Move smart data buffer to a different variable so UI
            # thread can read it, and remove the buffer so later events don't
            # try and modify it
            self.smart_data_copy = self.smart_cube_data
            self.smart_cube_data = None
            final_time = self.stop_solve(dnf=True)
            self.stop_solve_ui(final_time)
        else:
            event.ignore()
            return
        self.update_state_ui()

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key.Key_Space and self.state == State.SOLVE_PENDING:
            if time.time() - self.pending_start > TIMER_DEBOUNCE:
                self.state = State.SOLVING
                self.start_solve()
                self.start_solve_ui()
            else:
                self.state = State.SCRAMBLE
                self.stop_pending(False)
            self.update_state_ui()
        else:
            event.ignore()

    def update_timer(self):
        if self.start_time is not None:
            self.timer_widget.update_time(time.time() - self.start_time, 1)

    def calibrate_cube(self):
        self.gl_widget.base_quat = quat_invert(self.gl_widget.quat)

    # Change UI modes based on state
    def update_state_ui(self):
        # Hide things that are only shown conditionally below
        self.gl_widget.hide()
        self.smart_playback_widget.hide()
        self.scramble_widget.hide()
        self.scramble_view_widget.hide()
        self.instruction_widget.hide()
        self.timer_widget.hide()

        self.timer_widget.set_pending(False)

        if self.state == State.SCRAMBLE:
            self.scramble_widget.show()
            self.scramble_view_widget.show()
            self.timer_widget.show()
        elif self.state == State.SOLVE_PENDING:
            self.timer_widget.update_time(0, 3)
            self.timer_widget.show()
        elif self.state == State.SOLVING:
            self.timer_widget.show()
        elif self.state == State.SMART_SCRAMBLING:
            self.scramble_widget.show()
            self.gl_widget.show()
        elif self.state == State.SMART_SCRAMBLED:
            self.instruction_widget.setText("Start solving when you're ready!")
            self.instruction_widget.show()
            self.gl_widget.show()
        elif self.state == State.SMART_SOLVING:
            self.timer_widget.show()
            self.start_solve_ui()
        elif self.state == State.SMART_VIEWING:
            self.smart_playback_widget.show()
            self.gl_widget.show()

    def start_pending(self):
        self.pending_start = time.time()
        self.pending_timer = QTimer()
        self.pending_timer.timeout.connect(self.set_pending)
        self.pending_timer.start(int(1000 * TIMER_DEBOUNCE))

    def stop_pending(self, pending):
        if self.pending_timer:
            self.pending_timer.stop()
            self.pending_timer = None
        self.timer_widget.set_pending(pending)

    def set_pending(self):
        self.stop_pending(True)

    def gen_scramble(self):
        self.reset()
        if self.smart_device:
            self.state = State.SMART_SCRAMBLING
        else:
            self.state = State.SCRAMBLE
        self.scramble = []
        self.solve_moves = []
        self.start_time = None
        self.end_time = None

        all_faces = set(range(6))
        blocked_faces = set()
        turns = list(solver.TURN_STR.values())
        # Just do N random moves for now, not random state scrambles
        for i in range(SCRAMBLE_MOVES):
            face = random.choice(list(all_faces - blocked_faces))
            # Only allow one turn of each of an opposing pair of faces in a row.
            # E.g. F B' is allowed, F B' F is not
            if face ^ 1 not in blocked_faces:
                blocked_faces = set()
            blocked_faces.add(face)

            face = solver.FACE_STR[face]
            move = face + random.choice(turns)
            self.scramble.append(move)
        self.scramble_left = self.scramble[:]

        self.mark_changed()

    # Initialize some data to record smart solve data. We do this before starting
    # the actual solve, since that logic only fires once the first turn is completed
    def prepare_smart_solve(self):
        self.smart_cube_data = bytearray()
        self.smart_data_copy = None
        # Keep a copy of the initial turns so we can preserve any weird choppy
        # turning state, keeping the recording intact
        self.smart_start_turns = self.turns.copy()

        self.smart_start_quat = self.quat
        self.last_ts = 0

    def start_solve(self):
        self.start_time = time.time()
        self.end_time = None

    # Start UI timer to update timer view. This needs to be called from a Qt thread,
    # so it's scheduled separately when a bluetooth event comes in
    def start_solve_ui(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_timer)
        self.timer.start(100)

    def stop_solve(self, dnf=False):
        self.end_time = time.time()
        final_time = self.end_time - self.start_time
        self.start_time = None

        # Update database
        with db.get_session() as session:
            sesh = session.query_first(db.Settings).current_session
            # Create data blob, with header, base rotation, and compressed solve data
            data = None
            if self.smart_data_copy:
                data = gzip.compress(self.smart_data_copy, compresslevel=4)
                # Create header with initial solve state
                header = struct.pack('<B4f4f6b', SMART_DATA_VERSION,
                        *self.gl_widget.base_quat, *self.smart_start_quat,
                        *self.smart_start_turns)
                data = header + data

            session.insert(db.Solve, session=sesh,
                    scramble=' '.join(self.scramble),
                    time_ms=int(final_time * 1000), dnf=dnf,
                    smart_data_raw=data)

        self.gen_scramble()
        return final_time

    # Start UI timer to update timer view. This needs to be called from a Qt thread,
    # so it's scheduled separately when a bluetooth event comes in
    def start_solve_ui(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_timer)
        self.timer.start(100)

    def stop_solve_ui(self, final_time):
        # Stop UI timer
        if self.timer:
            self.timer.stop()
            self.timer = None
            self.timer_widget.update_time(final_time, 3)

        # Update session state
        self.session_widget.trigger_update()

    # Smart cube stuff

    def check_solved(self):
        return self.cube == SOLVED_CUBE

    def reset(self, playback=False):
        if playback:
            self.play_cube = solver.Cube()
            self.play_turns = [0] * 6
            self.play_quat = [1, 0, 0, 0]
        else:
            self.cube = solver.Cube()
            self.turns = [0] * 6
            self.quat = [1, 0, 0, 0]
            self.smart_cube_data = None
            self.smart_data_copy = None

    # Notify the cube widget that we've updated. We copy all the rendering
    # data to a new object so it can pick up a consistent view at its leisure
    def mark_changed(self, playback=False):
        # Ignore smart cube events during a playback
        if self.state == State.SMART_VIEWING and not playback:
            return

        # XXX copy only the stuff that's modified in place. Don't introduce
        # bugs here later OK
        if playback:
            cube = self.play_cube.copy()
            turns = self.play_turns[:]
            quat = self.play_quat
        else:
            cube = self.cube.copy()
            turns = self.turns[:]
            quat = self.quat

        self.scramble_widget.set_scramble(self.scramble, self.scramble_left)
        self.scramble_view_widget.set_scramble(self.scramble)
        self.gl_widget.set_render_data(cube, turns, quat)

    # Make a move and update any state for either a scramble or a solve
    def make_turn(self, face, turn):
        face = solver.FACE_STR[face]
        alg = face + solver.TURN_STR[turn]
        self.cube.run_alg(alg)
        old_state = self.state
        # Scrambling: see if this is the next move of the scramble
        if self.state == State.SMART_SCRAMBLING:
            # If this is the first move of the scramble, check if
            # we should auto-calibrate the cube
            if self.scramble_left == self.scramble:
                with db.get_session() as session:
                    settings = session.query_first(db.Settings)
                    if settings.auto_calibrate:
                        self.calibrate_cube()

            # Failsafe for weird bugs: if the cube is scrambled already, but we're
            # still in scrambling mode for some reason, allow the inverse move
            # to get added to the scramble (like an incorrect scramble move).
            # This shouldn't be necessary, but there have been two or three other
            # bugs that indirectly cause an IndexError below and it's annoying.
            # Belt and suspenders.
            if not self.scramble_left:
                s_face = s_turn = None
            else:
                s_face = self.scramble_left[0][0]
                s_turn = solver.INV_TURN_STR[self.scramble_left[0][1:]]

            if face == s_face:
                s_turn = (s_turn - turn) % 4
                if not s_turn:
                    self.scramble_left.pop(0)
                    # Done scrambling: start recording data
                    if not self.scramble_left:
                        self.state = State.SMART_SCRAMBLED
                        # Start a timer so we wait a bit to start the
                        # solve in case they overturn
                        self.smart_pending_start = time.time()
                else:
                    new_turn = solver.TURN_STR[s_turn]
                    self.scramble_left[0] = face + new_turn
            else:
                new_turn = solver.TURN_STR[-turn % 4]
                self.scramble_left.insert(0, face + new_turn)
        # If we're in a scrambled state, this move needs to move us to
        # unscrambled. If it started a solve, that would be handled earlier in
        # make_turn(), which checks the debounce timer. So if we're here
        # and in a scrambled state, add the inverse turn back to the scramble.
        elif self.state == State.SMART_SCRAMBLED:
            new_turn = solver.TURN_STR[-turn % 4]
            self.scramble_left.insert(0, face + new_turn)
            self.state = State.SMART_SCRAMBLING
        # Solving: check for a complete solve
        elif self.state == State.SMART_SOLVING:
            if self.check_solved():
                self.state = State.SMART_SCRAMBLING
                # Move smart data buffer to a different variable so UI
                # thread can read it, and remove the buffer so later events don't
                # try and modify it
                self.smart_data_copy = self.smart_cube_data
                self.smart_cube_data = None

                final_time = self.stop_solve()
                self.schedule_fn_args.emit(self.stop_solve_ui, (final_time,))

        # XXX handle after-solve-but-pre-scramble moves

        if self.state != old_state:
            self.schedule_fn.emit(self.update_state_ui)

    # Get a 16-bit timestamp in milliseconds for recording smart solve events.
    # The timestamps in the messages seem pretty wonky, so this is more robust
    def get_ts(self):
        if self.start_time is None:
            return 0
        # Only encode timestamp deltas for slightly better compressibility and
        # so we don't have to care about overflows. OK if someone pauses for
        # 2^16 milliseconds (64 seconds) during a solve it can overflow, but
        # I hope they don't need a smart cube to tell them to not do that
        ts = int((time.time() - self.start_time) * 1000)
        result = (ts - self.last_ts) & 0xFFFF
        self.last_ts = ts
        return result

    # XXX for now we use weilong units, 1/36th turns
    def update_turn(self, face, turn, ts):
        # Update state if this is the first partial turn after a scramble
        if self.state == State.SMART_SCRAMBLED:
            # ...and enough time has elapsed since the scramble finished
            now = time.time()
            if now - self.smart_pending_start > TIMER_DEBOUNCE:
                self.state = State.SMART_SOLVING
                self.prepare_smart_solve()
                self.start_solve()
                # Update state in a Qt thread
                self.schedule_fn.emit(self.update_state_ui)
            # Otherwise, restart the timer
            else:
                self.smart_pending_start = now

        # Record data if we're in a solve
        if self.smart_cube_data is not None:
            turn_byte = face * 2 + {-1: 0, 1: 1}[turn]
            data = struct.pack('<BH', turn_byte, self.get_ts())
            self.smart_cube_data.extend(data)

        # Add up partial turns
        self.turns[face] += turn

        # 9 incremental turns make a full quarter turn
        if abs(self.turns[face]) >= 9:
            turn = self.turns[face] // 9
            self.make_turn(face, turn)

            # Zero out everything but the opposite face as a sanity
            # check. Use a threshold so that a partial turn doesn't
            # mess up later turn accounting (if the turning is choppy,
            # say, one turn might start before the last completes)
            opp = face ^ 1
            for f in range(6):
                if f != opp and abs(self.turns[f]) > 4:
                    self.turns[f] = 0

        self.mark_changed()

    def update_rotation(self, quat, ts):
        # Record data if we're in a solve
        if self.smart_cube_data is not None:
            # Hacky half-float format: truncate the low bytes
            qdata = struct.pack('<ffff', *quat)
            qdata = [b for [i, b] in enumerate(qdata) if i & 2]

            data = struct.pack('<BH8B', 0xFF, self.get_ts(), *qdata)
            self.smart_cube_data.extend(data)

        self.quat = quat
        self.mark_changed()

    def play_events(self, events):
        quat = None
        for e in events:
            if e.cube:
                self.play_cube = e.cube.copy()
                self.play_turns = e.turns.copy()
            if e.face is not None:
                self.play_turns[e.face] += e.turn
        # Only show the last rotation, since they're not incremental
        if e.quat:
            self.play_quat = e.quat
        self.mark_changed(playback=True)

    def start_playback(self, solve_id, solve_nb):
        # Save the state we're in so we can go back after playback. Make sure
        # to not overwrite it if we're already in playback mode
        if self.state != State.SMART_VIEWING:
            self.prev_state = self.state

        self.state = State.SMART_VIEWING
        with db.get_session() as session:
            solve = session.query_first(db.Solve, id=solve_id)
            self.reset(playback=True)
            self.smart_playback_widget.update_solve(SmartSolve(solve, solve_nb))
        self.update_state_ui()

    def stop_playback(self):
        self.state = self.prev_state
        self.prev_state = None
        self.mark_changed()
        self.update_state_ui()

class SettingsDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent

        buttons = QDialogButtonBox(QDialogButtonBox.Ok)
        buttons.accepted.connect(self.accept)

        # Ugh python scoping etc etc, bind the local axis variable
        def create(axis):
            slider = QSlider()
            slider.setMinimum(-180)
            slider.setMaximum(180)

            # Icky: reach into parent then into gl_widget
            attr = 'view_rot_%s' % axis
            slider.setValue(getattr(self.parent.gl_widget, attr))

            slider.setOrientation(Qt.Horizontal)
            slider.valueChanged.connect(lambda v: self.update_rotation(axis, v))
            return slider

        sliders = [[QLabel('Rotation %s:' % axis.upper()), create(axis)]
                for axis in 'xyz']

        reset_button = QPushButton('Reset Camera')
        reset_button.pressed.connect(self.parent.gl_widget.reset_camera)

        # Initialize persistent settings
        with db.get_session() as session:
            settings = session.query_first(db.Settings)

            auto_cal = QCheckBox('Automatically calibrate smart cube gyro '
                    'when beginning a scramble')
            auto_cal.setChecked(settings.auto_calibrate)
            auto_cal.stateChanged.connect(self.set_auto_calibrate)

        make_grid(self, sliders + [
            [reset_button],
            [auto_cal],
            [buttons],
        ])

    def set_auto_calibrate(self, value):
        with db.get_session() as session:
            settings = session.query_first(db.Settings)
            settings.auto_calibrate = bool(value)

    def update_rotation(self, axis, value):
        # Icky: reach into parent then into gl_widget, then update it
        attr = 'view_rot_%s' % axis
        setattr(self.parent.gl_widget, attr, value)
        self.parent.gl_widget.update()

class BluetoothConnectionDialog(QDialog):
    def __init__(self, parent, bt_handler):
        super().__init__(parent)
        self.bt_handler = bt_handler

        scan = QPushButton('Scan')
        scan.pressed.connect(self.start_scan)

        self.status = QLabel()
        self.disconnect = QPushButton('Disconnect')
        self.disconnect.hide()
        self.disconnect.pressed.connect(self.disconnect_device)

        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderItem(0, cell('Name'))
        self.table.setHorizontalHeaderItem(1, cell('Status'))
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.cellDoubleClicked.connect(self.connect_device)

        ok = QDialogButtonBox(QDialogButtonBox.Ok)
        ok.accepted.connect(self.accept)

        make_grid(self, [
            [self.status, self.disconnect],
            [scan], [self.table], [ok]])

        self.timer = QTimer()
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.hide)

        self.devices = {}
        self.current_device = None

    def sizeHint(self):
        return QSize(400, 400)

    # Automatically start a scan when the settings are opened
    def exec(self):
        self.start_scan()
        return super().exec()

    def start_scan(self):
        self.devices = {}
        self.bt_handler.start_bt()

    def connect_device(self, row, col):
        device = self.table.item(row, 0).secret_data
        self.bt_handler.connect(device)

    def disconnect_device(self):
        self.bt_handler.disconnect_bt()

    def set_device(self, device):
        self.current_device = device
        if device:
            # XXX reaching into corebluetooth API
            self.status.setText('Connected to %s' % device.name())
            self.disconnect.show()
            self.timer.start(1000)
        else:
            self.status.setText('')
            self.disconnect.hide()
        self.render()

    def update_device(self, name, device):
        self.devices[name] = device
        self.render()

    def render(self):
        self.table.clearContents()
        self.table.setRowCount(len(self.devices))
        for [i, [name, device]] in enumerate(sorted(self.devices.items())):
            self.table.setItem(i, 0, cell(name, secret_data=device))
            status = 'Connected' if self.current_device == device else ''
            self.table.setItem(i, 1, cell(status))
        self.update()

class BluetoothStatusWidget(QLabel):
    def __init__(self, parent):
        super().__init__(parent)
        self.hide()

        # Set up fade out animation
        self.anim = QVariantAnimation(self)
        self.anim.setDuration(800)
        self.anim.setStartValue(100)
        self.anim.setEndValue(0)
        self.anim.valueChanged.connect(self.set_opacity)
        self.anim.finished.connect(self.hide)

        self.fade_timer = QTimer()
        self.fade_timer.setSingleShot(True)
        self.fade_timer.timeout.connect(self.fade)

    def set_opacity(self, value):
        self.setStyleSheet('background: rgba(80,80,255,%s%%);' % value)
        self.update()

    def update_status(self, status):
        self.setText(status)
        self.show()
        self.fade_timer.stop()
        self.fade_timer.start(3000)
        self.update()

    def fade(self):
        self.anim.start(QAbstractAnimation.KeepWhenStopped)

    def hide(self):
        super().hide()
        self.setStyleSheet('background: rgba(80,80,255,100%);')

class TimerWidget(QLabel):
    def __init__(self, parent):
        super().__init__(parent)
        # This shit should really be in the stylesheet, but not supported?!
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.update_time(0, 3)

    def set_pending(self, pending):
        color = 'red' if pending else 'black'
        self.setStyleSheet('color: %s;' % color)
        self.update()

    def update_time(self, t, prec):
        self.setText(ms_str(t * 1000, prec=prec))
        self.update()

class InstructionWidget(QLabel):
    def __init__(self, parent):
        super().__init__(parent)
        self.setStyleSheet('InstructionWidget { font: 40px; max-height: 100px }')
        # This shit should really be in the stylesheet, but not supported?!
        self.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.setWordWrap(True)

# Display the scramble moves

class ScrambleWidget(QLabel):
    def __init__(self, parent):
        super().__init__(parent)
        self.setStyleSheet('ScrambleWidget { font: 48px Courier; }')
        # This shit should really be in the stylesheet, but not supported?!
        self.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.setWordWrap(True)

    def set_scramble(self, scramble, scramble_left):
        offset = max(len(scramble) - len(scramble_left), 0)
        left = ['-'] * len(scramble)
        for i in range(min(len(left), len(scramble_left))):
            left[offset+i] = scramble_left[i]
        self.setText(' '.join('% -2s' % s for s in left))
        self.update()

class ScrambleViewWidget(QFrame):
    def __init__(self, parent):
        super().__init__(parent)
        self.scramble = None
        self.last_scramble = None

        # Set up a GL widget to render little scrambled cubes
        self.gl_widget = GLWidget(self)
        self.gl_widget.resize(200, 200)
        self.gl_widget.hide()

        # Build layout for scramble view
        label = QLabel('Scramble')
        label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.top_pic = QLabel()
        self.bottom_pic = QLabel()
        make_grid(self, [
            [label],
            [self.top_pic, self.bottom_pic],
        ])

        self.setStyleSheet('ScrambleViewWidget { background-color: #999; '
                'font: 24px; }')

    def set_scramble(self, scramble):
        self.scramble = scramble
        self.update()

    # Handle the paint event so we can use an OpenGL context to render the
    # scramble. The scramble can get updated from the bluetooth thread and
    # we can't render from there.
    def paintEvent(self, event):
        if self.scramble is not self.last_scramble:
            cube = solver.Cube()
            cube.run_alg(' '.join(self.scramble))
            for [pic, ax, ay] in [[self.top_pic, 30, 45],
                    [self.bottom_pic, -30, 225]]:
                self.gl_widget.set_render_data(cube, [0]*6, [1, 0, 0, 0])
                self.gl_widget.bg_color = [.6, .6, .6, 1]
                self.gl_widget.set_ortho(ax, ay)
                picture = self.gl_widget.grab()
                pic.setPixmap(picture)
            self.last_scramble = self.scramble

class SessionWidget(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.setStyleSheet('SessionWidget { max-width: 350px; }')

        self.label = QLabel('Session:')
        self.selector = QComboBox()
        self.selector.currentIndexChanged.connect(self.change_session)

        self.stats = QWidget()

        action = QAction('Move to session...', self)
        action.triggered.connect(self.move_selection)
        self.ctx_menu = QMenu(self)
        self.ctx_menu.addAction(action)

        self.table = QTableWidget()
        self.table.setStyleSheet('font: 16px')
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderItem(0, cell('Time'))
        self.table.setHorizontalHeaderItem(1, cell('ao5'))
        self.table.setHorizontalHeaderItem(2, cell('ao12'))
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.cellDoubleClicked.connect(self.edit_solve)
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self.show_ctx_menu)

        title = QWidget()
        make_hbox(title, [self.label, self.selector])
        self.layout = make_vbox(self, [title, self.stats, self.table])

        self.session_editor = SessionEditorDialog(self)
        self.session_selector = SessionSelectorDialog(self)
        self.solve_editor = SolveEditorDialog(self)
        self.average_viewer = AverageDialog(self)
        self.graph_viewer = None

    def edit_solve(self, row, col):
        solve_id = self.table.item(row, 0).secret_data
        # Change behavior based on whether the click is on the single/ao5/ao12
        if col == 0:
            self.solve_editor.update_solve(solve_id)
            # Start a DB session here just so we can rollback if the user cancels
            with db.get_session() as session:
                if self.solve_editor.exec():
                    self.trigger_update()
                else:
                    session.rollback()
        else:
            n_solves = 5 if col == 1 else 12
            self.average_viewer.update_solve(solve_id, n_solves)
            self.average_viewer.exec()

    def show_graph(self):
        with db.get_session() as session:
            sesh = session.query_first(db.Settings).current_session
            if self.graph_viewer is None:
                self.graph_viewer = GraphDialog(self)
            self.graph_viewer.update_data({sesh.name: sesh.solves})
            self.graph_viewer.exec()

    def show_ctx_menu(self, pos):
        self.ctx_menu.popup(self.table.viewport().mapToGlobal(pos))

    def move_selection(self):
        with db.get_session() as session:
            rows = {item.row() for item in self.table.selectedItems()}
            solves = [self.table.item(row, 0).secret_data for row in rows]
            msg = 'Move %s solve(s) to session:' % len(solves)
            query = session.query(db.Solve).filter(db.Solve.id.in_(solves))

            merge_id = show_merge_dialog(session, self.session_selector, msg,
                    query, allow_new=True)
            if merge_id is not None:
                self.trigger_update()

    def change_session(self, index):
        with db.get_session() as session:
            settings = session.upsert(db.Settings, {})

            id = self.session_ids[index]
            if id == 'new':
                sesh = session.insert(db.Session, name='New Session',
                        scramble_type='3x3')
                settings.current_session = sesh
            elif id == 'edit':
                self.session_editor.update_items()
                if not self.session_editor.exec():
                    session.rollback()
            else:
                settings.current_session = session.query_first(db.Session, id=id)
            session.flush()
            self.trigger_update()

    def trigger_update(self):
        with db.get_session() as session:
            sesh = session.query_first(db.Settings).current_session

            # Block signal handler so we don't trigger a recursive update
            self.selector.blockSignals(True)

            # Set up dropdown
            self.selector.clear()
            self.session_ids = []
            sessions = session.query_all(db.Session)
            sessions = sorted(sessions, key=session_sort_key)
            for s in sessions:
                self.session_ids.append(s.id)
                self.selector.addItem(s.name)
                if s.id == sesh.id:
                    self.selector.setCurrentIndex(len(self.session_ids) - 1)
            for cmd in ['new', 'edit']:
                self.session_ids.append(cmd)
                self.selector.addItem(cmd.title() + '...')

            self.selector.blockSignals(False)

            # Get solves
            solves = (session.query(db.Solve).filter_by(session=sesh)
                    .order_by(db.Solve.created_at.desc()).all())

            self.table.clearContents()
            self.table.setRowCount(0)

            # Clear the stats
            self.layout.removeWidget(self.stats)
            self.stats = QWidget()
            self.stats.setStyleSheet('QLabel { font: 16px; }')
            self.layout.insertWidget(1, self.stats)

            if not solves:
                self.update()
                return

            graph_button = QPushButton('Graph')
            graph_button.pressed.connect(self.show_graph)

            stat_table = [[graph_button, QLabel('current'), QLabel('best')]]

            # Calculate statistics

            all_times = [solve_time(s) for s in solves]

            stats_current = sesh.cached_stats_current or {}
            stats_best = sesh.cached_stats_best or {}
            for [stat_idx, size] in enumerate(STAT_AO_COUNTS):
                label = stat_str(size)
                mean = calc_ao(solves, 0, size)
                stats_current[label] = mean

                # Update best stats, recalculating if necessary
                if label not in stats_best:
                    best = None

                    averages = None
                    if size > 1:
                        averages = iter(calc_rolling_ao(solves, all_times, size))

                    for i in range(len(solves)):
                        if size > 1:
                            m = next(averages)
                        else:
                            m = all_times[i]

                        # Update rolling cache stats
                        if solves[i].cached_stats is None:
                            solves[i].cached_stats = {}
                        solves[i].cached_stats[label] = m

                        if not best or (m and m < best):
                            best = m
                    stats_best[label] = best
                else:
                    best = stats_best[label]
                    if not best or mean < best:
                        best = stats_best[label] = mean

                stat_table.append([QLabel(label), QLabel(ms_str(mean)),
                        QLabel(ms_str(best))])

            make_grid(self.stats, stat_table)

            sesh.cached_stats_current = stats_current
            sesh.cached_stats_best = stats_best
            solves[0].cached_stats = stats_current.copy()

            # Build the table of actual solves
            self.table.setRowCount(len(solves))

            for [i, solve] in enumerate(solves):
                self.table.setVerticalHeaderItem(i,
                        cell('%s' % (len(solves) - i)))
                # HACK: set the secret data so the solve editor gets the ID
                self.table.setItem(i, 0, cell(ms_str(solve_time(solve)),
                        secret_data=solve.id))
                stats = solve.cached_stats or {}
                self.table.setItem(i, 1,
                        cell(ms_str(stats.get('ao5'))))
                self.table.setItem(i, 2,
                        cell(ms_str(stats.get('ao12'))))

            self.update()

class SolveEditorDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        self.session_label = QLabel()
        self.time_label = QLabel()
        self.result_label = QLabel()
        self.scramble_label = QLabel()
        self.scramble_label.setWordWrap(True)
        self.notes = QPlainTextEdit()
        self.notes.textChanged.connect(lambda: self.make_edit('notes',
                self.notes.toPlainText()))
        self.smart_widget = None

        self.dnf = QCheckBox('DNF')
        self.dnf.stateChanged.connect(lambda v: self.make_edit('dnf', bool(v)))
        self.plus_2 = QCheckBox('+2')
        self.plus_2.stateChanged.connect(lambda v: self.make_edit('plus_2', bool(v)))

        delete = QPushButton('Delete')
        delete.pressed.connect(self.delete_solve)
        merge = QPushButton('Move to session...')
        merge.pressed.connect(self.move_solve)

        self.layout = make_grid(self, [
            [QLabel('Session:'), self.session_label],
            [QLabel('Time:'), self.time_label],
            [QLabel('Result:'), self.result_label],
            [QLabel('Scramble:'), self.scramble_label],
            [QLabel('Notes:'), self.notes],
            [QLabel('Smart data:'), self.smart_widget],
            [self.dnf, self.plus_2],
            [delete, merge],
            [buttons],
        ])

        self.session_selector = SessionSelectorDialog(self)

        self.solve_id = None

    def make_edit(self, key, value):
        with db.get_session() as session:
            # Edit solve
            solve = session.query_first(db.Solve, id=self.solve_id)
            setattr(solve, key, value)

            self.result_label.setText(solve_time_str(solve))

            # Invalidate statistics on session
            # XXX might need to invalidate individual solve stats later, but
            # for now when the 'best' stat cache is cleared it recalculates all
            # solves
            solve.session.cached_stats_current = None
            solve.session.cached_stats_best = None

    def start_playback(self, solve_id, solve_nb):
        # HACK
        window = self.parent()
        while not isinstance(window, CubeWindow):
            window = window.parent()
        window.schedule_fn_args.emit(window.start_playback, (solve_id, solve_nb))
        self.accept()

    def update_solve(self, solve_id):
        self.solve_id = solve_id
        with db.get_session() as session:
            solve = session.query_first(db.Solve, id=solve_id)
            self.solve = solve
            self.dnf.setChecked(solve.dnf)
            self.plus_2.setChecked(solve.plus_2)
            self.session_label.setText(solve.session.name)
            self.notes.setPlainText(solve.notes)
            self.scramble_label.setText(solve.scramble)
            self.time_label.setText(str(solve.created_at))
            self.result_label.setText(solve_time_str(solve))
            # Show a playback button if there's smart data
            if self.smart_widget:
                self.layout.removeWidget(self.smart_widget)
            if solve.smart_data_raw:
                solve_nb = get_solve_nb(session, solve)
                self.smart_widget = QPushButton('View Playback')
                self.smart_widget.pressed.connect(
                        lambda: self.start_playback(solve_id, solve_nb))
            else:
                self.smart_widget = QLabel('None')
            self.layout.addWidget(self.smart_widget, 5, 1)
        self.update()

    def delete_solve(self):
        response = QMessageBox.question(self, 'Confirm delete',
                'Are you sure you want to delete this solve? This cannot '
                ' be undone!', QMessageBox.Ok | QMessageBox.Cancel)
        if response == QMessageBox.Ok:
            with db.get_session() as session:
                solve = session.query_first(db.Solve, id=self.solve_id)
                solve.session.cached_stats_current = None
                solve.session.cached_stats_best = None

                session.query(db.Solve).filter_by(id=self.solve_id).delete()
            self.accept()

    def move_solve(self):
        with db.get_session() as session:
            solve = session.query_first(db.Solve, id=self.solve_id)
            msg = 'Move solve %s to session:' % get_solve_nb(session, solve)
            query = session.query(db.Solve).filter_by(id=self.solve_id)

            merge_id = show_merge_dialog(session, self.session_selector, msg,
                    query, exclude_session=solve.session_id, allow_new=True)
            if merge_id is not None:
                self.accept()

    def sizeHint(self):
        return QSize(400, 100)

class AverageDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok)
        buttons.accepted.connect(self.accept)

        self.result_label = QLabel()
        self.result_label.setWordWrap(True)

        make_grid(self, [
            [QLabel('Result:'), self.result_label],
            [buttons],
        ])

    def update_solve(self, solve_id, n_solves):
        self.solve_id = solve_id
        with db.get_session() as session:
            # Get solves
            solve = session.query_first(db.Solve, id=solve_id)

            solves = (session.query(db.Solve).filter_by(session=solve.session)
                    .filter(db.Solve.created_at <= solve.created_at)
                    .order_by(db.Solve.created_at.desc()).limit(n_solves).all())
            solves = solves[::-1]

            self.result_label.setText(get_ao_str(solves, 0, n_solves))
        self.update()

    def sizeHint(self):
        return QSize(400, 100)

GRAPH_TYPES = ['adaptive', 'date', 'count']

class GraphDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent)

        # Only import this crap here since it's pretty slow to import, and
        # this widget is instantiated lazily, all to improve startup time
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        self.title = QLabel()
        self.title.setStyleSheet('font: 24px')
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        select_label = QLabel('Graph Type:')
        self.selector = QComboBox()
        for t in GRAPH_TYPES:
            self.selector.addItem(t)
        self.selector.currentIndexChanged.connect(self.change_type)

        self.figure = Figure()
        self.canvas = FigureCanvasQTAgg(self.figure)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok)
        buttons.accepted.connect(self.accept)

        self.type = 'adaptive'
        self.stat = 'ao100'
        self.solves = None

        make_grid(self, [
            [None, None, self.title, select_label, self.selector],
            [self.canvas],
            [buttons],
        ], stretch=[0, 0, 1, 0, 0])

    def change_type(self):
        self.type = GRAPH_TYPES[self.selector.currentIndex()]
        self.render()

    def update_data(self, solve_sets, stat='single'):
        self.stat = stat
        self.solve_sets = {name: [(s.created_at, s.cached_stats) for s in ss]
                for [name, ss] in solve_sets.items()}

        self.render()

    def render(self):
        if len(self.solve_sets) == 1:
            for name in self.solve_sets.keys():
                self.title.setText('%s: %s' % (name, self.stat))
        else:
            self.title.setText('%s sessions: %s' % (len(self.solve_sets), self.stat))

        self.figure.clear()
        plot = self.figure.add_subplot()

        # Preprocessing for different graph types
        if self.type == 'count':
            maxlen = max(len(s) for s in self.solve_sets.values())
        elif self.type == 'adaptive':
            solves_per_day = collections.Counter(d.date()
                for [name, solves] in self.solve_sets.items() for [d, s] in solves)
            day_ordinal = {d: i for [i, d] in enumerate(sorted(solves_per_day))}
            first_day = min(solves_per_day)

        # Create plot for each session's solves
        for [name, solves] in self.solve_sets.items():
            # Create x series based on graph type

            # Date: just use the solve time
            if self.type == 'date':
                x = [d for [d, s] in solves]

            # Count: solve number, but right aligned among all sessions
            elif self.type == 'count':
                x = range(maxlen - len(solves), maxlen)

            # Adaptive: for every day that had solves, stretch all solves out
            # evenly throughout the day
            elif self.type == 'adaptive':
                sesh_solves_per_day = collections.Counter(d.date()
                        for [d, s] in solves)

                x = []
                last_day = None
                dc = 0
                for [d, s] in solves:
                    d = d.date()
                    if last_day and d > last_day:
                        dc = 0
                    last_day = d
                    x.append(day_ordinal[d] +
                            dc / sesh_solves_per_day[d])
                    dc += 1

            # Create y series from the given stat
            y = [s[self.stat] / 1000 if s[self.stat] else None
                    for [d, s] in solves]

            plot.plot(x, y, '.', label=name, markersize=1)

        plot.legend(loc='upper right')
        self.canvas.draw()

# Based on https://stackoverflow.com/a/43789304
class ReorderTableWidget(QTableWidget):
    def __init__(self, parent, reorder_cb):
        super().__init__(parent)

        self.reorder_cb = reorder_cb

        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.viewport().setAcceptDrops(True)
        self.setDragDropOverwriteMode(False)
        self.setDropIndicatorShown(True)

        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setDragDropMode(QAbstractItemView.InternalMove)

    def dropEvent(self, event):
        if not event.isAccepted() and event.source() == self:
            drop_row = self.get_drop_row(event)

            rows = list(sorted({item.row() for item in self.selectedItems()}))

            # Make a copy of all items in the selected rows. We need to copy
            # over the wacky hidden attributes we sometimes use, too...
            def copy_cell(item):
                cell = QTableWidgetItem(item)
                cell.secret_data = getattr(item, 'secret_data', None)
                return cell

            rows_to_move = [[copy_cell(self.item(row, column))
                    for column in range(self.columnCount())] for row in rows]
            headers = [copy_cell(self.verticalHeaderItem(row))
                    for row in rows]

            # Disconnect signals so we don't get spurious edit signals
            self.blockSignals(True)

            # Remove the old rows
            for row in reversed(rows):
                self.removeRow(row)
                if row < drop_row:
                    drop_row -= 1

            # Re-insert old items and select them
            for [row_idx, [row, header]] in enumerate(zip(rows_to_move, headers)):
                row_idx += drop_row
                self.insertRow(row_idx)
                self.setVerticalHeaderItem(row_idx, header)
                for [column, cell] in enumerate(row):
                    self.setItem(row_idx, column, cell)
                    self.item(row_idx, column).setSelected(True)

            self.blockSignals(False)

            # Notify parent of reordering
            self.reorder_cb()

            event.accept()
        else:
            event.ignore()

    def get_drop_row(self, event):
        index = self.indexAt(event.pos())
        if not index.isValid():
            return self.rowCount()
        return index.row() + 1 if self.is_below(event.pos(), index) else index.row()

    def is_below(self, pos, index):
        rect = self.visualRect(index)
        margin = 2
        if pos.y() - rect.top() < margin:
            return False
        elif rect.bottom() - pos.y() < margin:
            return True
        return (rect.contains(pos, True) and pos.y() >= rect.center().y() and
                not self.model().flags(index) & Qt.ItemIsDropEnabled)

class SessionEditorDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent)

        # Create context menu items for graphing stats
        self.ctx_menu = QMenu(self)
        for s in STAT_AO_COUNTS:
            # Meh, bind local stat variable
            def create(stat):
                action = QAction('Graph %s' % stat, self)
                action.triggered.connect(lambda: self.graph_selection(stat))
                self.ctx_menu.addAction(action)
            create(stat_str(s))

        self.table = ReorderTableWidget(self, self.rows_reordered)
        self.table.setColumnCount(4 + len(STAT_AO_COUNTS))
        self.table.setHorizontalHeaderItem(0, cell('Name'))
        self.table.setHorizontalHeaderItem(1, cell('Scramble'))
        self.table.setHorizontalHeaderItem(2, cell('# Solves'))
        for [i, stat] in enumerate(STAT_AO_COUNTS):
            self.table.setHorizontalHeaderItem(3+i, cell(stat_str(stat)))
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self.show_ctx_menu)

        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        self.table.itemChanged.connect(self.edit_attr)

        button = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button.accepted.connect(self.accept_edits)
        button.rejected.connect(self.reject)

        make_vbox(self, [self.table, button])

        self.session_selector = SessionSelectorDialog(self)
        self.graph_viewer = None

    def show_ctx_menu(self, pos):
        self.ctx_menu.popup(self.table.viewport().mapToGlobal(pos))

    def graph_selection(self, stat):
        with db.get_session() as session:
            rows = {item.row() for item in self.table.selectedItems()}
            solve_sets = {}
            for row in rows:
                id = self.table.item(row, 0).secret_data[0] 
                sesh = session.query_first(db.Session, id=id)
                solve_sets[sesh.name] = sesh.solves

            if self.graph_viewer is None:
                self.graph_viewer = GraphDialog(self)
            self.graph_viewer.update_data(solve_sets, stat=stat)
            self.graph_viewer.exec()

    def edit_attr(self, item):
        [id, attr] = item.secret_data
        with db.get_session() as session:
            sesh = session.query_first(db.Session, id=id)
            setattr(sesh, attr, item.text())

    def rows_reordered(self):
        with db.get_session() as session:
            for row in range(self.table.rowCount()):
                item = self.table.item(row, 0)
                sesh = session.query_first(db.Session, id=item.secret_data[0])
                sesh.sort_id = row + 1

        # Rebuild the table so it can get new buttons. We can't easily copy a
        # widget, so the ReorderTableWidget class just leaves them blank
        self.update_items()

    def accept_edits(self):
        # Make sure to change focus to the window in case the user clicked OK
        # while still editing. Without the focus change, the itemChanged signal
        # is fired after we accept changes, so the last edit would go in a
        # separate DB transaction, ugh
        self.setFocus()

        self.accept()

    def sizeHint(self):
        return QSize(700, 800)

    def merge_sessions(self, session_id):
        with db.get_session() as session:
            sesh = session.query_first(db.Session, id=session_id)
            msg = 'Merge %s into session:' % sesh.name
            query = session.query(db.Solve).filter_by(session_id=session_id)
            merge_id = show_merge_dialog(session, self.session_selector,
                    msg, query, exclude_session=session_id)
            if merge_id is not None:
                assert merge_id != session_id
                # Delete old session
                session.query(db.Session).filter_by(id=session_id).delete()

                # Update current session if necessary
                settings = session.query_first(db.Settings)
                if settings.current_session_id == session_id:
                    settings.current_session_id = merge_id

                self.update_items()

    def update_items(self):
        # Disconnect signals so we don't get spurious edit signals
        self.table.blockSignals(True)

        self.table.clearContents()
        self.table.setRowCount(0)
        with db.get_session() as session:
            # Get all sessions along with their number of solves. Kinda tricky.
            # We could do len(sesh.solves), but that's real slow
            stmt = (session.query(db.Solve.session_id,
                    db.sa.func.count('*').label('n_solves'))
                    .group_by(db.Solve.session_id).subquery())
            sessions = (session.query(db.Session, stmt.c.n_solves)
                    .outerjoin(stmt, db.Session.id == stmt.c.session_id).all())
            sessions = list(sorted(sessions, key=lambda s: session_sort_key(s[0])))

            self.table.setRowCount(len(sessions))
            for [i, [sesh, n_solves]] in enumerate(sessions):
                stats = sesh.cached_stats_best or {}
                sesh_id = session_sort_key(sesh)
                self.table.setVerticalHeaderItem(i, cell(str(sesh_id)))
                self.table.setItem(i, 0, cell(sesh.name, editable=True,
                        secret_data=(sesh.id, 'name')))
                self.table.setItem(i, 1, cell(sesh.scramble_type, editable=True,
                        secret_data=(sesh.id, 'scramble_type')))
                self.table.setItem(i, 2, cell(str(n_solves)))
                for [j, stat] in enumerate(STAT_AO_COUNTS):
                    stat = stat_str(stat)
                    self.table.setItem(i, 3+j,
                            cell(ms_str(stats.get(stat))))
                offset = 3 + len(STAT_AO_COUNTS)
                # Ugh, Python scoping: create a local function to capture the
                # current session ID
                def bind_button(s_id):
                    button = QPushButton('Merge...')
                    button.pressed.connect(lambda: self.merge_sessions(s_id))
                    return button
                self.table.setCellWidget(i, offset+0, bind_button(sesh.id))

        self.table.blockSignals(False)

class SmartPlaybackWidget(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent

        self.title = QLabel()
        self.title.setStyleSheet('QLabel { font: 24px; }')
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.play_button = QPushButton('Play')
        self.stop_button = QPushButton('Exit')
        self.current_time_label = QLabel()
        self.end_time_label = QLabel()
        self.play_button.pressed.connect(self.play_pause)
        self.stop_button.pressed.connect(self.stop_playback)

        slider = QSlider()
        slider.setOrientation(Qt.Horizontal)
        slider.valueChanged.connect(self.scrub)
        self.slider = slider

        grid = make_grid(self, [
            [None] * 6, # Placeholder for title
            [None, self.current_time_label, self.end_time_label, self.play_button,
                None, self.stop_button],
            [slider]
        ], stretch=[1, 0, 0, 0, 1, 0], widths=[0, 60, 60, 80, 0, 80])
        # Title takes up the middle three non-stretchy columns
        grid.addWidget(self.title, 0, 1, 1, 3)

        self.timer = QTimer()
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.play_event)

        self.events = None
        self.event_idx = None
        self.playing = False
        self.base_time = 0
        self.base_ts = 0

    def set_playing(self, playing):
        self.playing = playing
        if playing:
            self.play_button.setText('Pause')
            # Get current time and first timestamp so we can keep in sync
            self.base_time = time.time()
            if self.event_idx >= len(self.events):
                self.event_idx = 0
            self.base_ts = self.events[self.event_idx].ts
            self.play_event()
        else:
            self.play_button.setText('Play')
            self.timer.stop()
            self.base_ts = 0

    def play_pause(self):
        self.set_playing(not self.playing)

    def play_event(self):
        # Grab all the events that should have happened by this point (at least one)
        max_ts = self.base_ts + (time.time() - self.base_time) * 1000
        ts = self.events[self.event_idx].ts
        events = [self.events[self.event_idx]]
        self.event_idx += 1
        while self.event_idx < len(self.events):
            event = self.events[self.event_idx]
            if event.ts > max_ts:
                break
            events.append(event)
            self.event_idx += 1

        self.parent.playback_events.emit(events)

        self.slider.blockSignals(True)
        self.slider.setValue(self.event_idx)
        self.slider.blockSignals(False)

        self.current_time_label.setText(ms_str(ts))

        if self.playing:
            # Stop playing if we're done
            if self.event_idx >= len(self.events):
                self.set_playing(False)
            # Otherwise, schedule next event
            else:
                ts = self.events[self.event_idx].ts
                diff_ts = ts - self.base_ts
                diff_time = (time.time() - self.base_time) * 1000
                next_time = diff_ts - diff_time
                self.timer.start(max(0, int(next_time)))

    def stop_playback(self):
        self.parent.schedule_fn.emit(self.parent.stop_playback)

    def scrub(self, event_idx):
        # Search for the last cube state before this event ("keyframe"), then
        # play forward from there til the index
        i = event_idx
        while i > 0 and self.events[i].cube is None:
            i -= 1
        events = self.events[i:event_idx+1]
        self.event_idx = event_idx

        if events:
            ts = events[-1].ts
            self.base_time = time.time()
            self.base_ts = ts
            self.current_time_label.setText(ms_str(ts))

            self.parent.playback_events.emit(events)
        else:
            self.set_playing(False)

    def update_solve(self, solve):
        self.playing = False
        self.event_idx = 0
        self.solve = solve
        self.events = solve.events
        self.title.setText('Solve %s (%s)' % (solve.solve_nb, solve.session_name))

        self.slider.setMaximum(len(self.events) - 1)

        # Meh, sometimes the last event has a zero timestamp
        end_ts = max(self.events[-1].ts, self.events[-2].ts)
        self.end_time_label.setText('/ %s' % ms_str(end_ts))

        # Play the first event to set the scramble
        self.parent.playback_events.emit(solve.events[:1])

        # Ick, reach into parent, then into gl_widget to set this...
        self.parent.gl_widget.base_quat = solve.base_quat

class SessionSelectorDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent)

        self.label = QLabel('Session:')
        self.selector = QComboBox()

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept_session)
        buttons.rejected.connect(self.reject)

        top = QWidget()
        make_hbox(top, [self.label, self.selector])
        make_vbox(self, [top, buttons])

        self.selected_session = None
        self.session_ids = []

    def update_data(self, message, sessions, allow_new=False):
        self.selected_session = None

        self.label.setText(message)
        self.selector.clear()
        self.session_ids = []
        for s in sessions:
            self.session_ids.append(s.id)
            self.selector.addItem(s.name)
        if allow_new:
            self.session_ids.append(None)
            self.selector.addItem('New...')

    def accept_session(self):
        i = self.selector.currentIndex()
        self.selected_session_id = self.session_ids[i]
        if self.selected_session_id is None:
            [name, accepted] = QInputDialog.getText(self, 'New session name',
                    'Enter new session name:', text='New Session')
            if not accepted:
                self.reject()
            self.selected_session_id = name
        self.accept()

# Display the cube

class GLWidget(QOpenGLWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.reset()
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def reset(self):
        self.gl_init = False
        self.quat = self.base_quat = [1, 0, 0, 0]
        self.zoom = -4
        self.view_rot_x = 30
        self.view_rot_y = 30
        self.view_rot_z = 0
        self.size = None
        self.ortho = None
        self.drag_start_vector = None
        self.bg_color = [.7, .7, .7, 1]
        self.cam_quat = [1, 0, 0, 0]
        self.cam_base = [1, 0, 0, 0]

    def set_render_data(self, cube, turns, quat):
        self.cube = cube
        self.turns = turns
        self.quat = quat
        self.update()

    def set_ortho(self, ax, ay):
        self.ortho = [ax, ay]

    def initializeGL(self):
        self.gl_init = True
        render.setup()

    def resizeGL(self, w, h):
        self.size = [w, h]

    def paintGL(self):
        render.BG_COLOR = self.bg_color
        render.reset()

        if self.ortho is not None:
            render.set_ortho(self.size, *self.ortho)
        else:
            render.set_persective(self.size, self.zoom)

            q = quat_mul(self.cam_quat, self.cam_base)
            matrix = quat_matrix(quat_normalize(q))
            render.rotate_camera(matrix)

            q = quat_mul(self.base_quat, self.quat)
            matrix = quat_matrix(quat_normalize(q))

            render.set_rotation(matrix)

            render.glRotatef(self.view_rot_x, 1, 0, 0)
            render.glRotatef(self.view_rot_y, 0, -1, 0)
            render.glRotatef(self.view_rot_z, 0, 0, 1)

        render.render_cube(self.cube, self.turns)

    def reset_camera(self):
        self.cam_quat = [1, 0, 0, 0]
        self.cam_base = [1, 0, 0, 0]
        self.update()

    def mousePressEvent(self, event):
        pos = event.windowPos()
        [x, y, z] = render.gluUnProject(pos.x(), pos.y(), .1)
        self.drag_start_vector = (x, -y, z)

    def mouseMoveEvent(self, event):
        pos = event.windowPos()
        v1 = self.drag_start_vector
        [x, y, z] = render.gluUnProject(pos.x(), pos.y(), .1)
        v2 = (x, -y, z)
        v1, v2 = v2, v1

        if v1 == v2:
            self.cam_quat = [1, 0, 0, 0]
        else:
            cross = vec_cross_prod(v1, v2)
            w = (vec_len_sq(v1) * vec_len_sq(v2)) ** .5 + vec_dot_prod(v1, v2)
            self.cam_quat = quat_normalize([w, *cross])

        self.update()

    def mouseReleaseEvent(self, event):
        # Just multiply the current base
        self.cam_base = quat_normalize(quat_mul(self.cam_quat, self.cam_base))
        self.cam_quat = [1, 0, 0, 0]
        self.update()

    def wheelEvent(self, event):
        delta = -event.pixelDelta().y()
        self.zoom *= 1.002 ** delta
        self.zoom = min(-1, max(-25, self.zoom))
        self.update()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setApplicationName('CubingB')
    icon = QIcon('pix/cubingb-icon-small.png')
    app.setWindowIcon(icon)
    window = CubeWindow()
    window.show()
    sys.exit(app.exec_())
