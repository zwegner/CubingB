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

import enum
import gzip
import random
import struct
import sys
import time

from PyQt5.QtCore import QSize, Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (QApplication, QMainWindow, QHBoxLayout, QVBoxLayout,
        QWidget, QOpenGLWidget, QLabel, QTableWidget, QTableWidgetItem,
        QSizePolicy, QGridLayout, QComboBox, QDialog, QDialogButtonBox,
        QAbstractItemView, QHeaderView, QFrame, QCheckBox, QPushButton,
        QSlider)
from PyQt5.QtGui import QIcon

import bluetooth
import config
import db
import render
import solver

# Constants

WINDOW_SIZE = [1600, 1000]

# Should make UI for this
USE_SMART_CUBE = False

State = enum.Enum('State', 'SCRAMBLE SOLVE_PENDING SOLVING SMART_SCRAMBLING '
        'SMART_SCRAMBLED SMART_SOLVING SMART_VIEWING')

SOLVED_CUBE = solver.Cube()

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

def cell(text, editable=False):
    item = QTableWidgetItem(text)
    if not editable:
        item.setFlags(item.flags() & ~Qt.ItemIsEditable)
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

# Smart cube analysis stuff

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
        new_events = [[0, cube, turns.copy(), quat, None, None]]
        # Variables to track what the cube/turns were before an event
        for [i, [ts, quat, face, turn]] in enumerate(events):
            # Add the updated event
            new_events.append([ts, None, None, quat, face, turn])

            if face is not None:
                turns[face] += turn

                # 9 incremental turns make a full quarter turn
                if abs(turns[face]) >= 9:
                    turn = turns[face] // 9

                    # Zero out everything but the opposite face as a sanity
                    # check. Use a threshold so that a partial turn doesn't
                    # mess up later turn accounting (if the turning is choppy,
                    # say, one turn might start before the last completes)
                    opp = face ^ 1
                    for f in range(6):
                        if f != opp and abs(turns[f]) > 4:
                            turns[f] = 0

                    alg = solver.FACE_STR[face] + solver.TURN_STR[turn]

                    turns = turns.copy()
                    cube = cube.copy()
                    cube.run_alg(alg)

                    # Copy the new cube/turns to an event if they just changed
                    new_events.append([ts, cube, turns.copy(), None, None, None])

        self.scramble = solve.scramble
        self.base_quat = base_quat
        self.events = new_events
        self.solve_nb = solve_nb
        self.session_name = solve.session.name

# Giant main class that handles the main window, receives bluetooth messages,
# deals with cube logic, etc.
class CubeWindow(QMainWindow):
    # Signal to just run a specified function in a Qt thread, since Qt really
    # cares deeply which thread you're using when starting timers etc.
    schedule_fn = pyqtSignal([object])
    schedule_fn_args = pyqtSignal([object, object])

    playback_events = pyqtSignal([list])

    def __init__(self):
        super().__init__()
        self.timer = None
        self.pending_timer = None

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

        self.settings_dialog = SettingsDialog(self)

        # Annoying: set this here so it can be overridden
        self.setStyleSheet('TimerWidget { font: 240px Courier; }')

        main = QWidget()

        # Create an overlapping widget thingy so the scramble is above the timer
        timer_container = QWidget(main)
        timer_layout = QGridLayout(timer_container)
        timer_layout.addWidget(self.timer_widget, 0, 0)
        timer_layout.addWidget(self.scramble_view_widget, 0, 0,
                Qt.AlignRight | Qt.AlignTop)

        right = QWidget()
        right.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        make_vbox(right, [self.instruction_widget, self.smart_playback_widget,
                self.scramble_widget, self.gl_widget, timer_container])

        make_hbox(main, [self.session_widget, right])

        settings_button = QPushButton('Settings')
        settings_button.pressed.connect(self.settings_dialog.exec)

        # Create another overlapping thingy with the settings button
        central = QWidget()
        central_layout = QGridLayout(central)
        central_layout.addWidget(main, 0, 0)
        central_layout.addWidget(settings_button, 0, 0, Qt.AlignRight | Qt.AlignTop)

        self.setCentralWidget(central)

        self.gen_scramble()

        self.update_state_ui()

        # Initialize bluetooth
        # Capture the return value just so it doesn't get GC'd and stop listening
        if USE_SMART_CUBE:
            self.bt = bluetooth.init_bluetooth(self)

        self.setWindowTitle('CubingB')
        self.setFocus()
        self.setFocusPolicy(Qt.StrongFocus)

        self.schedule_fn.connect(self.run_scheduled_fn)
        self.schedule_fn_args.connect(self.run_scheduled_fn_args)
        self.playback_events.connect(self.play_events)

    def run_scheduled_fn(self, fn):
        fn()

    def run_scheduled_fn_args(self, fn, args):
        fn(*args)

    def sizeHint(self):
        return QSize(*WINDOW_SIZE)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_C:
            self.gl_widget.base_quat = quat_invert(self.gl_widget.quat)
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
            self.finish_solve(dnf=event.key() == Qt.Key.Key_Escape)
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
        self.timer_widget.update_time(time.time() - self.start_time, 1)

    # Change UI modes based on state
    def update_state_ui(self):
        self.mark_changed()

        # Hide things that are only shown conditionally below
        self.gl_widget.hide()
        self.smart_playback_widget.hide()
        self.scramble_widget.hide()
        self.scramble_view_widget.hide()
        self.instruction_widget.hide()
        self.timer_widget.hide()

        self.timer_widget.set_pending(False)

        self.session_widget.trigger_update()

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
        if USE_SMART_CUBE:
            self.state = State.SMART_SCRAMBLING
        else:
            self.state = State.SCRAMBLE
        self.scramble = []
        self.solve_moves = []
        self.start_time = None
        self.end_time = None
        self.final_time = None

        all_faces = set(range(6))
        blocked_faces = set()
        turns = list(solver.TURN_STR.values())
        # Just do 25 random moves for now, not random state scrambles
        for i in range(25):
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
        self.final_time = None

    # Start UI timer to update timer view. This needs to be called from a Qt thread,
    # so it's scheduled separately when a bluetooth event comes in
    def start_solve_ui(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_timer)
        self.timer.start(100)

    def finish_solve(self, dnf=False):
        self.end_time = time.time()
        self.final_time = self.end_time - self.start_time
        self.start_time = None

        # Stop UI timer
        if self.timer:
            self.timer.stop()
            self.timer = None
            self.timer_widget.update_time(self.final_time, 3)

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
                    time_ms=int(self.final_time * 1000), dnf=dnf,
                    smart_data_raw=data)

        self.gen_scramble()

    # Smart cube stuff

    def check_solved(self):
        return self.cube == SOLVED_CUBE

    def reset(self):
        self.cube = solver.Cube()
        self.turns = [0] * 6
        self.quat = [1, 0, 0, 0]
        self.smart_cube_data = None
        self.smart_data_copy = None

    # Notify the cube widget that we've updated. We copy all the rendering
    # data to a new object so it can pick up a consistent view at its leisure
    def mark_changed(self):
        # XXX copy only the stuff that's modified in place. Don't introduce
        # bugs here later OK
        cube = self.cube.copy()
        turns = self.turns[:]

        self.scramble_widget.set_scramble(self.scramble, self.scramble_left)
        self.scramble_view_widget.set_scramble(self.scramble)
        self.gl_widget.set_render_data(cube, turns, self.quat)

    # Make a move and update any state for either a scramble or a solve
    def make_turn(self, face, turn):
        face = solver.FACE_STR[face]
        alg = face + solver.TURN_STR[turn]
        self.cube.run_alg(alg)
        old_state = self.state
        # Scrambling: see if this is the next move of the scramble
        if self.state == State.SMART_SCRAMBLING:
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

                self.schedule_fn.emit(self.finish_solve)

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
        # Add up partial turns
        self.turns[face] += turn

        # Record data if we're in a solve
        if self.smart_cube_data is not None:
            turn_byte = face * 2 + {-1: 0, 1: 1}[turn]
            data = struct.pack('<BH', turn_byte, self.get_ts())
            self.smart_cube_data.extend(data)

        # Update state if this is the first partial turn after a scramble
        if self.state == State.SMART_SCRAMBLED:
            # ...and enough time has elapsed since the scramble finished
            now = time.time()
            if now - self.smart_pending_start > TIMER_DEBOUNCE:
                self.state = State.SMART_SOLVING
                self.prepare_smart_solve()
                self.start_solve()
                # Have to start timers in a qt thread
                self.schedule_fn.emit(self.start_solve_ui)
            # Otherwise, restart the timer
            else:
                self.smart_pending_start = now

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
        for [ts, cube, turns, quat, face, turn] in events:
            if cube:
                self.cube = cube.copy()
                self.turns = turns.copy()
            if face is not None:
                self.turns[face] += turn
        # Only show the last rotation, since they're not incremental
        if quat:
            self.quat = quat
        self.mark_changed()

    def start_playback(self, solve_id, solve_nb):
        self.state = State.SMART_VIEWING
        with db.get_session() as session:
            solve = session.query_first(db.Solve, id=solve_id)
            self.reset()
            self.smart_playback_widget.update_solve(SmartSolve(solve, solve_nb))
        self.update_state_ui()

    def stop_playback(self):
        self.state = State.SMART_SCRAMBLING if USE_SMART_CUBE else State.SCRAMBLE
        self.reset()
        self.gen_scramble()
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
        make_grid(self, sliders + [[reset_button], [buttons]])

    def update_rotation(self, axis, value):
        # Icky: reach into parent then into gl_widget, then update it
        attr = 'view_rot_%s' % axis
        setattr(self.parent.gl_widget, attr, value)
        self.parent.gl_widget.update()

class TimerWidget(QLabel):
    def __init__(self, parent):
        super().__init__(parent)
        # This shit should really be in the stylesheet, but not supported?!
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.update_time(0, 3)

    def set_pending(self, pending):
        color = 'red' if pending else 'black'
        self.setStyleSheet('TimerWidget { color: %s }' % color)
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
        # Only show next 5 moves, so it looks fancy
        if USE_SMART_CUBE:
            for i in range(min(5, len(scramble_left), len(left))):
                left[offset+i] = scramble_left[i]
        else:
            for i in range(len(scramble_left)):
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
        self.setStyleSheet('SessionWidget { max-width: 300px; }')

        self.label = QLabel('Session:')
        self.selector = QComboBox()
        self.selector.currentIndexChanged.connect(self.change_session)

        self.stats = QWidget()

        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderItem(0, cell('Time'))
        self.table.setHorizontalHeaderItem(1, cell('ao5'))
        self.table.setHorizontalHeaderItem(2, cell('ao12'))
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.cellDoubleClicked.connect(self.edit_solve)

        title = QWidget()
        make_hbox(title, [self.label, self.selector])
        self.layout = make_vbox(self, [title, self.stats, self.table])

        self.session_editor = SessionEditorDialog(self)
        self.solve_editor = SolveEditorDialog(self)
        self.average_viewer = AverageDialog(self)

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
            elif id == 'delete':
                assert 0
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
            self.session_ids = {}
            sessions = session.query_all(db.Session)
            sessions = sorted(sessions, key=session_sort_key)
            for [i, s] in enumerate(sessions):
                self.session_ids[i] = s.id
                self.selector.addItem(s.name)
                if s.id == sesh.id:
                    self.selector.setCurrentIndex(i)
            for cmd in ['new', 'edit', 'delete']:
                self.session_ids[self.selector.count()] = cmd
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
            self.layout.insertWidget(1, self.stats)

            if not solves:
                return

            # Calculate statistics

            stat_table = [[None, QLabel('current'), QLabel('best')]]

            stats_current = sesh.cached_stats_current or {}
            stats_best = sesh.cached_stats_best or {}
            for [stat_idx, size] in enumerate(STAT_AO_COUNTS):
                label = stat_str(size)
                mean = calc_ao(solves, 0, size)

                # Update best stats, recalculating if necessary. This
                # recalculation is pretty inefficient, but should rarely
                # happen--I guess just when solves are imported or when the
                # session is edited
                stats_current[label] = mean
                if label not in stats_best:
                    best = None
                    for i in range(0, len(solves)):
                        m = calc_ao(solves, i, size)
                        # Update rolling cache stats
                        if solves[i].cached_stats is None:
                            solves[i].cached_stats = {}
                        solves[i].cached_stats[label] = m

                        if not best or m and m < best:
                            best = m
                    stats_best[label] = best
                else:
                    best = stats_best[label]
                    if not best or mean < best:
                        best = stats_best[label] = mean

                i = stat_idx + 1
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
                solve_str = cell(ms_str(solve_time(solve)))
                solve_str.secret_data = solve.id
                self.table.setItem(i, 0, solve_str)
                stats = solve.cached_stats or {}
                self.table.setItem(i, 1,
                        cell(ms_str(stats.get('ao5'))))
                self.table.setItem(i, 2,
                        cell(ms_str(stats.get('ao12'))))

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
        self.smart_widget = None

        self.dnf = QCheckBox('DNF')
        self.dnf.stateChanged.connect(lambda v: self.make_edit('dnf', v))
        self.plus_2 = QCheckBox('+2')
        self.plus_2.stateChanged.connect(lambda v: self.make_edit('plus_2', v))

        self.layout = make_grid(self, [
            [QLabel('Session:'), self.session_label],
            [QLabel('Time:'), self.time_label],
            [QLabel('Result:'), self.result_label],
            [QLabel('Scramble:'), self.scramble_label],
            [QLabel('Smart data:'), self.smart_widget],
            [self.dnf, self.plus_2],
            [buttons],
        ])

        self.solve_id = None

    def make_edit(self, key, value):
        with db.get_session() as session:
            # Edit solve
            solve = session.query_first(db.Solve, id=self.solve_id)
            setattr(solve, key, bool(value))

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
            self.scramble_label.setText(solve.scramble)
            self.time_label.setText(str(solve.created_at))
            self.result_label.setText(solve_time_str(solve))
            # Show a playback button if there's smart data
            if self.smart_widget:
                self.layout.removeWidget(self.smart_widget)
            if solve.smart_data_raw:
                # Get the solve number (i.e. the number within the session, not the
                # database id). This is an extra query, so only do it here
                solve_nb = (session.query(db.Solve).filter_by(session=solve.session)
                        .filter(db.Solve.created_at <= solve.created_at).count())

                self.smart_widget = QPushButton('View Playback')
                self.smart_widget.pressed.connect(
                        lambda: self.start_playback(solve_id, solve_nb))
            else:
                self.smart_widget = QLabel('None')
            self.layout.addWidget(self.smart_widget, 4, 1)
        self.update()

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

        self.table = ReorderTableWidget(self, self.rows_reordered)
        self.table.setColumnCount(4 + len(STAT_AO_COUNTS))
        self.table.setHorizontalHeaderItem(0, cell('Name'))
        self.table.setHorizontalHeaderItem(1, cell('Scramble'))
        self.table.setHorizontalHeaderItem(2, cell('# Solves'))
        for [i, stat] in enumerate(STAT_AO_COUNTS):
            self.table.setHorizontalHeaderItem(3+i, cell(stat_str(stat)))

        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        self.table.itemChanged.connect(self.edit_attr)

        button = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button.accepted.connect(self.accept_edits)
        button.rejected.connect(self.reject)

        make_vbox(self, [self.table, button])

        self.session_selector = SessionSelectorDialog(self)

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
            # Update session selector dialog
            sesh = session.query_first(db.Session, id=session_id)
            sessions = (session.query(db.Session).filter(
                    db.Session.id != session_id).all())
            sessions = sorted(sessions, key=session_sort_key)
            self.session_selector.update_data('Merge %s into session:' % sesh.name, 
                    sessions)
            # Show dialog and merge sessions if accepted
            if self.session_selector.exec():
                # Update solve IDs and delete old session
                merge_id = self.session_selector.selected_session_id
                assert session_id != merge_id
                session.query(db.Solve).filter_by(session_id=session_id).update(
                        {db.Solve.session_id: merge_id})
                session.query(db.Session).filter_by(id=session_id).delete()

                # Clear stat cache
                new_sesh = session.query_first(db.Session, id=merge_id)
                new_sesh.cached_stats_current = None
                new_sesh.cached_stats_best = None

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
                name_widget = cell(sesh.name, editable=True)
                scramble_widget = cell(sesh.scramble_type, editable=True)
                # Just set an attribute on the cell to pass data around?
                # Probably not supposed to do this but it works
                name_widget.secret_data = (sesh.id, 'name')
                scramble_widget.secret_data = (sesh.id, 'scramble_type')
                self.table.setItem(i, 0, name_widget)
                self.table.setItem(i, 1, scramble_widget)
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
            self.base_ts = self.events[self.event_idx][0]
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
        ts = self.events[self.event_idx][0]
        events = [self.events[self.event_idx]]
        self.event_idx += 1
        while self.event_idx < len(self.events):
            event = self.events[self.event_idx]
            if event[0] > max_ts:
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
                ts = self.events[self.event_idx][0]
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
        while i > 0 and self.events[i][1] is None:
            i -= 1
        events = self.events[i:event_idx]
        self.event_idx = event_idx

        if events:
            ts = events[-1][0]
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
        end_ts = max(self.events[-1][0], self.events[-2][0])
        self.end_time_label.setText('/ %s' % ms_str(end_ts))

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
        self.session_ids = {}

    def update_data(self, message, sessions):
        self.selected_session = None

        self.label.setText(message)
        self.selector.clear()
        self.session_ids = {}
        for [i, s] in enumerate(sessions):
            self.session_ids[i] = s.id
            self.selector.addItem(s.name)

    def accept_session(self):
        i = self.selector.currentIndex()
        self.selected_session_id = self.session_ids[i]
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
