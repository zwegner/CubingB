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
import functools
import gzip
import os
import random
import struct
import sys
import time

from PyQt5.QtCore import (QSize, Qt, QTimer, pyqtSignal, QAbstractAnimation,
        QVariantAnimation, QBuffer, QByteArray, QPoint)
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QOpenGLWidget,
        QLabel, QTableWidget, QTableWidgetItem, QSizePolicy, QGridLayout,
        QComboBox, QDialog, QDialogButtonBox, QAbstractItemView, QHeaderView,
        QFrame, QCheckBox, QPushButton, QSlider, QMessageBox, QInputDialog,
        QMenu, QAction, QPlainTextEdit, QTabBar, QToolTip)
from PyQt5.QtGui import (QIcon, QFont, QFontDatabase, QCursor, QPainter, QImage,
        QRegion, QColor)
from PyQt5.QtSvg import QSvgWidget

import analyze
import bluetooth
import config
import db
import render
import solver
from util import *

# Constants

WINDOW_SIZE = [1600, 1000]

Mode = enum.IntEnum('Mode', 'TIMER PLAYBACK ALG_TRAIN ALG_VIEW', start=0)
State = enum.Enum('State', 'SCRAMBLE SOLVE_PENDING SOLVING SMART_SCRAMBLING '
        'SMART_SCRAMBLED SMART_SOLVING')

SCRAMBLE_MOVES = 25

TIMER_DEBOUNCE = .5

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

def session_sort_key(s):
    if s.sort_id is not None:
        return s.sort_id
    return s.id

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
        self.mode = None

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
        self.alg_trainer_widget = analyze.AlgTrainer(self)
        self.alg_viewer_widget = analyze.AlgViewer(self)
        self.session_widget = SessionWidget(self)
        self.smart_playback_widget = SmartPlaybackWidget(self)
        self.bt_status_widget = BluetoothStatusWidget(self)

        # Set up bluetooth. This doesn't actually scan or anything yet
        self.bt_handler = bluetooth.BluetoothHandler(self)

        self.bt_connection_dialog = BluetoothConnectionDialog(self, self.bt_handler)
        self.settings_dialog = SettingsDialog(self)

        # Set up settings buttons
        settings_icon = QIcon('rsrc/material/settings_black_24dp.svg')
        settings_button = QPushButton(settings_icon, '')
        settings_button.clicked.connect(self.settings_dialog.exec)
        bt_icon = QIcon('rsrc/material/bluetooth_black_24dp.svg')
        bt_button = QPushButton(bt_icon, '')
        bt_button.clicked.connect(self.bt_connection_dialog.exec)

        buttons = QWidget()
        buttons.setStyleSheet('QPushButton { icon-size: 20px 20px; max-width: 30px; '
                'border: 1px solid #777; border-radius: 3px; '
                'border-style: outset; padding: 5px; margin: 0px; '
                'background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, '
                '   stop: 0 #fff, stop: 1 #eee); } '
                'QPushButton::pressed { '
                'background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, '
                '   stop: 0 #ccc, stop: 1 #aaa); } ')
        make_grid(buttons, [[settings_button], [bt_button]], margin=0)

        # Make main layout
        main = QWidget()

        # Set up left
        self.tab_bar = QTabBar()
        for mode in ['Timer', 'Watch', 'Train', 'Algs']:
            self.tab_bar.addTab(mode)
        self.tab_bar.currentChanged.connect(self.change_tab)

        title = QLabel('CubingB')
        title.setStyleSheet('font: 48px; padding: 8px;')
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left = QWidget()
        make_grid(left, [
            [self.tab_bar], 
            [title, buttons],
            [self.session_widget]
        ], margin=0, widths=[0, 40])
        left.setStyleSheet('SessionWidget, QTabBar { min-width: 300px; max-width: 350px; }')

        top = QWidget()
        top.setObjectName('top')
        make_grid(top, [[self.instruction_widget, self.smart_playback_widget,
                self.scramble_widget]], margin=0)

        # Create an overlapping widget thingy so the scramble is above the timer
        timer_container = QWidget(main)
        timer_layout = QGridLayout(timer_container)
        timer_layout.addWidget(self.timer_widget, 0, 0)
        timer_layout.addWidget(self.scramble_view_widget, 0, 0,
                Qt.AlignRight | Qt.AlignTop)

        right = QWidget()
        right.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        make_vbox(right, [top, self.gl_widget, timer_container,
                self.alg_trainer_widget, self.alg_viewer_widget], margin=0)

        # Make grid and overlapping status widget
        grid = make_grid(main, [[left, right]])
        grid.addWidget(self.bt_status_widget, 0, 1,
                Qt.AlignRight | Qt.AlignBottom)

        self.setCentralWidget(main)

        # Annoying: set up style here in the parent, so it can be overridden
        # in children
        self.setStyleSheet('BluetoothStatusWidget { font: 24px; '
                '   color: #FFF; background: rgb(80,80,255); padding: 5px; }'
                '#top { min-width: 300px; }')

        # Set up default mode
        self.set_mode(Mode.TIMER)
        self.gen_scramble()
        # Also set up initial render data
        self.mark_changed()

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

        # Initialize session state asynchronously
        self.schedule_fn.emit(self.session_widget.trigger_update)

    def run_scheduled_fn(self, fn):
        fn()

    def run_scheduled_fn_args(self, fn, args):
        fn(*args)

    def set_mode(self, mode):
        # Set session viewer to only show smart solves if we're in playback mode
        if self.mode == Mode.PLAYBACK or mode == Mode.PLAYBACK:
            self.session_widget.set_playback_mode(mode == Mode.PLAYBACK)
        self.mode = mode
        self.tab_bar.setCurrentIndex(int(mode))

    def change_tab(self, tab):
        mode = Mode(tab)
        if mode != self.mode:
            self.set_mode(mode)
        self.update_state_ui()

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

    def resizeEvent(self, event):
        size = event.size()
        size = min(size.width() - 350, size.height())
        self.timer_widget.resize(int(size * .22))
        self.scramble_widget.resize(int(size * .07))
        self.scramble_view_widget.resize(int(size * .18))

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
        self.alg_trainer_widget.hide()
        self.alg_viewer_widget.hide()

        self.timer_widget.set_pending(False)

        # Timer mode: main scrambling/solving interface
        if self.mode == Mode.TIMER:
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
        # Playback mode
        elif self.mode == Mode.PLAYBACK:
            self.smart_playback_widget.show()
            self.gl_widget.show()
        # Alg Training mode
        elif self.mode == Mode.ALG_TRAIN:
            self.alg_trainer_widget.init()
            self.alg_trainer_widget.show()
        # Alg Viewer mode
        elif self.mode == Mode.ALG_VIEW:
            self.alg_viewer_widget.init()
            self.alg_viewer_widget.show()

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
        turns = [-1, 1, 2]
        # Just do N random moves for now, not random state scrambles
        for i in range(SCRAMBLE_MOVES):
            face = random.choice(list(all_faces - blocked_faces))
            # Only allow one turn of each of an opposing pair of faces in a row.
            # E.g. F B' is allowed, F B' F is not
            if face ^ 1 not in blocked_faces:
                blocked_faces = set()
            blocked_faces.add(face)

            turn = random.choice(turns)
            self.scramble.append(solver.move_str(face, turn))
        self.scramble_left = self.scramble[:]

        self.mark_scramble_changed()

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
        return self.cube == solver.SOLVED_CUBE

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

    def mark_scramble_changed(self):
        self.scramble_widget.set_scramble(self.scramble, self.scramble_left)
        self.scramble_view_widget.set_scramble(self.scramble)
        self.update()

    # Notify the cube widget that we've updated. We copy all the rendering
    # data to a new object so it can pick up a consistent view at its leisure
    def mark_changed(self, playback=False):
        # Ignore smart cube events during a playback
        if self.mode == Mode.PLAYBACK and not playback:
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

        self.gl_widget.set_render_data(cube, turns, quat)
        self.update()

    # Make a move and update any state for either a scramble or a solve
    def make_turn(self, face, turn):
        alg = solver.move_str(face, turn)
        self.cube.run_alg(alg)
        old_state = self.state

        # Alg training mode: just let the trainer handle all the logic
        if self.mode == Mode.ALG_TRAIN:
            self.alg_trainer_widget.make_move(face, turn)
            return

        assert self.mode == Mode.TIMER

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
                [s_face, s_turn] = solver.parse_move(self.scramble_left[0])

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
                    self.scramble_left[0] = solver.move_str(face, s_turn)
            else:
                self.scramble_left.insert(0, solver.move_str(face, -turn % 4))
        # If we're in a scrambled state, this move needs to move us to
        # unscrambled. If it started a solve, that would be handled earlier in
        # make_turn(), which checks the debounce timer. So if we're here
        # and in a scrambled state, add the inverse turn back to the scramble.
        elif self.state == State.SMART_SCRAMBLED:
            self.scramble_left.insert(0, solver.move_str(face, -turn % 4))
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

        self.mark_scramble_changed()

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
        self.set_mode(Mode.PLAYBACK)
        with db.get_session() as session:
            solve = session.query_first(db.Solve, id=solve_id)
            self.reset(playback=True)
            self.smart_playback_widget.update_solve(analyze.SmartSolve(solve, solve_nb))
        self.update_state_ui()

class SettingsDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent

        buttons = QDialogButtonBox(QDialogButtonBox.Ok)
        buttons.accepted.connect(self.accept)

        sliders = []
        for axis in 'xyz':
            slider = QSlider()
            slider.setMinimum(-180)
            slider.setMaximum(180)

            # Icky: reach into parent then into gl_widget
            attr = 'view_rot_%s' % axis
            slider.setValue(getattr(self.parent.gl_widget, attr))

            slider.setOrientation(Qt.Horizontal)
            update = functools.partial(self.update_rotation, axis)
            slider.valueChanged.connect(update)
            sliders.append([QLabel('Rotation %s:' % axis.upper()), slider])

        reset_button = QPushButton('Reset Camera')
        reset_button.clicked.connect(self.parent.gl_widget.reset_camera)

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
        scan.clicked.connect(self.start_scan)

        self.status = QLabel()
        self.disconnect = QPushButton('Disconnect')
        self.disconnect.hide()
        self.disconnect.clicked.connect(self.disconnect_device)

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

        path = os.path.abspath('./rsrc/lcd.ttf')
        font = QFontDatabase.addApplicationFont(path)
        self.setFont(QFont(QFontDatabase.applicationFontFamilies(font)[0]))

        self.color = 'black'
        self.size = 240
        self.update_font()

    def update_font(self):
        self.setStyleSheet('font: %spx; color: %s;' % (self.size, self.color))
        self.update()

    def resize(self, size):
        self.size = size
        self.update_font()

    def set_pending(self, pending):
        self.color = 'red' if pending else 'black'
        self.update_font()

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
        self.scramble = None

        self.resize(48)

        # This shit should really be in the stylesheet, but not supported?!
        self.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.setWordWrap(True)

        self.scramble_popup = ScrambleViewWidget(self)
        self.scramble_popup.hide()
        self.linkHovered.connect(self.hover_move)

    def resize(self, size):
        self.setStyleSheet('font: %spx Courier;' % size)

    def hover_move(self, link):
        if link:
            scramble = self.scramble[:int(link) + 1]
            self.scramble_popup.set_scramble(scramble)
            text = ' '.join(self.scramble_popup.get_b64_pics(100, 10))
            QToolTip.showText(QCursor.pos(), text)

    def set_scramble(self, scramble, scramble_left):
        self.scramble = scramble
        offset = max(len(scramble) - len(scramble_left), 0)
        left = ['-'] * len(scramble)
        for i in range(min(len(left), len(scramble_left))):
            left[offset+i] = scramble_left[i]
        # Generate the scramble moves with invisible links on each move.
        # The links trigger a tooltip popup through the linkHovered event.
        # And no, setting this style in the main stylesheet doesn't work...
        moves = []
        for [i, s] in enumerate(left):
            m = ('<a style="color: #000; text-decoration: none;"'
                    'href="%s">%s</a>' % (i, s))
            if len(s) == 1:
                m += '&nbsp;'
            moves.append(m)
        self.setText(' '.join(moves))
        self.update()

class ScrambleViewWidget(QFrame):
    def __init__(self, parent, size=200):
        super().__init__(parent)

        # Build layout for scramble view
        self.svg_top = QSvgWidget()
        self.svg_top.setFixedSize(size, size)
        self.svg_bottom = QSvgWidget()
        self.svg_bottom.setFixedSize(size, size)
        make_grid(self, [
            [self.svg_top, self.svg_bottom],
        ])
        self.setStyleSheet('ScrambleViewWidget { background-color: #bbb; '
                'border: 2px solid #777; border-style: outset; } ')

    def resize(self, size):
        self.svg_top.setFixedSize(size, size)
        self.svg_bottom.setFixedSize(size, size)

        self.update()

    def set_scramble(self, scramble):
        cube = solver.Cube()
        cube.run_alg(' '.join(scramble))

        top_diag = render.gen_cube_diagram(cube)
        bottom_diag = render.gen_cube_diagram(cube.run_alg('x2 z'),
                transform='rotate(60)')

        self.svg_top.load(top_diag.encode('ascii'))
        self.svg_bottom.load(bottom_diag.encode('ascii'))

        self.update()

    # Render top/bottom pics into img tags with base64-encoded data. Pretty dirty
    # but this is seemingly required to get images in tooltips (and using widgets
    # as makeshift tooltips seemed even worse).
    # More info at https://stackoverflow.com/a/34300771
    def get_b64_pics(self, size, margin):
        for svg in [self.svg_top, self.svg_bottom]:
            svg.setFixedSize(size, size)
            isize = size + 2*margin
            img = QImage(isize, isize, QImage.Format_RGB32)
            img.fill(QColor(255, 255, 255))
            ba = QByteArray()
            buf = QBuffer(ba)
            painter = QPainter(img)
            svg.render(painter, QPoint(margin, margin), QRegion(),
                    QWidget.RenderFlags(0))
            # Uhh we get a segfault and this crazy warning without this del?!
            # QPaintDevice: Cannot destroy paint device that is being painted
            del painter
            img.save(buf, 'PNG', 100)
            data = bytes(ba.toBase64()).decode()
            yield '<img src="data:image/png;base64, %s">' % data

class SessionWidget(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.setStyleSheet('SessionWidget { max-width: 350px; }')

        self.label = QLabel('Session:')
        self.label.setStyleSheet('font: 18px; font-weight: bold;')
        self.selector = QComboBox()
        self.selector.currentIndexChanged.connect(self.change_session)
        new_icon = QIcon('rsrc/material/add_black_24dp.svg')
        new = QPushButton(new_icon, '')
        new.setStyleSheet('border: none; max-width: 20px;')
        new.clicked.connect(self.new_session)
        edit_icon = QIcon('rsrc/material/edit_note_black_24dp.svg')
        edit = QPushButton(edit_icon, '')
        edit.setStyleSheet('border: none; max-width: 20px;')
        edit.clicked.connect(self.edit_sessions)

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

        self.layout = make_grid(self, [
            [self.label, self.selector, new, edit],
            [self.stats],
            [self.table],
        ], stretch=[0, 1, 0, 0], margin=0)

        self.session_editor = SessionEditorDialog(self)
        self.session_selector = SessionSelectorDialog(self)
        self.solve_editor = SolveEditorDialog(self)
        self.average_viewer = AverageDialog(self)
        self.graph_viewer = analyze.GraphDialog(self)
        self.playback_mode = False

    def edit_solve(self, row, col):
        solve_id = self.table.item(row, 0).secret_data
        # Playback mode: just open the solve immediately
        if self.playback_mode:
            with db.get_session() as session:
                solve = session.query_first(db.Solve, id=solve_id)
                solve_nb = get_solve_nb(session, solve)
                # HACK
                window = self.parent()
                while not isinstance(window, CubeWindow):
                    window = window.parent()
                window.schedule_fn_args.emit(window.start_playback,
                        (solve_id, solve_nb))
            return

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

    def show_graph(self, stat):
        with db.get_session() as session:
            sesh = session.query_first(db.Settings).current_session
            self.graph_viewer.update_data({sesh.name: sesh.solves}, stat=stat)
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

    def new_session(self):
        with db.get_session() as session:
            settings = session.upsert(db.Settings, {})
            sesh = session.insert(db.Session, name='New Session',
                    scramble_type='3x3')
            settings.current_session = sesh
            session.flush()
            self.trigger_update()

    def edit_sessions(self):
        with db.get_session() as session:
            self.session_editor.update_items()
            if not self.session_editor.exec():
                session.rollback()
            self.trigger_update()

    def change_session(self, index):
        with db.get_session() as session:
            settings = session.upsert(db.Settings, {})

            id = self.session_ids[index]
            settings.current_session = session.query_first(db.Session, id=id)
            self.trigger_update()

    def set_playback_mode(self, mode):
        self.playback_mode = mode
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

            self.selector.blockSignals(False)

            # Get solves
            solves = analyze.get_session_solves(session, sesh)

            self.table.clearContents()
            self.table.setRowCount(0)

            # Clear the stats
            self.layout.removeWidget(self.stats)
            self.stats = QWidget()
            self.stats.setStyleSheet('QLabel { font: 16px; }')
            self.layout.addWidget(self.stats, 1, 0, 1, 4)

            if self.playback_mode:
                self.stats.hide()
            else:
                self.stats.show()

            if not solves:
                self.update()
                return

            if not self.playback_mode:
                # Calculate statistics, build stat table
                stat_table = [[None, QLabel('current'), QLabel('best'), None]]
                graph_icon = QIcon('rsrc/graph.svg')
                analyze.calc_session_stats(sesh, solves)
                for size in STAT_AO_COUNTS:
                    stat = stat_str(size)
                    graph_button = QPushButton(graph_icon, '')
                    graph_button.setStyleSheet('border: none;')
                    graph_button.clicked.connect(
                            functools.partial(self.show_graph, stat))
                    mean = sesh.cached_stats_current[stat]
                    best = sesh.cached_stats_best[stat]
                    stat_table.append([QLabel(stat), QLabel(ms_str(mean)),
                            QLabel(ms_str(best)), graph_button])

                make_grid(self.stats, stat_table, stretch=[1, 1, 1, 0])

            # Skip non-smart solves if we're in playback mode. We have the
            # skip logic here (as opposed to SQL) just to keep the solve
            # numbers the same
            filtered_solves = []
            for [i, solve] in enumerate(solves):
                if self.playback_mode and solve.smart_data_raw is None:
                    continue
                filtered_solves.append((len(solves) - i, solve))

            # Build the table of actual solves
            self.table.setRowCount(len(filtered_solves))

            for [i, [n, solve]] in enumerate(filtered_solves):
                # Skip non-smart solves if we're in playback mode. We have the
                # skip logic here (as opposed to SQL) just to keep the solve
                # numbers the same
                if self.playback_mode and solve.smart_data_raw is None:
                    continue
                self.table.setVerticalHeaderItem(i,
                        cell('%s' % n))
                # HACK: set the secret data so the solve editor gets the ID
                self.table.setItem(i, 0, cell(ms_str(solve_time(solve)),
                        secret_data=solve.id))
                stats = solve.cached_stats or {}
                self.table.setItem(i, 1, cell(ms_str(stats.get('ao5'))))
                self.table.setItem(i, 2, cell(ms_str(stats.get('ao12'))))

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
        delete.clicked.connect(self.delete_solve)
        merge = QPushButton('Move to session...')
        merge.clicked.connect(self.move_solve)

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
                self.smart_widget.clicked.connect(
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

            self.result_label.setText(analyze.get_ao_str(solves, 0, n_solves))
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

        # Create context menu items for graphing stats
        self.ctx_menu = QMenu(self)
        for s in STAT_AO_COUNTS:
            stat = stat_str(s)
            action = QAction('Graph %s' % stat, self)
            action.triggered.connect(functools.partial(self.graph_selection, stat))
            self.ctx_menu.addAction(action)

        self.table = ReorderTableWidget(self, self.rows_reordered)
        self.table.setColumnCount(4 + len(STAT_AO_COUNTS))
        self.table.setHorizontalHeaderItem(0, cell('Name'))
        self.table.setHorizontalHeaderItem(1, cell('Scramble'))
        self.table.setHorizontalHeaderItem(2, cell('# Solves'))
        for [i, stat] in enumerate(STAT_AO_COUNTS):
            self.table.setHorizontalHeaderItem(3+i, cell(stat_str(stat)))
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self.show_ctx_menu)

        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)

        self.table.itemChanged.connect(self.edit_attr)

        button = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button.accepted.connect(self.accept_edits)
        button.rejected.connect(self.reject)

        make_vbox(self, [self.table, button])

        self.session_selector = SessionSelectorDialog(self)
        self.graph_viewer = analyze.GraphDialog(self)

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
        return QSize(1000, 800)

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
                # Calculate stats if there's no cache
                if sesh.cached_stats_best is None:
                    solves = analyze.get_session_solves(session, sesh)
                    analyze.calc_session_stats(sesh, solves)

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
                button = QPushButton('Merge...')
                button.clicked.connect(functools.partial(self.merge_sessions, sesh.id))
                self.table.setCellWidget(i, offset+0, button)

        self.table.blockSignals(False)

        self.table.resizeColumnsToContents()

class SmartPlaybackWidget(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent

        # Set up play/pause/etc. controls. We hang on to the play/pause icons and
        # button (in the self object) since that changes dynamically
        self.play_icon = QIcon('rsrc/material/play_arrow_black_24dp.svg')
        self.pause_icon = QIcon('rsrc/material/pause_black_24dp.svg')
        self.play_button = QPushButton(self.play_icon, '')
        self.play_button.clicked.connect(self.play_pause)

        start_icon = QIcon('rsrc/material/skip_previous_black_24dp.svg')
        start_button = QPushButton(start_icon, '')
        start_button.clicked.connect(lambda: self.scrub(0))
        end_icon = QIcon('rsrc/material/skip_next_black_24dp.svg')
        end_button = QPushButton(end_icon, '')
        end_button.clicked.connect(lambda: self.events and
                self.scrub(len(self.events) - 1))

        controls = QWidget()
        controls.setStyleSheet('QPushButton { icon-size: 40px 40px; '
                'border: none; }')
        make_grid(controls, [[None, start_button, self.play_button, end_button, None,
                ]], stretch=[1, 0, 0, 0, 1])

        self.title = QLabel()
        self.title.setStyleSheet('QLabel { font: 24px; }')
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        time_labels = QWidget()
        time_labels.setStyleSheet('QLabel { font: 14px Courier; } ')
        self.current_time_label = QLabel()
        self.end_time_label = QLabel()
        make_hbox(time_labels, [self.current_time_label, self.end_time_label])

        slider = QSlider()
        slider.setOrientation(Qt.Horizontal)
        slider.valueChanged.connect(self.scrub)
        self.slider = slider

        grid = make_grid(self, [
            [self.title],
            [controls],
            [time_labels, slider]
        ])

        self.timer = QTimer()
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.play_event)

        self.solve = None
        self.events = None
        self.event_idx = None
        self.playing = False
        self.base_time = 0
        self.base_ts = 0

    def set_playing(self, playing):
        if self.solve is None:
            return
        self.playing = playing
        if playing:
            self.play_button.setIcon(self.pause_icon)
            # Get current time and first timestamp so we can keep in sync
            self.base_time = time.time()
            if self.event_idx >= len(self.events):
                self.event_idx = 0
            self.base_ts = self.events[self.event_idx].ts
            self.play_event()
        else:
            self.play_button.setIcon(self.play_icon)
            self.timer.stop()
            self.base_ts = 0

    def play_pause(self):
        self.set_playing(not self.playing)

    def play_event(self):
        if self.solve is None:
            return
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

    def scrub(self, event_idx):
        if self.solve is None:
            return
        # In case this came from start/end buttons, set the slider to new position
        self.slider.blockSignals(True)
        self.slider.setValue(event_idx)
        self.slider.blockSignals(False)

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

        self.slider.blockSignals(True)
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        self.slider.setMaximum(len(self.events) - 1)

        self.current_time_label.setText(ms_str(0))
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
        self.drag_start_vector = None
        self.bg_color = [.7, .7, .7, 1]
        self.cam_quat = [1, 0, 0, 0]
        self.cam_base = [1, 0, 0, 0]

    def set_render_data(self, cube, turns, quat):
        self.cube = cube
        self.turns = turns
        self.quat = quat
        self.update()

    def initializeGL(self):
        self.gl_init = True
        render.setup()

    def resizeGL(self, w, h):
        self.size = [w, h]

    def paintGL(self):
        render.BG_COLOR = self.bg_color
        render.reset()

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
    icon = QIcon('rsrc/cubingb-icon-small.png')
    app.setWindowIcon(icon)
    window = CubeWindow()
    window.show()
    sys.exit(app.exec_())
