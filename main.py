#!/usr/bin/env python
import copy
import enum
import random
import sys
import time

from PyQt5.QtCore import QSize, Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (QApplication, QMainWindow, QHBoxLayout, QVBoxLayout,
        QWidget, QOpenGLWidget, QLabel, QTableWidget, QTableWidgetItem,
        QSizePolicy, QGridLayout, QComboBox, QDialog, QDialogButtonBox,
        QAbstractItemView, QHeaderView, QFrame, QCheckBox)

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
        'SMART_SCRAMBLED SMART_SOLVING')

SOLVED_CUBE = solver.Cube()

STAT_AO_COUNTS = [1, 5, 12, 100]
STAT_OUTLIER_PCT = 5

TIMER_DEBOUNCE = .5

INF = float('+inf')

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

def ms_str(ms):
    if not ms:
        return '-'
    if ms == INF:
        return 'DNF'
    if ms > 60000:
        [minutes, ms] = divmod(ms, 60000)
        return '%d:%06.3f' % (minutes, ms / 1000)
    return '%.3f' % (ms / 1000)

def cell(text, editable=False):
    item = QTableWidgetItem(text)
    if not editable:
        item.setFlags(item.flags() & ~Qt.ItemIsEditable)
    return item

def session_sort_key(s):
    if s.sort_id is not None:
        return s.sort_id
    return s.id

def make_h_layout(parent, children):
    layout = QHBoxLayout(parent)
    for c in children:
        layout.addWidget(c)
    return layout

def make_v_layout(parent, children):
    layout = QVBoxLayout(parent)
    for c in children:
        layout.addWidget(c)
    return layout

# Giant main class that handles the main window, receives bluetooth messages,
# deals with cube logic, etc.
class CubeWindow(QMainWindow):
    # Signal to just run a specified function in a Qt thread, since Qt really
    # cares deeply which thread you're using when starting timers etc.
    schedule_fn = pyqtSignal([object])

    def __init__(self):
        super().__init__()
        self.timer = None
        self.pending_timer = None

        # Initialize DB and make sure there's a current session
        db.init_db(config.DB_PATH)
        with db.get_session() as session:
            settings = session.upsert(db.Settings, {})
            if not settings.current_session:
                sesh = session.insert(db.CubeSession, name='New Session',
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

        # Annoying: set this here so it can be overridden
        self.setStyleSheet('TimerWidget { font: 240px Courier; }')

        main = QWidget()


        timer_container = QWidget(main)
        timer_layout = QGridLayout(timer_container)
        timer_layout.addWidget(self.timer_widget, 0, 0)
        timer_layout.addWidget(self.scramble_view_widget, 0, 0,
                Qt.AlignRight | Qt.AlignTop)

        right = QWidget()
        right.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        make_v_layout(right, [self.instruction_widget, self.scramble_widget,
                self.gl_widget, timer_container])

        make_h_layout(main, [self.session_widget, right])
        self.setCentralWidget(main)

        self.gen_scramble()

        self.update_state_ui()

        # Initialize bluetooth
        # Capture the return value just so it doesn't get GC'd and stop listening
        if USE_SMART_CUBE:
            self.bt = bluetooth.init_bluetooth(self)

        self.setWindowTitle('CubingB')
        self.setFocus()

        self.schedule_fn.connect(self.run_scheduled_fn)

    def run_scheduled_fn(self, fn):
        fn()

    def sizeHint(self):
        return QSize(*WINDOW_SIZE)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_R:
            self.gl_widget.base_quat = quat_invert(self.gl_widget.quat)
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

        last_face = None
        turns = list(solver.TURN_STR.values())
        # Just do 25 random moves for now, not random state scrambles
        for i in range(25):
            faces = [f for f in solver.FACE_STR if f != last_face]
            face = random.choice(faces)
            last_face = face
            move = face + random.choice(turns)
            self.scramble.append(move)
        self.scramble_left = self.scramble[:]

        self.mark_changed()

    def start_solve(self):
        self.start_time = time.time()
        self.end_time = None
        self.final_time = None
        # Start UI timer to update timer view
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_timer)
        self.timer.start(100)

    def finish_solve(self, dnf=False):
        self.end_time = time.time()
        self.final_time = self.end_time - self.start_time

        # Stop UI timer
        if self.timer:
            self.timer.stop()
            self.timer = None
            self.timer_widget.update_time(self.final_time, 3)

        # Update database
        with db.get_session() as session:
            sesh = session.query_first(db.Settings).current_session
            session.insert(db.Solve, session=sesh,
                    scramble=' '.join(self.scramble),
                    time_ms=int(self.final_time * 1000), dnf=dnf)

        self.gen_scramble()

    # Smart cube stuff

    def check_solved(self):
        return self.cube == SOLVED_CUBE

    def reset(self):
        self.cube = solver.Cube()
        self.turns = [0] * 6
        self.quat = [1, 0, 0, 0]

    # Notify the cube widget that we've updated. We copy all the rendering
    # data to a new object so it can pick up a consistent view at its leisure
    def mark_changed(self):
        # XXX copy only the stuff that's modified in place. Don't introduce
        # bugs here later OK
        cube = copy.deepcopy(self.cube)
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
                    if not self.scramble_left:
                        self.state = State.SMART_SCRAMBLED
                else:
                    new_turn = solver.TURN_STR[s_turn]
                    self.scramble_left[0] = face + new_turn
            else:
                new_turn = solver.TURN_STR[-turn % 4]
                self.scramble_left.insert(0, face + new_turn)
        # Scrambled: begin a solve
        elif self.state == State.SMART_SCRAMBLED:
            self.state = State.SMART_SOLVING
            # Have to start timers in a qt thread
            self.schedule_fn.emit(self.start_solve)
        # Solving: check for a complete solve
        elif self.state == State.SMART_SOLVING:
            if self.check_solved():
                self.state = State.SMART_SCRAMBLING
                self.schedule_fn.emit(self.finish_solve)

        # XXX handle after-solve-but-pre-scramble moves

        if self.state != old_state:
            self.schedule_fn.emit(self.update_state_ui)

    # XXX for now we use weilong units, 1/36th turns
    def update_turn(self, face, turn):
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

    def update_rotation(self, quat):
        self.quat = quat
        self.mark_changed()

class TimerWidget(QLabel):
    def __init__(self, parent):
        super().__init__(parent)
        # This shit should really be in the stylesheet, but not supported?!
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_pending(self, pending):
        color = 'red' if pending else 'black'
        self.setStyleSheet('TimerWidget { color: %s }' % color)
        self.update()

    def update_time(self, t, prec):
        # Set up a format string since there's no .*f formatting
        fmt = '%%.0%sf' % prec
        self.setText(fmt % t)
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
        layout = QGridLayout(self)
        layout.addWidget(label, 0, 0, 1, 2)
        layout.addWidget(self.top_pic, 1, 0)
        layout.addWidget(self.bottom_pic, 1, 1)

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
        make_h_layout(title, [self.label, self.selector])
        self.layout = make_v_layout(self, [title, self.stats, self.table])

        self.session_editor = SessionEditorWidget(self)
        self.solve_editor = SolveEditorWidget(self)

    def edit_solve(self, row, col):
        solve_id = self.table.item(row, 0).secret_data
        self.solve_editor.update_solve(solve_id)
        self.solve_editor.exec()
        self.trigger_update()

    def change_session(self, index):
        with db.get_session() as session:
            settings = session.upsert(db.Settings, {})

            id = self.session_ids[index]
            if id == 'new':
                sesh = session.insert(db.CubeSession, name='New Session',
                        scramble_type='3x3')
                settings.current_session = sesh
            elif id == 'edit':
                self.session_editor.update_items()
                self.session_editor.exec()
            elif id == 'delete':
                assert 0
            else:
                settings.current_session = session.query_first(db.CubeSession, id=id)
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
            sessions = session.query_all(db.CubeSession)
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

            # Clear the stats in an annoying way
            self.layout.removeWidget(self.stats)
            self.stats = QWidget()
            self.stat_grid = QGridLayout(self.stats)
            self.layout.insertWidget(1, self.stats)

            if not solves:
                return

            # Calculate statistics

            self.stat_grid.addWidget(QLabel('current'), 0, 1)
            self.stat_grid.addWidget(QLabel('best'), 0, 2)

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
                self.stat_grid.addWidget(QLabel(label), i, 0)
                self.stat_grid.addWidget(QLabel(ms_str(mean)), i, 1)
                self.stat_grid.addWidget(QLabel(ms_str(best)), i, 2)

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

class SolveEditorWidget(QDialog):
    def __init__(self, parent):
        super().__init__(parent)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept_edits)
        buttons.rejected.connect(self.reject_edits)

        self.session_label = QLabel()
        self.time_label = QLabel()
        self.result_label = QLabel()
        self.scramble_label = QLabel()
        self.scramble_label.setWordWrap(True)

        self.dnf = QCheckBox('DNF')
        self.dnf.stateChanged.connect(lambda v: self.make_edit('dnf', v))
        self.plus_2 = QCheckBox('+2')
        self.plus_2.stateChanged.connect(lambda v: self.make_edit('plus_2', v))

        layout = QGridLayout(self)
        layout.addWidget(QLabel('Session:'), 0, 0)
        layout.addWidget(self.session_label, 0, 1)
        layout.addWidget(QLabel('Time:'), 1, 0)
        layout.addWidget(self.time_label, 1, 1)
        layout.addWidget(QLabel('Result:'), 2, 0)
        layout.addWidget(self.result_label, 2, 1)
        layout.addWidget(QLabel('Scramble:'), 3, 0)
        layout.addWidget(self.scramble_label, 3, 1)
        layout.addWidget(self.dnf, 4, 0)
        layout.addWidget(self.plus_2, 4, 1)
        layout.addWidget(buttons, 5, 0, 1, 3)

        self.solve_id = None
        self.edits = {}

    def make_edit(self, key, value):
        self.edits[key] = bool(value)
        setattr(self.solve, key, value)
        self.result_label.setText(solve_time_str(self.solve))

    def update_solve(self, solve_id):
        self.solve_id = solve_id
        with db.get_session() as session:
            solve = session.query_first(db.Solve, id=solve_id)

            solve.session # Attribute access so it still works after make_transient
            db.make_transient(solve)

            self.solve = solve
            self.dnf.setChecked(solve.dnf)
            self.plus_2.setChecked(solve.plus_2)
            self.session_label.setText(solve.session.name)
            self.scramble_label.setText(solve.scramble)
            self.time_label.setText(str(solve.created_at))
            self.result_label.setText(solve_time_str(solve))
        self.update()

    def sizeHint(self):
        return QSize(400, 100)

    def reset(self):
        self.solve_id = None
        self.solve = None
        self.edits = {}

    def accept_edits(self):
        with db.get_session() as session:
            # Edit solve
            solve = session.query_first(db.Solve, id=self.solve_id)
            for [k, v] in self.edits.items():
                setattr(solve, k, v)

            # Invalidate statistics on session
            # XXX might need to invalidate individual solve stats later, but
            # for now when the 'best' stat cache is cleared it recalculates all
            # solves
            solve.session.cached_stats_current = None
            solve.session.cached_stats_best = None

        self.reset()
        self.accept()

    def reject_edits(self):
        self.reset()
        self.reject()

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

class SessionEditorWidget(QDialog):
    def __init__(self, parent):
        super().__init__(parent)

        self.table = ReorderTableWidget(self, self.rows_reordered)
        self.table.setColumnCount(3 + len(STAT_AO_COUNTS))
        self.table.setHorizontalHeaderItem(0, cell('Name'))
        self.table.setHorizontalHeaderItem(1, cell('Scramble'))
        self.table.setHorizontalHeaderItem(2, cell('# Solves'))
        for [i, stat] in enumerate(STAT_AO_COUNTS):
            self.table.setHorizontalHeaderItem(3+i, cell(stat_str(stat)))

        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        self.table.itemChanged.connect(self.item_edited)

        button = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button.accepted.connect(self.accept_edits)
        button.rejected.connect(self.reject_edits)

        make_v_layout(self, [self.table, button])

        self.name_edits = {}
        self.reorder_edits = {}

    def item_edited(self, item):
        self.name_edits[item.secret_data] = item.text()

    def rows_reordered(self):
        # Update sort IDs
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            self.reorder_edits[item.secret_data] = row + 1

    def accept_edits(self):
        # Make sure to change focus to the window in case the user clicked OK
        # while still editing. Without the focus change, the itemChanged signal
        # is fired after we accept changes, so we'd lose the last edit, and it
        # would get silently added to the next edit action
        self.setFocus()

        with db.get_session() as session:
            # Update names
            for [id, name] in self.name_edits.items():
                sesh = session.query_first(db.CubeSession, id=id)
                sesh.name = name
            self.name_edits = {}

            # Update ordering
            for [id, sort_id] in self.reorder_edits.items():
                sesh = session.query_first(db.CubeSession, id=id)
                sesh.sort_id = sort_id
            self.reorder_edits = {}
        self.accept()

    def reject_edits(self):
        self.name_edits = {}
        self.reorder_edits = {}
        self.reject()

    def sizeHint(self):
        return QSize(700, 800)

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
            sessions = (session.query(db.CubeSession, stmt.c.n_solves)
                    .outerjoin(stmt, db.CubeSession.id == stmt.c.session_id).all())
            sessions = list(sorted(sessions, key=lambda s: session_sort_key(s[0])))

            self.table.setRowCount(len(sessions))
            for [i, [sesh, n_solves]] in enumerate(sessions):
                stats = sesh.cached_stats_best or {}
                sesh_id = session_sort_key(sesh)
                self.table.setVerticalHeaderItem(i, cell(str(sesh_id)))
                name_widget = cell(sesh.name, editable=True)
                # Just set an attribute on the cell to pass data around?
                # Probably not supposed to do this but it works
                name_widget.secret_data = sesh.id
                self.table.setItem(i, 0, name_widget)
                self.table.setItem(i, 1, cell(sesh.scramble_type))
                self.table.setItem(i, 2, cell(str(n_solves)))
                for [j, stat] in enumerate(STAT_AO_COUNTS):
                    stat = stat_str(stat)
                    self.table.setItem(i, 3+j,
                            cell(ms_str(stats.get(stat))))

        self.table.blockSignals(False)

# Display the cube

class GLWidget(QOpenGLWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.reset()
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def reset(self):
        self.quat = self.base_quat = [1, 0, 0, 0]
        self.gl_init = False
        self.size = None
        self.ortho = None
        self.bg_color = [.1, .1, .1, 1]

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
            render.set_persective(self.size)

            q = quat_mul(self.base_quat, self.quat)
            matrix = quat_matrix(quat_normalize(q))

            render.set_rotation(matrix)

        render.render_cube(self.cube, self.turns)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CubeWindow()
    window.show()
    sys.exit(app.exec_())
