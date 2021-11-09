#!/usr/bin/env python
import copy
import enum
import random
import sys
import time

from PyQt5.QtCore import QSize, Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (QApplication, QMainWindow, QHBoxLayout, QVBoxLayout,
        QWidget, QOpenGLWidget, QLabel, QTableWidget, QTableWidgetItem,
        QSizePolicy, QGridLayout, QComboBox, QDialog, QDialogButtonBox)

import bluetooth
import db
import render
import solver

# Constants

WINDOW_SIZE = [1600, 1000]

DB_PATH = 'sqlite:///cubingb.db'

# Should make UI for this
USE_SMART_CUBE = False

State = enum.Enum('State', 'SCRAMBLE SOLVE_PENDING SOLVING SMART_SCRAMBLING '
        'SMART_SCRAMBLED SMART_SOLVING')

SOLVED_CUBE = solver.Cube()

STAT_AO_COUNTS = [1, 5, 12, 100]
STAT_OUTLIER_PCT = 5

TIMER_DEBOUNCE = .5

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

def stat_str(size):
    if size == 1:
        return 'single'
    else:
        return 'ao%s' % size

def ms_str(mean):
    if not mean:
        return '-'
    return '%.3f' % (mean / 1000)

def cell(text, editable=False):
    item = QTableWidgetItem(text)
    if not editable:
        item.setFlags(item.flags() & ~Qt.ItemIsEditable)
    return item

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
        db.init_db(DB_PATH)
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
        self.instruction_widget = InstructionWidget(self)
        self.timer_widget = TimerWidget(self)
        self.session_widget = SessionWidget(self)

        # Annoying: set this here so it can be overridden
        self.setStyleSheet('TimerWidget { font: 240px Courier; }')

        main = QWidget()

        layout = QHBoxLayout(main)
        layout.addWidget(self.session_widget)

        right = QWidget()
        right.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right_layout = QVBoxLayout(right)
        right_layout.addWidget(self.instruction_widget)
        right_layout.addWidget(self.scramble_widget)
        right_layout.addWidget(self.gl_widget)
        right_layout.addWidget(self.timer_widget)
        layout.addWidget(right)

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
        elif event.key() == Qt.Key.Key_Space and self.state in {State.SCRAMBLE,
                State.SOLVING}:
            if self.state == State.SCRAMBLE:
                self.state = State.SOLVE_PENDING
                self.start_pending()
            elif self.state == State.SOLVING:
                self.state = State.SCRAMBLE
                self.finish_solve()
            self.update_state_ui()
        else:
            event.ignore()

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
        self.instruction_widget.hide()
        self.timer_widget.hide()

        self.timer_widget.set_pending(False)

        self.session_widget.trigger_update()

        if self.state == State.SCRAMBLE:
            self.scramble_widget.show()
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

    def finish_solve(self):
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
                    time_ms=int(self.final_time * 1000))

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

# Display the scramble

class ScrambleWidget(QLabel):
    def __init__(self, parent):
        super().__init__(parent)
        self.setStyleSheet('ScrambleWidget { font: 48px Courier; max-height: 100px }')
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

class SessionWidget(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.setStyleSheet('SessionWidget { max-width: 300px; }')

        title = QWidget()
        title_layout = QHBoxLayout(title)
        self.label = QLabel('Session:')
        self.selector = QComboBox()
        self.selector.currentIndexChanged.connect(self.change_session)
        title_layout.addWidget(self.label)
        title_layout.addWidget(self.selector)

        self.stats = QWidget()

        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderItem(0, cell('Time'))
        self.table.setHorizontalHeaderItem(1, cell('ao5'))
        self.table.setHorizontalHeaderItem(2, cell('ao12'))

        layout = QVBoxLayout(self)
        layout.addWidget(title)
        layout.addWidget(self.stats)
        layout.addWidget(self.table)
        self.layout = layout

        self.session_editor = SessionEditorWidget(self)

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

            # HACK: disconnect the signal handler so we don't trigger a recursive
            # update
            self.selector.currentIndexChanged.disconnect(self.change_session)

            # Set up dropdown
            self.selector.clear()
            self.session_ids = {}
            for [i, s] in enumerate(session.query_all(db.CubeSession)):
                self.session_ids[i] = s.id
                self.selector.addItem(s.name)
                if s.id == sesh.id:
                    self.selector.setCurrentIndex(i)
            for cmd in ['new', 'edit', 'delete']:
                self.session_ids[self.selector.count()] = cmd
                self.selector.addItem(cmd.title() + '...')

            # Restore signal handler per hack above
            self.selector.currentIndexChanged.connect(self.change_session)

            # Get solves
            # XXX assumes the query returns in chronological order
            solves = list(reversed(session.query_all(db.Solve, session=sesh)))

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
            def calc_ao(start, size):
                if len(solves) - start < size:
                    mean = None
                else:
                    times = [s.time_ms for s in solves[start:start+size]]
                    if size > 1:
                        times.sort()
                        outliers = (size * STAT_OUTLIER_PCT + 99) // 100
                        times = times[outliers:-outliers]
                    mean = sum(times) / len(times)

                return mean

            self.stat_grid.addWidget(QLabel('current'), 0, 1)
            self.stat_grid.addWidget(QLabel('best'), 0, 2)

            all_times = [s.time_ms for s in solves]
            stats_current = sesh.cached_stats_current or {}
            stats_best = sesh.cached_stats_best or {}
            for [stat_idx, size] in enumerate(STAT_AO_COUNTS):
                label = stat_str(size)
                mean = calc_ao(0, size)

                # Update best stats, recalculating if necessary. This
                # recalculation is pretty inefficient, but should rarely
                # happen--I guess just when solves are imported or when the
                # session is edited
                stats_current[label] = mean
                if label not in stats_best:
                    best = None
                    for i in range(0, len(all_times)):
                        m = calc_ao(i, size)
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
            solves[0].cached_stats = stats_current

            # Build the table of actual solves
            self.table.setRowCount(len(solves))

            for [i, solve] in enumerate(solves):
                self.table.setVerticalHeaderItem(i,
                        cell('%s' % (len(solves) - i)))
                self.table.setItem(i, 0,
                        cell('%.3f' % (solve.time_ms / 1000)))
                stats = solve.cached_stats or {}
                self.table.setItem(i, 1,
                        cell(ms_str(stats.get('ao5'))))
                self.table.setItem(i, 2,
                        cell(ms_str(stats.get('ao12'))))

class SessionEditorWidget(QDialog):
    def __init__(self, parent):
        super().__init__(parent)

        self.table = QTableWidget()
        self.table.setColumnCount(3 + len(STAT_AO_COUNTS))
        self.table.setHorizontalHeaderItem(0, cell('ID'))
        self.table.setHorizontalHeaderItem(1, cell('Name'))
        self.table.setHorizontalHeaderItem(2, cell('Scramble type'))
        for [i, stat] in enumerate(STAT_AO_COUNTS):
            self.table.setHorizontalHeaderItem(3+i, cell(stat_str(stat)))

        self.table.itemChanged.connect(self.item_edited)

        button = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button.accepted.connect(self.accept_edits)
        button.rejected.connect(self.reject_edits)

        layout = QVBoxLayout(self)
        layout.addWidget(self.table)
        layout.addWidget(button)

        self.current_edits = {}

    def item_edited(self, item):
        self.current_edits[item.session_id] = item.text()

    def accept_edits(self):
        # Make sure to change focus to the window in case the user clicked OK
        # while still editing. Without the focus change, the itemChanged signal
        # is fired after we accept changes, so we'd lose the last edit, and it
        # would get silently added to the next edit action
        self.setFocus()

        with db.get_session() as session:
            for [id, name] in self.current_edits.items():
                sesh = session.query_first(db.CubeSession, id=id)
                sesh.name = name
            self.current_edits = {}
        self.accept()

    def reject_edits(self):
        self.current_edits = {}
        self.reject()

    def sizeHint(self):
        return QSize(600, 500)

    def update_items(self):
        # HACK: disconnect the edit signal
        self.table.itemChanged.disconnect(self.item_edited)

        self.table.clearContents()
        self.table.setRowCount(0)
        with db.get_session() as session:
            sessions = session.query_all(db.CubeSession)
            self.table.setRowCount(len(sessions))
            for [i, sesh] in enumerate(sessions):
                stats = sesh.cached_stats_best or {}
                self.table.setItem(i, 0, cell(str(sesh.id)))
                name_widget = cell(sesh.name, editable=True)
                # Just set an attribute on the cell to pass data around?
                # Probably not supposed to do this but it works
                name_widget.session_id = sesh.id
                self.table.setItem(i, 1, name_widget)
                self.table.setItem(i, 2, cell(sesh.scramble_type))
                for [j, stat] in enumerate(STAT_AO_COUNTS):
                    stat = stat_str(stat)
                    self.table.setItem(i, 3+j,
                            cell(ms_str(stats.get(stat))))

        self.table.itemChanged.connect(self.item_edited)

# Display the cube

class GLWidget(QOpenGLWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.quat = self.base_quat = [1, 0, 0, 0]
        self.gl_init = False
        self.size = None

    def set_render_data(self, cube, turns, quat):
        self.cube = cube
        self.turns = turns
        self.quat = quat
        self.update()

    def initializeGL(self):
        self.gl_init = True
        render.setup()
        render.set_persective(self.size)

    def resizeEvent(self, event):
        s = event.size()
        self.size = [s.width(), s.height()]
        if self.gl_init:
            render.set_persective(self.size)

    def paintGL(self):
        render.reset()

        q = quat_mul(self.base_quat, self.quat)
        matrix = quat_matrix(quat_normalize(q))

        render.set_rotation(matrix)
        render.render_cube(self.cube, self.turns)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CubeWindow()
    window.show()
    sys.exit(app.exec_())
