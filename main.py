#!/usr/bin/env python
import copy
import enum
import random
import sys
import time

from PyQt5.QtCore import QSize, Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (QApplication, QMainWindow, QHBoxLayout, QVBoxLayout,
        QWidget, QOpenGLWidget, QLabel, QTableWidget, QTableWidgetItem,
        QSizePolicy, QGridLayout)

import bluetooth
import db
import render
import solver

WINDOW_SIZE = [1600, 1000]

DB_PATH = 'sqlite:///cubingb.db'

State = enum.Enum('State', 'SCRAMBLING SCRAMBLED SOLVING SOLVED')

SOLVED_CUBE = solver.Cube()

STAT_AO_COUNTS = [1, 5, 12, 100]
STAT_OUTLIER_PCT = 5

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

# Giant main class that handles the main window, receives bluetooth messages,
# deals with cube logic, etc.
class CubeWindow(QMainWindow):
    update_signal = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.timer = None

        # Initialize DB, upsert a session and set it as current
        db.init_db(DB_PATH)
        with db.get_session() as session:
            sesh = session.upsert(db.CubeSession, {'name': '3x3'})
            session.upsert(db.Settings, {}, current_session=sesh)

        # Create basic layout/widgets. We do this first because the various
        # initialization functions can send data to the appropriate widgets to
        # update the view, and it's simpler to just always have them available
        self.gl_widget = GLWidget(self)
        self.scramble_widget = ScrambleWidget(self)
        self.instruction_widget = InstructionWidget(self)
        self.timer_widget = TimerWidget(self)
        self.session_widget = SessionWidget(self)

        layout = QHBoxLayout()
        layout.addWidget(self.session_widget)

        right = QWidget()
        right.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right_layout = QVBoxLayout()
        right_layout.addWidget(self.instruction_widget)
        right_layout.addWidget(self.scramble_widget)
        right_layout.addWidget(self.gl_widget)
        right_layout.addWidget(self.timer_widget)
        right.setLayout(right_layout)
        layout.addWidget(right)

        main = QWidget()
        main.setLayout(layout)
        self.setCentralWidget(main)

        self.gen_scramble()

        self.update_state_ui()

        # Initialize bluetooth
        # Capture the return value just so it doesn't get GC'd and stop listening
        self.bt = bluetooth.init_bluetooth(self)

        self.setWindowTitle('CubingB')
        self.grabKeyboard()

        self.update_signal.connect(self.update_state_ui)

    def sizeHint(self):
        return QSize(*WINDOW_SIZE)

    def keyPressEvent(self, key):
        if key.key() == Qt.Key.Key_Space:
            self.gl_widget.base_quat = quat_invert(self.gl_widget.quat)
        elif key.key() == Qt.Key.Key_Return and self.state == State.SOLVED:
            self.gen_scramble()
            self.update_state_ui()

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

        self.session_widget.trigger_update()

        if self.state == State.SCRAMBLING:
            self.scramble_widget.show()
            self.gl_widget.show()
        elif self.state == State.SCRAMBLED:
            self.instruction_widget.setText("Start solving when you're ready!")
            self.instruction_widget.show()
            self.gl_widget.show()
        elif self.state == State.SOLVING:
            self.timer_widget.show()
            # Start UI timer to update timer view
            self.timer = QTimer()
            self.timer.timeout.connect(self.update_timer)
            self.timer.start(100)
        elif self.state == State.SOLVED:
            self.instruction_widget.setText("Press Enter for next solve")
            self.instruction_widget.show()
            self.timer_widget.show()

            # Stop UI timer
            self.timer.stop()
            self.timer = None
            self.timer_widget.update_time(self.final_time, 3)

    def gen_scramble(self):
        self.reset()
        self.state = State.SCRAMBLING
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

        if self.scramble_widget:
            self.scramble_widget.set_scramble(self.scramble, self.scramble_left)
        if self.gl_widget:
            self.gl_widget.set_render_data(cube, turns, self.quat)

    # Make a move and update any state for either a scramble or a solve
    def make_turn(self, face, turn):
        face = solver.FACE_STR[face]
        alg = face + solver.TURN_STR[turn]
        self.cube.run_alg(alg)
        old_state = self.state
        # Scrambling: see if this is the next move of the scramble
        if self.state == State.SCRAMBLING:
            s_face = self.scramble_left[0][0]
            s_turn = solver.INV_TURN_STR[self.scramble_left[0][1:]]
            if face == s_face:
                s_turn = (s_turn - turn) % 4
                if not s_turn:
                    self.scramble_left.pop(0)
                    if not self.scramble_left:
                        self.state = State.SCRAMBLED
                else:
                    new_turn = solver.TURN_STR[s_turn]
                    self.scramble_left[0] = face + new_turn
            else:
                new_turn = solver.TURN_STR[-turn % 4]
                self.scramble_left.insert(0, face + new_turn)
        # Scrambled: begin a solve
        elif self.state == State.SCRAMBLED:
            self.state = State.SOLVING
            self.start_time = time.time()
        # Solving: check for a complete solve
        elif self.state == State.SOLVING:
            if self.check_solved():
                self.state = State.SOLVED
                self.end_time = time.time()
                self.final_time = self.end_time - self.start_time
                # Update database
                with db.get_session() as session:
                    sesh = session.query_first(db.Settings).current_session
                    session.insert(db.Solve, session=sesh,
                            scramble=' '.join(self.scramble),
                            time_ms=int(self.final_time * 1000))

        # XXX handle after-solve-but-pre-scramble moves

        if self.state != old_state:
            self.update_signal.emit()

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
        self.setStyleSheet('TimerWidget { font: 240px Courier; }')
        # This shit should really be in the stylesheet, but not supported?!
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

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
        for i in range(min(5, len(scramble_left), len(left))):
            left[offset+i] = scramble_left[i]
        self.setText(' '.join('% -2s' % s for s in left))
        self.update()

class SessionWidget(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.setStyleSheet('SessionWidget { max-width: 300px; }')

        self.label = QLabel()
        self.stats = QWidget()
        self.stat_grid = QGridLayout()
        self.stats.setLayout(self.stat_grid)
        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderItem(0, QTableWidgetItem('Time'))
        self.table.setHorizontalHeaderItem(1, QTableWidgetItem('ao5'))
        self.table.setHorizontalHeaderItem(2, QTableWidgetItem('ao12'))

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.stats)
        layout.addWidget(self.table)
        self.setLayout(layout)

    def trigger_update(self):
        with db.get_session() as session:
            sesh = session.query_first(db.Settings).current_session
            self.label.setText(sesh.name)

            # Get solves
            # XXX assumes the query returns in chronological order
            solves = list(reversed(session.query_all(db.Solve, session=sesh)))

            if not solves:
                return

            # Calculate statistics
            def calc_ao(start, size):
                if size == 1:
                    label = 'single'
                else:
                    label = 'ao%s' % size

                if len(solves) - start < size:
                    mean = None
                else:
                    times = [s.time_ms for s in solves[start:start+size]]
                    if size > 1:
                        times.sort()
                        outliers = (size * STAT_OUTLIER_PCT + 99) // 100
                        times = times[outliers:-outliers]
                    mean = sum(times) / len(times)

                return [label, mean]

            def mean_str(mean):
                if not mean:
                    return '-'
                return '%.3f' % (mean / 1000)

            self.stat_grid.addWidget(QLabel('current'), 0, 1)
            self.stat_grid.addWidget(QLabel('best'), 0, 2)

            all_times = [s.time_ms for s in solves]
            stats_current = sesh.cached_stats_current or {}
            stats_best = sesh.cached_stats_best or {}
            for [stat_idx, size] in enumerate(STAT_AO_COUNTS):
                [label, mean] = calc_ao(0, size)

                # Update best stats, recalculating if necessary. This
                # recalculation is pretty inefficient, but should rarely
                # happen--I guess just when solves are imported or when the
                # session is edited
                stats_current[label] = mean
                if label not in stats_best:
                    best = None
                    for i in range(0, len(all_times)):
                        [_, m] = calc_ao(i, size)
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
                self.stat_grid.addWidget(QLabel(mean_str(mean)), i, 1)
                self.stat_grid.addWidget(QLabel(mean_str(best)), i, 2)

            sesh.cached_stats_current = stats_current
            sesh.cached_stats_best = stats_best
            solves[0].cached_stats = stats_current

            # Build the table of actual solves
            self.table.setRowCount(len(solves))

            for [i, solve] in enumerate(solves):
                self.table.setVerticalHeaderItem(i,
                        QTableWidgetItem('%s' % (len(solves) - i)))
                self.table.setItem(i, 0,
                        QTableWidgetItem('%.3f' % (solve.time_ms / 1000)))
                stats = solve.cached_stats or {}
                self.table.setItem(i, 1,
                        QTableWidgetItem(mean_str(stats.get('ao5'))))
                self.table.setItem(i, 2,
                        QTableWidgetItem(mean_str(stats.get('ao12'))))

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
