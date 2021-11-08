#!/usr/bin/env python
import copy
import enum
import random
import sys
import time

from PyQt5.QtCore import QSize, Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget,
        QOpenGLWidget, QLabel)

import bluetooth
import db
import render
import solver

WINDOW_SIZE = [1600, 1000]

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

State = enum.Enum('State', 'SCRAMBLING SCRAMBLED SOLVING SOLVED')

SOLVED_CUBE = solver.Cube()

# Giant main class that handles the main window, receives bluetooth messages,
# deals with cube logic, etc.
class CubeWindow(QMainWindow):
    update_signal = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.timer = None
        self.gl_widget = None
        self.scramble_widget = None
        self.instruction_widget = None
        self.timer_widget = None
        self.gen_scramble()

        db.init_db('sqlite:///cubingb.db')
        # Create a session and set it as current
        with db.get_session() as session:
            sesh = session.upsert(db.CubeSession, {'name': '3x3'})
            session.upsert(db.Settings, {}, current_session=sesh)

        self.update_state_ui()

        # Initialize bluetooth
        # Capture the return value just so it doesn't get GC'd and stop listening
        self.bt = bluetooth.init_bluetooth(self)

        self.setWindowTitle('CubingB')
        self.grabKeyboard()

        # Set up styles
        self.setStyleSheet('''ScrambleWidget { font: 48px Courier; }
                InstructionWidget { font: 40px; }
                TimerWidget { font: 240px Courier; }''')


        self.update_signal.connect(self.update_state_ui)

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
        self.widget = QWidget()
        self.layout = QVBoxLayout()

        self.gl_widget = None
        self.scramble_widget = None
        self.instruction_widget = None
        self.timer_widget = None

        self.gl_widget = GLWidget(self)
        self.mark_changed()

        if self.state == State.SCRAMBLING:
            self.scramble_widget = ScrambleWidget(self)
            self.layout.addWidget(self.scramble_widget)
            self.layout.addWidget(self.gl_widget)
        elif self.state == State.SCRAMBLED:
            self.instruction_widget = InstructionWidget(self)
            self.instruction_widget.setText("Start solving when you're ready!")
            self.layout.addWidget(self.instruction_widget)
            self.layout.addWidget(self.gl_widget)
        elif self.state == State.SOLVING:
            self.instruction_widget = InstructionWidget(self)
            self.timer_widget = TimerWidget(self)
            self.instruction_widget.setText("")
            self.layout.addWidget(self.instruction_widget)
            self.layout.addWidget(self.timer_widget)
            self.timer = QTimer()
            self.timer.timeout.connect(self.update_timer)
            self.timer.start(100)
        elif self.state == State.SOLVED:
            self.instruction_widget = InstructionWidget(self)
            self.timer.stop()
            self.timer = None
            self.timer_widget = TimerWidget(self)
            self.timer_widget.update_time(self.final_time, 3)
            self.instruction_widget.setText("That was pretty cool.")
            self.layout.addWidget(self.instruction_widget)
            self.layout.addWidget(self.timer_widget)

        self.setCentralWidget(self.widget)
        self.widget.setLayout(self.layout)

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
        #self.setVerticalPolicy(Qt.QSizePolicy.Maximum)
        # This shit should really be in the stylesheet, but not supported?!
        self.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.setWordWrap(True)

    def sizeHint(self):
        return QSize(WINDOW_SIZE[0], 100)

# Display the scramble

class ScrambleWidget(QLabel):
    def __init__(self, parent):
        super().__init__(parent)
        # This shit should really be in the stylesheet, but not supported?!
        self.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.setWordWrap(True)

    def set_scramble(self, scramble, scramble_left):
        offset = max(len(scramble) - len(scramble_left), 0)
        left = ['-'] * len(scramble)
        for i in range(min(5, len(scramble_left))):
            left[offset+i] = scramble_left[i]
        self.setText(' '.join('% -2s' % s for s in left))
        self.update()

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

    def sizeHint(self):
        return QSize(*WINDOW_SIZE)

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
