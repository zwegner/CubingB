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
import functools
import gzip
import itertools
import math
import re
import struct
import time

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtCore import (QSize, Qt)
from PyQt5.QtWidgets import (QLabel, QComboBox, QDialog, QDialogButtonBox,
        QWidget, QSizePolicy, QScrollArea, QTableWidget, QPushButton,
        QHeaderView, QTabBar, QCheckBox)
from PyQt5.QtSvg import QSvgWidget

import db
import flow_layout
import render
import solver
from util import *

STAT_OUTLIER_PCT = 5

F2L_SLOTS = ['Front Right', 'Front Left', 'Back Left', 'Back Right']

def calc_ao(all_times, start, size):
    if len(all_times) - start < size:
        mean = None
    else:
        times = all_times[start:start+size]
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

def get_session_solves(session, sesh):
    return (session.query(db.Solve).filter_by(session=sesh)
            .order_by(db.Solve.created_at.desc()).all())

# Given a session and an ordered list of solves, update all single/aoX statistics
def calc_session_stats(sesh, solves):
    all_times = [solve_time(s) for s in solves]

    stats_current = sesh.cached_stats_current or {}
    stats_best = sesh.cached_stats_best or {}
    stats_best_solve = sesh.cached_stats_best_solve_id or {}
    for [stat_idx, size] in enumerate(STAT_AO_COUNTS):
        stat = stat_str(size)
        mean = calc_ao(all_times, 0, size)
        stats_current[stat] = mean

        # Update best stats, recalculating if necessary
        if stat not in stats_best:
            best = None
            best_id = None

            averages = None
            if size > 1:
                averages = iter(calc_rolling_ao(solves,
                        all_times, size))

            for i in range(len(solves)):
                if size > 1:
                    m = next(averages)
                else:
                    m = all_times[i]

                # Update rolling cache stats
                if solves[i].cached_stats is None:
                    solves[i].cached_stats = {}
                solves[i].cached_stats[stat] = m

                if m and (not best or m < best):
                    best = m
                    best_id = solves[i].id
            stats_best[stat] = best
            stats_best_solve[stat] = best_id
        else:
            best = stats_best[stat]
            if mean and (not best or mean < best):
                best = stats_best[stat] = mean
                stats_best_solve[stat] = solves[0].id

    sesh.cached_stats_current = stats_current
    sesh.cached_stats_best = stats_best
    sesh.cached_stats_best_solve_id = stats_best_solve
    if solves:
        solves[0].cached_stats = stats_current.copy()

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

                    alg = solver.move_str(face, turn)

                    # Merge moves of the same made within a short time window
                    # for purposes of reconstruction
                    if face == last_face and ts - last_ts < 500 and (last_turn + turn) % 4:
                        last_turn += turn
                        last_turn %= 4
                        moves[-1] = solver.move_str(orient[face], last_turn)
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
                        moves.append(solver.move_str(orient[face], turn))

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
        self.solved = (c == solver.SOLVED_CUBE)

# Show a diagram and a label for a certain alg case. This is just a widget
# subclass so we can handle mouse clicks (there's no clicked signal in QWidget?!)
class CaseCard(QWidget):
    def __init__(self, parent, case):
        super().__init__(parent)
        self.parent = parent
        self.case_id = case.id
        self.is_f2l = ('F2L' in case.alg_set)

        # Generate diagram
        diag = render.gen_cube_diagram(case.diagram, type=case.diag_type)
        svg = QSvgWidget()
        svg.setFixedSize(60, 60)
        svg.load(diag.encode('ascii'))

        label = QLabel('%s - %s' % (case.alg_set, case.alg_nb))
        label.setStyleSheet('font: 14px; font-weight: bold;')
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        make_vbox(self, [svg, label])

    def mouseReleaseEvent(self, event):
        self.parent.select_case(self.case_id, self.is_f2l)

class AlgViewer(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.initialized = False
        self.case_id = None
        self.f2l_slot = None

    def init(self):
        if self.initialized:
            return
        self.initialized = True

        # Build cards for each case
        all_algs = collections.defaultdict(list)
        with db.get_session() as session:
            for case in session.query_all(db.AlgCase):
                all_algs[case.alg_set].append(CaseCard(self, case))

        # For each alg set, create a title label and a flow layout with all
        # its cases
        set_widgets = []
        for [alg_set, algs] in all_algs.items():
            title = QLabel(alg_set)
            title.setStyleSheet('font: 18px; font-weight: bold;')
            set_widgets.append(title)

            widget = QWidget()
            layout = flow_layout.FlowLayout(widget)
            for s in algs:
                layout.addWidget(s)
            set_widgets.append(widget)

        # Make main scrollable alg view
        contents = QWidget()
        make_vbox(contents, set_widgets)
        scroll_area = QScrollArea(self)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setWidget(contents)
        scroll_area.setWidgetResizable(True)
        scroll_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        title = QLabel('All Algs')
        title.setStyleSheet('font: 24px; font-weight: bold;')

        self.main_view = QWidget()
        make_vbox(self.main_view, [title, scroll_area])

        # Make alg detail view, left side
        self.alg_label = QLabel()
        self.alg_label.setStyleSheet('font: 24px; font-weight: bold;')
        self.alg_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.alg_icon = QSvgWidget()
        self.alg_icon.setFixedSize(120, 120)
        back = QPushButton('Return to cases')
        back.clicked.connect(functools.partial(self.select_case, None, False))
        left = QWidget()
        layout = make_vbox(left, [back, self.alg_icon, self.alg_label])
        layout.addStretch(1)

        # Make alg detail view, right side
        self.f2l_tabs = QTabBar()
        for slot in F2L_SLOTS:
            self.f2l_tabs.addTab(slot)
        self.f2l_tabs.currentChanged.connect(self.change_f2l_tab)

        self.alg_table = QTableWidget()
        self.alg_table.setColumnCount(3)
        self.alg_table.setHorizontalHeaderItem(0, cell('Alg'))
        self.alg_table.setHorizontalHeaderItem(1, cell('Known?'))
        self.alg_table.setHorizontalHeaderItem(2, cell('Ignore'))
        self.alg_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.alg_table.setStyleSheet('font: 20px;')
        right = QWidget()
        make_vbox(right, [self.f2l_tabs, self.alg_table], margin=0)

        self.alg_detail = QWidget()
        make_hbox(self.alg_detail, [left, right])
        self.alg_detail.hide()

        make_vbox(self, [self.main_view, self.alg_detail], margin=0)

    def change_f2l_tab(self, tab):
        self.f2l_slot = F2L_SLOTS[tab]
        self.render()

    def select_case(self, case_id, is_f2l):
        self.case_id = case_id
        self.is_f2l = is_f2l
        if is_f2l:
            self.f2l_slot = 'Front Right'
            self.f2l_tabs.setCurrentIndex(0)
        else:
            self.f2l_slot = None
        self.render()

    def change_alg_attr(self, alg_id, attr, value):
        with db.get_session() as session:
            alg = session.query_first(db.Algorithm, id=alg_id)
            setattr(alg, attr, bool(value))

    def render(self):
        if self.case_id is not None:
            self.main_view.hide()
            self.alg_detail.show()
            self.f2l_tabs.setVisible(self.is_f2l)

            with db.get_session() as session:
                case = session.query_first(db.AlgCase, id=self.case_id)
                self.alg_label.setText('%s - %s' % (case.alg_set, case.alg_nb))

                diag = render.gen_cube_diagram(case.diagram, type=case.diag_type)
                self.alg_icon.load(diag.encode('ascii'))

                algs = [alg for alg in case.algs if alg.f2l_slot == self.f2l_slot]

                self.alg_table.clearContents()
                self.alg_table.setRowCount(len(algs))
                for [i, alg] in enumerate(algs):
                    self.alg_table.setItem(i, 0, cell(alg.moves))
                    for [j, attr] in [[1, 'known'], [2, 'ignore']]:
                        cb = QCheckBox('')
                        cb.setChecked(bool(getattr(alg, attr)))
                        cb.stateChanged.connect(
                                functools.partial(self.change_alg_attr, alg.id, attr))
                        self.alg_table.setCellWidget(i, j, cb)
        else:
            self.main_view.show()
            self.alg_detail.hide()
        self.update()

# Alg training stuff

N_RECENT_ALGS = 10

class AlgTrainer(QWidget):
    # Signal to async call on main thread (like in main.py)
    schedule_fn = pyqtSignal([object])

    def __init__(self, parent):
        super().__init__(parent)
        self.initialized = False
        self.recent_algs = []

    def init(self):
        if self.initialized:
            return
        self.initialized = True

        title = QLabel('Alg Trainer')
        title.setStyleSheet('font: 24px;')
        self.current = QLabel()
        self.current.setWordWrap(True)

        recents = QWidget(self)
        recents.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.recent = [[QLabel(), QLabel(), QSvgWidget()]
                for y in range(N_RECENT_ALGS)]
        make_grid(recents, self.recent)

        make_vbox(self, [title, self.current, recents])

        self.reset()

        self.schedule_fn.connect(self.run_scheduled_fn, type=Qt.QueuedConnection)

        self.build_regex()

    def run_scheduled_fn(self, fn):
        fn()

    # Build a regex of all the algs in the db. I had started to build a trie to
    # match the current moves to a list of known algorithms, and seeing the
    # complications of that approach when dealing with the needs of the trainer
    # feature (like recognizing slice moves as simultaneous U/D etc moves, or
    # ignoring an incorrect turn that is reversed (like U U') in the middle of
    # an alg, or matching algs that have another alg as a substring, etc.),
    # it became clear just using a regex combined with some preprocessing would
    # be simpler and probably faster
    def build_regex(self):
        regs = []
        with db.get_session() as session:
            for alg in session.query_all(db.Algorithm):
                moves = []
                # Parse the alg to remove rotations
                cube = solver.Cube()
                alg_moves = alg.moves.split()
                if alg_moves[0].startswith('y'):
                    alg_moves.pop(0)
                for move in alg_moves:
                    [m] = solver.parse_alg(move)
                    [_, _, f1, t1, f2, t2] = m
                    if f1 is not None:
                        moves.append((cube.centers[f1], t1))
                        if f2 is not None:
                            moves.append((cube.centers[f2], t2))
                    cube.run_alg(move)

                regex_parts = []
                while moves:
                    [f1, t1] = moves.pop(0)
                    # Pop off another move if it's on the opposite face. This
                    # happens with slice moves, but also algs like G perms with
                    # simultaneous U/D moves.
                    if moves and moves[0][0] == f1 ^ 1:
                        [f2, t2] = moves.pop(0)

                        # Split double moves into component parts: two consecutive
                        # turns in either direction
                        turns_1 = []
                        if t1 == 2:
                            turns_1.append([1, 1])
                            turns_1.append([-1, -1])
                        else:
                            turns_1.append([t1])

                        turns_2 = []
                        if t2 == 2:
                            turns_2.append([1, 1])
                            turns_2.append([-1, -1])
                        else:
                            turns_2.append([t2])

                        # Now for all combinations of the component parts of both
                        # turns, create all sequences composed of the components
                        # interleaved together
                        alt_parts = []
                        for [pt1, pt2] in itertools.product(turns_1, turns_2):
                            # Generate all interleaved orderings. If all the
                            # components together have length n, and m of those
                            # are components of the first move, then there are
                            # n choose m orderings. We take range(n) as all the
                            # indices of the final list and take all possible ways
                            # of choosing m of them to come from the first move.
                            n = len(pt1) + len(pt2)
                            for i1 in itertools.combinations(range(n), len(pt1)):
                                # Copy the components so we can pop them
                                p1 = pt1[:]
                                p2 = pt2[:]
                                part = []
                                for i in range(n):
                                    if i in i1:
                                        part.append((f1, p1.pop(0)))
                                    else:
                                        part.append((f2, p2.pop(0)))
                                assert p1 == [] and p2 == []
                                # And more annoying^Wfun complications! Merge
                                # consecutive turns of the same type, so
                                # R L' L' R becomes R L2 R. Go backwards so we
                                # can modify in place.
                                for i in range(n - 2, -1, -1):
                                    if part[i][0] == part[i+1][0]:
                                        part[i:i+2] = [(part[i][0],
                                                (part[i][1] + part[i+1][1]) % 4)]

                                alt_parts.append(' '.join(solver.move_str(f, t)
                                        for [f, t] in part))

                        regex_parts.append('(%s)' % '|'.join(alt_parts))
                    else:
                        regex_parts.append(solver.move_str(f1, t1))

                regs.append('(?P<g%s>%s)' % (alg.id, ' '.join(regex_parts)))

        self.matcher = re.compile('|'.join(regs))

    def reset(self):
        self.current_moves = []
        self.move_times = []
        self.last_face = None
        self.last_turn = None
        self.schedule_fn.emit(self.render_current)
        self.schedule_fn.emit(self.render_recent)

    def render_current(self):
        self.current.setText(' '.join(self.current_moves))
        self.update()

    def render_recent(self):
        for [l, [t, alg, diag]] in zip(self.recent, self.recent_algs):
            l[0].setText(ms_str(t * 1000))
            l[1].setText(alg)
            l[2].setFixedSize(50, 50)
            l[2].load(diag.encode('ascii'))
        self.update()

    def make_move(self, face, turn):
        # Update current list of moves, collapsing double turns, etc.
        if face == self.last_face:
            self.last_turn = (self.last_turn + turn) % 4
            # Last move was reversed: pop it off the move list
            if not self.last_turn:
                self.current_moves.pop(-1)
                self.move_times.pop(-1)
                self.last_face = None
                if self.current_moves:
                    m = self.current_moves[-1]
                    [self.last_face, self.last_turn] = solver.parse_move(m)
            # Otherwise, merge the moves
            else:
                self.current_moves[-1] = solver.move_str(face, self.last_turn)
        else:
            self.last_face = face
            self.last_turn = turn
            self.current_moves.append(solver.move_str(face, turn))
            self.move_times.append(time.time())

        move_str = ' '.join(self.current_moves)
        match = self.matcher.search(move_str)
        done = False
        if match:
            alg_id = match.lastgroup
            moves = match.group(alg_id)
            with db.get_session() as session:
                alg = session.query_first(db.Algorithm, id=int(alg_id[1:]))
                assert len(self.move_times) == len(self.current_moves)
                start = self.move_times[-len(moves.split())]
                t = time.time() - start
                a = '%s - %s' % (alg.case.alg_set, alg.case.alg_nb)

                # Generate diagram (should cache this or something)
                diag = render.gen_cube_diagram(alg.case.diagram,
                        type=alg.case.diag_type)

                self.recent_algs.insert(0, (t, a, diag))
                self.recent_algs = self.recent_algs[:N_RECENT_ALGS]

            # Reset will also render
            self.reset()
        else:
            self.render_current()

GRAPH_TYPES = ['adaptive', 'date', 'count']

class GraphDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.initialized = False
        self.solve_sets = None

    def init(self):
        if self.initialized:
            return
        self.initialized = True

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
        self.init()

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

            # Count: solve number
            elif self.type == 'count':
                x = range(len(solves))

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
