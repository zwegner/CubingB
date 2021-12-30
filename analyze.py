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
import functools
import gzip
import itertools
import math
import random
import re
import struct
import time

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtCore import (QSize, Qt)
from PyQt5.QtWidgets import (QLabel, QDialog, QDialogButtonBox,
        QWidget, QSizePolicy, QScrollArea, QTableWidget, QPushButton,
        QCheckBox)
from PyQt5.QtSvg import QSvgWidget

import db
import flow_layout
import render
import solver
from util import *

STAT_OUTLIER_PCT = 5

F2L_SLOTS = ['Front Right', 'Front Left', 'Back Left', 'Back Right']

TrainMode = enum.IntEnum('TrainMode', 'RECOGNIZE DRILL', start=0)
AlgSet = enum.IntEnum('AlgSet', 'F2L OLL PLL', start=0)

# Normal cube colors to autodetect in session names. White is intentionally
# left out since the graphs are on white backgrounds
COLORS = ['yellow', 'red', 'orange', 'green', 'blue']
OTHER_COLORS = ['black', 'grey', 'brown', 'purple', 'cyan', 'magenta']

def calc_ao(all_times, start, size):
    if len(all_times) - start < size:
        mean = None
    else:
        times = all_times[start:start+size]
        if size >= 5:
            times.sort()
            outliers = (size * STAT_OUTLIER_PCT + 99) // 100
            times = times[outliers:-outliers]
        mean = sum(times) / len(times)

    return mean

def get_ao_str(solves, start, size):
    if len(solves) - start < size or size < 5:
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
            if size >= 5:
                averages = iter(calc_rolling_ao(solves,
                        all_times, size))

            for i in range(len(solves)):
                if size >= 5:
                    m = next(averages)
                elif size > 1:
                    m = calc_ao(all_times, i, size)
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

class AlgTable(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.case_id = None
        self.f2l_slot = None

        self.f2l_tabs = make_tabs(*F2L_SLOTS, change=self.change_f2l_tab)

        self.alg_table = QTableWidget()
        set_table_columns(self.alg_table, ['Alg', 'Known?', 'Ignore'], stretch=0)
        self.alg_table.setStyleSheet('font: 20px;')

        make_vbox(self, [self.f2l_tabs, self.alg_table], margin=0)

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
        if self.case_id is None:
            return
        self.f2l_tabs.setVisible(self.is_f2l)

        with db.get_session() as session:
            case = session.query_first(db.AlgCase, id=self.case_id)

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

        self.update()

class AlgViewer(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.initialized = False

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

        # Make alg detail view, right side
        self.alg_table = AlgTable(self)

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

        self.alg_detail = QWidget()
        make_hbox(self.alg_detail, [left, self.alg_table])
        self.alg_detail.hide()

        make_vbox(self, [self.main_view, self.alg_detail], margin=0)

    def select_case(self, case_id, is_f2l):
        # Just forward message to the alg table and rerender
        self.alg_table.select_case(case_id, is_f2l)
        self.render()

    def render(self):
        selected = (self.alg_table.case_id is not None)
        self.main_view.setVisible(not selected)
        self.alg_detail.setVisible(selected)

        if selected:
            with db.get_session() as session:
                case = session.query_first(db.AlgCase, id=self.alg_table.case_id)
                self.alg_label.setText('%s - %s' % (case.alg_set, case.alg_nb))

                diag = render.gen_cube_diagram(case.diagram, type=case.diag_type)
                self.alg_icon.load(diag.encode('ascii'))

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
        self.mode = TrainMode.RECOGNIZE
        self.alg_set = AlgSet.OLL
        self.cube_diag = None
        self.status = ''
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.schedule_fn.connect(self.run_scheduled_fn, type=Qt.QueuedConnection)

    def run_scheduled_fn(self, fn):
        fn()

    def init(self):
        if self.initialized:
            return
        self.initialized = True

        title = QLabel('Alg Trainer')
        title.setStyleSheet('font: 24px; padding: 10px')
        title.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        # Make top option bar
        self.mode_tabs = make_tabs('Recognize', 'Drill', change=self.change_mode)
        self.alg_set_tabs = make_tabs(*AlgSet.__members__,
                change=self.change_alg_set)
        top = QWidget(self)
        make_hbox(top, [self.mode_tabs, self.alg_set_tabs])

        # Make view for recognition mode
        self.recog_view = QWidget(self)
        self.cube_view = QSvgWidget()
        self.cube_view.setFixedSize(200, 200)
        self.recog_status = QLabel()
        title.setStyleSheet('font: 18px;')
        make_vbox(self.recog_view, [self.recog_status, self.cube_view])

        # Make view for drill mode
        self.drill_view = QWidget(self)
        self.current = QLabel()
        self.current.setWordWrap(True)
        recents = QWidget()
        recents.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.recent = [[QLabel(), QLabel(), QSvgWidget()]
                for y in range(N_RECENT_ALGS)]
        make_grid(recents, self.recent)
        make_vbox(self.drill_view, [self.current, recents])

        make_vbox(self, [title, top, self.recog_view,
                self.drill_view], margin=0)

        # Collect all the alg cases for training
        self.alg_cases = collections.defaultdict(list)
        with db.get_session() as session:
            for case in session.query_all(db.AlgCase):
                if case.alg_set in AlgSet.__members__:
                    alg_set = AlgSet.__members__[case.alg_set]
                    self.alg_cases[alg_set].append((case.alg_nb, case.algs[0].moves))

        self.reset()

        self.build_regex()

    def change_mode(self, tab):
        self.mode = TrainMode(tab)
        if self.mode == TrainMode.RECOGNIZE:
            self.alg_set_tabs.show()
        else:
            self.alg_set_tabs.hide()
        self.render()

    def change_alg_set(self, tab):
        alg_set = AlgSet(tab)
        if alg_set != self.alg_set:
            self.alg_set = alg_set
            self.reset()

    def resizeEvent(self, event):
        size = event.size()
        size = max(0, min(size.width(), size.height() - 100) - 100)
        self.cube_view.setFixedSize(size, size)

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
                    [f1, t1, f2, t2] = solver.MOVE_COMPONENTS[move]
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

    def run_inverse_alg(self, moves):
        # Add random pre/post AUF
        AUFS = ['', 'U', 'U2', "U'"]
        moves = '%s %s %s' % (random.choice(AUFS), moves,
                random.choice(AUFS))

        # Run the inverse algorithm on the cube, then reorient
        self.cube.run_alg(solver.invert_alg(moves))
        self.cube.reorient()

    def reset(self):
        self.current_moves = []
        self.move_times = []
        self.last_face = None
        self.last_turn = None

        if self.mode == TrainMode.RECOGNIZE:
            self.cube = solver.Cube()
            self.cross_color = solver.Y

            # Choose a random case. We pick random cases for each alg set
            # later in the CFOP solve, and invert those first, so as to scramble
            # all the pieces. (e.g. for a random F2L case, we go from a solved cube,
            # do an inverse PLL, inverse OLL, then an inverted F2L alg)

            # Choose a random PLL
            [case, alg] = random.choice(self.alg_cases[AlgSet.PLL])
            self.run_inverse_alg(alg)
            if self.alg_set <= AlgSet.OLL:
                # Choose a random OLL
                [case, alg] = random.choice(self.alg_cases[AlgSet.OLL])
                self.run_inverse_alg(alg)
                # Choose a random FLL
                if self.alg_set == AlgSet.F2L:
                    [case, alg] = random.choice(self.alg_cases[AlgSet.F2L])
                    self.run_inverse_alg(alg)

            # Generate diagram
            self.cube_diag = render.gen_cube_diagram(self.cube)

        self.schedule_fn.emit(self.render)

    def render(self):
        if self.mode == TrainMode.RECOGNIZE:
            self.recog_view.show()
            self.drill_view.hide()
            self.cube_view.load(self.cube_diag.encode('ascii'))
            self.recog_status.setText(self.status)
        elif self.mode == TrainMode.DRILL:
            self.recog_view.hide()
            self.drill_view.show()

            self.render_current()
            self.render_recent()
        else:
            assert False
        self.update()

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

        # Recognize mode: update internal cube, see if it's solved
        if self.mode == TrainMode.RECOGNIZE:
            alg = solver.move_str(face, turn)
            self.cube.run_alg(alg)
            # Check for the cube being partially solved based on which
            # alg set we're using (e.g. only first two layers are solved in
            # an F2L alg)
            solved = False
            if (solver.is_cross_solved(self.cube, self.cross_color) and
                    solver.is_f2l_solved(self.cube, self.cross_color)):
                if self.alg_set == AlgSet.F2L:
                    solved = True
                elif solver.is_oll_solved(self.cube, self.cross_color):
                    if self.alg_set == AlgSet.OLL:
                        solved = True
                    elif self.cube == solver.SOLVED_CUBE:
                        solved = True

            if solved:
                self.status = 'Nice!'
                self.reset()
            else:
                if self.status != '':
                    self.status = ''
                    self.schedule_fn.emit(self.render)

        elif self.mode == TrainMode.DRILL:
            # Check for a completed alg
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

        select_label = QLabel('Graph Type:')
        self.selector = make_dropdown(GRAPH_TYPES, change=self.change_type)

        session_button = QPushButton('Select sessions...')
        session_button.clicked.connect(self.change_sessions)
        self.session_selector = SessionSelectorDialog(self)

        stat_label = QLabel('Stat:')
        self.stat_selector = make_dropdown([stat_str(s) for s in STAT_AO_COUNTS],
                change=self.change_stat)

        record_cb = QCheckBox('Only records')
        record_cb.setChecked(False)
        record_cb.stateChanged.connect(self.change_record)

        self.figure = Figure()
        self.canvas = FigureCanvasQTAgg(self.figure)

        self.figure.canvas.mpl_connect('scroll_event', self.handle_scroll)
        self.figure.canvas.mpl_connect('button_press_event', self.handle_press)
        self.figure.canvas.mpl_connect('button_release_event', self.handle_release)
        self.figure.canvas.mpl_connect('motion_notify_event', self.handle_move)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok)
        buttons.accepted.connect(self.accept)

        self.type = 'adaptive'
        self.stat = 'ao100'

        self.plot = None
        self.line = None
        self.index_map = None
        self.tooltip = None
        self.record = False

        # Pan/zoom stuff
        self.pressed = None
        self.bottom_limit = -1e100
        self.top_limit = 1e100
        self.left_limit = -1e100
        self.right_limit = 1e100

        make_grid(self, [
            [session_button, None, record_cb, None, stat_label, self.stat_selector,
                    None, select_label, self.selector],
            [self.canvas],
            [buttons],
        ], stretch=[0, 1, 0, 1, 0, 0, 1, 0, 0])

    def change_record(self, value):
        self.record = bool(value)
        self.render(keep_ranges=True)

    def change_stat(self):
        self.stat = STAT_AO_COUNTS[self.stat_selector.currentIndex()]
        self.render(keep_ranges=True)

    def change_type(self):
        self.type = GRAPH_TYPES[self.selector.currentIndex()]
        self.render()

    def change_sessions(self):
        # Update session selector with all sessions
        with db.get_session() as session:
            sessions = session.query(db.Session)
            sessions = sorted(sessions.all(), key=session_sort_key)
            self.session_selector.update_data('Sessions:', sessions, allow_multi=True)

        if self.session_selector.exec():
            ids = self.session_selector.selected_session_id
            self.update_data(ids, stat=self.stat)

    def update_data(self, session_ids, stat=1):
        self.init()

        self.stat = stat

        with block_signals(self.stat_selector):
            self.stat_selector.setCurrentIndex(STAT_AO_COUNTS.index(stat))

        with db.get_session() as session:
            self.solve_sets = []
            for id in session_ids:
                sesh = session.query_first(db.Session, id=id)
                self.solve_sets.append((sesh.name,
                        [(i+1, s.created_at, s.cached_stats)
                            for [i, s] in enumerate(sesh.solves)]))

        self.render()

    def handle_move(self, event):
        # See if the user is hovering over a point, and show a tooltip if so
        for [name, solves] in self.solve_sets:
            line = self.lines[name]

            [contained, indices] = line.contains(event)
            if contained:
                # Get first point that contains cursor and set tooltip there
                i = indices['ind'][0]
                [x, y] = line.get_data()
                self.tooltip.xy = [x[i], y[i]]

                # Remap index if we're not showing all solves
                if name in self.index_map:
                    i = self.index_map[name][i]

                # Show some neat data there
                [i, d, s] = solves[i]
                t = s[stat_str(self.stat)]
                self.tooltip.set_text('%s, %s\nSolve %s: %s' % (name,
                        d, i, ms_str(t)))

                self.tooltip.set_visible(True)
                self.figure.canvas.draw_idle()
                break
        # No hovered point: hide tooltip if it's visible
        # Too bad Python doesn't have for..elif syntax
        else:
            if self.tooltip.get_visible():
                self.tooltip.set_visible(False)
                self.figure.canvas.draw_idle()

        # Handle panning
        if not self.pressed or event.xdata is None or event.ydata is None:
            return
        [ox, oy] = self.pressed
        dx = ox - event.xdata
        dy = oy - event.ydata

        [left, right] = self.plot.get_xlim()
        dx = min(self.right_limit - right, max(self.left_limit - left, dx))
        self.plot.set_xlim(left + dx, right + dx)

        [bottom, top] = self.plot.get_ylim()
        dy = min(self.top_limit - top, max(self.bottom_limit - bottom, dy))
        self.plot.set_ylim(bottom + dy, top + dy)

        self.figure.canvas.draw_idle()

    def handle_scroll(self, event):
        scale = 1.1
        if event.step > 0:
            scale = 1 / scale

        [cx, cy] = [event.xdata, event.ydata]

        [left, right] = self.plot.get_xlim()
        left = max(self.left_limit, cx - scale * (cx - left))
        right = min(self.right_limit, cx + scale * (right - cx))
        self.plot.set_xlim(left, right)

        [bottom, top] = self.plot.get_ylim()
        bottom = max(self.bottom_limit, cy - scale * (cy - bottom))
        top = min(self.top_limit, cy + scale * (top - cy))
        self.plot.set_ylim(bottom, top)

        self.figure.canvas.draw_idle()

    def handle_press(self, event):
        self.pressed = (event.xdata, event.ydata)

    def handle_release(self, event):
        self.pressed = None

    def render(self, keep_ranges=False):
        self.init()

        limit_x = None
        limit_y = None
        if keep_ranges and self.plot:
            limit_x = self.plot.get_xlim()
            limit_y = self.plot.get_ylim()

        self.figure.clear()
        self.plot = self.figure.add_subplot()

        # Set up tooltip, to be activated in the handle_move() function
        self.tooltip = self.plot.annotate('', xy=(0, 0), xytext=(20, 10),
                textcoords='offset points', bbox={'boxstyle': 'round', 'fc': 'w'})
        self.tooltip.set_visible(False)
        self.tooltip.set_wrap(True)

        # Preprocessing for different graph types
        if self.type == 'count':
            maxlen = max(len(s) for [_, s] in self.solve_sets)
        elif self.type == 'adaptive':
            solves_per_day = collections.Counter(d.date()
                for [_, solves] in self.solve_sets for [i, d, s] in solves)
            day_ordinal = {d: i for [i, d] in enumerate(sorted(solves_per_day))}
            first_day = min(solves_per_day)

        self.lines = {}
        self.index_map = {}
        color_index = 0

        stat = stat_str(self.stat)

        # Create plot for each session's solves
        for [name, solves] in self.solve_sets:
            # Autodetect colors from the session name
            color = None
            for c in COLORS:
                if c in name:
                    color = c
                    break
            # Otherwise, choose a neutral color
            else:
                if color_index < len(OTHER_COLORS):
                    color = OTHER_COLORS[color_index]
                    color_index += 1

            # Create x series based on graph type

            # Date: just use the solve time
            if self.type == 'date':
                x = [d for [i, d, s] in solves]

            # Count: solve number
            elif self.type == 'count':
                x = range(len(solves))

            # Adaptive: for every day that had solves, stretch all solves out
            # evenly throughout the day
            elif self.type == 'adaptive':
                sesh_solves_per_day = collections.Counter(d.date()
                        for [i, d, s] in solves)

                x = []
                last_day = None
                dc = 0
                for [i, d, s] in solves:
                    d = d.date()
                    if last_day and d > last_day:
                        dc = 0
                    last_day = d
                    x.append(day_ordinal[d] +
                            dc / sesh_solves_per_day[d])
                    dc += 1

            # Create y series from the given stat
            y = [s[stat] / 1000 if s[stat] else None
                    for [i, d, s] in solves]

            # Record mode: only show the best stat up to a given point
            if self.record:
                best = 1e100
                indices = []
                new_x = []
                new_y = []
                for [i, [xx, yy]] in enumerate(zip(x, y)):
                    if yy and yy < best:
                        indices.append(i)
                        best = yy
                        new_x.append(xx)
                        new_y.append(yy)

                self.index_map[name] = indices
                [line] = self.plot.step(new_x, new_y, label=name, where='post',
                        color=color)
            else:
                [line] = self.plot.plot(x, y, '.', label=name, markersize=1,
                        color=color)

            self.lines[name] = line

        # Set the zoom limits
        [self.left_limit, self.right_limit] = self.plot.get_xlim()
        [self.bottom_limit, self.top_limit] = self.plot.get_ylim()

        # Set the old view box if we're just changing graph parameters.
        # Have to do this after we get the plot limits because the parameter
        # change can affect the range of the data.
        if keep_ranges:
            self.plot.set_xlim(*limit_x)
            self.plot.set_ylim(*limit_y)

        self.plot.legend(loc='upper right')
        self.canvas.draw()
