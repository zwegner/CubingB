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
import gzip
import math
import struct

from PyQt5.QtCore import (QSize, Qt)
from PyQt5.QtWidgets import (QLabel, QComboBox, QDialog, QDialogButtonBox)

import solver
from util import *

STAT_OUTLIER_PCT = 5

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
        self.solved = (c == solver.SOLVED_CUBE)

# UI Widgets

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
