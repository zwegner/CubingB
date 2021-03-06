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
import datetime
import enum
import functools
import gzip
import itertools
import math
import random
import re
import struct
import time

from PySide6.QtCore import Signal
from PySide6.QtCore import (QSize, Qt)
from PySide6.QtWidgets import (QLabel, QDialog, QDialogButtonBox,
        QWidget, QSizePolicy, QScrollArea, QTableWidget, QFrame)
from PySide6.QtSvgWidgets import QSvgWidget

import db
import flow_layout
import render
import solver
from util import *

STAT_OUTLIER_PCT = 5

F2L_SLOTS = ['Front Right', 'Front Left', 'Back Left', 'Back Right']

TrainMode = enum.IntEnum('TrainMode', 'RECOGNIZE DRILL', start=0)
AlgSet = enum.IntEnum('AlgSet', 'PLL OLL F2L AF2L', start=0)

DIAGRAM_TYPE = {AlgSet.PLL: 'pll', AlgSet.OLL: 'oll'}

# Normal cube colors to autodetect in session names. White is intentionally
# left out since the graphs are on white backgrounds
COLORS = ['yellow', 'red', 'orange', 'green', 'blue']
OTHER_COLORS = ['black', 'grey', 'brown', 'purple', 'cyan', 'magenta']

def calc_ao(all_times, start, size):
    if start < size - 1:
        return None
    else:
        times = all_times[start-size+1:start+1]
        if size >= 5:
            times.sort()
            outliers = (size * STAT_OUTLIER_PCT + 99) // 100
            times = times[outliers:-outliers]
        return sum(times) / len(times)

SESSION_ROLLING_CACHE = collections.defaultdict(dict)
SESSION_ROLLING_CACHE_DATE = {}

# Clear any cached information for this session, mainly used when e.g. solves
# are edited or deleted
def clear_session_caches(session):
    # XXX might need to invalidate individual solve stats later, but for now
    # when the 'best' stat cache is cleared it recalculates all solves
    session.cached_stats_current = None
    session.cached_stats_best = None

    if session.id in SESSION_ROLLING_CACHE:
        del SESSION_ROLLING_CACHE[session.id]
        del SESSION_ROLLING_CACHE_DATE[session.id]

# Binary search helper, mainly used for RollingAverage(). Slightly annoying in
# that it needs to return the insertion position for new elements.
def binary_search(l, i):
    lo = 0
    hi = len(l) - 1
    while lo < hi:
        mid = (hi + lo + 1) // 2
        if l[mid] > i:
            hi = mid - 1
        elif l[mid] < i:
            lo = mid
        else:
            return mid
    if l[lo] < i:
        return lo + 1
    return lo

# To speed up rolling aoX calculations, we maintain a incrementally updated
# sliding window of solve times. This is more efficient (at least for bigger
# X), but the code is a bit annoying. We use a binary search on the sorted
# solves within the sliding window, so overall the runtime is O(n log size).
# This class maintains the state of the sliding window so we can keep it
# around in memory and not need to reload a crapload of solves whenever a
# new solve comes in.
class RollingAverage:
    def __init__(self, size, times=[]):
        self.size = size
        outliers = (size * STAT_OUTLIER_PCT + 99) // 100
        self.low = outliers
        self.high = size - outliers
        self.n_samples = self.high - self.low

        # Build initial sliding window, if we have enough solves
        if len(times) >= size:
            self.sliding_window = collections.deque(times[-size:])
            self.sorted_times = list(sorted(self.sliding_window))
            # Keep track of the sum of all solve times within the window
            self.total = sum(self.sorted_times[self.low:self.high])
        else:
            self.sliding_window = collections.deque(times)
            self.sorted_times = None
            self.total = None

    def current(self):
        if self.total is None:
            return None
        return self.total / self.n_samples

    def update(self, new_time):
        self.sliding_window.append(new_time)

        # See if we just got a full sliding window's worth of solves
        if self.sorted_times is None:
            if len(self.sliding_window) == self.size:
                self.sorted_times = list(sorted(self.sliding_window))
                self.total = sum(self.sorted_times[self.low:self.high])
                return self.total / self.n_samples
            # Not enough solves yet, just return None
            assert len(self.sliding_window) < self.size
            return None

        # Get the old time we're pushing out of the sliding window, and add the
        # new time
        old_time = self.sliding_window.popleft()

        # Pull out some attributes for cleaner code below
        [total, times, low, high] = [self.total, self.sorted_times,
                self.low, self.high]
        assert not math.isnan(total)
        # Remove old solve time
        o = binary_search(times, old_time)
        assert times[o] == old_time, (times, old_time, o)
        times.pop(o)

        # Insert new solve
        n = binary_search(times, new_time)
        times.insert(n, new_time)

        # Recalculate total if the average is moving from DNF to non-DNF. If
        # this happens a lot, it negates the speed advantage of this rolling
        # calculation. I guess that's punishment for having too many DNFs.
        if total == INF and times[high - 1] < INF:
            total = sum(times[low:high])
        else:
            # Update total based on what solves moved in/out the window. There
            # are 9 cases, with old and new each in high/mid/low

            # Old in low
            if o < low:
                if n < low:
                    pass
                elif n < high:
                    total += new_time
                    total -= times[low - 1]
                else:
                    total += times[high - 1]
                    total -= times[low - 1]
            # Old in mid
            elif o < high:
                total -= old_time
                if n < low:
                    total += times[low]
                elif n < high:
                    total += new_time
                else:
                    total += times[high - 1]
            # Old in high
            else:
                if n < low:
                    total += times[low]
                    total -= times[high]
                elif n < high:
                    total += new_time
                    total -= times[high]
                else:
                    pass

            # Guard against inf-inf
            if math.isnan(total):
                total = INF

        self.total = total
        return total / self.n_samples

    def feed_times(self, times):
        for t in times:
            yield self.update(t)

def get_session_solves(session, sesh, *filter_args, newest_first=True,
        newer_than=None, limit=None, **filter_kwargs):
    date = db.Solve.created_at
    sort = date.desc() if newest_first else date.asc()
    query = session.query(db.Solve).filter_by(session=sesh)
    if newer_than:
        query = query.filter(db.Solve.created_at > newer_than)
    query = query.filter(*filter_args, **filter_kwargs)
    return query.order_by(sort).limit(limit).all()

def get_session_solve_count(session, sesh):
    if 'solve_count' in sesh.cached_stats_best:
        return sesh.cached_stats_best['solve_count']
    return session.query(db.Solve).filter_by(session=sesh).count()

# Given a DB session and a cubing session, update all single/aoX statistics.
# This is rather annoyingly complicated, all for the purpose of being fast,
# loading very little from the database in most cases. There's all the weird
# cases handled in the RollingAverage class, but also lots of code for keeping
# the caches (both in-memory and database) in sync, and also caching solve numbers.
def calc_session_stats(session, sesh):
    solves = None
    all_times = None

    stats_current = sesh.cached_stats_current or {}
    stats_best = sesh.cached_stats_best or {}
    stats_best_solve = sesh.cached_stats_best_solve_id or {}

    # Check that the 'best' stats are up to date in the database cache. If not,
    # we have to load all the solves and do a rolling calculation for each stat.
    for [size, stat] in STAT_AO_COUNTS_STR:
        if stat not in stats_best:
            best = None
            best_id = None

            # Load solves if we need to
            if all_times is None:
                if solves is None:
                    solves = get_session_solves(session, sesh, newest_first=False)
                all_times = [solve_time(s) for s in solves]

            averages = None
            ctx = None
            if size >= 5:
                ctx = RollingAverage(size)
                SESSION_ROLLING_CACHE[sesh.id][size] = ctx
                averages = iter(ctx.feed_times(all_times))

            last_date = None
            last_time = None
            for i in range(len(all_times)):
                if size >= 5:
                    m = next(averages)
                # For mo3, just do a direct calculation. There are no outliers
                # anyways, so no need for the fancy incremental stuff
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
                last_time = m
                last_date = solves[i].created_at

            # Update cache to go in the database later
            stats_best[stat] = best
            stats_best_solve[stat] = best_id
            stats_current[stat] = last_time
            SESSION_ROLLING_CACHE_DATE[sesh.id] = last_date

    # Calculate newest stats. If we have a rolling average cached, we can just
    # add the newest time(s) and update. We pull all solves from the database
    # newer than the last seen date. Also, we keep track of the number of solves
    # we need to pull afterwards (max_stat_size) for starting the cache, or for
    # computing the stats that don't use rolling caches
    if sesh.id in SESSION_ROLLING_CACHE:
        date = SESSION_ROLLING_CACHE_DATE[sesh.id]
        new_solves = get_session_solves(session, sesh, newest_first=False,
                newer_than=date)

        max_stat_size = 0

        for [size, stat] in STAT_AO_COUNTS_STR:
            if size >= 5:
                ctx = SESSION_ROLLING_CACHE[sesh.id][size]
                value = ctx.current()
                for s in new_solves:
                    value = ctx.update(solve_time(s))
                    if s.cached_stats is None:
                        s.cached_stats = {}
                    s.cached_stats[stat] = value
                stats_current[stat] = value
            elif size > max_stat_size:
                max_stat_size = size
    else:
        max_stat_size = STAT_AO_COUNTS[-1]

    # Grab the last N solves for starting the cache up. If the in-memory cache
    # is empty for this session, we get the maximum stat size (now ao1000).
    # Otherwise, we get biggest size that doesn't use the rolling cache (now mo3).
    times = None
    partial_solves = False
    if solves is None:
        # Get the *newest* solves and then reverse so the limit works
        solves = get_session_solves(session, sesh, limit=max_stat_size)
        solves = solves[::-1]
        partial_solves = True
    times = [solve_time(s) for s in solves[-max_stat_size:]]

    # Calculate new averages, either rolling if it's not cached or single/mo3
    for [size, stat] in STAT_AO_COUNTS_STR:
        if size <= max_stat_size:
            if size >= 5:
                ctx = RollingAverage(size, times=times)
                SESSION_ROLLING_CACHE[sesh.id][size] = ctx
                value = ctx.current()
            else:
                value = calc_ao(times, len(times) - 1, size)
            stats_current[stat] = value

    # Mark the latest solve date in the rolling average for later updates
    if solves:
        SESSION_ROLLING_CACHE_DATE[sesh.id] = solves[-1].created_at

    # Now update best stat for each size
    for [size, stat] in STAT_AO_COUNTS_STR:
        best = stats_best[stat]
        current = stats_current[stat]
        if current and (not best or current < best):
            best = stats_best[stat] = current
            stats_best_solve[stat] = solves[-1].id

    # Also, calculate solve numbers within a session. This is kind of annoying,
    # but we don't want to have to run weird SQL queries to get solve numbers,
    # and I could not manage to get bulk solve number calculations to be efficient
    # *at all* with SQLAlchemy/SQLite. So instead of that "clean" solution, we
    # keep solve numbers cached in the solve entries, along with a "current solve
    # number" in the stats_best dict. This allows us to detect cache clearing (like
    # when a solve is deleted) and re-run the solve numbering.
    if 'solve_count' not in stats_best:
        if solves is None or partial_solves:
            solves = get_session_solves(session, sesh, newest_first=False)
            partial_solves = False
        offset = 0
    else:
        # Solve numbers are mostly good, but grab any new unnumbered solves.
        # This will probably always be the same as the 'new_solves' variable
        # used above, but keeping those in sync would be weird
        solves = get_session_solves(session, sesh, db.Solve.solve_nb == None,
                newest_first=False)
        offset = stats_best['solve_count']

    for [i, s] in enumerate(solves):
        s.solve_nb = i + offset + 1
    stats_best['solve_count'] = len(solves) + offset

    # Update session with cached stats
    sesh.cached_stats_current = stats_current
    sesh.cached_stats_best = stats_best
    sesh.cached_stats_best_solve_id = stats_best_solve
    if solves:
        solves[-1].cached_stats = stats_current.copy()

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
    def __init__(self, parent, case, algs=None):
        super().__init__(parent)
        self.parent = parent
        self.case_id = case.id
        self.is_f2l = ('F2L' in case.alg_set)
        self.algs = algs

        # Generate diagram
        diag = render.gen_cube_diagram(case.diagram, type=case.diag_type)
        svg = QSvgWidget()
        svg.setFixedSize(60, 60)
        svg.load(diag.encode('ascii'))

        label = QLabel('%s - %s' % (case.alg_set, case.alg_nb))
        label.setStyleSheet('font: 14px; font-weight: bold;')
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        left = QWidget()
        make_vbox(left, [svg, label])

        # Show an alg table if requested
        items = [left]
        if algs:
            self.table = AlgTable(self, algs=algs)#, compact=True)
            self.table.select_case(self.case_id, self.is_f2l)
            items.append(self.table)
        else:
            self.table = None

        make_hbox(self, items)

    def mouseReleaseEvent(self, event):
        self.parent.select_case(self.case_id, self.is_f2l)

class AlgTable(QWidget):
    def __init__(self, parent, algs=None, compact=False):
        super().__init__(parent)
        self.case_id = None
        self.f2l_slot = None
        self.algs = algs
        self.compact = compact

        self.f2l_tabs = make_tabs(*F2L_SLOTS, change=self.change_f2l_tab)

        self.alg_table = QTableWidget()
        if compact:
            set_table_columns(self.alg_table, [''], visible=False, stretch=0)
        else:
            set_table_columns(self.alg_table, ['Alg', 'Known', 'Ignore'], stretch=0)
        self.alg_table.verticalHeader().hide()
        self.alg_table.setStyleSheet('QTableWidget { font: 12px; }'
                'QHeaderView::section { font: 10px; }')

        # Not totally sure why this is necessary, but stylesheets weren't cutting it
        self.alg_table.horizontalHeader().resizeSection(1, 40)
        self.alg_table.horizontalHeader().resizeSection(2, 40)

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

        if self.algs:
            algs = self.algs
        else:
            with db.get_session() as session:
                case = session.query_first(db.AlgCase, id=self.case_id)
                algs = [db.make_transient(alg) for alg in case.algs]

        algs = [alg for alg in algs if alg.f2l_slot == self.f2l_slot]

        self.alg_table.clearContents()
        self.alg_table.setRowCount(len(algs))
        for [i, alg] in enumerate(algs):
            a = cell(alg.moves)
            a.setToolTip(alg.moves)
            self.alg_table.setItem(i, 0, a)
            if not self.compact:
                for [j, attr] in [[1, 'known'], [2, 'ignore']]:
                    cb = make_checkbox('',
                            functools.partial(self.change_alg_attr, alg.id, attr),
                            checked=bool(getattr(alg, attr)))
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

        self.alg_set = AlgSet.PLL
        self.alg_set_init = collections.defaultdict(bool)

        self.alg_set_tabs = make_tabs(*AlgSet.__members__,
                change=self.change_alg_set)

        # For each alg set, create an empty scroll area that we'll lazily fill
        # in with algs as the tabs are clicked
        self.alg_set_tables = {}
        alg_tables = []
        for alg_set in AlgSet:
            scroll_area = QScrollArea(self)
            scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            scroll_area.setWidgetResizable(True)
            scroll_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

            scroll_area.hide()
            self.alg_set_tables[alg_set] = scroll_area
            alg_tables.append(scroll_area)

        self.main_view = QWidget()
        make_vbox(self.main_view, alg_tables)

        # Make alg detail view, right side
        self.alg_table = AlgTable(self)

        # Make alg detail view, left side
        self.alg_label = QLabel()
        self.alg_label.setStyleSheet('font: 24px; font-weight: bold;')
        self.alg_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.alg_icon = QSvgWidget()
        self.alg_icon.setFixedSize(120, 120)
        back = make_button('Return to cases',
                functools.partial(self.select_case, None, False))
        left = QWidget()
        layout = make_vbox(left, [back, self.alg_icon, self.alg_label])
        layout.addStretch(1)

        self.alg_detail = QWidget()
        make_hbox(self.alg_detail, [left, self.alg_table])
        self.alg_detail.hide()

        make_vbox(self, [self.alg_set_tabs, self.main_view,
                self.alg_detail], margin=0)

        self.render()

    def change_alg_set(self, tab):
        alg_set = AlgSet(tab)
        if alg_set != self.alg_set:
            self.alg_set = alg_set
            self.render()

    def select_case(self, case_id, is_f2l):
        # Just forward message to the alg table and rerender
        self.alg_table.select_case(case_id, is_f2l)
        self.render()

    def render(self):
        selected = (self.alg_table.case_id is not None)
        self.main_view.setVisible(not selected)
        self.alg_detail.setVisible(selected)

        for alg_set in AlgSet:
            self.alg_set_tables[alg_set].setVisible(self.alg_set == alg_set)

        # Lazy load the current tab's algs for snappier startup
        if not self.alg_set_init[self.alg_set]:
            self.alg_set_init[self.alg_set] = True

            # Build cards for each case
            alg_cards = []
            with db.get_session() as session:
                query = (session.query(db.AlgCase)
                        .filter_by(alg_set=self.alg_set.name)
                        .options(db.sa.orm.joinedload(db.AlgCase.algs)))
                for case in query:
                    db.make_transient(case)
                    algs = [db.make_transient(alg) for alg in case.algs]
                    alg_cards.append(CaseCard(self, case, algs=algs))

            # For this alg set, create a flow layout with all its cases
            contents = QWidget()
            layout = flow_layout.FlowLayout(contents)
            for s in alg_cards:
                layout.addWidget(s)
            self.alg_set_tables[self.alg_set].setWidget(contents)

        if selected:
            with db.get_session() as session:
                case = session.query_first(db.AlgCase, id=self.alg_table.case_id)
                self.alg_label.setText('%s - %s' % (case.alg_set, case.alg_nb))

                diag = render.gen_cube_diagram(case.diagram, type=case.diag_type)
                self.alg_icon.load(diag.encode('ascii'))

        self.update()

class TrainingResultCard(QFrame):
    def __init__(self):
        super().__init__()

        self.setStyleSheet('TrainingResultCard { max-height: 120px; '
                'background-color: #ccc; border-radius: 5px; }')

        self.case = QLabel('')
        self.case.setStyleSheet('font-weight: bold;')
        self.case.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.diagram = QSvgWidget()
        self.diagram.setFixedSize(60, 60)
        left = QWidget(self)
        make_vbox(left, [self.diagram, self.case], margin=0)

        self.message= QLabel('')
        self.recog_time = QLabel('')
        self.time = QLabel('')
        self.tps = QLabel('')
        self.qtm_label = QLabel('Extra moves:')
        self.qtm = QLabel('')
        right = QWidget(self)
        make_grid(right, [
            [self.message],
            [QLabel('Recog. Time:'), self.recog_time],
            [QLabel('Exec. Time:'), self.time],
            [QLabel('TPS:'), self.tps],
            [self.qtm_label, self.qtm],
        ], margin=3)

        self.wrong = QWidget()
        self.message_2 = QLabel('')
        self.wrong_diagram = QSvgWidget()
        self.wrong_diagram.setFixedSize(40, 40)
        make_vbox(self.wrong, [self.wrong_diagram, self.message_2], margin=0)

        make_grid(self, [[left, right, self.wrong]], stretch=[0, 1, 0])

    def set_data(self, case, diagram, recog_time, time, moves, n_qtm,
            wrong_diagram, message, message_2):
        self.case.setText(case)
        self.diagram.load(diagram.encode('ascii'))

        fail = message or wrong_diagram
        color = '#C00' if fail else '#0C0'
        mark = '\u2717' if fail else '\u2713'
        if not message:
            message = 'Incorrect' if fail else 'Correct'

        self.message.setStyleSheet('font-weight: bold; color: %s;' % color)
        self.message.setText('%s %s' % (mark, message))

        self.recog_time.setText(ms_str(recog_time * 1000) if recog_time else '')
        self.time.setText(ms_str(time * 1000) if time else '')
        self.tps.setText('%.2f' % (len(moves.split()) / time) if time else '')

        self.qtm_label.setVisible(n_qtm is not None)
        self.qtm.setVisible(n_qtm is not None)
        self.qtm.setText(str(n_qtm or 0))
        stylesheet = 'color: #C00; font-weight: bold' if n_qtm else ''
        self.qtm_label.setStyleSheet(stylesheet)
        self.qtm.setStyleSheet(stylesheet)

        self.wrong.setVisible(bool(wrong_diagram))
        if wrong_diagram:
            self.wrong_diagram.show()
            self.wrong_diagram.load(wrong_diagram.encode('ascii'))
        self.message_2.setVisible(bool(message_2))
        if message_2:
            self.message_2.setText(message_2)

# Alg training stuff

N_RECENT_ALGS = 10

AUFS = ['', 'U', "U'", 'U2']
PRE_AUFS = [[pre, ''] for pre in AUFS]
PRE_POST_AUFS = [[pre, post] for pre in AUFS for post in AUFS]
PRE_POST_AUFS.sort(key=lambda pp: solver.move_qtm(pp[0]) + solver.move_qtm(pp[1]))

class AlgTrainer(QWidget):
    # Signal to async call on main thread (like in main.py)
    schedule_fn = Signal([object])

    def __init__(self, parent):
        super().__init__(parent)
        self.initialized = False
        self.recent_algs = []
        self.mode = TrainMode.RECOGNIZE
        self.alg_set = AlgSet.PLL
        self.allow_auf = True
        self.stop_after_wrong = True
        self.cube_diag = None
        self.result_history = []
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

        # Make view for recognition mode
        self.recog_view = QWidget(self)
        options = QWidget(self.recog_view)
        self.alg_set_tabs = make_tabs(*AlgSet.__members__,
                change=self.change_alg_set)
        checks = []
        for [attr, text] in [['allow_auf', 'Show AUFs'],
                ['stop_after_wrong', 'Stop after wrong alg']]:
            checks.append(make_checkbox(text,
                    functools.partial(self.change_bool_attr, attr),
                    checked=getattr(self, attr)))
        make_hbox(options, [self.alg_set_tabs] + checks)
        self.cube_view = QSvgWidget()
        self.cube_view.setFixedSize(200, 200)
        result_title = QLabel('Results:')
        result_title.setStyleSheet('font: 20px;')
        self.result_cards = [TrainingResultCard() for i in range(20)]
        for c in self.result_cards:
            c.hide()
        result_cards = QWidget()
        make_vbox(result_cards, [*self.result_cards, Spacer()])
        scroll_area = QScrollArea(self)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(result_cards)
        scroll_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        results = QWidget()
        make_vbox(results, [result_title, scroll_area])
        make_grid(self.recog_view, [
            [options],
            [self.cube_view, results],
        ])

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

        make_vbox(self, [title, self.mode_tabs, self.recog_view,
                self.drill_view], margin=0)

        self.build_regex()

        self.reset()

    def change_bool_attr(self, attr, value):
        setattr(self, attr, bool(value))

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
        size = max(0, min(size.width() - 300, size.height() - 100) - 100)
        self.cube_view.setFixedSize(size, size)

    # Build a regex of all the algs in the db. I had started to build a trie to
    # match the current moves to a list of known algorithms, and seeing the
    # complications of that approach when dealing with the needs of the trainer
    # feature (like recognizing slice moves as simultaneous U/D etc moves, or
    # ignoring an incorrect turn that is reversed (like U U') in the middle of
    # an alg, or matching algs that have another alg as a substring, etc.),
    # it became clear just using a regex combined with some preprocessing would
    # be simpler and probably faster. This also builds up a couple of tables
    # for keeping alg/case data around without querying the DB.
    def build_regex(self):
        self.case_data = {}
        self.alg_set_cases = collections.defaultdict(list)
        self.alg_data = {}
        alg_set_regs = collections.defaultdict(list)
        all_regs = []
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
                alg_moves = ' '.join(alg_moves)

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

                # Build the regex for this entire alg, with the regex capture
                # group keeping track of the alg ID, and add it to the lists
                regex = '(?P<g%s>%s)' % (alg.id, ' '.join(regex_parts))
                all_regs.append(regex)
                if alg.case.alg_set in AlgSet.__members__:
                    alg_set = AlgSet.__members__[alg.case.alg_set]
                    alg_set_regs[alg_set].append(regex)

                    # Add this case to the caches
                    if alg.alg_case_id not in self.case_data:
                        self.alg_set_cases[alg_set].append((alg.alg_case_id,
                                alg_moves))

                        # Generate diagram
                        diag = render.gen_cube_diagram(alg.case.diagram,
                                type=alg.case.diag_type or 'normal')

                        self.case_data[alg.alg_case_id] = DummyObj(
                                alg_set=alg.case.alg_set, case_nb=alg.case.alg_nb,
                                diagram=diag)
                self.alg_data[alg.id] = DummyObj(case_id=alg.alg_case_id,
                        moves=alg_moves, alg_set=alg.case.alg_set,
                        case_nb=alg.case.alg_nb, diagram=diag)

        # Compile the regex parts into big regexes, for all algs, and for each
        # alg set
        self.matcher = re.compile('|'.join(all_regs))
        self.alg_set_matcher = {}
        for [k, v] in alg_set_regs.items():
            self.alg_set_matcher[k] = re.compile('|'.join(v))

        self.current_case_id = list(self.case_data.keys())[0]
        self.move_times = [time.time(), time.time()]
        self.current_moves = ['U']
        self.start_time = time.time() - 1

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
        self.n_qturns = 0
        self.init_time = time.time()

        if self.mode == TrainMode.RECOGNIZE:
            self.cube = solver.Cube()
            self.cross_color = solver.Y

            # Choose a random case. We pick random cases for each alg set
            # later in the CFOP solve, and invert those first, so as to scramble
            # all the pieces. (e.g. for a random F2L case, we go from a solved cube,
            # do an inverse PLL, inverse OLL, then an inverted F2L alg)

            # Choose a random PLL
            [case, alg] = random.choice(self.alg_set_cases[AlgSet.PLL])
            self.run_inverse_alg(alg)
            if self.alg_set >= AlgSet.OLL:
                # Choose a random OLL
                [case, alg] = random.choice(self.alg_set_cases[AlgSet.OLL])
                self.run_inverse_alg(alg)
                # Choose a random F2L
                if self.alg_set == AlgSet.F2L or self.alg_set == AlgSet.AF2L:
                    [case, alg] = random.choice(self.alg_set_cases[self.alg_set])
                    self.run_inverse_alg(alg)

            self.current_case_id = case
            self.current_case_nb = self.case_data[case].case_nb

            self.initial_cube = self.cube.copy()

            # Generate diagram
            self.cube_diag = render.gen_cube_diagram(self.cube)

        self.schedule_fn.emit(self.render)

    def render(self):
        if self.mode == TrainMode.RECOGNIZE:
            self.recog_view.show()
            self.drill_view.hide()
            diag = self.cube_diag
            if not diag:
                diag = render.gen_cube_diagram(None)
            self.cube_view.load(diag.encode('ascii'))
            self.render_results()
        elif self.mode == TrainMode.DRILL:
            self.recog_view.hide()
            self.drill_view.show()

            self.render_current()
            self.render_recent()
        else:
            assert False
        self.update()

    def render_results(self):
        # Maybe want to do something smarter here, actually moving the cards
        # down instead of moving the content down? Unclear if that's faster,
        # though.
        for [card, args] in zip(self.result_cards, self.result_history):
            card.show()
            card.set_data(*args)
        for card in self.result_cards[len(self.result_history):]:
            card.hide()

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

    def get_current_match(self, alg_set=None):
        matcher = self.matcher if alg_set is None else self.alg_set_matcher

        move_str = ' '.join(self.current_moves)
        match = self.alg_set_matcher[self.alg_set].search(move_str)
        if not match:
            return None
        alg_id = match.lastgroup
        moves = match.group(alg_id)
        return self.alg_data[int(alg_id[1:])]

    def push_recog_result(self, wrong_diagram=None, message=None, message_2=None,
            match=None):
        case = self.case_data[self.current_case_id]

        # XXX Should handle AUFs/recog time better here--should recog time stop
        # after the first AUF move? Right now AUFs can cancel out and will just
        # stay in recognition time
        if self.move_times:
            t = time.time() - self.move_times[0]
            r = self.move_times[0] - self.init_time
        else:
            t = r = None

        # Calculate extra moves in quarter-turn metric. We use QTM both since it's
        # easy to calculate, but also because it makes the most sense for recording
        # mistakes: if you do U U' before an algorithm, it shouldn't cancel out.
        n_qtm = None
        if match:
            # Annoying: figure out the optimal AUFs for this particular alg.
            # We scrambled the cube with possibly a different alg, so we can't
            # use the number of AUFs we actually did for it (since some algs work
            # from different angles), and also some cases are symmetric, so multiple
            # pre-AUFs would work. So we just brute force all the different
            # pre/post AUFs and check for solved cubes (post only for PLLs, so
            # this is either 4 or 16 cases).
            aufs = PRE_POST_AUFS if self.alg_set == AlgSet.PLL else PRE_AUFS
            for [pre, post] in aufs:
                cube = self.initial_cube.copy()
                cube.run_alg(pre)
                cube.run_alg(match.moves)
                cube.reorient()
                cube.run_alg(post)
                if self.case_is_solved(cube):
                    alg = ('%s %s %s' % (pre, match.moves, post)).strip()
                    alg_qtm = sum(solver.move_qtm(m) for m in alg.split())
                    n_qtm = self.n_qturns - alg_qtm

                    break
            else:
                # Case isn't solved by alg?? This shouldn't happen but it will
                # and I haven't figured it out yet
                alg_qtm = 0

        # Insert a big tuple at the beginning of the history. This is a bit ugly,
        # the tuple elements are just the args to TrainingResultCard.set_data()
        self.result_history.insert(0, ('%s - %s' % (case.alg_set, case.case_nb),
                case.diagram, r, t, ' '.join(self.current_moves), n_qtm,
                wrong_diagram, message, message_2))

        self.result_history = self.result_history[:len(self.result_cards)]

    def abort(self):
        self.push_recog_result(message='Cancelled')
        self.reset()

    # Check for the cube being partially solved based on which alg set we're
    # using (e.g. only first two layers are solved in an F2L alg)
    def case_is_solved(self, cube):
        if (solver.is_cross_solved(cube, self.cross_color) and
                solver.is_f2l_solved(cube, self.cross_color)):
            if self.alg_set == AlgSet.F2L or self.alg_set == AlgSet.AF2L:
                return True
            elif solver.is_oll_solved(cube, self.cross_color):
                if self.alg_set == AlgSet.OLL:
                    return True
                elif cube == solver.SOLVED_CUBE:
                    return True
        return False

    def make_move(self, face, turn):
        self.n_qturns += 1
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

        match = self.get_current_match()

        # Recognize mode: update internal cube, see if it's solved
        if self.mode == TrainMode.RECOGNIZE:
            self.cube.turn(face, turn % 4)

            # Show AUF turns if that's allowed
            if self.cube_diag:
                if self.allow_auf and face == 0:
                    self.cube_diag = render.gen_cube_diagram(self.cube)
                else:
                    self.cube_diag = None
                self.schedule_fn.emit(self.render)

            if self.case_is_solved(self.cube):
                self.push_recog_result(match=match)
                self.reset()
            # See if the user did a different alg in the same set, and
            # error out if so. If we got here (since the solved checks
            # above failed), either the matched alg here is wrong, or
            # the user still needs to do an AUF. So check this isn't
            # the right alg.
            elif match and self.stop_after_wrong:
                if match.case_nb != self.current_case_nb:
                    self.push_recog_result(wrong_diagram=match.diagram,
                            message='Incorrect',
                            message_2='Did %s - %s' % (match.alg_set,
                            match.case_nb))
                    self.reset()
                # But wait! There's another case here: the user did the
                # right alg, but with extra moves first (e.g. wrong pre-AUF).
                # So check if any U turn solves the cube. If it does, we're
                # just waiting for them to do the post-AUF, otherwise we error
                else:
                    cube = self.cube.copy()
                    for i in range(3):
                        cube.move('U')
                        if self.case_is_solved(cube):
                            break
                    else:
                        # This is a tad weird, just show the cube pre-alg
                        cube = solver.Cube()
                        cube.run_alg(solver.invert_alg(' '.join(self.current_moves)))
                        dt = DIAGRAM_TYPE.get(self.alg_set, 'normal')
                        diagram = render.gen_cube_diagram(cube, type=dt)
                        self.push_recog_result(wrong_diagram=diagram,
                                message='Incorrect', message_2='Wrong pre-moves')
                        self.reset()

        elif self.mode == TrainMode.DRILL:
            # Check for a completed alg
            done = False
            if match:
                assert len(self.move_times) == len(self.current_moves)
                start = self.move_times[-len(match.moves.split())]
                t = time.time() - start
                a = '%s - %s' % (match.alg_set, match.case_nb)

                self.recent_algs.insert(0, (t, a, match.diagram))
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

        session_button = make_button('Select sessions...', self.change_sessions)
        self.session_selector = SessionSelectorDialog(self)

        record_cb = make_checkbox('Only records', self.change_record, checked=False)

        stat_label = QLabel('Stat:')
        # Build a dropdown and lookup tables for stats
        self.stat_table = STAT_AO_STR
        self.stat_table.append('Solves per day')
        self.inv_stat_table = STAT_AO_COUNTS + ['count']
        self.stat_selector = make_dropdown(self.stat_table,
                change=self.change_stat)

        self.type_label = QLabel('Graph Type:')
        self.type_selector = make_dropdown(GRAPH_TYPES, change=self.change_type)

        self.cutoff_label = QLabel('Day cutoff')
        hours = [12] + list(range(1, 12))
        ampm_hours = ['%s%s' % (h, ampm) for ampm in ['am', 'pm'] for h in hours]
        self.cutoff_selector = make_dropdown(ampm_hours,
                change=self.change_cutoff)
        self.cutoff_label.hide()
        self.cutoff_selector.hide()

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
        self.cutoff = 0

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
                    None, self.type_label, self.type_selector, self.cutoff_label,
                    self.cutoff_selector],
            [self.canvas],
            [buttons],
        ], stretch=[0, 1, 0, 1, 0, 0, 1, 0, 0])

    def change_record(self, value):
        self.record = bool(value)
        self.render(keep_ranges=True)

    def change_stat(self):
        was_count = (self.stat == 'count')
        self.stat = self.inv_stat_table[self.stat_selector.currentIndex()]
        is_count = (self.stat == 'count')
        self.render(keep_ranges=not was_count ^ is_count)

    def change_type(self):
        self.type = GRAPH_TYPES[self.type_selector.currentIndex()]
        self.render()

    def change_cutoff(self):
        self.cutoff = self.cutoff_selector.currentIndex()
        self.render(keep_ranges=True)

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
            self.stat_selector.setCurrentIndex(self.inv_stat_table.index(stat))

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
                # The indices data for a stackplot are pretty much useless, so
                # do some hacking to get coordinates
                if self.stat == 'count':
                    if event.xdata:
                        self.tooltip.xy = [event.xdata, event.ydata]

                        [x, y] = self.line_data[name]

                        date = (datetime.date(1970, 1, 1) +
                                datetime.timedelta(days=int(event.xdata)))
                        i = (date - x[0]).days
                        value = y[i] if i < len(x) and x[i] == date else 0

                        self.tooltip.set_text('%s, %s\n%s solves' % (name,
                                date, value))
                else:
                    # Get first point that contains cursor and set tooltip there
                    i = indices['ind'][0]
                    [x, y] = line.get_data()
                    self.tooltip.xy = [x[i], y[i]]

                    # Remap index if we're not showing all solves
                    if name in self.index_map:
                        i = self.index_map[name][i]

                    # Show some neat data in the tooltip
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

        # Get x and y position of the scroll event in the coordinates of the
        # data. We'd usually use event.xdata/.ydata, but that is only valid
        # when scrolling inside the axis
        [cx, cy] = self.plot.transData.inverted().transform([event.x, event.y])

        [left, right] = self.plot.get_xlim()
        if left <= cx <= right:
            left = max(self.left_limit, cx - scale * (cx - left))
            right = min(self.right_limit, cx + scale * (right - cx))
            self.plot.set_xlim(left, right)

        [bottom, top] = self.plot.get_ylim()
        if bottom <= cy <= top:
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

        is_count = self.stat == 'count'
        self.type_label.setVisible(not is_count)
        self.type_selector.setVisible(not is_count)
        self.cutoff_label.setVisible(is_count)
        self.cutoff_selector.setVisible(is_count)

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
        self.line_data = {}
        self.index_map = {}
        color_index = 0

        # Helper to get graph color for a session
        def get_color(name):
            nonlocal color_index
            # Autodetect colors from the session name
            for c in COLORS:
                if c in name:
                    return c
            # Otherwise, choose a neutral color
            else:
                if color_index < len(OTHER_COLORS):
                    color = OTHER_COLORS[color_index]
                    color_index += 1
                    return color

        # Handle 'solves per day' metric
        if is_count:
            # Count all solves based on the day
            dates = {}
            counts = {}
            all_dates = set()
            # Get a delta to add to each time based on the cutoff
            delta = datetime.timedelta(hours=-self.cutoff)
            for [name, solves] in self.solve_sets:
                dates[name] = [(d + delta).date() for [i, d, s] in solves]
                counts[name] = collections.Counter(dates[name])
                all_dates.update(counts[name].keys())

            # Make a list of all dates in the range of the min/max dates
            x = []
            date = min(all_dates)
            last_date = max(all_dates)
            while date <= last_date:
                date += datetime.timedelta(days=1)
                x.append(date)

            # Collect counts per session on each day
            y = {}
            for [name, solves] in self.solve_sets:
                y[name] = [counts[name][xx] for xx in x]
                self.line_data[name] = [x, y[name]]

            # Make a stack plot, and save the individual stack elements
            lines = self.plot.stackplot(x, y.values(), labels=y.keys(),
                    colors=[get_color(name) for [name, _] in self.solve_sets])
            for [[name, _], line] in zip(self.solve_sets, lines):
                self.lines[name] = line
        else:
            stat = stat_str(self.stat)
            # Create plot for each session's solves
            for [name, solves] in self.solve_sets:
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
                            color=get_color(name))
                else:
                    [line] = self.plot.plot(x, y, '.', label=name, markersize=1,
                            color=get_color(name))

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
