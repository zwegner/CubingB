![icon](rsrc/cubingb-icon-small.png)

CubingB
=======

CubingB is a timer/analyzer for speedsolving Rubik's cubes (and related
puzzles). It focuses on supporting "smart cubes" (i.e. bluetooth cubes) for
recording the exact moves of a solve in real time.

CubingB is at a very early stage, and only supports macOS, and only the MoYu
Weilong AI for smart cube functionality. It's written in Python, and uses
PyQt and PyOpenGL for all sorts of graphical goodness, and pyobjc for binding
to macOS' native CoreBluetooth APIs for communicating with smart cubes. It
uses a SQLite database, managed with SQLAlchemy and Alembic.

This is just a hobby project written for my own use, so don't expect an easy
installation experience or much technical support. If you're not a programmer
that knows how to install Python packages, don't bug me with issues yet, please.

![icon](rsrc/screenshot.png)

Features
---
* Normal cubing timer features: basic scramble generation, single/aoX calculations, etc.
* Session management: renaming, drag-n-drop to reorder, and merge sessions
* Full incremental turn and gyroscope data recorded and timestamped throughout
  solves on a smart cube. This can be viewed later like a video, with
  play/pause/scrubbing. The data is stored in a compact binary format, but can
  still accumulate rather quickly (about 1K of data for 2-3 seconds of solving)
* Click-and-drag to rotate, scroll to zoom on 3D cube viewer
* Smart cube hotkeys: **r** to reset cube state (i.e. make the virtual cube
  solved), **c** to calibrate gyroscope (takes the current gyroscope position
  and assumes its the standard white top, green front orientation)
* All solve data stored in a SQLite database for easy analytics (if you're
  nerdy enough to know SQL)
* CSTimer importing (no interface, just a Python script for now)
* Free and open source, yo

In the near future, the priority will be in analyzing solves (i.e. making sane
reconstructions incorporating the turn and gyroscope data, tracking algorithm
performance, etc.), as well as creating training exercises. That's the whole
reason I wanted a smart cube in the first place. This other stuff was just the
basics to get a decent timer that's good enough for typical usage.

Issues
---
* Cross platform support: PyQt can theoretically run on most platforms. Qt
  supports bluetooth, but apparently it doesn't support the BTLE
  advertisement/scanning, which I think is needed for the MoYu cube (at least
  I couldn't get it to work).
* Other smart cube support: Maybe later? I really like the MoYu cube for now.
* Minor weirdnesses with rotations/gyroscope: no idea! I'm out of my depth on
the math here at the moment. If you know how quaternions work, I'd love some help!

Misc
---

**CubingB doesn't work!** That sucks!

**What does CubingB mean?** Well, it could potentially stand for CubingBuddy, 
CubingBenchmarker, or CubingBehemoth, but really it's just a dumb variation
on a [dumb joke](https://www.youtube.com/watch?v=VJMV-FFKcPU)
