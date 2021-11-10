import argparse
import json
import datetime
import sys

import config
import db

# XXX add more scrambles
SCRAMBLE_TYPE = {
    '222so': '2x2',
    'sqrs': 'sq1',
    None: '3x3',
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(action='store', dest='backup_file', help='filename of CSTimer backup')
    parser.add_argument('-p', '--prop-file', action='store', help='separate CSTimer backup '
            'file with session properties, in case a backup lost names')
    args = parser.parse_args()

    # Load backup file JSON and properties

    with open(args.backup_file) as f:
        data = json.load(f)

    if args.prop_file:
        with open(args.prop_file) as f:
            properties = json.load(f)
        session_data = json.loads(properties['properties']['sessionData'])
    else:
        session_data = json.loads(data['properties']['sessionData'])

    db.init_db(config.DB_PATH)

    # Parse solve records
    with db.get_session() as session:
        for [k, v] in data.items():
            if not k.startswith('session'):
                continue
            s_id = k[7:]
            name = k
            scr_type = None
            if s_id in session_data:
                s_data = session_data[s_id]
                name = s_data['name']
                scr_type = s_data['opt'].get('scrType')
                scr_type = SCRAMBLE_TYPE[scr_type]

            sesh = session.insert(db.CubeSession, name=name, scramble_type=scr_type)
            for solve in v:
                [result, scramble, notes, ts] = solve
                dnf = (result[0] == -1)
                plus_2 = (result[0] == 2000)
                time_ms = result[1]
                seg_ms = result[-1:1:-1] or None
                dt = datetime.datetime.fromtimestamp(ts)

                session.insert(db.Solve, session=sesh, scramble=scramble, 
                    time_ms=time_ms, segment_time_ms=seg_ms, dnf=dnf, plus_2=plus_2,
                    created_at=dt, notes=notes)

            print('imported %s solves from %s' % (len(v), name))

if __name__ == '__main__':
    main()
