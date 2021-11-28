# Generate an LCD font in SVG since I'm hardcore like that
# Converted to ttf with https://icomoon.io, which is a pretty nice tool

digits = ['1011111', '0000101', '1110110', '1110101', '0101101', '1111001',
    '1111011', '1000101', '1111111', '1111101']
paths = []
d = .4
# Horizontal
for y in [d, 4, 8-d]:
    x1 = 2*d
    x2 = 4-2*d
    paths.append([[x1, y + d], [x1 - d, y], [x1, y - d],
            [x2, y - d], [x2 + d, y], [x2, y + d]])

# Vertical
for y1 in [2*d, 4+d]:
    for x in [d, 4-d]:
        y2 = y1 + 4-3*d
        paths.append([[x - d, y1], [x, y1 - d], [x + d, y1],
                [x + d, y2], [x, y2 + d], [x - d, y2]])

for [i, digit] in enumerate(digits):
    with open('digits/%s.svg' % i, 'w') as f:
        f.write('<svg xmlns="http://www.w3.org/2000/svg" viewBox="-.4 0 4.4 8" width="48" height="80">')

        for [seg, path] in zip(digit, paths):
            if seg != '1':
                continue
            f.write('<path d="M %s z" />' %
                    ' L '.join('%s %s' % (x, y) for [x, y] in path))

        f.write('</svg>')

with open('digits/dot.svg', 'w') as f:
    f.write('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1 8" width="10" height="80">')

    f.write('<circle cx="%s" cy="%s" r="%s" />' % (.5, 8-d, d))

    f.write('</svg>')

with open('digits/colon.svg', 'w') as f:
    f.write('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1 8" width="10" height="80">')

    f.write('<circle cx="%s" cy="%s" r="%s" />' % (.5, 4, d))
    f.write('<circle cx="%s" cy="%s" r="%s" />' % (.5, 8-d, d))

    f.write('</svg>')
