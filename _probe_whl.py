import urllib.request as u
import re

tags = ['pt26cu124', 'pt26cu121', 'pt26cu118', 'pt25cu124', 'pt24cu124']
for tag in tags:
    url = f'https://docs.gsplat.studio/whl/{tag}/gsplat/'
    try:
        html = u.urlopen(url, timeout=30).read().decode('utf-8', 'ignore')
        whls = sorted(set(re.findall(r'gsplat[^"<>]*\.whl', html)))
        wins = [w for w in whls if 'win' in w.lower()]
        print(f'{tag}: {len(whls)} wheels, {len(wins)} windows')
        for w in wins:
            print('    WIN:', w)
        if whls and not wins:
            print('    (linux-only sample):', whls[0])
    except Exception as e:
        print(f'{tag}: ERR {e}')
