# -*- coding: utf-8 -*-

import requests
for i in range(86):
    print(i)
    file_name = "../xml/price"+str(i) + ".xml"
    r = requests.get("http://api.e-stat.go.jp/rest/2.0/app/getStatsData?appId=c18396ba52648c00fa21d7962d6f7440f8618f9c&statsDataId=0003013703&startPosition=" + str(i * 100000 + 1))
    f = open(file_name, 'w')
    f.write(r.text.encode('utf_8'))
    f.close
