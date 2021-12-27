# This script is for personal use. It is not an intended component of the EphysAnalysisTools package.
# 
# Copyright (c) 2021 Delbert Yip
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import json 
import os 

F431A_dates = dict(
    F431A__env_20=['20910004', '20911007', '20o08003', '20o16001'],
    F431A__env_20_pub=['20o16001', '20910004'],

    F431A__env_55=['20o16002', '21521006'],
    F431A__env_55_pub=['20o16002', '21521006'],

    F431A__ramp_dt=['20o16003', '21121007', '21521001'],
    F431A__ramp_dt_pub=['20o16003', '21121007', '21521001'],

    F431A__de=['19d27002', '20116002', '20117008', '21521005', '20n20002'],
    F431A__de_pub=['20117008', '21521005'],

    F431A__act=['20910002', '20910005', '20911000',
                '21521000', '21521004', '20o16004', '21121006'],
    F431A__act_pub=['20910005', '20911000', '21521000'],
)

WT_dates = dict(
    WT__env_20=['19d22010', '21416009', '20917001', '20917007', '21618000'],
    WT__env_20_pub=['19d22010'],

    WT__env_55=['20917002', '20917010', '21416008'],
    WT__env_55_pub=['20917002', '21416008'],

    WT__ramp_dt=['20917000', '20d10010', '21416006', '21618001'],
    WT__ramp_dt_pub=['20d10010'],

    WT__de=['20917009', '20d03004', '20d10001',
            '20d10009', '21416007', '21430005'],
    WT__de_pub=['20917009', '20d10001', '20d10009', '21416007'],

    WT__act=['20917008', '20d10000', '20d10002', '20d10005',
             '21416005', '21430003', '21430004', '21430012'],
    WT__act_pub=['21416005', '21430003', '20d10002'],
)

f = r"./data/curatedFileNames.json"
if os.path.isfile(f):
    with open(f, 'r') as io: print(json.load(io))
else:
    res = dict(WT=WT_dates, F431A=F431A_dates)
    with open(f, "w") as io: json.dump(res, io)
    
    