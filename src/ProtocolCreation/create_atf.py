import pandas as pd 
import numpy as np 
from pathlib import Path 
import pyabf 

"""
Last update: April 2, 2021
How to use this file to create arbitrary stimulus waveforms for pClamp.

1. Read in a protocol as a numpy array. The first column is time, the rest are voltages. 
2. Edit the string `ATF_HEADER` to include "Trace #x" for however many traces there are. These should be placed in the last row of `ATF_HEADER` with a tab in between. 
3. Edit the function `create_atf` to use the appropriate save location. Finally, edit the line 
    `out+="\n%.05f\t%.05f"%(i/rate,val[1], val[2])`
    so that there is a `\n%0.5f' for each trace. For instance, there are two traces being constructed in the example above. 
4. At this stage, the ATF file is ready to import into pClamp. However, you still need to:
    a. Open the file in Clampex, 
    b. Create a new Episodic protocol. 
    c. Specify the desired .atf file as the stimulus file, and adjust the following:
        i. Holding level. 
        ii. Sweep-to-sweep interval.
        iii. Pre-trigger length (supposedly; can't seem to access in pClamp 10)
            `Edit Protocol > Trigger > Trigger Settings > Pretrigger length (samples)`
    d. FInally, ensure that the following match what was used to make the protocol:
        i. Sampling frequency 
        ii. Number of sweeps (change the second digit in second line of ATF_HEADER to match the number of sweeps, change the number of Trace # in the last row of the header, and then change the values appended in the create_atf function)
    
    At this stage, make sure to preview the protocol between edits to ensure the protocol was created as intended.
    
That's it.
"""

def default_header():
    
    ATF_HEADER="""
    ATF	1.0
    8	NUMBER_OF_SWEEPS
    "AcquisitionMode=Episodic Stimulation"
    "Comment="
    "YTop=200"
    "YBottom=-200"
    "SyncTimeUnits=20"
    "SweepStartTimesMS=0.000"
    "SignalsExported=IN 0   OUT 0"
    "Signals="	"IN 0"
    "Time (s)"
    """.strip()
    # "Time (s)"	"Trace #1"  "Trace #2"
    
    return ATF_HEADER

# path to protocol files 
protocol_path = r"C:/Users/delbe/Downloads/wut/wut/Post_grad/UBC/Research/lab/Github_repos/hcn-gating-kinetics/data/protocols/"
class CreateATF():
    def __init__(self, filename, protocol_path=protocol_path):
        """
        `filename` = name of created .ATF file 
        `protocol_path` = path to directory containing corresponding .CSV file 
        """
        self.fname = filename 
        
        # path to .CSV file 
        protocol_path += "%s.csv" % filename
        # check .CSV file exists 
        if not Path(protocol_path).isfile():
            raise Exception("\n Could not find protocol at {p}".format(p=protocol_path))
            
        # read in csv file, then convert to numpy array 
        df_data = pd.read_csv(protocol_path, header=0).to_numpy()
        print(" Shape of data: \n", df_data.shape)
        self.df_data = df_data 

        # create and save .ATF file 
        self.save_path = protocol_path[:-4] + ".atf"
        create_atf(df_data, filename=save_path)
        print(" Successfully created .atf file at \n", save_path)
        
    def make_header(self):
        """
        Create ATF header with appropriate number of sweeps and column labels \\
            
        Returns string containing ATF header 
        """
        ATF_HEADER = default_header()
        head = ATF_HEADER.replace("NUMBER_OF_SWEEPS", str(self.df_data.shape[1]-1))
            
        for i in range(1, self.df_data.shape[1]):
            head += "\t" + """	"Trace #{i}" """"".format(i=i).strip()
        
        return head 
    
    def make_row(self, t, row):
        """
        Convert single `row` of data into string format for .ATF file. \\
        `t` = time (index of row divided by sampling frequency) \\
            
        Returns string `s` representing respective row in ATF file 
        """
        s = "\n%0.5f" % t 
        for v in row[1:]:
            s += "\t%0.5f" % v
        
        return s 
    
    def create_atf(self, filename=r"./output.atf"):
        """
        Save a stimulus waveform array as an ATF 1.0 file with filename `filename`
        """
        # create ATF header 
        header = self.make_header()
        
        # instantiate output string, beginning with ATF header 
        # data for each trace and tiempoint will be added as newlines (rows) 
        out = header
        
        # sampling frequency
        rate = 1000/(self.df_data[1,0] - self.df_data[0,0])
        print(" The sampling frequency is %.0f kHz" % rate)
        
        for i, val in enumerate(self.df_data):
            out += self.make_row(i/rate, val)
            
        with open(filename,'w') as f:
            f.write(out)
            print("wrote", filename)
        return

# name of output protocol in csv format 
filename = "WT_stag-act_RR1_d15"
# path where output csv will be saved 
protocol_path = r"C:/Users/delbe/Downloads/wut/wut/Post_grad/UBC/Research/lab/Github_repos/hcn-gating-kinetics/data/protocols/%s.csv" % filename 
