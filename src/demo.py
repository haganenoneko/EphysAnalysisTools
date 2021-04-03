from EphysProcessing import process 
import post_process as post
import ActivationCurves
from EphysInfoFilter import EphysInfoFiltering



def ghk_example():
        ion_set = {"Na" : [14, 110], "K" : [135, 35], "Cl" : [141, 144.6]} 
        P_K = 8.5e-5
        E = math.exp(-15*96.485/8.314/298)
        P_Na = P_K*((E*135 - 35)/(110 - E*14)) 

        def ghk(v):
                K = ion_set["K"]
                Na = ion_set["Na"]
                
                e = (v*96.485)/(8.3145*298) 
                
                if v == 0:
                        return 96.485*(P_K*(K[0] - K[1]) + P_Na*(Na[0] - Na[1])) 
                else:
                        return 96.485*e*( 
                                (P_K*(K[0] - K[1]*math.exp(-e)) + P_Na*(Na[0] - Na[1]*math.exp(-e)))/
                                (1 - math.exp(-e))                 
                        )

        vrange = range(-35, 50)
        I = [ghk(v) for v in vrange]
        plt.plot(vrange, I, ls='--', alpha=0.5)
        plt.show()
        exit()

a = ActivationCurves.Summarize_GV(
        ["21326-21304__WT_act_stag__Y459A__act_norm.csv",
        # "all__WT_act_stag__WT mHCN2__act_norm.csv"
        ],
        individual=True
        )
a.go(fname_as_label=-1)
exit()

P = process(
        filter_criteria = {
                "Dates" : ["all"],
                "Protocol_Name" : ["WT_act_stag_R_"],
                "Files_To_Skip" : [
                        "20903003", "20903004", "21304025",
                        "20d03002", "20d04000",
                        "21304019", "21304020", "21304021", "21326022"],
                "Construct" : ["I344A"]
        },        
        show_abf_segments = False,
        show_csv_segments = False,
        do_pubplots = False,
        show_leak_subtraction = False,
        show_Cm_estimation = False,
        show_MT_estimation = False,
        do_ramp_stuff = False,
        do_exp_kinetics = False,
        do_activation_curves = True,
        do_inst_IV = False,
        normalize = False,
        remove_after_normalize = {
                "20910002" : [-145],
                "20910005" : [-70],
                "20d10000" : [(-55, [0, -1600] )], 
                "20d10008" : [-160, -145, (-70, [0, 1500]), (-55, [0, 1500]), (-40, [0, 1500])], 
                # "20d10009": [-55, -40, -25, -10, 5, 20, 35],
                "20d10009": [-25, -10, 5],
                "21121006" : [-40],
                }, 
        save_AggregatedPDF = False 
        )
P.go(idle=False, save_csv=True, append=False)
# P.summarize(title="20d10", output=False) 
# print(P.filenames)

exit()
# path to extracted, leak-subtracted files 
leaksub_ex_path = "C:\\Users\\delbe\\Downloads\\wut\\wut\\Post_grad\\UBC\\Research\\lab\\Github_repos\\hcn-gating-kinetics\\output\\Processing\\Processed_Time_Courses\\leaksub\\extracted\\"

# files to reduce
to_reduce = ["20o16003", "20d10010", "20903005", "21121007"]

G_mu = 0 
for f in to_reduce:
        path = r"{p}{f}_leaksub_extracted.csv".format(p=leaksub_ex_path, f=f)
        # df = post.reduce_rampdt(file=path, dv=0.1, 
        #                 filename=f, save=False, re_save=False, 
        #                 output_dir = leaksub_ex_path + "reduced\\",
        #                 show=False)
        
        G_mu += post.estimate_g(path)

G_mu = G_mu/len(to_reduce)
print(" mean estimate of g = {cond}".format(cond=G_mu))
