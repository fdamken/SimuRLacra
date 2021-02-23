import pyrado
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_performance(timestamp):
    return np.load(f'{pyrado.TEMP_DIR}/perf/sums_{timestamp}.npy'), np.load(f'{pyrado.TEMP_DIR}/perf/names_{timestamp}.npy')

def plot_performance(sums, names, goalReward=7000):
    perf = np.array([ [idx, np.mean(sum), np.std(sum)] for idx, sum in enumerate(sums) ])
    horiz_line_data = np.array([goalReward for _ in perf[:,0]])
    colors = ["blue" if 'teacher' in name else "red" for name in names]
    plt.figure()
    sns.set()
    for x, y, err, color in zip(perf[:,0], perf[:,1], perf[:,2], colors):
        plt.errorbar(x, y, err, linestyle='None', marker='o', capsize=4, color=color)

    plt.xticks(perf[:,0], names, rotation=90)
    plt.plot(perf[:,0], horiz_line_data, 'r--', color="green", label="solved") 
    plt.ylabel("AVG cum reward")
    plt.legend()
    plt.show()




if __name__ == "__main__":
    
    # MEAN, STD, MIN, MAX, MEDIAN
    perf_16 = np.array([   [0 , 1457.056253211687 , 1741.8858856697886 , 262.10796440972223 , 7279.3271484375 , 735.5743713378906       ]    
                    ,   [1 , 3023.0676960697224 , 2569.919219981348 , 234.2699462890625 , 7361.181640625 , 1705.7252807617188]
                    ,   [2 , 1990.4416506696427 , 665.5065922240642 , 785.6869419642857 , 3614.63427734375 , 1715.3070678710938]
                    ,   [3 , 234.3679110150946 , 27.465919583881078 , 169.9569302262931 , 325.83551025390625 , 230.99917300981966]
                    ,   [4 , 3871.74013247324 , 2459.2701318700565 , 411.10224609375 , 7254.736328125 , 3465.8651123046875]
                    ,   [5 , 5457.77137562779 , 2325.293664082956 , 621.9422200520834 , 7405.12890625 , 7213.306884765625]
                    ,   [6 , 5161.982745279947 , 2238.2839623351233 , 1033.1259765625 , 7220.912109375 , 6973.61083984375]
                    ,   [7 , 5811.66101196289 , 1954.2423721013508 , 1534.4410400390625 , 7355.9482421875 , 7032.908447265625]
                    ,   [8 , 2315.9912565414593 , 2035.9934063573955 , 321.56207275390625 , 7457.283203125 , 1433.7962890625]
                    ,   [9 , 6189.972897135417 , 1695.9022598100476 , 1511.435546875 , 7371.22509765625 , 7077.74365234375]
                    ,   [10,  6603.527403971354 , 1362.257433899913 , 1601.2080078125 , 7365.4375 , 7134.4521484375]
                    ,   [11,  6788.997939453125 , 617.435041608416 , 2968.1650390625 , 7332.53857421875 , 6877.10986328125]
                    ,   [12,  6440.9896875 , 561.0623700621123 , 4991.1943359375 , 7309.255859375 , 6567.1826171875]
                    ,   [13,  2799.8634502185637 , 1671.4739722032207 , 724.08740234375 , 7205.212890625 , 2200.562744140625]
                    ,   [14,  2011.8399658476576 , 2131.01776864805 , 184.31144575639203 , 7366.0166015625 , 1333.7095703125]
                    ,   [15,  429.92245800438826 , 138.16036837936477 , 245.92855674342104 , 1068.408203125 , 401.59547061011904]
                    ,   [16, 5533.6359248383305, 2079.3770669397923 ,184.4332361391546 , 7238.02932818355 , 6437.613689801959]    #student
                ])  
    
    names_16 =[ f'teacher {t}' for t in range(len(perf_16)-1) ]
    names_16.append('student')

    #teacher_8
    perf_8 = np.array([   [0,  4286.8188708961125    , 2429.295767995145 , 563.76259765625       , 7277.6640625      , 3524.7752685546875    ]
        ,[1,  5024.408484409877     , 2244.7105922259198,  755.3080357142857    , 7297.86328125     , 6800.9287109375]
        ,[2,  4781.027062959217     , 2409.310854490281 , 680.9456176757812     , 7239.302734375    , 3561.618896484375]
        ,[3,  5314.705980526878     , 2340.499707123239 , 807.3968505859375     , 7403.1611328125   , 7129.25390625]
        ,[4,  3686.0701989499485    , 2279.7309339775707,  471.0094105113636    , 7258.470703125    , 3450.1146240234375]
        ,[5,  2499.4549628523077    , 2205.636222663871 , 275.63605813419116    , 7319.53857421875  , 1722.4091186523438]
        ,[6,  4505.052232340495     , 2295.108964274523 , 849.1624348958334     , 7338.365234375    , 3482.9237060546875]
        ,[7,  5458.91764351981      , 2109.5437227524203,  887.5000697544643    , 7294.2236328125   , 7076.64453125]
        ,[8, 5165.306386376839 , 1012.3684624261585 , 288.0529235942477 , 7242.72953417727 , 5190.278203093616    ] #student
        ])
    
    names_8 =[ f'teacher {t}' for t in range(len(perf_8)-1) ]
    names_8.append('student')


    #plot_performance(perf_8,names_8)
    #plot_performance(perf_16,names_16)

    sums, names = load_performance('2021-01-16_13:41:39')
    plot_performance(sums, names)

"""
horiz_line_data = np.array([7000 for i in perf[:,0]])

plt.figure()
plt.errorbar(perf[:-1,0], perf[:-1,1], yerr=perf[:-1,2], linestyle='None', marker='o', capsize=4, label='teacher mean (std)', color='blue')
plt.errorbar(perf[-1,0], perf[-1,1], yerr=perf[-1,2], linestyle='None', marker='o', capsize=4, label='student mean (std)', color='red')
plt.plot(perf[:,0], horiz_line_data, 'r--', color="green", label="solved") 
plt.legend()
plt.ylabel("AVG cum reward")
plt.show()


"""


""" 20210115_2
Endsumme ( teacher_real 0 from 100 reps): MEAN = 1457.056253211687 STD = 1741.8858856697886 MIN = 262.10796440972223 MAX = 7279.3271484375 MEDIAN = 735.5743713378906
Endsumme ( teacher_real 1 from 100 reps): MEAN = 3023.0676960697224 STD = 2569.919219981348 MIN = 234.2699462890625 MAX = 7361.181640625 MEDIAN = 1705.7252807617188
Endsumme ( teacher_real 2 from 100 reps): MEAN = 1990.4416506696427 STD = 665.5065922240642 MIN = 785.6869419642857 MAX = 3614.63427734375 MEDIAN = 1715.3070678710938
Endsumme ( teacher_real 3 from 100 reps): MEAN = 234.3679110150946 STD = 27.465919583881078 MIN = 169.9569302262931 MAX = 325.83551025390625 MEDIAN = 230.99917300981966
Endsumme ( teacher_real 4 from 100 reps): MEAN = 3871.74013247324 STD = 2459.2701318700565 MIN = 411.10224609375 MAX = 7254.736328125 MEDIAN = 3465.8651123046875
Endsumme ( teacher_real 5 from 100 reps): MEAN = 5457.77137562779 STD = 2325.293664082956 MIN = 621.9422200520834 MAX = 7405.12890625 MEDIAN = 7213.306884765625
Endsumme ( teacher_real 6 from 100 reps): MEAN = 5161.982745279947 STD = 2238.2839623351233 MIN = 1033.1259765625 MAX = 7220.912109375 MEDIAN = 6973.61083984375
Endsumme ( teacher_real 7 from 100 reps): MEAN = 5811.66101196289 STD = 1954.2423721013508 MIN = 1534.4410400390625 MAX = 7355.9482421875 MEDIAN = 7032.908447265625
Endsumme ( teacher_real 8 from 100 reps): MEAN = 2315.9912565414593 STD = 2035.9934063573955 MIN = 321.56207275390625 MAX = 7457.283203125 MEDIAN = 1433.7962890625
Endsumme ( teacher_real 9 from 100 reps): MEAN = 6189.972897135417 STD = 1695.9022598100476 MIN = 1511.435546875 MAX = 7371.22509765625 MEDIAN = 7077.74365234375
Endsumme ( teacher_real 10 from 100 reps): MEAN = 6603.527403971354 STD = 1362.257433899913 MIN = 1601.2080078125 MAX = 7365.4375 MEDIAN = 7134.4521484375
Endsumme ( teacher_real 11 from 100 reps): MEAN = 6788.997939453125 STD = 617.435041608416 MIN = 2968.1650390625 MAX = 7332.53857421875 MEDIAN = 6877.10986328125
Endsumme ( teacher_real 12 from 100 reps): MEAN = 6440.9896875 STD = 561.0623700621123 MIN = 4991.1943359375 MAX = 7309.255859375 MEDIAN = 6567.1826171875
Endsumme ( teacher_real 13 from 100 reps): MEAN = 2799.8634502185637 STD = 1671.4739722032207 MIN = 724.08740234375 MAX = 7205.212890625 MEDIAN = 2200.562744140625
Endsumme ( teacher_real 14 from 100 reps): MEAN = 2011.8399658476576 STD = 2131.01776864805 MIN = 184.31144575639203 MAX = 7366.0166015625 MEDIAN = 1333.7095703125
Endsumme ( teacher_real 15 from 100 reps): MEAN = 429.92245800438826 STD = 138.16036837936477 MIN = 245.92855674342104 MAX = 1068.408203125 MEDIAN = 401.59547061011904

Endsumme ( student_before_real from 100 reps): MEAN = 223.84089289295164 STD = 19.67185345313823 MIN = 180.2774861653646 MAX = 268.4387394831731 MEDIAN = 224.67336989182692

Endsumme (student_after from 100 reps): MEAN = 5590.898537998371 STD = 2180.4279308208265 MIN = 428.1935547538154 MAX = 7232.392024033101 MEDIAN = 6740.698948677419
"""
""" 20210115_2
Endsumme (student_after from 100 reps): MEAN = 5655.006931339948 STD = 1814.2686164368863 MIN = 446.6063265898098 MAX = 7232.147710608331 MEDIAN = 6306.515245162642


student_16
Endsumme (student_after from 1000 reps): MEAN = 5533.6359248383305 VAR = 4323808.986515133 STD = 2079.3770669397923 MIN = 184.4332361391546 MAX = 7238.02932818355 MEDIAN = 6437.613689801959


student_8
Endsumme (student_after from 1000 reps): MEAN = 5165.306386376839 VAR = 1024889.9037151043 STD = 1012.3684624261585 MIN = 288.0529235942477 MAX = 7242.72953417727 MEDIAN = 5190.278203093616
"""