import numpy as np
import pandas as pd
from statistics import mode
import math

s = '''
    1  2.0187462 0.14625955     A
    2  1.8157475 1.92205393     A
    3  0.6286695 2.96856634     A
    4  1.4008323 2.18492596     A
    5  2.2945451 0.62005642     A
    6  2.3897943 0.56448564     A
    7  0.7919238 2.36208723     A
    8  1.6363240 0.24091325     A
    9  0.3733273 1.67545599     A
    10 1.7435216 1.34843701     A
    11 3.1017795 3.08655140     A
    12 2.7557815 1.23745512     A
    13 1.7617664 1.17133746     A
    14 2.9874447 2.83447390     A
    15 2.7413901 1.03234801     A
    16 4.0893473 0.97118466     B
    17 3.0450561 1.23252515     B
    18 3.8048496 0.69879132     B
    19 4.9255213 0.32238542     B
    20 4.4829785 1.65522764     B
    21 3.4036894 0.59936245     B
    22 1.8147132 0.66544343     B
    23 3.3251341 2.36795395     B
    24 1.8809388 3.13776710     B
    25 2.7348020 1.50581926     B
    26 3.6263384 1.78634238     B
    27 3.3124446 0.09778806     B
    28 3.1278412 1.53289699     B
    29 3.8982390 0.35410575     B
    30 3.7462195 1.29098749     B
    '''

def build_data_basic():
    l = s.split()
    data = []
    for i in range(0, len(l), 4):
        data.append(([float(l[i+1]), float(l[i+2])], l[i+3]))
    return data

def build_data_vectorized():
    l = s.split()
    data = []
    for i in range(0, len(l), 4):
        row = [l[i+1], l[i+2], l[i+3]]
        data.append(np.array(row))
    data = np.array(data)
    data = pd.DataFrame(data)
    data.iloc[:, 0] = pd.to_numeric(data.iloc[:, 0])
    data.iloc[:, 1] = pd.to_numeric(data.iloc[:, 1])
    return pd.DataFrame(data)

iris =  '''
        1           3.5          1.4     setosa
        2           3.0          1.4     setosa
        3           3.2          1.3     setosa
        4           3.1          1.5     setosa
        5           3.6          1.4     setosa
        6           3.9          1.7     setosa
        7           3.4          1.4     setosa
        8           3.4          1.5     setosa
        9           2.9          1.4     setosa
        10          3.1          1.5     setosa
        11          3.7          1.5     setosa
        12          3.4          1.6     setosa
        13          3.0          1.4     setosa
        14          3.0          1.1     setosa
        15          4.0          1.2     setosa
        16          4.4          1.5     setosa
        17          3.9          1.3     setosa
        18          3.5          1.4     setosa
        19          3.8          1.7     setosa
        20          3.8          1.5     setosa
        21          3.4          1.7     setosa
        22          3.7          1.5     setosa
        23          3.6          1.0     setosa
        24          3.3          1.7     setosa
        25          3.4          1.9     setosa
        26          3.0          1.6     setosa
        27          3.4          1.6     setosa
        28          3.5          1.5     setosa
        29          3.4          1.4     setosa
        30          3.2          1.6     setosa
        31          3.1          1.6     setosa
        32          3.4          1.5     setosa
        33          4.1          1.5     setosa
        34          4.2          1.4     setosa
        35          3.1          1.5     setosa
        36          3.2          1.2     setosa
        37          3.5          1.3     setosa
        38          3.6          1.4     setosa
        39          3.0          1.3     setosa
        40          3.4          1.5     setosa
        41          3.5          1.3     setosa
        42          2.3          1.3     setosa
        43          3.2          1.3     setosa
        44          3.5          1.6     setosa
        45          3.8          1.9     setosa
        46          3.0          1.4     setosa
        47          3.8          1.6     setosa
        48          3.2          1.4     setosa
        49          3.7          1.5     setosa
        50          3.3          1.4     setosa
        51          3.2          4.7 versicolor
        52          3.2          4.5 versicolor
        53          3.1          4.9 versicolor
        54          2.3          4.0 versicolor
        55          2.8          4.6 versicolor
        56          2.8          4.5 versicolor
        57          3.3          4.7 versicolor
        58          2.4          3.3 versicolor
        59          2.9          4.6 versicolor
        60          2.7          3.9 versicolor
        61          2.0          3.5 versicolor
        62          3.0          4.2 versicolor
        63          2.2          4.0 versicolor
        64          2.9          4.7 versicolor
        65          2.9          3.6 versicolor
        66          3.1          4.4 versicolor
        67          3.0          4.5 versicolor
        68          2.7          4.1 versicolor
        69          2.2          4.5 versicolor
        70          2.5          3.9 versicolor
        71          3.2          4.8 versicolor
        72          2.8          4.0 versicolor
        73          2.5          4.9 versicolor
        74          2.8          4.7 versicolor
        75          2.9          4.3 versicolor
        76          3.0          4.4 versicolor
        77          2.8          4.8 versicolor
        78          3.0          5.0 versicolor
        79          2.9          4.5 versicolor
        80          2.6          3.5 versicolor
        81          2.4          3.8 versicolor
        82          2.4          3.7 versicolor
        83          2.7          3.9 versicolor
        84          2.7          5.1 versicolor
        85          3.0          4.5 versicolor
        86          3.4          4.5 versicolor
        87          3.1          4.7 versicolor
        88          2.3          4.4 versicolor
        89          3.0          4.1 versicolor
        90          2.5          4.0 versicolor
        91          2.6          4.4 versicolor
        92          3.0          4.6 versicolor
        93          2.6          4.0 versicolor
        94          2.3          3.3 versicolor
        95          2.7          4.2 versicolor
        96          3.0          4.2 versicolor
        97          2.9          4.2 versicolor
        98          2.9          4.3 versicolor
        99          2.5          3.0 versicolor
        100         2.8          4.1 versicolor
        101         3.3          6.0  virginica
        102         2.7          5.1  virginica
        103         3.0          5.9  virginica
        104         2.9          5.6  virginica
        105         3.0          5.8  virginica
        106         3.0          6.6  virginica
        107         2.5          4.5  virginica
        108         2.9          6.3  virginica
        109         2.5          5.8  virginica
        110         3.6          6.1  virginica
        111         3.2          5.1  virginica
        112         2.7          5.3  virginica
        113         3.0          5.5  virginica
        114         2.5          5.0  virginica
        115         2.8          5.1  virginica
        116         3.2          5.3  virginica
        117         3.0          5.5  virginica
        118         3.8          6.7  virginica
        119         2.6          6.9  virginica
        120         2.2          5.0  virginica
        121         3.2          5.7  virginica
        122         2.8          4.9  virginica
        123         2.8          6.7  virginica
        124         2.7          4.9  virginica
        125         3.3          5.7  virginica
        126         3.2          6.0  virginica
        127         2.8          4.8  virginica
        128         3.0          4.9  virginica
        129         2.8          5.6  virginica
        130         3.0          5.8  virginica
        131         2.8          6.1  virginica
        132         3.8          6.4  virginica
        133         2.8          5.6  virginica
        134         2.8          5.1  virginica
        135         2.6          5.6  virginica
        136         3.0          6.1  virginica
        137         3.4          5.6  virginica
        138         3.1          5.5  virginica
        139         3.0          4.8  virginica
        140         3.1          5.4  virginica
        141         3.1          5.6  virginica
        142         3.1          5.1  virginica
        143         2.7          5.1  virginica
        144         3.2          5.9  virginica
        145         3.3          5.7  virginica
        146         3.0          5.2  virginica
        147         2.5          5.0  virginica
        148         3.0          5.2  virginica
        149         3.4          5.4  virginica
        150         3.0          5.1  virginica
        '''

def build_iris():
    l = iris.split()
    data = []
    for i in range(0, len(l), 4):
        row = [l[i+1], l[i+2], l[i+3]]
        data.append(np.array(row))
    data = np.array(data)
    data = pd.DataFrame(data)
    data.iloc[:, 0] = pd.to_numeric(data.iloc[:, 0])
    data.iloc[:, 1] = pd.to_numeric(data.iloc[:, 1])
    return pd.DataFrame(data)
