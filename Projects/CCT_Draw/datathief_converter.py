import numpy as np 

def calculate_rates(T0,points):
    rates = []
    for k,x in enumerate(points['X']):
        rates.append((T0 - points['Y'][k])/points['X'][k])
    points['Rates'] = rates
    return points

def stavex_analysis():
    C_curve = {
        'X':[20.588,23.0643,59.4332,140.6534,362.2827,1153.1619],
        'Y':[601.58,691.3466,788.31,826.6521,861.316,888.6168]
    }
    Pf_curve = {
        'X':[514.7711,999.5338,1913.8901,3182.5074],
        'Y':[628.5528,633.9442,661.3259,703.3888]
    }

    Ps_curve = {
        'X':[405.1326,636.7348,1104.556,1862.6535],
        'Y':[698.2198,751.2858,787.8449,817.0792]
    }

    Ms_curve = {
        'X':[3.9927,84.267,818.2745,5505.1409],
        'Y':[233.5405,249.5463,287.6636,338.6658]
    }

    print(calculate_rates(1030.0,Ms_curve))

def H11_analysis():
    Bf_curve = {
        'X':[7794.4394,15412.8237,24982.4297,72492.9627],
        'Y':[383.3406,388.5877,386.8387,392.0857]
    }

    Bf2_curve = {
        'X':[7794.4394,15412.8237,26836.9644],
        'Y':[383.3406,330.87,304.6349]
    }

    Bs_curve = {
        'X':[7575.4607,15639.7307,26836.9644],
        'Y':[383.3406,437.5601,469.0424,539.0031]
    }

    Ms_curve = {
        'X':[15.4821,154.6032,1500.4647,7157.2094],
        'Y':[357.1054,355.3564,367.5995,392.0857]
    }

    print(calculate_rates(1050.0,Bf2_curve))


if __name__=='__main__':
    H11_analysis()