from enum import IntEnum

class TempScale(IntEnum):
    Celsius = 0
    Fahrenheit = 1
    Reaumur = 2
    Newton = 3
    Delisle = 4
    Kelvin = 5
    Romer = 6
    Rankine = 7

def TConvert(T,begin_scale,end_scale):
    if(begin_scale==end_scale):
        return T
    else:
        if(end_scale==TempScale.Celsius):
            if(begin_scale == TempScale.Kelvin):
                return T - 273.15
            elif(begin_scale == TempScale.Fahrenheit):
                return (T - 32.0)*5.0/9.0
            elif(begin_scale == TempScale.Rankine):
                return (T - 491.67)*5.0/9.0
            elif(begin_scale == TempScale.Delisle):
                return 100.0 - T*2.0/3.0
            elif(begin_scale == TempScale.Newton):
                return 100.0/33.0*T
            elif(begin_scale == TempScale.Reaumur):
                return T*5.0/4.0
            elif(begin_scale == TempScale.Romer):
                return (T - 7.5)*40.0/21.0

        elif(end_scale==TempScale.Kelvin):
            if(begin_scale == TempScale.Celsius):
                return T + 273.15
            elif(begin_scale == TempScale.Fahrenheit):
                return (T + 459.67)*5.0/9.0
            elif(begin_scale == TempScale.Rankine):
                return T*5.0/9.0
            elif(begin_scale == TempScale.Delisle):
                return 373.15 - T*2.0/3.0
            elif(begin_scale == TempScale.Newton):
                return T*100.0/33.0 + 273.15
            elif(begin_scale == TempScale.Reaumur):
                return T*5.0/4.0 + 273.15
            elif(begin_scale == TempScale.Romer):
                return (T - 7.5)*40.0/21.0 + 273.15
            
        elif(end_scale==TempScale.Fahrenheit):
            if(begin_scale == TempScale.Celsius):
                return T*9.0/5.0 + 32.0
            elif(begin_scale == TempScale.Kelvin):
                return T*9.0/5.0 - 459.67
            elif(begin_scale == TempScale.Rankine):
                return T - 459.67
            elif(begin_scale == TempScale.Delisle):
                return 212.0 - T*6.0/5.0
            elif(begin_scale == TempScale.Newton):
                return T*60.0/11.0 + 32.0
            elif(begin_scale == TempScale.Reaumur):
                return T*9.0/4.0 + 32.0
            elif(begin_scale == TempScale.Romer):
                return (T - 7.5)*24.7 + 32.0
            