from math import *

angles_2theta = [43.0,50.0,90.0]

miller_indexes = [(1,1,1),(2,0,0),(3,1,1)]

def mod(x):
    return (x[0]**2 + x[1]**2 + x[2]**2)**(0.5)

def deg_to_rad(angle):
    return pi*angle/180.0

a = 3.597 # Angstrons
sum = 0.0
for i in xrange(len(angles_2theta)):
    sum += sin(deg_to_rad(angles_2theta[i]/2.0))/mod(miller_indexes[i])

sum *= 2.0*a

print sum

n_max = 20
sum_n = range(len(miller_indexes),n_max)
l = []
for i in sum_n:
    l.append(sum/i)

n = []
for i in range(len(l)):
    s = []
    for j in range(len(miller_indexes)):
        s.append(2.0*a*sin(deg_to_rad(angles_2theta[j]/2.0))/mod(miller_indexes[j])/l[i])
    n.append(s)

print n
print l

bool_l = []
for i in range(len(l)):
    n_choice = 1
    error = 2.0
    for k in range(0,n_max):
        for j in range(len(miller_indexes)):
            if(abs(n[i,j] - k) < error):
                n_choice = k
                error = abs(n[i,j] - k)
    
        
        
