from math import *
import re
import os

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

def add_prime(list_of_primes, n):
    x = list_of_primes[len(list_of_primes) - 1]
    for i in xrange(n):
        x += 2
        got_a_prime = True
        for prime in list_of_primes:
            if(x%prime==0):
                got_a_prime = False
                break
        if(got_a_prime):
            list_of_primes.append(x)

    return list_of_primes

def prime_list(_file):
    x = []
    for line in _file:
        y = re.match('\d*',line)
        if(y!=None):
            x.append(int(y.string))
    return x

def save_prime_file(list_of_primes):
    prime_file = open(os.path.join(__location__,'primes.txt'),'w')
    for i in list_of_primes:
        prime_file.write(str(i) + "\n")
    prime_file.close()

def find_pairs_of_primes(list_of_primes,x):
    tmp = []
    if(x%2==1):
        return 0
    else:
        if(x > list_of_primes[len(list_of_primes) - 1]):
            return 0
        else:
            index = 0
            for i in xrange(len(list_of_primes)):
                if(list_of_primes[i]>x):
                    index = i
                    break

            for i in xrange(index):
                for j in xrange(index):
                    if(list_of_primes[i] + list_of_primes[j] == x):
                        tmp.append((list_of_primes[j],list_of_primes[i]))
            return tmp

prime_f = open(os.path.join(__location__,'primes.txt'),'r')
primes = prime_list(prime_f)
#primes = add_prime(primes,10000)
#save_prime_file(primes)
t = find_pairs_of_primes(primes,56944)
print t
