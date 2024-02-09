import random as rnd

reverse = True

def randomElement(sequence):
    index = rnd.randrange(len(sequence))
    return index,sequence[index]

rnd.seed(6)
add = lambda x,y: int(x + y)
sub = lambda x,y: int(x - y)
mult = lambda x,y: int(x*y)
div = lambda x,y: int(x/y)
power = lambda x,y: int(x**y)
concat = lambda x,y: int(str(x) + str(y))

operations = [add,sub,mult,div,power,concat]
operations_names = ['add','sub','mult','div','power','concat']
operations_symbols = ['+','-','*','/','**','||']
sequence = [1,2,3,4,5,6,7,8,9]
if(reverse):
    sequence.reverse()
total = sequence[0]
total_sequence = ''
for k in sequence[1:]:
    index,op = randomElement(operations)
    tmp_string = str(total) + operations_symbols[index] + str(k)
    total = op(total,k)
    print(tmp_string + ' = ' + str(total))


