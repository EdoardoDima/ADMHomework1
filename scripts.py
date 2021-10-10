#PART 1------------------------------

#Say "Hello, World!" With Python
print("Hello, World!")


#Python If-Else
#!/bin/python3

import math
import os
import random
import re
import sys
if __name__ == '__main__':
    n = int(input().strip())
    if n % 2 != 0:
        print('Weird')
    elif n % 2 == 0 and n in range(2, 6):
        print('Not Weird')
    elif n % 2 == 0 and n in range(5, 21):
        print('Weird')
    else:
        print('Not Weird')


#Arithmetic Operators
if __name__ == '__main__':
    a = int(input())
    b = int(input())
    print('%s\n%s\n%s\n' % (a+b, a-b, a*b))


#Python: Division
if __name__ == '__main__':
    a = int(input())
    b = int(input())
    print('%s\n%s' % (a//b, a/b))


#Loops
if __name__ == '__main__':
    n = int(input())
    for i in range(0, n):
        print(i**2, end = '\n')


#Write a function
def is_leap(year):
    leap = False
    if (year % 4 == 0) and (year % 100 != 0 or year % 400 == 0):
        leap = True
    return leap


#Print Function
if __name__ == '__main__':
    n = int(input())
    for i in range (1, n+1):
        print(i, end = '')


#List Comprehensions
if __name__ == '__main__':
    x = int(input())
    y = int(input())
    z = int(input())
    n = int(input())
permutations = [[a, b, c] for a in range(x+1) for b in range(y+1) for c in range(z+1) if a+b+c != n]
print(permutations)


#Find the Runner-Up Score!
if __name__ == '__main__':
    n = int(input())
    arr = map(int, input().split())
    new_set = sorted(set(arr))
    print(new_set[-2])
    

#Finding the percentage
if __name__ == '__main__':
    n = int(input())
    student_marks = {}
    for _ in range(n):
        name, *line = input().split()
        scores = list(map(float, line))
        student_marks[name] = scores
    query_name = input()
    personal_scores = (student_marks[query_name])
    #average = sum(personal_scores) / len(personal_scores)
    #print(average)
    #This up here is how I originally did it but I couldn't round it to the 2 decimal
    print("{0:.2f}".format(sum(personal_scores)/(len(personal_scores))))


#Lists
if __name__ == '__main__':
    N = int(input())
a = []
for i in range(N):       
    command = input().split()
    if command[0] == 'insert':
        a.insert(int(command[1]), int(command[2]))
    elif command[0] == 'remove':
        a.remove(int(command[1]))
    elif command[0] == 'append':
        a.append(int(command[1]))
    elif command[0] == 'sort':
        a.sort()
    elif command[0] == 'pop':
        a.pop()
    elif command[0] == 'reverse':
        a.reverse()
    elif command[0] == 'print':
        print(a)
            
        
#Tuples
if __name__ == '__main__':
    n = int(input())
    integer_list = map(int, input().split())
    t = tuple(integer_list)
    print(hash(t))
    

#Nested lists
if __name__ == '__main__':
    nested_list = []
    for _ in range(int(input())):
        name = input()
        score = float(input())
        nested_list.append([name, score])
        
    second_lowest = sorted(set([i[1] for i in nested_list]))[1]    
    second_lowest_total = [i[0] for i in nested_list if i[1] == second_lowest]
    print('\n'.join(sorted(second_lowest_total)))


#sWAP cASE
def swap_case(s):
    new_s = ''
    for letter in s:
        if letter == letter.lower(): 
            new_s += letter.upper()
        else:
            new_s += letter.lower()
    return(new_s)


#String Split and Join
def split_and_join(line):
    new_line = line.split(' ')
    new_line2 = '-'.join(new_line)
    return(new_line2)


#What's your name?
def print_full_name(first, last):
    print('Hello %s %s! You just delved into python.' % (first, last))


#Mutations
def mutate_string(string, position, character):
    new_string = string[:position] + character + string[position+1:]
    return(new_string)


#Find a string
def count_substring(string, sub_string):
    counter = 0
    for i in range(0, len(string)):
        if string[i:i+len(sub_string)] == sub_string:
            counter += 1
    return(counter)


#String Validators
if __name__ == '__main__':
    s = input()
    #print(s.isalnum(), s.isalpha(), s.isdigit(), s.islower(), s.isupper(), sep = '\n')
    '''
    I first did it like this but then noticed this fails because it returns fail if any character fails the check
    Had to search online to discover the any method
    '''
    print(any(c.isalnum() for c in s), any(c.isalpha() for c in s), any(c.isdigit() for c in s), any(c.islower() for c in s), any(c.isupper() for c in s), sep = '\n')


#Text Alignment
#Replace all ______ with rjust, ljust or center. 
thickness = int(input()) #This must be an odd number
c = 'H'
#Top Cone
for i in range(thickness):
    print((c*i).rjust(thickness-1)+c+(c*i).ljust(thickness-1))
#Top Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))
#Middle Belt
for i in range((thickness+1)//2):
    print((c*thickness*5).center(thickness*6))    
#Bottom Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))    
#Bottom Cone
for i in range(thickness):
    print(((c*(thickness-i-1)).rjust(thickness)+c+(c*(thickness-i-1)).ljust(thickness)).rjust(thickness*6))
#Text Wrap
def wrap(string, max_width):
    wrapped = textwrap.fill(string, max_width)
    return(wrapped)


#String Formatting
def print_formatted(number):
    l=len(bin(number)[2:])
    for i in range(1,number+1):
        dec=str(i)
        octal=oct(i)
        hexadec=hex(i)
        binary=bin(i)
        print(dec.rjust(l,' '),octal[2:].rjust(l,' '),hexadec[2:].upper().rjust(l,' '),binary[2:].rjust(l,' '))
        '''
        I didn't quite know where to start with this, I checked
        the discussions and I get the logic now but the solutions
        isn't mine
        '''


#Designer Door Mat
n, m = map(int, input().split(" "))
#Done by looking at the discussion section 
for i in range(n):
    pattern = ".|."
    if i < (n-1)/2:
        print((pattern * (2*i+1)).center(m, "-"))
    elif i == (n-1)/2:
        print("WELCOME".center(m, "-"))
    else:
        print((pattern * (2*(n-1-i)+1)).center(m, "-"))


#Alphabet Rangoli
import string 
'''
I didn't really know what to do with this exercise and ended up 
taking some heavy inspiration from the solutions in the discussion
'''
def print_rangoli(size):
    a = string.ascii_lowercase
    l = []
    for i in range(size):
        s = "-".join(a[i:size])
        l.append((s[::-1]+s[1:]).center(4*size-3, '-'))
    for i in range(size-1):
        print(l[size-1-i])  
    print(l[0])
    for i in range(1, size):
        print(l[i])


#Capitalize!
def solve(s):
    a = s.split()
    b = [i.capitalize() for i in a]
    for i in range(len(b)):
        s = s.replace(a[i], b[i])
    return s


#The Minion Game
def minion_game(string):
    vowels = 'AEIOU'
    stuart = 0
    kevin = 0
    for i in range(len(string)):
        if string[i] in vowels:
            kevin += len(string)-i
        else:
            stuart += len(string)-i
    if stuart > kevin:
        print('Stuart %s' %(stuart))
    elif kevin > stuart:
        print('Kevin %s' %(kevin))
    else:
        print('Draw')


#Merge the Tools!
def merge_the_tools(string, k):
    for i in range(0, len(string), k):
        a = ''
        for x in string[i:i + k]:
            if x not in a:
                a += x
        print(a)


#Introduction to sets
def average(array):
    new_set = set(array)
    avg = sum(new_set) / len(new_set)
    return(avg)


#Symmetric Difference
M = int(input())
m = input()
N = int(input())
n = input()
m_list = map(int, m.split())
n_list = map(int, n.split())
m_set = set(m_list)
n_set = set(n_list)
diff1 = m_set.difference(n_set)
diff2 = n_set.difference(m_set)
sym_diff = diff1.union(diff2)
result = list(sym_diff)
result.sort()
print(*result, sep = '\n')


#Set .add()
N = int(input())
stamp_set = set()
for i in range(N):
    stamp_set.add(input())  
#I originally put it like i = input().add(stamp_set) but on the output it told me that I couldn't use .add on strings. It works this way though  
print(len(stamp_set))


#Set .discard(), .remove() & .pop()
n = int(input())
s = set(map(int, input().split()))
N = int(input())
for i in range(N):
    command = input().split()
    if command[0] == 'remove':
        s.remove(int(command[1]))    
    elif command[0] == 'discard':
        s.discard(int(command[1]))  
    elif command[0] == 'pop':
        s.pop()       
print(sum(s))


#Set .union() Operation
n = int(input())
s = set(map(int, input().split()))
N = int(input())
for i in range(N):
    command = input().split()
    if command[0] == 'remove':
        s.remove(int(command[1]))    
    elif command[0] == 'discard':
        s.discard(int(command[1]))  
    elif command[0] == 'pop':
        s.pop()        
print(sum(s))


#Set .intersection() Operation
n = int(input())
roll_n = set(map(int, input().split()))
m = int(input())
roll_m = set(map(int, input().split()))
print(len(roll_n & roll_m))


#Set .difference() Operation
n = int(input())
roll_n = set(map(int, input().split()))
m = int(input())
roll_m = set(map(int, input().split()))
print(len(roll_n - roll_m))


#Set .symmetric_difference() Operation
n = int(input())
roll_n = set(map(int, input().split()))
m = int(input())
roll_m = set(map(int, input().split()))
print(len(roll_n ^ roll_m))


#Set mutations
a = int(input())
set_a = set(map(int, input().split()))
n = int(input())
for i in range(n):
    operation, other = input().split() 
    '''
    #I had to check the discussion for this input here 
    (I didn't put 'other' and so the sum at the end was wrong), 
    the rest I did correctly on my own
    '''
    operation_set = set(map(int, input().split()))
    if operation == 'intersection_update':
        set_a.intersection_update(operation_set)
    elif operation == 'update':
        set_a.update(operation_set)
    elif operation == 'difference_update':
        set_a.difference_update(operation_set)
    elif operation == 'symmetric_difference_update':
        set_a.symmetric_difference_update(operation_set)      
print(sum(set_a))
    

#The Captain's Room
import collections
'''
I tried to solve this using sets but I couldn't find a way, 
I checked the discussion and saw that many people suggested 
trying using collections and I tried myself
'''
K = int(input())
rooms = list(map(int, input().split()))
counter_rooms = collections.Counter(rooms)
for x, y in counter_rooms.items():
    if y == 1:
        print(x)


#Check Subset
T = int(input())
for i in range(T):
    A = int(input())
    a = set(map(int, input().split()))
    B = int(input())
    b = set(map(int, input().split()))
    subset = a.difference(b)
    if a.issubset(b):
        print('True')
    else:
        print('False')


#Check Strict Superset
a = set(map(int, input().split()))
n = int(input())
sets = []
for i in range(n):
    sets.append(set(map(int, input().split())))
counter = 0   
for i in sets:
    if a.issuperset(i):
        counter += 1   
if counter == n:
    print('True')
else:
    print('False')


#No Idea!
n, m = map(int, input().split())
arr = list(map(int, input().split()))  
A = set(map(int, input().split()))
B = set(map(int, input().split()))
happiness = 0
for i in arr:
    if i in A:
        happiness += 1
    elif i in B:
        happiness -= 1       
print(happiness)


#collections.Counter()
import collections
X = int(input())
shoes = collections.Counter(map(int, input().split()))
N = int(input())
money_earned = 0
for i in range(N):
    size, x = map(int, input().split())
    if shoes[size] > 0:
        money_earned += x
        shoes[size] -= 1      
print(money_earned)


#Collections.namedtuple()
from collections import namedtuple
n = int(input())
columns = map(str, input().split())
marks_sum = 0
students = namedtuple('student', columns)
for i in range(n):
    student = students(*input().split())
    marks_sum += int(student.MARKS)
print(marks_sum/n)


#Calendar Module
import calendar 
month, day, year = map(int, input().split())
text_days = ['MONDAY', 'TUESDAY', 'WEDNESDAY', 'THURSDAY', 'FRIDAY', 'SATURDAY', 'SUNDAY']
day = (calendar.weekday(year, month, day))
print(text_days[day])
#I know this probably isn't optimal but it's the easiest way I found to complete it


#Time Delta
import math
import os
import random
import re
import sys
from datetime import datetime
def time_delta(t1, t2):
    t1_date = datetime.strptime(t1, '%a %d %b %Y %H:%M:%S %z')
    t2_date = datetime.strptime(t2, '%a %d %b %Y %H:%M:%S %z')
    time_difference = str(abs(int((t2_date - t1_date).total_seconds())))
    return time_difference

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    t = int(input())
    for t_itr in range(t):
        t1 = input()
        t2 = input()
        delta = time_delta(t1, t2)
        fptr.write(delta + '\n')
    fptr.close()


#Exceptions
t = int(input())
for i in range(t):
    a,b=(input().split())    
    try:
        print(int(a)//int(b))
    except (ZeroDivisionError, ValueError) as e:
        print("Error Code:", e)
    

#Zipped
n, x = map(int, input().split())
marks = []
for i in range(x):
    marks.append(map(float, input().split()))   
student_marks = zip(*marks)
for i in student_marks:
    print(sum(i)/len(i))


#Athlete Sort
#!/bin/python3
from operator import itemgetter
import math
import os
import random
import re
import sys
if __name__ == '__main__':
    nm = input().split()
    n = int(nm[0])
    m = int(nm[1])
    arr = []
    for _ in range(n):
        arr.append(list(map(int, input().rstrip().split())))
    k = int(input())
    '''
    I decided to use itemgetter for this challenge since I already knew how
    it worked from a previous one
    '''   
    arr2 = sorted(arr, key=itemgetter(k)) 
    for i in arr2:
        print(*i, sep=' ') 
    

#ginortS
s =input()
lowercase = ''
uppercase = ''
odd = ''
even = ''
for i in s:
    if i.islower():
        lowercase = lowercase + i
    if i.isupper():
        uppercase = uppercase + i
    if i.isnumeric() and int(i) % 2 != 0:
        odd = odd + i
    elif i.isnumeric() and int(i) % 2 == 0:
        even = even + i           
print(''.join(sorted(lowercase))+''.join(sorted(uppercase))+''.join(sorted(odd))+''.join(sorted(even)))


#Map and Lambda Function
cube = lambda x: x*x*x
def fibonacci(n):
    fib_list = []
    if n == 0:
        fib_list = []
    elif n == 1:
        fib_list = [0]
    else:
        fib_list.append(0)
        fib_list.append(1)   
        a = 0
        b = 1
        for i in range(0, n-2):
            c = a + b
            a = b
            b = c
            fib_list.append(b)
    return fib_list


#Detect Floating Point Number
import re
test_cases = int(input())
for i in range(test_cases):
    print(bool(re.match('^[-+]?[0-9]*\.[0-9]+$', input())))


#Re.split()
regex_pattern = r"[,.]"	


#Group(), Groups() & Groupdict()
import re
s = input()
m = re.search(r"([a-zA-Z0-9])\1+", s)
print(m.group(1) if m else -1) 
'''
had to check discussions for that last bit (if m else -1) because 
it failed on some test cases without it
'''


#Re.findall() & Re.finditer()
#Checked the discussion section
import re
vowels = "aeiou"
consonants = "qwrtypsdfghjklzxcvbnm"
m = re.findall(r"(?<=[%s])([%s]{2,})[%s]" % (consonants, vowels, consonants), input(), flags = re.I)
print('\n'.join(m or ['-1']))


#Regex Substitution
#Taken from discussion
from sys import stdin
import re
n = input()
print(re.sub( r"(?<= )(&&|\|\|)(?= )", lambda x: 'and' if x.group()=='&&' else 'or', stdin.read()))


#Validating Roman Numerals
#Checked discussion
regex_pattern = r""	# Do not delete 'r'.
thousand = "(?:(M){0,3})?"
hundred  = "(?:(D?(C){0,3})|(CM)|(CD))?"
ten      = "(?:(L?(X){0,3})|(XC)|(XL))?"
unit     = "(?:(V?(I){0,3})|(IX)|(IV))?"
regex_pattern = r"^" + thousand + hundred + ten + unit + "$"


#Validating phone numbers
#Took heavy inspiration from discussion section
import re
n = int(input())
for i in range(n):
    if re.match("^[789][0-9]{9}$", input()):
        print("YES") 
    else:
         print("NO")


#Validating and Parsing Email Addresses
#Saw the solution from the discussion section
import re
n = int(input())
for i in range(n):
    x, y = input().split(' ')
    m = re.match(r'<[A-Za-z](\w|-|\.|_)+@[A-Za-z]+\.[A-Za-z]{1,3}>', y)
    if m:
        print(x,y)

#Hex Color Code
#Checked discussion for the pattern solution
import re
pattern=r'(#[0-9a-fA-F]{3,6}){1,2}[^\n ]'
for i in range(int(input())):
    for x in re.findall(pattern,input()):
        print(x)

#HTML Parser - Part 1
#Solution taken from discussion section
from html.parser import HTMLParser
class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):        
        print ('Start :',tag)
        for ele in attrs:
            print ('->',ele[0],'>',ele[1])          
    def handle_endtag(self, tag):
        print ('End   :',tag)    
    def handle_startendtag(self, tag, attrs):
        print ('Empty :',tag)
        for ele in attrs:
            print ('->',ele[0],'>',ele[1])           
MyParser = MyHTMLParser()
MyParser.feed(''.join([input().strip() for _ in range(int(input()))]))

#HTML Parser - Part 2
#Solution taken from discussion section 
from html.parser import HTMLParser
class MyHTMLParser(HTMLParser):
  def handle_comment(self, data):
    if '\n' in data:
        print('>>> Multi-line Comment')
        print(data)
    else:
        print('>>> Single-line Comment')
        print(data)

  def handle_data(self, data):
    if '\n' not in data:
        print(">>> Data")
        print(data)
html = ''       
for i in range(int(input())):
    html += input().rstrip()
    html += '\n'
parser = MyHTMLParser()
parser.feed(html)
parser.close()

#Detect HTML Tags, Attributes and Attribute Values
#Solution taken from discussion section 
from html.parser import HTMLParser
class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print(tag)
        [print('-> {} > {}'.format(*attr)) for attr in attrs]       
html = '\n'.join([input() for _ in range(int(input()))])
parser = MyHTMLParser()
parser.feed(html)
parser.close()

#Validating UID
#Solution taken from discussion section
import re
for _ in range(int(input())):
    u = ''.join(sorted(input()))
    try:
        assert re.search(r'[A-Z]{2}', u)
        assert re.search(r'\d\d\d', u)
        assert not re.search(r'[^a-zA-Z0-9]', u)
        assert not re.search(r'(.)\1', u)
        assert len(u) == 10
    except:
        print('Invalid')
    else:
        print('Valid')

#Validating Credit Card Numbers
#Checked solution in the discussion section
import re
TESTER = re.compile(
    r"^"
    r"(?!.*(\d)(-?\1){3})"
    r"[456]"
    r"\d{3}"
    r"(?:-?\d{4}){3}"
    r"$")
for _ in range(int(input().strip())):
    print("Valid" if TESTER.search(input().strip()) else "Invalid")


#Validating Postal Codes
#Solution taken from discussion section
regex_integer_in_range = r"_________"	# Do not delete 'r'.
regex_alternating_repetitive_digit_pair = r"_________"	# Do not delete 'r'.
import re
import re
s=input()
print (bool(re.match(r'^[1-9][\d]{5}$',s) and len(re.findall(r'(\d)(?=\d\1)',s))<2 ))


#XML 1 - Find the Score
def get_attr_number(node):
    counter = len(node.attrib)
    for child in node:
        counter += get_attr_number(child)
    return counter    
    '''
    Had to check the discussion for this,
    the logic behind it was easy to understand
    but I didn't fully grasp how the etree library worked
    and it didn't occurr to me to use recursion
    '''

#XML2 - Find the Maximum Depth
#This solution is taken from the discussion section
maxdepth = 0
def depth(elem, level):
    global maxdepth
    level += 1
    if level >= maxdepth:
        maxdepth = level
    for child in elem:
        depth(child, level)


#Standardize Mobile Number Using Decorators
def wrapper(f):
    def fun(l):
        return f([('+91 %s %s' % (num[-10:-5], num[-5:])) for num in l])
    return fun


#Decorators 2 - Name Directory
def person_lister(f):
    def inner(people):
        #return [f(i) for i in sorted(people, key=operator.itemgetter(2))]
        '''
        This was my original solution using itemgetter as suggested on the exercise
        page, but it didn't pass three test cases. I checked on the discussion page
        and people were sugggesting that itemgetter doesn't work on this challenge
        and to use something like this:   
        '''
        return map(f, sorted(people, key=lambda x: int(x[2])))
    return inner


#Arrays
def arrays(arr):
    array = numpy.array(arr, float)
    return(array[::-1])


#Shape and Reshape
import numpy
my_array = numpy.array(input().split(), int)
print(numpy.reshape(my_array, (3, 3)))


#Transpose and Flatten
import numpy
n, m = map(int, input().split()) 
my_array = numpy.array([input().split() for i in range(n)], int)
print (numpy.transpose(my_array))
print(my_array.flatten())


#Concatenate
import numpy
n, m, p = map(int, input().split()) 
n_array = numpy.array([input().split() for i in range(n)], int)
m_array = numpy.array([input().split() for i in range(m)], int)
print(numpy.concatenate((n_array, m_array))) 


#Zeros and Ones
import numpy
my_array = numpy.array(input().split(), int)
print(numpy.zeros(my_array, int))
print(numpy.ones(my_array, int))


#Eye and Identity
import numpy
numpy.set_printoptions(legacy='1.13')
n, m = map(int, input().split()) 
print(numpy.eye(n, m))


#Array Mathematics
import numpy
n, m = map(int, input().split())
a = numpy.array([input().split() for i in range(n)], int)
b = numpy.array([input().split() for i in range(n)], int)
print(numpy.add(a, b))
print(numpy.subtract(a, b))
print(numpy.multiply(a, b))
print(a//b) #the function .divide didn't work for some reason
print(numpy.mod(a, b))
print(numpy.power(a, b))


#Floor, Ceil and Rint
import numpy
numpy.set_printoptions(legacy='1.13')
my_array = numpy.array(input().split(), float)
print(numpy.floor(my_array))
print(numpy.ceil(my_array))
print(numpy.rint(my_array))


#Sum and Prod
import numpy
n, m = map(int, input().split()) 
my_array = numpy.array([input().split() for i in range(n)], int)
a = (numpy.sum(my_array, axis = 0))
print(numpy.prod(a))


#Min and Max
import numpy
n, m = map(int, input().split()) 
my_array = numpy.array([input().split() for i in range(n)], int)
a = numpy.min(my_array, axis = 1) 
print(numpy.max(a))


#Mean, Var, and Std
import numpy
n, m = map(int, input().split()) 
my_array = numpy.array([input().split() for i in range(n)], int)
print(numpy.mean(my_array, axis = 1))
print(numpy.var(my_array, axis = 0))
print(round(numpy.std(my_array), 11))
'''
The last output in this challenge is aproximated to what - I guess - is a non standard value 
for the numpy module, that's why I had to use the round function
'''


#Dot and Cross
import numpy
n = int(input())
a = numpy.array([input().split() for i in range(n)], int)
b = numpy.array([input().split() for i in range(n)], int)
print(numpy.dot(a, b))


#Inner and Outer
import numpy
a = numpy.array(input().split(), int)
b = numpy.array(input().split(), int)
print(numpy.inner(a, b))
print(numpy.outer(a, b))


#Polynomials
import numpy
p = list(map(float, input().split()))
x = int(input())
print(numpy.polyval(p, x))


#Linear Algebra
import numpy
n = int(input())
a = numpy.array([input().split() for i in range(n)], float)
print(round(numpy.linalg.det(a), 2)) 
'''
The challenge doesn't state this but the output should be rounded to the 
second digit after zero or the code would fail in one of the tests
'''

#PART2----------------------------


#Birhtday Cake Candles
#!/bin/python3
import math
import os
import random
import re
import sys
'''
def birthdayCakeCandles(candles):
    counter = 0
    for i in candles:
        if i == max(candles):
            counter += 1
    return counter
'''   
'''
#this up here was my original, naive algorithm, 
but the runtime was obviously OBSCENELY excessive so I had to change it
'''
def birthdayCakeCandles(candles):
    a = candles.count(max(candles))
    return a
'''
I checked Google for ways to count specific elements in a list, 
found this method and I'm not even entirely sure why it works but it does
'''
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    candles_count = int(input().strip())
    candles = list(map(int, input().rstrip().split()))
    result = birthdayCakeCandles(candles)
    fptr.write(str(result) + '\n')
    fptr.close()


#Number Line Jumps
def kangaroo(x1, v1, x2, v2):
    if x1 == x2 and v1 == v2: 
        return('YES')
        '''
        I put this line up here to optimize the runtime for edge cases
        but I don't really know if it's something useful to do or not
        '''
    elif x2 > x1 and v1 == v2:
        return('NO') #This line was to avoid a runtime error in one of the test cases
    elif (x1 - x2) % (v2 - v1) == 0 and v1>v2:
        return('YES')
    else:
        return('NO')
    
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')


#Viral Advertising
#!/bin/python3
import math
import os
import random
import re
import sys

def viralAdvertising(n):
    liked = 0
    shared = 5
    for i in range(n):
        liked += math.floor(shared/2)     
        shared = math.floor(shared/2)*3
    return liked

    if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    n = int(input().strip())
    result = viralAdvertising(n)
    fptr.write(str(result) + '\n')
    fptr.close()


#Recursive Digit Sum
#!/bin/python3
import math
import os
import random
import re
import sys

def super_d(n):
    sup_d = str(sum([int(i) for i in n])) #this is the superdigit
    return sup_d

def superDigit(n, k):
    if len(n) == 1 and k == 1: 
        return n
    else:
        p = str(int(super_d(n)) * k)   
        while len(p) > 1:
            p = super_d(p)
        return p
    '''
    I know this doesn't properly use recursion but 
    it was the best I could come up with because of the 
    somewhat strict runtimes of this challenge
    '''
    
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    first_multiple_input = input().rstrip().split()
    n = first_multiple_input[0]
    k = int(first_multiple_input[1])
    result = superDigit(n, k)
    fptr.write(str(result) + '\n')
    fptr.close()


#Insertion Sort - Part 1
#!/bin/python3
import math
import os
import random
import re
import sys
'''
def insertionSort1(n, arr):
    target = arr[-1]
    for i in range(len(arr)-2, -1, -1):  #This was my original solution but it failed test case 2
        if arr[i] > target:              #The problem was that it would fail when n was == 1 and 
            arr[i+1] = arr[i]            #it wouldn't be able to put it at index [0]
            print(*arr) 
        elif arr[i] < target:
            arr[i+1] = target
            print(*arr)
            break
'''        
def insertionSort1(n, arr):
    target = arr[n-1]
    for i in range(len(arr)-2, -2, -1):  #I checked for some hints in the discussion section and I found out
        if arr[i] > target:              #I needed to change the stop in the len() function to -2
            if i != -1:                  #And to add this double if to check wether it was reverse-traversing past index [0]
                arr[i+1] = arr[i]
                print(*arr) 
            else:          
                arr[i+1] = target
                print(*arr)   
        elif arr[i] < target:
            arr[i+1] = target
            print(*arr)
            break           

if __name__ == '__main__':
    n = int(input().strip())

    arr = list(map(int, input().rstrip().split()))

    insertionSort1(n, arr)


#Insertion Sort - Part 2
#!/bin/python3
import math
import os
import random
import re
import sys

def insertionSort2(n, arr):
    for i in range(1, len(arr)):
        for x in range(i):
            if arr[x] > arr[i]:
                arr[x], arr[i] = arr[i], arr[x]       
        print(*arr)

if __name__ == '__main__':
    n = int(input().strip())

    arr = list(map(int, input().rstrip().split()))

    insertionSort2(n, arr)



    



