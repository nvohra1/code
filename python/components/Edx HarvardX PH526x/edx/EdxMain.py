
# sets
nums = set([1,1,2,2,3,3,3,4])
print (nums)
print(len(nums))

newSet = set([1,2,3,4])
newSetUnordered = set([1,4,2,3])
if(nums == newSet):
    print('both sets have unique values & are same')
else:
    print('both sets are different')

print ('nums ' + str(nums))
print ('newSet ' + str(newSet))
print ('newSetUnordered ' + str(newSetUnordered))

###########
print('#################2#######<class \'list\'>### List is immutable ####')
a=[1,2,3]
print(a)
a[1]=4
print(a)
print(type(a))
print('#################3###### <class \'tuple\'> ####tupple is like like that is immutable ####')
print('#################3###### <class ''tuple''> ########')
a=(1,2,3)
print(a)
#a[1]=4 will give error
print(type(a))

###########
print('#################4#######')
a = [1,2,3]
b = a
a == b
True
a is b
True

b = a[:]
a == b
True
a is b
False

###########
print('#######')
x = "Hello, world!"
y = x[5:]
print(y)

###########
print('#######')
x = 1
def my_function():
  x = 2
  print(x)
print(x)
my_function()
print(x)

###########
print('#######')

x = 1
while x < 5:
  x *= 2
print(x)

###########
print('#######')
for integer in (-1,3,5):
  if integer < 0:
   print("negative")
  else:
   print("non-negative")

###########
print('#######')
x = 'String'
y = 10
z = 5.0
print(x + x)  # print command 1
print(y + y)  # print command 2
#print(y + x)  # print command 3      Error
print(y + z)  # print command 4