x = [1,2,3,0]
print("Input data: ", x)
def total_sum(x):
    length = len(x)
    sum = 0
    c = 0
    while c < length:
       sum = sum + x[c]
       c += 1
    return sum
    
print("Result: ", total_sum(x))