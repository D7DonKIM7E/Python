first = int(input())
second = input()

for digit in second[::-1] :
  print(first*int(digit))

print(first*int(second))
