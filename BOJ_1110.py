tmp = n = int(input())
count = 0

while True :
  n1 = tmp%10
  n2 = tmp//10
  output = n1 + n2
  count += 1
  tmp = int(str(tmp%10)+str(output%10))

  if (n==tmp):
    break

print(count)
