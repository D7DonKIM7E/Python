cn = int(input())
avg = []

for _ in range(cn) : 
  n = list(map(int, input().split()))

  sum = 0
  for i in range(n[0]):
    sum += n[i+1]
  
  count = 0
  for i in range(n[0]):
    if n[i+1] > (sum/n[0]):
      count += 1
  
  avg.append((count/n[0])*100)
  n.clear()

for i in range(cn):
    print("%.3f" %avg[i]+"%")

