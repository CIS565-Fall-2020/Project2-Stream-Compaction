import csv
import matplotlib.pyplot as plt

data = []
with open('lab2.csv', 'r') as csvfile:
  spamreader = csv.reader(csvfile, delimiter=',')
  for row in spamreader:
    drow = []
    for data_str in row:
      drow.append(float(data_str))
    data.append(drow)

start = 10
end = 21

cpu_scan = []
naive_scan = []
efficient_scan = []
efficient_scan_npot = []
thrust_scan = []
thrust_scan_npot = []
cpu_compact = []
efficient_compact = []

x = []

# import pdb; pdb.set_trace()

for i in range(start, end + 1):
  x.append(i)
  row = data[i - 10]
  cpu_scan.append(row[0])
  naive_scan.append(row[2])
  efficient_scan.append(row[4])
  efficient_scan_npot.append(row[6])
  thrust_scan.append(row[6])
  thrust_scan_npot.append(row[7])
  cpu_compact.append(row[8])
  efficient_compact.append(row[11])  

fig, ax = plt.subplots()

### Scan 
# plt.plot(x, cpu_scan, label="cpu_scan")
# plt.plot(x, naive_scan, label="naive_scan")
# plt.plot(x, efficient_scan, label="work_efficient")
# plt.plot(x, thrust_scan, label="thrust_scane")
# plt.title("Scan time")

### NPOT
# plt.plot(x, efficient_scan, label="work-efficient scan")
# plt.plot(x, efficient_scan_npot, label="work-efficient scan NPOT")
# plt.title("Scan time")


### Thrust NPOT
# plt.plot(x, thrust_scan, label="thrust scan")
# plt.plot(x, thrust_scan_npot, label="thrust scan NPOT")
# plt.title("Scan time")

### Compact
# plt.plot(x, cpu_compact, label="cpu compact")
# plt.plot(x, efficient_compact, label="work efficient compact")
# plt.title("Compact time")

### GPU
plt.plot(x, naive_scan, label="naive_scan")
plt.plot(x, efficient_scan, label="work_efficient")
plt.plot(x, thrust_scan, label="thrust_scan")
plt.title("GPU Scan time")

plt.xlabel("Number (in base 2)")
plt.ylabel("Time (ms)")
plt.legend()
plt.show()