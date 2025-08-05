import csv

input_file = "./test_power.csv"
output_file = "./fixed_power.csv"

with open(input_file ,'r') as infile, open(output_file, 'w', newline='')as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    header = next(reader)
    writer.writerow(header)

    for i, row in enumerate(reader):
        row[0] = str(i)
        writer.writerow(row)
print("DONE")
