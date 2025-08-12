import os
import argparse

def encode_label(label, num_sm, bits):
    num_steps = (2**bits) -1
    if num_steps == 0:
        return 0
    step = num_sm // num_steps

    return label * step
def create_label_files(bin_file_path, output_dir, bits_list, num_sm):
    # read bin file and create label using given bits list
    try:
        with open(bin_file_path, 'r') as f:
            bin_string = f.read()
    except FileNotFoundError:
        print(f"Error: '{bin_file_path}' not found")
        return

    os.makedirs(output_dir, exist_ok=True)

    for bits in bits_list:
        labels = []
        for i in range(0, len(bin_string), bits):
            chunk = bin_string[i:i+bits]
            if len(chunk) == bits: # Only considering complete bit chunks
                decimal_value = int(chunk, 2)
                encoded_value = encode_label(decimal_value, num_sm, bits)
                labels.append(encoded_value)
        output_file_path = os.path.join(output_dir, f'labels{bits}bit.csv')
        header = list(range(0,num_sm+1))
        with open(output_file_path, 'w')as f:
            f.write(','.join(map(str, header)))
            f.write('\n')
            f.write(','.join(map(str,labels)))

parser = argparse.ArgumentParser(
    description = "Convert a binary file to multi-bit label files"
)
parser.add_argument('-i', '--input', default='bin.txt')
parser.add_argument('-o', '--output',help="output dir")
parser.add_argument('--num_sm', type=int, default=14)
args = parser.parse_args()

max_bits = 0
while (2**(max_bits+1)) <= (args.num_sm+1):
    max_bits += 1
bit_list  = list(range(1, max_bits + 1))
print(bit_list)
create_label_files(args.input, args.output, bit_list, args.num_sm)
