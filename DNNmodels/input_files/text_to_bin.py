import argparse

def txt_to_bin(input_file_path, output_file_path):
    # Convert plain txt to bin file
    try:
        with open(input_file_path, 'rb') as f:
            byte_data = f.read()
        binary_string = ''.join(f'{byte:08b}' for byte in byte_data)

        with open(output_file_path, 'w') as f_out:
            f_out.write(binary_string)
        return True

    except FileNotFoundError:
        print(f"Error: '{input_file_path}' was not found")
        return False

parser = argparse.ArgumentParser(description = "Convert plain text to bin")
parser.add_argument('-i', '--input', default='input.txt', help="Path to the input txt file")
parser.add_argument('-o', '--output', default='bin.txt', help="Path to the input txt file")

args = parser.parse_args()

isTrue = txt_to_bin(args.input, args.output)
if isTrue:
    print("Success")
else:
    print("Fail")
