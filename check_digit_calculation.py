# Calculate the check-digit of the container number
# input: container_code, first 10 characters of the container number
# output: check-digit, the 11th chracter
import argparse

def calculate_check_digit(container_code):
    # Mapping letters to their corresponding values
    letter_values = {
        'A': 10, 'B': 12, 'C': 13, 'D': 14, 'E': 15, 'F': 16, 'G': 17, 'H': 18, 'I': 19, 'J': 20,
        'K': 21, 'L': 23, 'M': 24, 'N': 25, 'O': 26, 'P': 27, 'Q': 28, 'R': 29, 'S': 30, 'T': 31,
        'U': 32, 'V': 34, 'W': 35, 'X': 36, 'Y': 37, 'Z': 38
    }

    # Convert the container code into its corresponding values
    values = []
    for char in container_code:
        if char.isalpha():
            values.append(letter_values[char])
        else:
            values.append(int(char))

    # Calculate the sum of the values multiplied by 2^(position-1)
    total = sum(val * (2 ** i) for i, val in enumerate(values))

    # Calculate the check digit as the remainder of division by 11
    check_digit = total % 11
    # If the remainder is 10, the check digit is 0
    if check_digit == 10:
        check_digit = 0

    return check_digit


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="calculate check digit from container number")
    parser.add_argument("--cn", default="ABCD112233")

    args = parser.parse_args()

    print(args.cn + str(calculate_check_digit(args.cn)))
