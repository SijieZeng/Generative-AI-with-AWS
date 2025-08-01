# ask for name
#remove leading and trailing spaces and capitalize the first letter of each word
name = input("what is your name? ").strip().title()

# Split the name into first and last name
first, last = name.split()

# say hello to user
print(f"hello, {first}")
