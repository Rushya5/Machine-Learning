import re

S1 = "I am a great learner. I am going to have an awesome life."
#Q5
count = S1.count("am")
print("The Occurrence count of 'a':",count)
#Q6
S2 = "I work hard and shall be rewarded well"
S3 = S1 + S2
print("added string:",S3)
#Q6
Split_S3 = re.split(r'[ .]+',S3)
print(Split_S3)
Length = len(Split_S3)
print("length of the Words in array :", Length)
#Q8
remove = [Split for Split in Split_S3 if Split not in {"I","Am", "to","And"} and len(Split) <= 6]
remove_length = len(remove)
print("Removed Words from the string:",remove)
print("Length of the words after removing:",remove_length)