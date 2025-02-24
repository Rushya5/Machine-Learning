def highest_freq_count(st1):
    a1=list(set(st1))
    b1=[]
    for i in a1:
        c=0
        for j in st1:
            if i==j:
                c+=1
        b1.append(c)
    f1=list(zip(a1,b1))
    print(f1)
    print("The character with max occurances is: ",max(f1,key=lambda x:x[1]))

highest_freq_count("hippopotamus")
