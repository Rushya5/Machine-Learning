def CountArray(Array,Target_Count):
    count = 0
    for i in range(len(Array)):
        for j in range(i+1, len(Array)):
            if Array[i] + Array[j] == Target_Count:
                count += 1
    return count

if __name__ == "__main__":
    Array = [2, 7, 4, 1, 3, 6]
    Target_Count = 10
    Pair_Count = CountArray(Array, Target_Count)
    print("Pair Count :",Pair_Count)


